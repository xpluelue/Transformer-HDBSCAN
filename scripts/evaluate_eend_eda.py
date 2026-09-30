#!/usr/bin/env python3
"""Evaluate a chronological radar EEND-EDA checkpoint on 1024-pulse windows."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from experiment_config import load_config_defaults
from torch.utils.data import DataLoader
from train_eend_eda import (EDA_METRIC_NAMES, summarize_eda_validation,
                            validate_eda)
from train_transformer_metric import (ExactDistributedSampler,
                                      ExhaustivePulseWindowDataset,
                                      create_control_group,
                                      distributed_context,
                                      install_stop_signal_handlers,
                                      resolve_h5_files, stop_requested)

from turing_deinterleaving_challenge.models import (PDWStandardizer,
                                                    RadarEENDEDA)


def summarize_scores(
    local_scores: dict[str, list[float]],
    control_group: object | None,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]] | None:
    """Gather every DDP score shard and calculate distribution summaries."""
    score_shards: list[dict[str, list[float]] | None] | None
    if control_group is None:
        score_shards = [local_scores]
    else:
        score_shards = (
            [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        )
        dist.gather_object(
            local_scores,
            score_shards,
            dst=0,
            group=control_group,
        )
        if dist.get_rank() != 0:
            return None
    assert score_shards is not None
    complete_shards = [shard for shard in score_shards if shard is not None]
    merged = {
        name: [
            value
            for shard in complete_shards
            for value in shard.get(name, [])
        ]
        for name in EDA_METRIC_NAMES
    }
    summary = {
        name: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in merged.items()
        if values
    }
    return summary, merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--min-emitters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("results/transformer_eda_test.json")
    )
    parser.add_argument("--config", type=Path, default=None)
    valid_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(
        **load_config_defaults(
            argv,
            sections=("evaluation",),
            valid_destinations=valid_destinations,
        )
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    install_stop_signal_handlers()
    if args.batch_size <= 0 or args.n_jobs < 0 or args.min_emitters <= 0:
        raise ValueError("invalid EEND-EDA evaluation configuration")
    device, _, rank, world_size, is_distributed = distributed_context(args.device)
    if is_distributed and args.batch_size % world_size:
        raise ValueError("--batch-size must be divisible by the DDP world size")
    control_group = create_control_group(is_distributed)
    try:
        checkpoint = torch.load(
            args.checkpoint.expanduser().resolve(), map_location="cpu"
        )
        if checkpoint.get("normalization") != "per_window":
            raise ValueError("checkpoint does not use per-window normalization")
        if checkpoint.get("assignment") != "hungarian_softmax":
            raise ValueError("checkpoint is not a radar EEND-EDA model")
        model = RadarEENDEDA(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        files = resolve_h5_files(args.test_dir.expanduser().resolve(), "test")
        dataset = ExhaustivePulseWindowDataset(
            files, args.window_length, PDWStandardizer()
        )
        if rank == 0:
            print(
                f"Evaluating {len(dataset):,} ordered EEND-EDA candidates on "
                f"{world_size} rank(s) (min_emitters={args.min_emitters})."
            )
        sampler = (
            ExactDistributedSampler(
                dataset,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                seed=args.seed,
            )
            if is_distributed
            else None
        )
        loader_kwargs: dict[str, object] = {}
        if args.n_jobs:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
                multiprocessing_context="spawn",
            )
        loader = DataLoader(
            dataset,
            batch_size=(
                args.batch_size // world_size
                if is_distributed
                else args.batch_size
            ),
            sampler=sampler,
            num_workers=args.n_jobs,
            pin_memory=device.type == "cuda",
            **loader_kwargs,
        )
        result = validate_eda(
            model,
            loader,
            device,
            args.min_emitters,
            should_stop=stop_requested,
            show_progress=rank == 0,
            collect_window_scores=True,
        )
        interrupted = result is None
        if is_distributed:
            interrupted_tensor = torch.tensor(int(interrupted or stop_requested()))
            dist.all_reduce(
                interrupted_tensor, op=dist.ReduceOp.MAX, group=control_group
            )
            interrupted = bool(interrupted_tensor.item())
        if interrupted:
            if rank == 0:
                print("Test evaluation interrupted; no result file was written.")
            return
        assert result is not None
        mean_summary, window_count, skipped_partial, skipped_low_emitter = (
            summarize_eda_validation(
                result, control_group if is_distributed else None
            )
        )
        local_scores = result[4]
        if local_scores is None:
            raise RuntimeError("test evaluation did not collect per-window scores")
        score_result = summarize_scores(
            local_scores, control_group if is_distributed else None
        )
        if rank != 0:
            return
        assert score_result is not None
        score_summary, scores = score_result
        if len(scores["V-measure"]) != window_count:
            raise RuntimeError("gathered score count does not match window count")
        if not np.isclose(
            score_summary["V-measure"]["mean"], mean_summary["V-measure"]
        ):
            raise RuntimeError("gathered and distributed V-measure means differ")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "data_dir": str(files[0].parent),
                    "split": "test",
                    "model": "chronological_radar_eend_eda",
                    "window_length": args.window_length,
                    "min_emitters": args.min_emitters,
                    "max_attractors": model.max_attractors,
                    "existence_threshold": model.existence_threshold,
                    "runs": [
                        {
                            "summary": score_summary,
                            "scores": scores,
                        }
                    ],
                    "skipped_partial": skipped_partial,
                    "skipped_low_emitter": skipped_low_emitter,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "test V-measure: "
            f"mean={score_summary['V-measure']['mean']:.4f}, "
            f"median={score_summary['V-measure']['median']:.4f}; "
            f"count_MAE={score_summary['Count MAE']['mean']:.3f}; "
            f"count_accuracy={score_summary['Count Accuracy']['mean']:.3f}; "
            f"windows={window_count}; saved={args.output}"
        )
    finally:
        if is_distributed:
            if control_group is not None:
                dist.destroy_process_group(control_group)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
