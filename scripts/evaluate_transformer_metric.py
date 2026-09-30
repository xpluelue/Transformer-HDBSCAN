#!/usr/bin/env python3
"""Evaluate a saved Transformer-metric checkpoint with per-window HDBSCAN."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from turing_deinterleaving_challenge.models import PDWStandardizer, TransformerMetricEncoder
from train_transformer_metric import (
    ExhaustivePulseWindowDataset,
    ExactDistributedSampler,
    create_control_group,
    distributed_context,
    install_stop_signal_handlers,
    resolve_h5_files,
    stop_requested,
    summarize_validation,
    validate,
)
from experiment_config import load_config_defaults


def summarize_scores(
    local_scores: dict[str, list[float]],
    control_group: object | None,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]] | None:
    """Gather every DDP shard and reproduce the raw baseline statistics."""
    score_shards: list[dict[str, list[float]] | None] | None
    if control_group is None:
        score_shards = [local_scores]
    else:
        score_shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
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
        for name in local_scores
    }
    return (
        {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in merged.items()
            if values
        },
        merged,
    )


def prediction_shard_path(output: Path, rank: int) -> Path:
    """Return a rank-local temporary path next to the final prediction file."""
    return output.with_name(f".{output.name}.rank-{rank}.npz")


def save_prediction_shard(
    output: Path,
    predictions: dict[str, np.ndarray],
) -> None:
    """Persist one rank's predictions before rank zero performs the merge."""
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **predictions)


def merge_prediction_shards(
    output: Path,
    shard_paths: list[Path],
    dataset: ExhaustivePulseWindowDataset,
    checkpoint_path: Path,
    created_at: str,
) -> int:
    """Merge DDP prediction shards in original dataset-window order."""
    shard_arrays: list[dict[str, np.ndarray]] = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            shard_arrays.append(
                {
                    "window_indices": np.asarray(shard["window_indices"]),
                    "predicted_labels": np.asarray(shard["predicted_labels"]),
                    "true_labels": np.asarray(shard["true_labels"]),
                }
            )

    window_indices = np.concatenate(
        [shard["window_indices"] for shard in shard_arrays]
    )
    predicted_labels = np.concatenate(
        [shard["predicted_labels"] for shard in shard_arrays]
    )
    true_labels = np.concatenate(
        [shard["true_labels"] for shard in shard_arrays]
    )
    if not (
        len(window_indices) == len(predicted_labels) == len(true_labels)
    ):
        raise RuntimeError("prediction shard arrays have inconsistent lengths")
    if len(np.unique(window_indices)) != len(window_indices):
        raise RuntimeError("prediction shards contain duplicate window indices")

    order = np.argsort(window_indices)
    window_indices = window_indices[order]
    predicted_labels = predicted_labels[order]
    true_labels = true_labels[order]
    locations = np.asarray(
        [dataset.window_location(int(index)) for index in window_indices],
        dtype=np.int64,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temporary_output,
        format_version=np.asarray(1, dtype=np.int64),
        created_at=np.asarray(created_at),
        checkpoint=np.asarray(str(checkpoint_path)),
        window_length=np.asarray(dataset.window_length, dtype=np.int64),
        noise_label=np.asarray(-1, dtype=np.int64),
        source_files=np.asarray([str(path) for path in dataset.files]),
        window_indices=window_indices,
        file_indices=locations[:, 0],
        pulse_starts=locations[:, 1],
        predicted_labels=predicted_labels,
        true_labels=true_labels,
    )
    temporary_output.replace(output)
    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)
    return len(window_indices)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Concurrent HDBSCAN workers per evaluation rank (default: 0).",
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument(
        "--min-emitters",
        type=int,
        default=2,
        help=(
            "Minimum unique true emitters in a full test window (default: 2). "
            "Matches evaluate_hdbscan_scan.py."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Use cuda under torchrun for DDP evaluation, or cuda:0,1 for "
            "single-process DataParallel evaluation."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/transformer_metric_test.json")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help=(
            "Compressed NPZ prediction path. By default, write "
            "<checkpoint_stem>_predictions.npz next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file whose evaluation section provides command defaults.",
    )
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
        raise ValueError(
            "--batch-size and --min-emitters must be positive; --n-jobs cannot be negative"
        )
    device, device_ids, rank, world_size, is_distributed = distributed_context(args.device)
    if is_distributed and args.batch_size % world_size:
        raise ValueError("--batch-size must be divisible by the DDP world size")
    control_group = create_control_group(is_distributed)
    try:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        predictions_output = (
            args.predictions_output.expanduser().resolve()
            if args.predictions_output is not None
            else checkpoint_path.with_name(f"{checkpoint_path.stem}_predictions.npz")
        )
        if predictions_output == checkpoint_path:
            raise ValueError("--predictions-output must not overwrite the checkpoint")
        if predictions_output == args.output.expanduser().resolve():
            raise ValueError(
                "--predictions-output and --output must refer to different files"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("normalization") != "per_window":
            raise ValueError(
                "checkpoint uses an earlier normalization baseline; retrain it with "
                "the current per-window normalization implementation"
            )
        config = checkpoint["model_config"]
        standardizer = PDWStandardizer()
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        encoder = TransformerMetricEncoder(**config).to(device)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        model: TransformerMetricEncoder | DataParallel = encoder
        if device_ids is not None:
            model = DataParallel(encoder, device_ids=device_ids, output_device=device_ids[0])
        model.eval()

        files = resolve_h5_files(args.test_dir.expanduser().resolve(), "test")
        dataset = ExhaustivePulseWindowDataset(files, args.window_length, standardizer)
        if rank == 0:
            print(
                f"Evaluating eligible full test windows from {len(dataset):,} ordered candidates "
                f"on {world_size} rank(s) (min_emitters={args.min_emitters})."
            )
        sampler = (
            ExactDistributedSampler(
                dataset, rank=rank, world_size=world_size, shuffle=False, seed=args.seed
            )
            if is_distributed
            else None
        )
        local_window_indices = (
            list(iter(sampler)) if sampler is not None else list(range(len(dataset)))
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size // world_size if is_distributed else args.batch_size,
            sampler=sampler,
            # HDBSCAN owns the CPU budget. The dataset caches main-process HDF5
            # handles, so loader workers would only oversubscribe the CPU.
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        result = validate(
            model,
            loader,
            device,
            args.min_cluster_size,
            args.min_emitters,
            None if args.n_jobs == 0 else args.n_jobs,
            should_stop=stop_requested,
            show_progress=rank == 0,
            collect_window_scores=True,
            collect_predictions=True,
            window_indices=local_window_indices,
        )
        interrupted = result is None
        if is_distributed:
            interrupted_tensor = torch.tensor(int(interrupted or stop_requested()))
            dist.all_reduce(interrupted_tensor, op=dist.ReduceOp.MAX, group=control_group)
            interrupted = bool(interrupted_tensor.item())
        if interrupted:
            if rank == 0:
                print("Test evaluation interrupted; no result file was written.")
            return
        assert result is not None
        mean_summary, window_count, skipped_partial, skipped_low_emitter = (
            summarize_validation(result, control_group if is_distributed else None)
        )
        local_scores = result[4]
        if local_scores is None:
            raise RuntimeError("test evaluation did not collect per-window scores")
        local_predictions = result[5]
        if local_predictions is None:
            raise RuntimeError("test evaluation did not collect predictions")
        score_result = summarize_scores(
            local_scores,
            control_group if is_distributed else None,
        )
        shard_paths = [
            prediction_shard_path(predictions_output, shard_rank)
            for shard_rank in range(world_size)
        ]
        save_prediction_shard(shard_paths[rank], local_predictions)
        if is_distributed:
            dist.barrier(group=control_group)
        if rank != 0:
            return
        assert score_result is not None
        score_summary, scores = score_result
        if len(scores["V-measure"]) != window_count:
            raise RuntimeError("gathered score count does not match evaluated window count")
        if not np.isclose(
            score_summary["V-measure"]["mean"], mean_summary["V-measure"]
        ):
            raise RuntimeError("gathered score mean does not match distributed mean")
        created_at = datetime.now(UTC).isoformat()
        prediction_count = merge_prediction_shards(
            predictions_output,
            shard_paths,
            dataset,
            checkpoint_path,
            created_at,
        )
        if prediction_count != window_count:
            raise RuntimeError("saved prediction count does not match evaluated window count")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "created_at": created_at,
                    "data_dir": str(files[0].parent),
                    "split": "test",
                    "window_length": args.window_length,
                    "min_emitters": args.min_emitters,
                    "max_eval_per_epsilon": None,
                    "n_jobs": args.n_jobs * world_size,
                    "predictions_file": str(predictions_output),
                    "runs": [
                        {
                            "epsilon": 0.0,
                            "summary": score_summary,
                            "scores": scores,
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "test V-measure: "
            f"mean={score_summary['V-measure']['mean']:.4f}, "
            f"median={score_summary['V-measure']['median']:.4f}; "
            "AMI: "
            f"mean={score_summary['Adjusted Mutual Information']['mean']:.4f}, "
            f"median={score_summary['Adjusted Mutual Information']['median']:.4f}; "
            f"windows={window_count}, skipped_partial={skipped_partial}, "
            f"skipped_low_emitter={skipped_low_emitter}; "
            f"scores={args.output}; predictions={predictions_output}"
        )
    finally:
        if is_distributed:
            if control_group is not None:
                dist.destroy_process_group(control_group)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
