#!/usr/bin/env python3
"""Train and select a standalone Transformer metric-learning checkpoint."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone

import h5py
import numpy as np
import torch
from sklearn.metrics import v_measure_score
from torch.utils.data import DataLoader

try:  # Support package modules and direct script execution.
    from .data import (
        FileWindowBatchSampler,
        H5WindowDataset,
        fit_global_normalization,
        resolve_h5_files,
        split_train_files,
    )
    from .model import (
        AnchorSamplingStats,
        TransformerMetricEncoder,
        emitter_compactness_loss,
        file_aware_triplet_metric_loss,
        triplet_metric_loss,
    )
    from .predict import (
        cluster_embedding,
        effective_min_cluster_size,
        load_cuml_hdbscan,
        resolve_hdbscan_devices,
        resolve_inference_devices,
    )
except ImportError:  # pragma: no cover
    from data import (
        FileWindowBatchSampler,
        H5WindowDataset,
        fit_global_normalization,
        resolve_h5_files,
        split_train_files,
    )
    from model import (
        AnchorSamplingStats,
        TransformerMetricEncoder,
        emitter_compactness_loss,
        file_aware_triplet_metric_loss,
        triplet_metric_loss,
    )
    from predict import (
        cluster_embedding,
        effective_min_cluster_size,
        load_cuml_hdbscan,
        resolve_hdbscan_devices,
        resolve_inference_devices,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    parser.add_argument("--window-length", type=int, default=4096)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Sliding-window stride (default: half of window-length).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hdf5-cache-size", type=int, default=32)
    parser.add_argument(
        "--normalization",
        choices=("per_file", "global", "per_window"),
        default="per_file",
        help="Use one normalization transform per source file (default: per_file).",
    )
    parser.add_argument("--normalization-samples", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--training-scope",
        choices=("file", "window"),
        default="file",
        help="File mode samples positives across windows (default: file).",
    )
    parser.add_argument("--compactness-weight", type=float, default=0.05)
    parser.add_argument("--max-anchors-per-emitter", type=int, default=64)
    parser.add_argument(
        "--adaptive-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Select unique physical-pulse anchors per emitter using a bounded "
            "fraction instead of the fixed --max-anchors-per-emitter limit."
        ),
    )
    parser.add_argument("--anchor-min-per-emitter", type=int, default=128)
    parser.add_argument("--anchor-max-per-emitter", type=int, default=512)
    parser.add_argument("--anchor-fraction-per-emitter", type=float, default=0.05)
    parser.add_argument(
        "--architecture",
        choices=("rope_swiglu_v1", "rope_swiglu_rmsnorm_silu_v2"),
        default="rope_swiglu_v1",
    )
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=5e-4,
        help="Adaptive file-level HDBSCAN floor as a fraction of pulses.",
    )
    parser.add_argument(
        "--allow-single-cluster",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validation-max-files",
        type=int,
        default=32,
        help="Validation files used for loss/V-measure; 0 uses all (default: 32).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many epochs without selection-metric improvement.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("v_measure", "loss"),
        default="v_measure",
        help=(
            "Best-checkpoint and early-stopping metric. v_measure runs complete-file "
            "GPU cuML HDBSCAN on validation files (default: v_measure)."
        ),
    )
    parser.add_argument(
        "--validation-hdbscan-devices",
        default=None,
        help=(
            "Logical CUDA devices for validation cuML HDBSCAN, for example 0,1. "
            "Defaults to the Transformer training devices."
        ),
    )
    parser.add_argument(
        "--validation-hdbscan-parallel-files",
        type=int,
        default=None,
        help="Validation source files clustered concurrently (default: device count).",
    )
    parser.add_argument("--shuffle-train-windows", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices for DataParallel training.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory (default: <output directory>/logs).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable periodic progress lines and print only epoch summaries.",
    )
    return parser.parse_args()


class StableLineProgress:
    """Print periodic newline-terminated progress that remains clean in screen."""

    def __init__(
        self,
        description: str,
        total: int,
        enabled: bool,
        update_seconds: float = 30.0,
    ) -> None:
        self.description = description
        self.total = total
        self.enabled = enabled
        self.update_seconds = update_seconds
        self.started = time.perf_counter()
        self.last_printed = self.started
        self.current = 0
        self.lock = threading.RLock()
        if enabled:
            print(f"{description}: start, total={total:,}", flush=True)

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
        if minutes:
            return f"{minutes:d}m{seconds:02d}s"
        return f"{seconds:d}s"

    def update(self, current: int, postfix: str = "") -> None:
        if not self.enabled:
            return
        with self.lock:
            self.current = current
            now = time.perf_counter()
            if (
                current not in {1, self.total}
                and now - self.last_printed < self.update_seconds
            ):
                return
            elapsed = now - self.started
            rate = current / elapsed if elapsed > 0 else 0.0
            remaining = (
                (self.total - current) / rate if rate > 0 else float("inf")
            )
            percent = 100.0 * current / self.total if self.total else 100.0
            eta = "unknown" if not np.isfinite(remaining) else self._duration(remaining)
            suffix = f" {postfix}" if postfix else ""
            print(
                f"{self.description}: {current:,}/{self.total:,} "
                f"({percent:.1f}%), elapsed={self._duration(elapsed)}, "
                f"ETA={eta}, rate={rate:.2f}/s{suffix}",
                flush=True,
            )
            self.last_printed = now

    def advance(self, postfix: str = "") -> None:
        with self.lock:
            self.update(self.current + 1, postfix=postfix)


class TrainingRunLogger:
    """Keep terminal progress separate from clean, persistent run logs."""

    metric_fields = (
        "epoch",
        "train_loss",
        "train_triplet",
        "train_compactness",
        "train_anchor_count",
        "train_unique_anchor_count",
        "train_mean_anchors_per_emitter",
        "train_anchor_candidate_occurrences",
        "train_anchor_candidate_unique_pulses",
        "validation_loss",
        "validation_triplet",
        "validation_compactness",
        "validation_anchor_count",
        "validation_unique_anchor_count",
        "validation_mean_anchors_per_emitter",
        "validation_anchor_candidate_occurrences",
        "validation_anchor_candidate_unique_pulses",
        "validation_batches",
        "validation_v_measure",
        "validation_v_measure_weighted",
        "validation_clustered_files",
        "selection_metric",
        "selection_value",
        "best",
        "stale_epochs",
        "epoch_seconds",
    )

    def __init__(self, args: argparse.Namespace) -> None:
        default_dir = args.output.expanduser().resolve().parent / "logs"
        self.directory = (
            args.log_dir.expanduser().resolve()
            if args.log_dir is not None
            else default_dir
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.text_path = self.directory / "train.log"
        self.metrics_path = self.directory / "metrics.csv"
        self.config_path = self.directory / "run_config.json"

        self.logger = logging.getLogger("transformer_hdbscan.train")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for old_handler in list(self.logger.handlers):
            old_handler.close()
            self.logger.removeHandler(old_handler)
        handler = logging.FileHandler(self.text_path, mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

        self.metrics_file = self.metrics_path.open(
            "w", encoding="utf-8", newline=""
        )
        self.metrics_writer = csv.DictWriter(
            self.metrics_file, fieldnames=self.metric_fields
        )
        self.metrics_writer.writeheader()
        self.metrics_file.flush()

        config = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        temporary = self.config_path.with_name(f".{self.config_path.name}.tmp")
        temporary.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.config_path)

    def report(self, message: str) -> None:
        self.logger.info(message)
        print(message, flush=True)

    def write_epoch(self, values: dict[str, object]) -> None:
        self.metrics_writer.writerow(values)
        self.metrics_file.flush()

    def exception(self, message: str) -> None:
        self.logger.exception(message)

    def close(self) -> None:
        self.metrics_file.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


def validate_file_aware_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    training_scope: str,
    margin: float,
    compactness_weight: float,
    max_anchors_per_emitter: int,
    adaptive_anchors: bool,
    anchor_min_per_emitter: int,
    anchor_max_per_emitter: int,
    anchor_fraction_per_emitter: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    progress_enabled: bool,
    dataset: H5WindowDataset | None = None,
    collect_complete_embeddings: bool = False,
) -> tuple[
    float,
    float,
    float,
    int,
    AnchorSamplingStats,
    list[np.ndarray] | None,
]:
    """Evaluate metric loss and optionally reconstruct complete-file embeddings."""
    if collect_complete_embeddings and dataset is None:
        raise ValueError("dataset is required when collecting complete embeddings")
    model.eval()
    total_loss = 0.0
    total_triplet = 0.0
    total_compactness = 0.0
    total_sampling_stats = AnchorSamplingStats()
    batch_count = 0
    embedding_sums: list[np.ndarray] | None = None
    contribution_counts: list[np.ndarray] | None = None
    progress = StableLineProgress(
        "Validation embedding",
        len(loader),
        enabled=progress_enabled,
    )
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            (
                features,
                labels,
                padding_mask,
                file_indices,
                pulse_starts,
                valid_lengths,
                *_,
            ) = batch
            cpu_file_indices = np.asarray(file_indices, dtype=np.int64)
            cpu_pulse_starts = np.asarray(pulse_starts, dtype=np.int64)
            cpu_valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            file_indices = file_indices.to(device, non_blocking=True)
            pulse_starts = pulse_starts.to(device, non_blocking=True)
            source_indices = pulse_starts[:, None] + torch.arange(
                labels.shape[1], device=device
            )[None, :]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                embeddings = model(
                    features,
                    padding_mask=padding_mask,
                )
                if training_scope == "file":
                    triplet, sampling_stats = file_aware_triplet_metric_loss(
                        embeddings,
                        labels,
                        file_indices,
                        source_indices=source_indices,
                        padding_mask=padding_mask,
                        margin=margin,
                        max_anchors_per_emitter=max_anchors_per_emitter,
                        adaptive_anchors=adaptive_anchors,
                        anchor_min_per_emitter=anchor_min_per_emitter,
                        anchor_max_per_emitter=anchor_max_per_emitter,
                        anchor_fraction_per_emitter=anchor_fraction_per_emitter,
                        return_sampling_stats=True,
                    )
                    compactness = emitter_compactness_loss(
                        embeddings,
                        labels,
                        file_indices,
                        padding_mask=padding_mask,
                    )
                    loss = triplet + compactness_weight * compactness
                else:
                    triplet = triplet_metric_loss(
                        embeddings,
                        labels,
                        padding_mask=padding_mask,
                        margin=margin,
                    )
                    compactness = embeddings.sum() * 0.0
                    loss = triplet
                    sampling_stats = AnchorSamplingStats()
            if collect_complete_embeddings:
                assert dataset is not None
                embedding_array = embeddings.detach().float().cpu().numpy()
                if embedding_sums is None:
                    embedding_dim = embedding_array.shape[-1]
                    embedding_sums = [
                        np.zeros((int(length), embedding_dim), dtype=np.float32)
                        for length in dataset.lengths
                    ]
                    contribution_counts = [
                        np.zeros(int(length), dtype=np.int32)
                        for length in dataset.lengths
                    ]
                assert contribution_counts is not None
                for row_index, file_index in enumerate(cpu_file_indices):
                    start = int(cpu_pulse_starts[row_index])
                    valid_length = int(cpu_valid_lengths[row_index])
                    end = start + valid_length
                    embedding_sums[int(file_index)][start:end] += embedding_array[
                        row_index, :valid_length
                    ]
                    contribution_counts[int(file_index)][start:end] += 1
            total_loss += float(loss.item())
            total_triplet += float(triplet.item())
            total_compactness += float(compactness.item())
            total_sampling_stats.anchor_count += sampling_stats.anchor_count
            total_sampling_stats.unique_anchor_count += (
                sampling_stats.unique_anchor_count
            )
            total_sampling_stats.emitter_count += sampling_stats.emitter_count
            total_sampling_stats.candidate_occurrence_count += (
                sampling_stats.candidate_occurrence_count
            )
            total_sampling_stats.candidate_unique_pulse_count += (
                sampling_stats.candidate_unique_pulse_count
            )
            batch_count += 1
            progress.update(batch_number)
    if batch_count == 0:
        raise RuntimeError("Validation loader is empty")
    complete_embeddings: list[np.ndarray] | None = None
    if collect_complete_embeddings:
        assert embedding_sums is not None and contribution_counts is not None
        complete_embeddings = []
        for file_index, (embedding_sum, counts) in enumerate(
            zip(embedding_sums, contribution_counts, strict=True)
        ):
            if np.any(counts == 0):
                raise RuntimeError(
                    f"Validation windows do not cover file index {file_index}"
                )
            complete_embeddings.append(
                embedding_sum / counts[:, None].astype(np.float32)
            )
    return (
        total_loss / batch_count,
        total_triplet / batch_count,
        total_compactness / batch_count,
        batch_count,
        total_sampling_stats,
        complete_embeddings,
    )


def evaluate_validation_v_measure(
    complete_embeddings: list[np.ndarray],
    dataset: H5WindowDataset,
    min_cluster_size: int,
    min_cluster_fraction: float,
    allow_single_cluster: bool,
    gpu_devices: list[int],
    progress_enabled: bool,
) -> tuple[float, float, int]:
    """Run complete-file cuML HDBSCAN and return macro/weighted V-measure."""
    if len(complete_embeddings) != len(dataset.files):
        raise ValueError("Validation embedding/file counts do not match")
    if not gpu_devices:
        raise ValueError("GPU validation HDBSCAN requires at least one CUDA device")

    scores = np.full(len(dataset.files), np.nan, dtype=np.float64)
    pulse_counts = np.asarray(dataset.lengths, dtype=np.int64)
    progress = StableLineProgress(
        "Validation GPU HDBSCAN",
        len(dataset.files),
        enabled=progress_enabled,
    )

    def evaluate_partition(device_id: int, file_indices: list[int]) -> None:
        for file_index in file_indices:
            embedding = complete_embeddings[file_index]
            minimum = effective_min_cluster_size(
                len(embedding), min_cluster_size, min_cluster_fraction
            )
            predicted = cluster_embedding(
                embedding,
                minimum,
                allow_single_cluster=allow_single_cluster,
                backend="cuml",
                gpu_device=device_id,
            )
            with h5py.File(dataset.files[file_index], "r") as handle:
                truth = np.asarray(handle["labels"][:]).reshape(-1)
            if len(truth) != len(predicted):
                raise RuntimeError("Validation truth/prediction lengths do not match")
            scores[file_index] = v_measure_score(truth, predicted)
            progress.advance(
                postfix=f"last={dataset.files[file_index].name}, gpu=cuda:{device_id}"
            )

    partitions = [
        list(range(worker_index, len(dataset.files), len(gpu_devices)))
        for worker_index in range(len(gpu_devices))
    ]
    with ThreadPoolExecutor(max_workers=len(gpu_devices)) as executor:
        futures = [
            executor.submit(evaluate_partition, device_id, file_indices)
            for device_id, file_indices in zip(
                gpu_devices, partitions, strict=True
            )
            if file_indices
        ]
        for future in futures:
            future.result()
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Validation V-measure contains a non-finite value")
    return (
        float(np.mean(scores)),
        float(np.average(scores, weights=pulse_counts)),
        len(scores),
    )


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    model_config: dict[str, object],
    args: argparse.Namespace,
    epoch: int,
    validation_loss: float,
    validation_v_measure: float | None,
    selection_value: float,
    normalization_mean: np.ndarray | None,
    normalization_std: np.ndarray | None,
    normalization_sample_count: int,
) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    checkpoint_model = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    torch.save(
        {
            "training_design": (
                "rope_rmsnorm_silu_adaptive_anchor_v1"
                if args.architecture == "rope_swiglu_rmsnorm_silu_v2"
                else "rope_swiglu_gpu_vmeasure_v5"
            ),
            "embedding_normalization": "none",
            "model_state_dict": checkpoint_model.state_dict(),
            "model_config": model_config,
            "window_length": args.window_length,
            "window_stride": args.window_stride,
            "min_cluster_size": args.min_cluster_size,
            "min_cluster_fraction": args.min_cluster_fraction,
            "allow_single_cluster": args.allow_single_cluster,
            "normalization": args.normalization,
            "normalization_mean": (
                None
                if normalization_mean is None
                else torch.from_numpy(normalization_mean.copy())
            ),
            "normalization_std": (
                None
                if normalization_std is None
                else torch.from_numpy(normalization_std.copy())
            ),
            "normalization_sample_count": normalization_sample_count,
            "training_scope": args.training_scope,
            "compactness_weight": args.compactness_weight,
            "adaptive_anchors": args.adaptive_anchors,
            "max_anchors_per_emitter": args.max_anchors_per_emitter,
            "anchor_min_per_emitter": args.anchor_min_per_emitter,
            "anchor_max_per_emitter": args.anchor_max_per_emitter,
            "anchor_fraction_per_emitter": args.anchor_fraction_per_emitter,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_v_measure": validation_v_measure,
            "selection_metric": args.selection_metric,
            "selection_value": selection_value,
            "validation_hdbscan_backend": (
                "cuml" if args.selection_metric == "v_measure" else None
            ),
            "seed": args.seed,
        },
        temporary,
    )
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    if args.window_stride is None:
        args.window_stride = max(1, args.window_length // 2)
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.num_workers < 0
        or args.hdf5_cache_size <= 0
        or args.normalization_samples <= 0
        or args.min_cluster_size < 2
        or args.min_cluster_fraction < 0
        or args.validation_max_files < 0
        or args.compactness_weight < 0
        or args.max_anchors_per_emitter <= 0
        or args.anchor_min_per_emitter <= 0
        or args.anchor_max_per_emitter <= 0
        or args.anchor_min_per_emitter > args.anchor_max_per_emitter
        or args.anchor_fraction_per_emitter < 0
        or args.early_stopping_patience < 0
        or (
            args.validation_hdbscan_parallel_files is not None
            and args.validation_hdbscan_parallel_files <= 0
        )
        or not 0 < args.window_stride <= args.window_length
    ):
        raise ValueError("Invalid non-positive training or clustering argument")
    if args.adaptive_anchors and args.training_scope != "file":
        raise ValueError("--adaptive-anchors requires --training-scope file")

    run_log = TrainingRunLogger(args)
    run_log.report(
        f"Logs: text={run_log.text_path}, metrics={run_log.metrics_path}, "
        f"config={run_log.config_path}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device, data_parallel_device_ids = resolve_inference_devices(
        args.device, args.devices
    )
    validation_hdbscan_devices: list[int] = []
    if args.selection_metric == "v_measure":
        if device.type != "cuda":
            raise RuntimeError(
                "--selection-metric v_measure requires CUDA and RAPIDS cuML; "
                "use --selection-metric loss only for CPU tests or ablations"
            )
        default_validation_devices = ",".join(
            str(device_id)
            for device_id in (
                data_parallel_device_ids
                or [torch.cuda.current_device() if device.index is None else device.index]
            )
        )
        validation_hdbscan_devices = resolve_hdbscan_devices(
            "cuml",
            None,
            args.validation_hdbscan_devices or default_validation_devices,
            args.validation_hdbscan_parallel_files,
        )
        load_cuml_hdbscan()
    all_train_files = resolve_h5_files(args.train_dir, split="train")
    if args.validation_dir is None:
        train_files, validation_files = split_train_files(
            all_train_files, args.validation_fraction, args.seed
        )
    else:
        train_files = all_train_files
        validation_files = resolve_h5_files(
            args.validation_dir, split="validation"
        )

    if args.validation_max_files and len(validation_files) > args.validation_max_files:
        validation_order = np.random.default_rng(args.seed).permutation(
            len(validation_files)
        )[: args.validation_max_files]
        validation_files = [
            validation_files[int(index)] for index in sorted(validation_order)
        ]

    normalization_mean: np.ndarray | None = None
    normalization_std: np.ndarray | None = None
    normalization_sample_count = 0
    if args.normalization == "global":
        normalization_mean, normalization_std, normalization_sample_count = (
            fit_global_normalization(
                train_files,
                max_samples=args.normalization_samples,
                seed=args.seed,
            )
        )
        run_log.report(
            f"Fitted global normalization from {normalization_sample_count:,} pulses"
        )

    train_dataset = H5WindowDataset(
        train_files,
        args.window_length,
        window_stride=args.window_stride,
        require_labels=True,
        hdf5_cache_size=args.hdf5_cache_size,
        normalization=args.normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_progress=(
            args.normalization == "per_file" and not args.no_progress
        ),
        normalization_description="Fit train-file normalization",
    )
    validation_dataset = H5WindowDataset(
        validation_files,
        args.window_length,
        window_stride=args.window_stride,
        require_labels=True,
        hdf5_cache_size=args.hdf5_cache_size,
        normalization=args.normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_progress=(
            args.normalization == "per_file" and not args.no_progress
        ),
        normalization_description="Fit validation-file normalization",
    )
    if args.adaptive_anchors:
        for dataset_name, dataset in (
            ("training", train_dataset),
            ("validation", validation_dataset),
        ):
            previous = 0
            split_files: list[str] = []
            for file_index, offset in enumerate(dataset.offsets):
                window_count = int(offset) - previous
                previous = int(offset)
                if window_count > args.batch_size:
                    split_files.append(
                        f"{dataset.files[file_index].name} ({window_count:,} windows)"
                    )
            if split_files:
                examples = ", ".join(split_files[:3])
                raise ValueError(
                    "Adaptive anchors require every source file to fit in one "
                    f"batch, but {dataset_name} has {len(split_files)} split file(s): "
                    f"{examples}. Increase --batch-size above the largest file's "
                    "window count."
                )
    train_batch_sampler = FileWindowBatchSampler(
        train_dataset,
        args.batch_size,
        shuffle=args.shuffle_train_windows,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_batch_sampler = FileWindowBatchSampler(
        validation_dataset,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_batch_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_config: dict[str, object] = {
        "input_dim": int(train_dataset.feature_dim),
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "normalize_embeddings": False,
        "architecture": args.architecture,
    }
    model: torch.nn.Module = TransformerMetricEncoder(**model_config).to(device)
    if len(data_parallel_device_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=data_parallel_device_ids,
            output_device=data_parallel_device_ids[0],
        )
    run_log.report(
        "Training devices: "
        + (
            ", ".join(f"cuda:{index}" for index in data_parallel_device_ids)
            if data_parallel_device_ids
            else str(device)
        )
    )
    run_log.report(
        f"Sliding windows: length={args.window_length:,}, "
        f"stride={args.window_stride:,}, normalization={args.normalization}"
    )
    norm_name = (
        "RMSNorm(eps=1e-6)"
        if args.architecture == "rope_swiglu_rmsnorm_silu_v2"
        else "LayerNorm"
    )
    activation_name = (
        "SiLU"
        if args.architecture == "rope_swiglu_rmsnorm_silu_v2"
        else "GELU"
    )
    run_log.report(
        f"Architecture: Linear({train_dataset.feature_dim},{args.model_dim}) + "
        f"{norm_name} + {activation_name}; {args.num_layers}-layer "
        f"RoPE-QK/SwiGLU Transformer, heads={args.num_heads}, "
        f"SwiGLU hidden={args.feedforward_dim}, embedding_dim={args.embedding_dim}, "
        f"pre-norm={norm_name}, output_l2=none"
    )
    run_log.report(
        "Metric loss: "
        + (
            "adaptive unique-pulse anchors="
            f"min({args.anchor_max_per_emitter}, max("
            f"{args.anchor_min_per_emitter}, ceil(N*"
            f"{args.anchor_fraction_per_emitter:g})))"
            if args.adaptive_anchors
            else f"fixed anchors<= {args.max_anchors_per_emitter} per emitter"
        )
        + f", compactness_weight={args.compactness_weight:g}"
    )
    if args.selection_metric == "v_measure":
        run_log.report(
            "Checkpoint selection: validation V-measure macro average; "
            "cuML HDBSCAN devices="
            + ",".join(f"cuda:{device_id}" for device_id in validation_hdbscan_devices)
        )
    else:
        run_log.report("Checkpoint selection: validation metric-learning loss")
    run_log.report(
        f"Data: train_files={len(train_files):,}, "
        f"validation_files={len(validation_files):,}, "
        f"train_windows={len(train_dataset):,}, "
        f"validation_windows={len(validation_dataset):,}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_enabled = args.amp and device.type == "cuda"
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0/2.1 compatibility.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    best_selection_value = (
        -np.inf if args.selection_metric == "v_measure" else np.inf
    )
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_batch_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_triplet = 0.0
        total_compactness = 0.0
        total_sampling_stats = AnchorSamplingStats()
        progress = StableLineProgress(
            f"Train {epoch}/{args.epochs}",
            len(train_loader),
            enabled=not args.no_progress,
        )
        for batch_number, batch in enumerate(train_loader, start=1):
            (
                features,
                labels,
                padding_mask,
                file_indices,
                pulse_starts,
                *_,
            ) = batch
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            file_indices = file_indices.to(device, non_blocking=True)
            pulse_starts = pulse_starts.to(device, non_blocking=True)
            source_indices = pulse_starts[:, None] + torch.arange(
                labels.shape[1], device=device
            )[None, :]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                embeddings = model(features, padding_mask=padding_mask)
                if args.training_scope == "file":
                    triplet, sampling_stats = file_aware_triplet_metric_loss(
                        embeddings,
                        labels,
                        file_indices,
                        source_indices=source_indices,
                        padding_mask=padding_mask,
                        margin=args.margin,
                        max_anchors_per_emitter=args.max_anchors_per_emitter,
                        adaptive_anchors=args.adaptive_anchors,
                        anchor_min_per_emitter=args.anchor_min_per_emitter,
                        anchor_max_per_emitter=args.anchor_max_per_emitter,
                        anchor_fraction_per_emitter=(
                            args.anchor_fraction_per_emitter
                        ),
                        return_sampling_stats=True,
                    )
                    compactness = emitter_compactness_loss(
                        embeddings,
                        labels,
                        file_indices,
                        padding_mask=padding_mask,
                    )
                    loss = triplet + args.compactness_weight * compactness
                else:
                    triplet = triplet_metric_loss(
                        embeddings,
                        labels,
                        padding_mask=padding_mask,
                        margin=args.margin,
                    )
                    compactness = embeddings.sum() * 0.0
                    loss = triplet
                    sampling_stats = AnchorSamplingStats()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_triplet += float(triplet.item())
            total_compactness += float(compactness.item())
            total_sampling_stats.anchor_count += sampling_stats.anchor_count
            total_sampling_stats.unique_anchor_count += (
                sampling_stats.unique_anchor_count
            )
            total_sampling_stats.emitter_count += sampling_stats.emitter_count
            total_sampling_stats.candidate_occurrence_count += (
                sampling_stats.candidate_occurrence_count
            )
            total_sampling_stats.candidate_unique_pulse_count += (
                sampling_stats.candidate_unique_pulse_count
            )
            progress.update(
                batch_number,
                postfix=(
                    f"L={loss.item():.4f} T={triplet.item():.4f} "
                    f"C={compactness.item():.4f} A={sampling_stats.anchor_count:,}"
                ),
            )

        (
            validation_loss,
            validation_triplet,
            validation_compactness,
            validation_batches,
            validation_sampling_stats,
            complete_validation_embeddings,
        ) = validate_file_aware_loss(
            model,
            validation_loader,
            device,
            args.training_scope,
            args.margin,
            args.compactness_weight,
            args.max_anchors_per_emitter,
            args.adaptive_anchors,
            args.anchor_min_per_emitter,
            args.anchor_max_per_emitter,
            args.anchor_fraction_per_emitter,
            amp_enabled,
            amp_dtype,
            not args.no_progress,
            dataset=validation_dataset,
            collect_complete_embeddings=args.selection_metric == "v_measure",
        )
        validation_v_measure: float | None = None
        validation_v_measure_weighted: float | None = None
        validation_clustered_files = 0
        if args.selection_metric == "v_measure":
            assert complete_validation_embeddings is not None
            (
                validation_v_measure,
                validation_v_measure_weighted,
                validation_clustered_files,
            ) = evaluate_validation_v_measure(
                complete_validation_embeddings,
                validation_dataset,
                args.min_cluster_size,
                args.min_cluster_fraction,
                args.allow_single_cluster,
                validation_hdbscan_devices,
                not args.no_progress,
            )
        mean_loss = total_loss / len(train_loader)
        mean_triplet = total_triplet / len(train_loader)
        mean_compactness = total_compactness / len(train_loader)
        train_mean_anchors_per_emitter = (
            total_sampling_stats.anchor_count / total_sampling_stats.emitter_count
            if total_sampling_stats.emitter_count
            else 0.0
        )
        validation_mean_anchors_per_emitter = (
            validation_sampling_stats.anchor_count
            / validation_sampling_stats.emitter_count
            if validation_sampling_stats.emitter_count
            else 0.0
        )
        selection_value = (
            validation_v_measure
            if args.selection_metric == "v_measure"
            else validation_loss
        )
        assert selection_value is not None
        improved = (
            selection_value > best_selection_value
            if args.selection_metric == "v_measure"
            else selection_value < best_selection_value
        )
        if improved:
            best_selection_value = selection_value
            stale_epochs = 0
            save_checkpoint(
                args.output,
                model,
                model_config,
                args,
                epoch,
                validation_loss,
                validation_v_measure,
                selection_value,
                normalization_mean,
                normalization_std,
                normalization_sample_count,
            )
        else:
            stale_epochs += 1
        epoch_seconds = time.perf_counter() - epoch_started
        run_log.write_epoch(
            {
                "epoch": epoch,
                "train_loss": f"{mean_loss:.8f}",
                "train_triplet": f"{mean_triplet:.8f}",
                "train_compactness": f"{mean_compactness:.8f}",
                "train_anchor_count": total_sampling_stats.anchor_count,
                "train_unique_anchor_count": (
                    total_sampling_stats.unique_anchor_count
                ),
                "train_mean_anchors_per_emitter": (
                    f"{train_mean_anchors_per_emitter:.4f}"
                ),
                "train_anchor_candidate_occurrences": (
                    total_sampling_stats.candidate_occurrence_count
                ),
                "train_anchor_candidate_unique_pulses": (
                    total_sampling_stats.candidate_unique_pulse_count
                ),
                "validation_loss": f"{validation_loss:.8f}",
                "validation_triplet": f"{validation_triplet:.8f}",
                "validation_compactness": f"{validation_compactness:.8f}",
                "validation_anchor_count": validation_sampling_stats.anchor_count,
                "validation_unique_anchor_count": (
                    validation_sampling_stats.unique_anchor_count
                ),
                "validation_mean_anchors_per_emitter": (
                    f"{validation_mean_anchors_per_emitter:.4f}"
                ),
                "validation_anchor_candidate_occurrences": (
                    validation_sampling_stats.candidate_occurrence_count
                ),
                "validation_anchor_candidate_unique_pulses": (
                    validation_sampling_stats.candidate_unique_pulse_count
                ),
                "validation_batches": validation_batches,
                "validation_v_measure": (
                    ""
                    if validation_v_measure is None
                    else f"{validation_v_measure:.8f}"
                ),
                "validation_v_measure_weighted": (
                    ""
                    if validation_v_measure_weighted is None
                    else f"{validation_v_measure_weighted:.8f}"
                ),
                "validation_clustered_files": validation_clustered_files,
                "selection_metric": args.selection_metric,
                "selection_value": f"{selection_value:.8f}",
                "best": improved,
                "stale_epochs": stale_epochs,
                "epoch_seconds": f"{epoch_seconds:.2f}",
            }
        )
        run_log.report(
            f"epoch={epoch}/{args.epochs} train={mean_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            + (
                f"validation_v_measure={validation_v_measure:.6f} "
                if validation_v_measure is not None
                else ""
            )
            + f"selection={selection_value:.6f} best={improved} "
            f"anchors={total_sampling_stats.anchor_count:,} "
            f"unique={total_sampling_stats.unique_anchor_count:,} "
            f"mean_per_emitter={train_mean_anchors_per_emitter:.1f} "
            f"stale={stale_epochs}/{args.early_stopping_patience} "
            f"time={epoch_seconds:.1f}s"
        )
        if not improved:
            if (
                args.early_stopping_patience > 0
                and stale_epochs >= args.early_stopping_patience
            ):
                run_log.report(
                    f"Early stopping after {stale_epochs} stale epoch(s)"
                )
                break
    run_log.report(
        f"Saved best checkpoint: {args.output.expanduser().resolve()} "
        f"({args.selection_metric}={best_selection_value:.6f})"
    )
    run_log.close()


if __name__ == "__main__":
    main()
