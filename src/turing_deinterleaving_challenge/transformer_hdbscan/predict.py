#!/usr/bin/env python3
"""Embed by window, then cluster and save each complete source pulse file."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from torch.utils.data import DataLoader
from tqdm import tqdm

try:  # Support package modules and ``python predict.py`` from this directory.
    from .data import H5WindowDataset, resolve_h5_files
    from .metrics import evaluate_labels
    from .model import TransformerMetricEncoder
    from .output import (
        MetricsOnlyRunWriter,
        PDWStudioRunWriter,
        PredictionRunWriter,
        SourceFileMetadata,
    )
except ImportError:  # pragma: no cover
    from data import H5WindowDataset, resolve_h5_files
    from metrics import evaluate_labels
    from model import TransformerMetricEncoder
    from output import (
        MetricsOnlyRunWriter,
        PDWStudioRunWriter,
        PredictionRunWriter,
        SourceFileMetadata,
    )


@dataclass
class BufferedWindow:
    """One window retained until its complete source file can be clustered."""

    embedding: np.ndarray
    pulse_start: int
    valid_length: int


@dataclass(frozen=True)
class SourceFileTiming:
    """Measured complete-file work used for progress reporting and ETA."""

    pulse_count: int
    predicted_cluster_count: int
    effective_min_cluster_size: int
    effective_min_samples: int
    hdbscan_seconds: float
    preparation_seconds: float
    evaluation_seconds: float
    write_seconds: float

    @property
    def other_seconds(self) -> float:
        return self.preparation_seconds + self.evaluation_seconds + self.write_seconds

    @property
    def total_seconds(self) -> float:
        return self.hdbscan_seconds + self.other_seconds


@dataclass(frozen=True)
class ClusteredSourceFile:
    """A completed worker result waiting for serialized output writing."""

    file_index: int
    source_file: Path
    pulse_count: int
    window_count: int
    window_starts: np.ndarray
    window_valid_lengths: np.ndarray
    predicted_labels: np.ndarray
    effective_min_cluster_size: int
    effective_min_samples: int
    hdbscan_device: int | None
    hdbscan_seconds: float
    preparation_seconds: float


def format_duration(seconds: float) -> str:
    """Format a non-negative duration without hiding long-running days."""
    seconds = max(0, round(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def estimate_remaining_seconds(
    *,
    total_windows: int,
    embedded_windows: int,
    embedding_seconds: float,
    file_lengths: list[int],
    completed_files: int,
    hdbscan_seconds: float,
    hdbscan_work: float,
    other_seconds: float,
    finalized_pulses: int,
    hdbscan_parallel_files: int = 1,
) -> float | None:
    """Estimate remaining time from measured linear and quadratic stage rates."""
    if embedded_windows <= 0 or completed_files <= 0:
        return None

    remaining_embedding_windows = max(total_windows - embedded_windows, 0)
    embedding_eta = embedding_seconds / embedded_windows * remaining_embedding_windows

    del completed_files
    remaining_hdbscan_work = max(
        sum(float(length) ** 2 for length in file_lengths) - hdbscan_work,
        0.0,
    )
    hdbscan_eta = (
        hdbscan_seconds
        / hdbscan_work
        * remaining_hdbscan_work
        / max(hdbscan_parallel_files, 1)
        if hdbscan_work > 0
        else 0.0
    )

    remaining_pulses = max(sum(file_lengths) - finalized_pulses, 0)
    other_eta = (
        other_seconds / finalized_pulses * remaining_pulses
        if finalized_pulses > 0
        else 0.0
    )
    return embedding_eta + hdbscan_eta + other_eta


@lru_cache(maxsize=1)
def load_cuml_hdbscan() -> type[Any]:
    """Import optional cuML lazily and provide an actionable failure."""
    try:
        from cuml.cluster import HDBSCAN as CuMLHDBSCAN
    except ImportError as error:
        raise RuntimeError(
            "cuML HDBSCAN was requested but RAPIDS cuML is not installed. "
            "Install a cuML build matching the server CUDA environment, then "
            "verify it with: python -c 'from cuml.cluster import HDBSCAN'"
        ) from error
    return CuMLHDBSCAN


def cluster_embedding(
    embedding: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None = None,
    n_jobs: int = 1,
    allow_single_cluster: bool = False,
    backend: str = "sklearn",
    gpu_device: int = 0,
) -> np.ndarray:
    """Cluster one embedding sequence; -1 remains an ordinary output group."""
    if len(embedding) < min_cluster_size:
        return np.full(len(embedding), -1, dtype=np.int32)
    contiguous = np.ascontiguousarray(embedding, dtype=np.float32)
    if backend == "sklearn":
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_jobs=n_jobs,
            allow_single_cluster=allow_single_cluster,
            copy=False,
        ).fit_predict(contiguous)
    elif backend == "cuml":
        if not torch.cuda.is_available():
            raise RuntimeError("cuML HDBSCAN requires an available CUDA device")
        if not 0 <= gpu_device < torch.cuda.device_count():
            raise ValueError(
                f"HDBSCAN CUDA device {gpu_device} is unavailable; PyTorch sees "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
        CuMLHDBSCAN = load_cuml_hdbscan()
        with torch.cuda.device(gpu_device):
            torch.cuda.empty_cache()
            clusterer = CuMLHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                allow_single_cluster=allow_single_cluster,
                output_type="numpy",
            )
            labels = clusterer.fit_predict(contiguous)
            torch.cuda.synchronize(gpu_device)
            labels_array = np.asarray(labels).reshape(-1).astype(np.int32, copy=True)
            del labels
            del clusterer
            gc.collect()
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported HDBSCAN backend: {backend}")
    if backend == "sklearn":
        labels_array = np.asarray(labels).reshape(-1)
    if len(labels_array) != len(contiguous):
        raise RuntimeError("HDBSCAN returned a label count that does not match input")
    return labels_array.astype(np.int32, copy=False)


def effective_min_cluster_size(pulse_count: int, minimum: int, fraction: float) -> int:
    """Scale a file-level HDBSCAN floor while retaining a fixed lower bound."""
    if pulse_count < 0 or minimum < 2 or fraction < 0:
        raise ValueError("Invalid adaptive min-cluster-size parameters")
    return max(minimum, int(np.ceil(pulse_count * fraction)))


def effective_min_samples(pulse_count: int, minimum: int, fraction: float) -> int:
    """Compute max(minimum, ceil(pulse_count * fraction))."""
    if pulse_count < 0 or minimum < 1 or fraction < 0:
        raise ValueError("Invalid adaptive min-samples parameters")
    return max(minimum, int(np.ceil(pulse_count * fraction)))


def aggregate_overlapping_embeddings(
    windows: list[BufferedWindow], pulse_count: int
) -> np.ndarray:
    """Average repeated pulse embeddings without post-aggregation normalization."""
    if not windows or pulse_count <= 0:
        raise ValueError("Cannot aggregate empty windows or an empty source file")
    embedding_dim = windows[0].embedding.shape[1]
    embedding_sum = np.zeros((pulse_count, embedding_dim), dtype=np.float32)
    contribution_count = np.zeros(pulse_count, dtype=np.int32)
    for window in windows:
        start = window.pulse_start
        end = start + window.valid_length
        if start < 0 or end > pulse_count:
            raise ValueError("Window lies outside the source pulse file")
        if window.embedding.shape != (window.valid_length, embedding_dim):
            raise ValueError("Window embedding shape does not match its valid length")
        embedding_sum[start:end] += window.embedding
        contribution_count[start:end] += 1
    if np.any(contribution_count == 0):
        raise RuntimeError("Sliding windows did not cover every source pulse")
    averaged = embedding_sum / contribution_count[:, None].astype(np.float32)
    return averaged.astype(np.float32, copy=False)


def load_checkpoint(path: Path) -> dict:
    """Load tensor-only checkpoints across the supported PyTorch 2.x range."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
        )
        return torch.load(path, map_location="cpu", weights_only=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("predictions"),
        help="Empty output directory (default: predictions).",
    )
    parser.add_argument(
        "--output-format",
        choices=("native", "pdw_studio", "metrics_only"),
        default="native",
        help=(
            "native writes one HDF5 per source; pdw_studio writes "
            "<source stem>/<source stem>_<cluster id>.h5; metrics_only writes "
            "only metrics.csv and summary.json (default: native)."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--source-files",
        nargs="+",
        default=None,
        help=(
            "Optional reproducible subset of source basenames or stems, for "
            "example config_0.h5 config_10.h5. Comma-separated values are also "
            "accepted. By default every file in the split is processed."
        ),
    )
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Override checkpoint sliding-window stride.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hdf5-cache-size", type=int, default=32)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=None,
        help="Override checkpoint adaptive file-level HDBSCAN fraction.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=(
            "Explicit HDBSCAN density-neighbor count. When omitted, HDBSCAN "
            "keeps its default min_samples=min_cluster_size behavior."
        ),
    )
    parser.add_argument(
        "--min-samples-fraction",
        type=float,
        default=None,
        help=(
            "Optional adaptive density fraction. With --min-samples K, the "
            "effective value is max(K, ceil(source_pulses * fraction))."
        ),
    )
    parser.add_argument(
        "--allow-single-cluster",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether HDBSCAN may return one cluster.",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Permit HDF5 inputs without labels; true groups and metrics are omitted.",
    )
    evaluation_group = parser.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help=(
            "Also compute metrics while clustering. Disabled by default; prefer "
            "the separate evaluate_saved command."
        ),
    )
    evaluation_group.add_argument(
        "--skip-evaluation",
        "--skip-metrics",
        dest="evaluate",
        action="store_false",
        help=(
            "Only embed, cluster, and save each source file (the default). This "
            "compatibility flag may be omitted."
        ),
    )
    parser.set_defaults(evaluate=False)
    parser.add_argument(
        "--hdbscan-jobs",
        type=int,
        default=1,
        help=("CPU workers used by sklearn HDBSCAN; ignored by cuML " "(default: 1)."),
    )
    parser.add_argument(
        "--hdbscan-backend",
        choices=("sklearn", "cuml"),
        default="sklearn",
        help="Complete-file clustering backend (default: sklearn).",
    )
    parser.add_argument(
        "--hdbscan-device",
        type=int,
        default=None,
        help=(
            "Legacy single logical CUDA device used by cuML HDBSCAN after "
            "applying CUDA_VISIBLE_DEVICES. Cannot be combined with "
            "--hdbscan-devices."
        ),
    )
    parser.add_argument(
        "--hdbscan-devices",
        default=None,
        help=(
            "Comma-separated logical CUDA devices that independently cluster "
            "different source files, for example 0,1 (cuML only)."
        ),
    )
    parser.add_argument(
        "--hdbscan-parallel-files",
        type=int,
        default=None,
        help=(
            "Maximum source files clustered concurrently. Defaults to the "
            "number of --hdbscan-devices and cannot exceed it."
        ),
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--device",
        default=None,
        help="Single inference device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated CUDA devices for DataParallel inference, for example "
            "cuda:0,cuda:1. Cannot be combined with --device."
        ),
    )
    return parser.parse_args()


def select_source_files(
    files: list[Path], requested: list[str] | None
) -> list[Path]:
    """Select an explicit source-file subset while preserving request order."""
    if not requested:
        return files
    tokens = [
        token.strip()
        for item in requested
        for token in item.split(",")
        if token.strip()
    ]
    if not tokens:
        raise ValueError("--source-files did not contain any file names")
    by_name = {path.name: path for path in files}
    by_stem = {path.stem: path for path in files}
    selected: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for token in tokens:
        name = Path(token).name
        path = by_name.get(name) or by_stem.get(Path(name).stem)
        if path is None:
            missing.append(token)
            continue
        if path in seen:
            raise ValueError(f"Duplicate --source-files entry: {token}")
        seen.add(path)
        selected.append(path)
    if missing:
        available = ", ".join(path.name for path in files[:10])
        raise ValueError(
            "Requested source file(s) were not found: "
            + ", ".join(missing)
            + f". First available files: {available}"
        )
    return selected


def resolve_inference_devices(
    device_arg: str | None, devices_arg: str | None
) -> tuple[torch.device, list[int]]:
    """Resolve one primary device and optional CUDA DataParallel device IDs."""
    if device_arg is not None and devices_arg is not None:
        raise ValueError("--device and --devices cannot be used together")

    if devices_arg is None:
        device = torch.device(
            device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type != "cuda":
            return device, []
        device_id = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device_id} is unavailable; PyTorch sees "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
        return torch.device(f"cuda:{device_id}"), [device_id]

    specs = [item.strip() for item in devices_arg.split(",") if item.strip()]
    if not specs:
        raise ValueError("--devices must contain at least one CUDA device")
    resolved = [
        torch.device(f"cuda:{item}" if item.isdigit() else item) for item in specs
    ]
    if any(device.type != "cuda" for device in resolved):
        raise ValueError("--devices only accepts CUDA devices")
    device_ids = [
        torch.cuda.current_device() if device.index is None else device.index
        for device in resolved
    ]
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("--devices contains duplicate CUDA devices")
    unavailable = [
        device_id
        for device_id in device_ids
        if device_id < 0 or device_id >= torch.cuda.device_count()
    ]
    if unavailable:
        raise ValueError(
            f"CUDA device(s) {unavailable} unavailable; PyTorch sees "
            f"{torch.cuda.device_count()} CUDA device(s). Check CUDA_VISIBLE_DEVICES."
        )
    return torch.device(f"cuda:{device_ids[0]}"), device_ids


def resolve_hdbscan_devices(
    backend: str,
    device_arg: int | None,
    devices_arg: str | None,
    parallel_files_arg: int | None,
) -> list[int]:
    """Resolve the independent per-file cuML workers to logical CUDA IDs."""
    if backend != "cuml":
        if devices_arg is not None or parallel_files_arg not in (None, 1):
            raise ValueError(
                "--hdbscan-devices/--hdbscan-parallel-files require "
                "--hdbscan-backend cuml"
            )
        return []
    if device_arg is not None and devices_arg is not None:
        raise ValueError(
            "--hdbscan-device and --hdbscan-devices cannot be used together"
        )
    if device_arg is not None:
        device_ids = [device_arg]
    elif devices_arg is None:
        device_ids = [0]
    else:
        specs = [item.strip() for item in devices_arg.split(",") if item.strip()]
        if not specs:
            raise ValueError("--hdbscan-devices must contain a CUDA device")
        device_ids = []
        for spec in specs:
            normalized = spec.removeprefix("cuda:")
            if not normalized.isdigit():
                raise ValueError(
                    "--hdbscan-devices accepts comma-separated logical CUDA "
                    "indices, for example 0,1"
                )
            device_ids.append(int(normalized))
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("--hdbscan-devices contains duplicate CUDA devices")
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError("HDBSCAN CUDA device indices cannot be negative")
    unavailable = [
        device_id for device_id in device_ids if device_id >= torch.cuda.device_count()
    ]
    if unavailable:
        raise ValueError(
            f"HDBSCAN CUDA device(s) {unavailable} unavailable; PyTorch sees "
            f"{torch.cuda.device_count()} CUDA device(s). Check "
            "CUDA_VISIBLE_DEVICES."
        )
    parallel_files = (
        len(device_ids) if parallel_files_arg is None else parallel_files_arg
    )
    if not 1 <= parallel_files <= len(device_ids):
        raise ValueError(
            "--hdbscan-parallel-files must be between 1 and the number of "
            "HDBSCAN devices"
        )
    return device_ids[:parallel_files]


def cluster_source_file(
    file_index: int,
    windows: list[BufferedWindow],
    source_file: Path,
    min_cluster_size: int,
    min_cluster_fraction: float,
    min_samples: int | None,
    min_samples_fraction: float | None,
    allow_single_cluster: bool,
    hdbscan_jobs: int,
    hdbscan_backend: str,
    hdbscan_device: int,
) -> ClusteredSourceFile:
    """Aggregate and cluster one source file without touching shared writers."""
    preparation_started = time.perf_counter()
    if not windows:
        raise ValueError("Cannot finalize an empty source file")
    with h5py.File(source_file, "r") as handle:
        pulse_count = len(handle["data"])
    complete_embedding = aggregate_overlapping_embeddings(windows, pulse_count)
    window_starts = np.asarray(
        [window.pulse_start for window in windows], dtype=np.int64
    )
    window_valid_lengths = np.asarray(
        [window.valid_length for window in windows], dtype=np.int64
    )
    effective_minimum = effective_min_cluster_size(
        len(complete_embedding), min_cluster_size, min_cluster_fraction
    )
    explicit_min_samples = (
        None
        if min_samples is None
        else (
            min_samples
            if min_samples_fraction is None
            else effective_min_samples(
                len(complete_embedding), min_samples, min_samples_fraction
            )
        )
    )
    reported_min_samples = (
        effective_minimum if explicit_min_samples is None else explicit_min_samples
    )
    preparation_seconds = time.perf_counter() - preparation_started
    hdbscan_started = time.perf_counter()
    predicted = cluster_embedding(
        complete_embedding,
        effective_minimum,
        min_samples=explicit_min_samples,
        n_jobs=hdbscan_jobs,
        allow_single_cluster=allow_single_cluster,
        backend=hdbscan_backend,
        gpu_device=hdbscan_device,
    )
    hdbscan_seconds = time.perf_counter() - hdbscan_started
    return ClusteredSourceFile(
        file_index=file_index,
        source_file=source_file,
        pulse_count=pulse_count,
        window_count=len(windows),
        window_starts=window_starts,
        window_valid_lengths=window_valid_lengths,
        predicted_labels=predicted,
        effective_min_cluster_size=effective_minimum,
        effective_min_samples=reported_min_samples,
        hdbscan_device=(hdbscan_device if hdbscan_backend == "cuml" else None),
        hdbscan_seconds=hdbscan_seconds,
        preparation_seconds=preparation_seconds,
    )


def write_clustered_source_file(
    result: ClusteredSourceFile,
    window_length: int,
    window_stride: int,
    writer: PredictionRunWriter | PDWStudioRunWriter | MetricsOnlyRunWriter,
    compute_metrics: bool = True,
) -> SourceFileTiming:
    """Write one worker result; called serially to keep manifests thread-safe."""
    preparation_started = time.perf_counter()
    with h5py.File(result.source_file, "r") as handle:
        raw_pdws = (
            np.empty((0, 0), dtype=np.float32)
            if isinstance(writer, MetricsOnlyRunWriter)
            else np.asarray(handle["data"][:])
        )
        truth = (
            np.asarray(handle["labels"][:]).reshape(-1).astype(np.int64)
            if "labels" in handle
            else None
        )
    preparation_seconds = time.perf_counter() - preparation_started
    evaluation_started = time.perf_counter()
    metrics = (
        evaluate_labels(result.predicted_labels, truth)
        if truth is not None and compute_metrics
        else None
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    preparation_started = time.perf_counter()
    pulse_count = len(result.predicted_labels)
    if not isinstance(writer, MetricsOnlyRunWriter) and len(raw_pdws) != pulse_count:
        raise RuntimeError("Embeddings do not reconstruct the complete source file")
    metadata = SourceFileMetadata(
        source_file=str(result.source_file),
        file_index=result.file_index,
        pulse_count=pulse_count,
        window_count=result.window_count,
        window_length=window_length,
        window_stride=window_stride,
    )
    preparation_seconds += time.perf_counter() - preparation_started
    write_started = time.perf_counter()
    writer.write(
        metadata,
        raw_pdws,
        result.predicted_labels,
        truth,
        result.window_starts,
        result.window_valid_lengths,
        metrics,
    )
    write_seconds = time.perf_counter() - write_started
    return SourceFileTiming(
        pulse_count=pulse_count,
        predicted_cluster_count=len(np.unique(result.predicted_labels)),
        effective_min_cluster_size=result.effective_min_cluster_size,
        effective_min_samples=result.effective_min_samples,
        hdbscan_seconds=result.hdbscan_seconds,
        preparation_seconds=result.preparation_seconds + preparation_seconds,
        evaluation_seconds=evaluation_seconds,
        write_seconds=write_seconds,
    )


def finalize_source_file(
    file_index: int,
    windows: list[BufferedWindow],
    source_file: Path,
    window_length: int,
    window_stride: int,
    min_cluster_size: int,
    min_cluster_fraction: float,
    min_samples: int | None,
    min_samples_fraction: float | None,
    allow_single_cluster: bool,
    hdbscan_jobs: int,
    hdbscan_backend: str,
    hdbscan_device: int,
    writer: PredictionRunWriter | PDWStudioRunWriter | MetricsOnlyRunWriter,
    compute_metrics: bool = True,
) -> SourceFileTiming:
    """Synchronously cluster and write one complete source file."""
    result = cluster_source_file(
        file_index,
        windows,
        source_file,
        min_cluster_size,
        min_cluster_fraction,
        min_samples,
        min_samples_fraction,
        allow_single_cluster,
        hdbscan_jobs,
        hdbscan_backend,
        hdbscan_device,
    )
    return write_clustered_source_file(
        result,
        window_length,
        window_stride,
        writer,
        compute_metrics=compute_metrics,
    )


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.hdf5_cache_size <= 0
        or args.hdbscan_jobs <= 0
        or (args.hdbscan_device is not None and args.hdbscan_device < 0)
        or (
            args.hdbscan_parallel_files is not None and args.hdbscan_parallel_files <= 0
        )
        or (args.min_cluster_fraction is not None and args.min_cluster_fraction < 0)
        or (args.min_samples is not None and args.min_samples < 1)
        or (
            args.min_samples_fraction is not None
            and args.min_samples_fraction < 0
        )
    ):
        raise ValueError(
            "batch-size/hdf5-cache-size/hdbscan-jobs must be positive; "
            "num-workers cannot be negative"
        )
    if args.min_samples_fraction is not None and args.min_samples is None:
        raise ValueError("--min-samples-fraction requires --min-samples")
    device, data_parallel_device_ids = resolve_inference_devices(
        args.device, args.devices
    )
    if args.hdbscan_backend == "cuml" and not torch.cuda.is_available():
        raise RuntimeError("--hdbscan-backend cuml requires CUDA")
    hdbscan_device_ids = resolve_hdbscan_devices(
        args.hdbscan_backend,
        args.hdbscan_device,
        args.hdbscan_devices,
        args.hdbscan_parallel_files,
    )
    if args.hdbscan_backend == "cuml":
        # Fail before checkpoint/data loading instead of after the first file.
        load_cuml_hdbscan()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    model_config = dict(checkpoint["model_config"])
    if "architecture" not in model_config:
        state_keys = checkpoint["model_state_dict"].keys()
        model_config["architecture"] = (
            "rope_swiglu_v1"
            if any(key.startswith("blocks.") for key in state_keys)
            else "legacy_sinusoidal"
        )
    if "normalize_embeddings" not in model_config:
        model_config["normalize_embeddings"] = (
            checkpoint.get("embedding_normalization", "l2") == "l2"
        )
    model = TransformerMetricEncoder(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if len(data_parallel_device_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=data_parallel_device_ids,
            output_device=data_parallel_device_ids[0],
        )
    model.eval()

    window_length = (
        args.window_length
        if args.window_length is not None
        else int(checkpoint.get("window_length", 1024))
    )
    window_stride = (
        args.window_stride
        if args.window_stride is not None
        else (
            max(1, window_length // 2)
            if args.window_length is not None
            else int(checkpoint.get("window_stride", window_length))
        )
    )
    min_cluster_size = (
        args.min_cluster_size
        if args.min_cluster_size is not None
        else int(checkpoint.get("min_cluster_size", 5))
    )
    min_cluster_fraction = (
        args.min_cluster_fraction
        if args.min_cluster_fraction is not None
        else float(checkpoint.get("min_cluster_fraction", 0.0))
    )
    checkpoint_min_samples = checkpoint.get("min_samples")
    min_samples = (
        args.min_samples
        if args.min_samples is not None
        else (
            None
            if checkpoint_min_samples is None
            else int(checkpoint_min_samples)
        )
    )
    checkpoint_min_samples_fraction = checkpoint.get("min_samples_fraction")
    min_samples_fraction = (
        args.min_samples_fraction
        if args.min_samples_fraction is not None
        else (
            None
            if checkpoint_min_samples_fraction is None
            else float(checkpoint_min_samples_fraction)
        )
    )
    allow_single_cluster = (
        args.allow_single_cluster
        if args.allow_single_cluster is not None
        else bool(checkpoint.get("allow_single_cluster", False))
    )
    if min_cluster_size < 2:
        raise ValueError("min-cluster-size must be at least 2")
    if not 0 < window_stride <= window_length:
        raise ValueError("window-stride must be in [1, window-length]")
    normalization = str(checkpoint.get("normalization", "per_window"))
    normalization_mean = checkpoint.get("normalization_mean")
    normalization_std = checkpoint.get("normalization_std")
    if normalization == "global" and (
        normalization_mean is None or normalization_std is None
    ):
        raise ValueError("Global-normalized checkpoint is missing mean/std")
    all_files = resolve_h5_files(args.data_dir, split=args.split)
    files = select_source_files(all_files, args.source_files)
    if args.source_files:
        print(
            "Selected source files: "
            + ", ".join(path.name for path in files),
            flush=True,
        )
    dataset = H5WindowDataset(
        files,
        window_length,
        window_stride=window_stride,
        require_labels=not args.allow_unlabeled,
        return_raw=False,
        hdf5_cache_size=args.hdf5_cache_size,
        normalization=normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_progress=normalization == "per_file",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    run_metadata = {
        "checkpoint": str(checkpoint_path),
        "training_design": checkpoint.get("training_design", "legacy_window_v1"),
        "architecture": model_config["architecture"],
        "embedding_normalization": (
            "l2" if model_config["normalize_embeddings"] else "none"
        ),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "split": args.split,
        "available_source_file_count": len(all_files),
        "selected_source_files": [path.name for path in files],
        "window_length": window_length,
        "window_stride": window_stride,
        "min_cluster_size": min_cluster_size,
        "min_cluster_fraction": min_cluster_fraction,
        "min_samples": min_samples,
        "min_samples_fraction": min_samples_fraction,
        "min_samples_mode": (
            "coupled_to_min_cluster_size"
            if min_samples is None
            else (
                "explicit_fixed"
                if min_samples_fraction is None
                else "explicit_adaptive"
            )
        ),
        "allow_single_cluster": allow_single_cluster,
        "normalization": normalization,
        "labels_present": dataset.has_labels,
        "embedding_scope": "overlapping_sliding_windows",
        "overlap_aggregation": "arithmetic_mean_without_post_normalization",
        "clustering_scope": "complete_source_file",
        "hdbscan_backend": args.hdbscan_backend,
        "hdbscan_device": (
            hdbscan_device_ids[0]
            if args.hdbscan_backend == "cuml" and len(hdbscan_device_ids) == 1
            else None
        ),
        "hdbscan_devices": hdbscan_device_ids,
        "hdbscan_parallel_files": (
            len(hdbscan_device_ids) if args.hdbscan_backend == "cuml" else 1
        ),
        "hdbscan_jobs": (
            args.hdbscan_jobs if args.hdbscan_backend == "sklearn" else None
        ),
        "minus_one_is_ordinary_cluster": True,
        "evaluation_enabled": args.evaluate and dataset.has_labels,
        "inference_devices": (
            [f"cuda:{device_id}" for device_id in data_parallel_device_ids]
            if data_parallel_device_ids
            else [str(device)]
        ),
        "model_parallelism": (
            "torch_data_parallel"
            if len(data_parallel_device_ids) > 1
            else "single_device"
        ),
        "output_format": args.output_format,
    }
    writer: PredictionRunWriter | PDWStudioRunWriter | MetricsOnlyRunWriter
    if args.output_format == "pdw_studio":
        writer = PDWStudioRunWriter(args.output_dir, run_metadata)
    elif args.output_format == "metrics_only":
        if not args.evaluate:
            raise ValueError("--output-format metrics_only requires --evaluate")
        writer = MetricsOnlyRunWriter(args.output_dir, run_metadata)
    else:
        writer = PredictionRunWriter(args.output_dir, run_metadata)

    device_description = (
        ", ".join(
            f"cuda:{device_id} ({torch.cuda.get_device_name(device_id)})"
            for device_id in data_parallel_device_ids
        )
        if data_parallel_device_ids
        else "cpu (CPU)"
    )
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    hdbscan_description = (
        "cuML GPUs "
        + ", ".join(
            f"cuda:{device_id} ({torch.cuda.get_device_name(device_id)})"
            for device_id in hdbscan_device_ids
        )
        + f", parallel_files={len(hdbscan_device_ids)}"
        if args.hdbscan_backend == "cuml"
        else f"sklearn CPU workers={args.hdbscan_jobs}"
    )
    print(
        f"devices={device_description}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}, "
        f"source_files={len(files):,}, "
        f"windows={len(dataset):,}, batch_size={args.batch_size}, "
        f"window_length={window_length:,}, window_stride={window_stride:,}, "
        f"complete-file HDBSCAN={hdbscan_description}, "
        f"output_format={args.output_format}"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_enabled = args.amp and device.type == "cuda"
    current_file_index: int | None = None
    buffered_windows: list[BufferedWindow] = []
    embedded_window_count = 0
    embedding_seconds = 0.0
    completed_file_count = 0
    hdbscan_seconds = 0.0
    hdbscan_work = 0.0
    other_finalize_seconds = 0.0
    finalized_pulses = 0
    latest_eta_text = "calibrating"
    hdbscan_parallelism = (
        len(hdbscan_device_ids) if args.hdbscan_backend == "cuml" else 1
    )
    parallel_hdbscan = hdbscan_parallelism > 1
    single_hdbscan_device = hdbscan_device_ids[0] if hdbscan_device_ids else 0

    progress_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
    embedding_progress = tqdm(
        total=len(dataset),
        desc="Embedding windows",
        unit="win",
        dynamic_ncols=True,
        bar_format=progress_format,
        position=0,
        mininterval=1.0,
        file=sys.stdout,
    )
    file_progress = tqdm(
        total=len(files),
        desc="Complete files",
        unit="file",
        dynamic_ncols=True,
        bar_format=progress_format,
        position=1,
        mininterval=1.0,
        file=sys.stdout,
    )

    executors: dict[int, ThreadPoolExecutor] = {}
    available_hdbscan_devices: deque[int] = deque()
    cluster_futures: dict[Future[ClusteredSourceFile], tuple[int, int, Path]] = {}
    if parallel_hdbscan:
        for hdbscan_device in hdbscan_device_ids:
            executors[hdbscan_device] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"hdbscan-cuda-{hdbscan_device}",
            )
            available_hdbscan_devices.append(hdbscan_device)

    def active_hdbscan_text() -> str:
        active = sorted(
            (
                device_id,
                source_file.name,
            )
            for _, device_id, source_file in cluster_futures.values()
        )
        return (
            ", ".join(
                f"cuda:{device_id}={source_name}" for device_id, source_name in active
            )
            or "idle"
        )

    def record_completion(
        source_file: Path,
        timing: SourceFileTiming,
        hdbscan_device: int | None,
    ) -> None:
        nonlocal completed_file_count
        nonlocal finalized_pulses
        nonlocal hdbscan_seconds
        nonlocal hdbscan_work
        nonlocal latest_eta_text
        nonlocal other_finalize_seconds

        completed_file_count += 1
        finalized_pulses += timing.pulse_count
        hdbscan_seconds += timing.hdbscan_seconds
        hdbscan_work += float(timing.pulse_count) ** 2
        other_finalize_seconds += timing.other_seconds
        eta_seconds = estimate_remaining_seconds(
            total_windows=len(dataset),
            embedded_windows=embedded_window_count,
            embedding_seconds=embedding_seconds,
            file_lengths=dataset.lengths,
            completed_files=completed_file_count,
            hdbscan_seconds=hdbscan_seconds,
            hdbscan_work=hdbscan_work,
            other_seconds=other_finalize_seconds,
            finalized_pulses=finalized_pulses,
            hdbscan_parallel_files=hdbscan_parallelism,
        )
        eta_text = (
            format_duration(eta_seconds) if eta_seconds is not None else "calibrating"
        )
        latest_eta_text = eta_text
        file_progress.update(1)
        device_label = (
            f"cuda:{hdbscan_device}" if hdbscan_device is not None else "CPU"
        )
        gpu_text = f", GPU={device_label}" if hdbscan_device is not None else ""
        file_progress.set_postfix_str(
            f"last={source_file.name}, HDBSCAN={format_duration(timing.hdbscan_seconds)}, "
            f"total={format_duration(timing.total_seconds)}{gpu_text}, ETA~{eta_text}"
        )
        file_progress.refresh()
        tqdm.write(
            f"[file {completed_file_count}/{len(files)}] {source_file.name}: "
            f"pulses={timing.pulse_count:,}, "
            f"predicted_clusters={timing.predicted_cluster_count:,}, "
            f"min_cluster={timing.effective_min_cluster_size:,}, "
            f"min_samples={timing.effective_min_samples:,}, "
            f"device={device_label}, "
            f"HDBSCAN={format_duration(timing.hdbscan_seconds)}, "
            f"evaluation={format_duration(timing.evaluation_seconds) if args.evaluate else 'skipped'}, "
            f"write={format_duration(timing.write_seconds)}, "
            f"other={format_duration(timing.other_seconds)}, remaining~{eta_text}",
            file=sys.stdout,
        )

    def collect_cluster_results(block: bool) -> None:
        if not cluster_futures:
            return
        if block:
            completed, _ = wait(tuple(cluster_futures), return_when=FIRST_COMPLETED)
        else:
            completed = {future for future in cluster_futures if future.done()}
        for future in sorted(completed, key=lambda item: cluster_futures[item][0]):
            _, assigned_device, source_file = cluster_futures.pop(future)
            try:
                result = future.result()
                timing = write_clustered_source_file(
                    result,
                    window_length,
                    window_stride,
                    writer,
                    compute_metrics=args.evaluate,
                )
            finally:
                available_hdbscan_devices.append(assigned_device)
            record_completion(source_file, timing, assigned_device)
        if cluster_futures:
            file_progress.set_postfix_str(
                f"stage=HDBSCAN, active=[{active_hdbscan_text()}], "
                f"ETA~{latest_eta_text}"
            )
            file_progress.refresh()

    def finalize_or_submit(file_index: int, windows: list[BufferedWindow]) -> None:
        source_file = files[file_index]
        pulse_count = dataset.lengths[file_index]
        effective_minimum = effective_min_cluster_size(
            pulse_count, min_cluster_size, min_cluster_fraction
        )
        displayed_min_samples = (
            effective_minimum
            if min_samples is None
            else (
                min_samples
                if min_samples_fraction is None
                else effective_min_samples(
                    pulse_count, min_samples, min_samples_fraction
                )
            )
        )
        if not parallel_hdbscan:
            file_progress.set_postfix_str(
                f"stage=HDBSCAN, file={file_index + 1}/{len(files)} "
                f"{source_file.name}, pulses={pulse_count:,}, min_cluster="
                f"{effective_minimum:,}, min_samples={displayed_min_samples:,}, "
                f"ETA~{latest_eta_text}"
            )
            file_progress.refresh()
            timing = finalize_source_file(
                file_index,
                windows,
                source_file,
                window_length,
                window_stride,
                min_cluster_size,
                min_cluster_fraction,
                min_samples,
                min_samples_fraction,
                allow_single_cluster,
                args.hdbscan_jobs,
                args.hdbscan_backend,
                single_hdbscan_device,
                writer,
                compute_metrics=args.evaluate,
            )
            record_completion(
                source_file,
                timing,
                (single_hdbscan_device if args.hdbscan_backend == "cuml" else None),
            )
            return

        while not available_hdbscan_devices:
            collect_cluster_results(block=True)
        assigned_device = available_hdbscan_devices.popleft()
        future = executors[assigned_device].submit(
            cluster_source_file,
            file_index,
            windows,
            source_file,
            min_cluster_size,
            min_cluster_fraction,
            min_samples,
            min_samples_fraction,
            allow_single_cluster,
            args.hdbscan_jobs,
            args.hdbscan_backend,
            assigned_device,
        )
        cluster_futures[future] = (
            file_index,
            assigned_device,
            source_file,
        )
        file_progress.set_postfix_str(
            f"stage=HDBSCAN, active=[{active_hdbscan_text()}], "
            f"ETA~{latest_eta_text}"
        )
        file_progress.refresh()

    try:
        with torch.inference_mode():
            loader_iterator = iter(loader)
            while True:
                embedding_step_started = time.perf_counter()
                try:
                    batch = next(loader_iterator)
                except StopIteration:
                    break
                (
                    features,
                    _labels,
                    padding_mask,
                    file_indices,
                    pulse_starts,
                    valid_lengths,
                    _dataset_window_indices,
                ) = batch
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    embeddings = model(
                        features.to(device, non_blocking=True),
                        padding_mask=padding_mask.to(device, non_blocking=True),
                    )
                embeddings_np = embeddings.float().cpu().numpy()
                del embeddings
                embedding_seconds += time.perf_counter() - embedding_step_started
                batch_window_count = len(features)
                embedded_window_count += batch_window_count
                embedding_progress.update(batch_window_count)
                active_rate = embedded_window_count / max(embedding_seconds, 1e-9)
                embedding_progress.set_postfix_str(
                    f"active={format_duration(embedding_seconds)}, "
                    f"rate={active_rate:.1f} win/s"
                )
                embedding_progress.refresh()
                if parallel_hdbscan:
                    collect_cluster_results(block=False)
                for (
                    embedding,
                    file_index,
                    pulse_start,
                    valid_length,
                ) in zip(
                    embeddings_np,
                    file_indices.numpy(),
                    pulse_starts.numpy(),
                    valid_lengths.numpy(),
                    strict=True,
                ):
                    file_id = int(file_index)
                    if current_file_index is not None and file_id != current_file_index:
                        if file_id < current_file_index:
                            raise RuntimeError(
                                "Prediction windows are not in source order"
                            )
                        finalize_or_submit(
                            current_file_index,
                            buffered_windows,
                        )
                        buffered_windows = []
                    current_file_index = file_id
                    length = int(valid_length)
                    buffered_windows.append(
                        BufferedWindow(
                            embedding=embedding[:length].copy(),
                            pulse_start=int(pulse_start),
                            valid_length=length,
                        )
                    )
            if current_file_index is not None:
                finalize_or_submit(
                    current_file_index,
                    buffered_windows,
                )
            while cluster_futures:
                collect_cluster_results(block=True)
        summary = writer.close()
    except BaseException:
        for future in cluster_futures:
            future.cancel()
        writer.abort()
        raise
    finally:
        for executor in executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
        embedding_progress.close()
        file_progress.close()

    metrics = summary["metrics"]
    v_measure = metrics.get("V-measure", {}) if isinstance(metrics, dict) else {}
    v_text = (
        f", file-macro mean V-measure={v_measure['mean']:.4f}"
        if isinstance(v_measure, dict) and "mean" in v_measure
        else ""
    )
    print(
        f"Processed {summary['source_file_count']:,} complete source-file results "
        f"({summary['total_embedding_window_count']:,} embedding windows) "
        f"to {writer.output_dir}; "
        f"evaluated source files={summary['evaluated_source_file_count']:,}{v_text}"
    )


if __name__ == "__main__":
    main()
