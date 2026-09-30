#!/usr/bin/env python3
"""Train the Transformer metric-learning baseline on all ordered HDF5 windows."""

from __future__ import annotations

import argparse
import faulthandler
import logging
import multiprocessing as mp
import os
import signal
import sys
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from collections.abc import Sequence
from typing import Callable, TextIO

import h5py
import numpy as np
import torch
import torch.distributed as dist
from sklearn.cluster import HDBSCAN
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from turing_deinterleaving_challenge.models import (
    PDWStandardizer,
    TransformerMetricEncoder,
    triplet_metric_loss,
)
from turing_deinterleaving_challenge.models.evaluate import evaluate_labels

from experiment_config import load_config_defaults


_FAULT_LOG_FILE: TextIO | None = None
_STOP_REQUESTED = False
IMPLEMENTATION_TAG = "ddp-validation-v3-process-hdbscan-embed8"
METRIC_NAMES = (
    "Homogeneity",
    "Completeness",
    "V-measure",
    "Adjusted Rand Index",
    "Adjusted Mutual Information",
    "MCC",
    "F1",
    "discount",
)


def install_stop_signal_handlers() -> None:
    """Exit a DDP rank immediately when the user interrupts the run.

    CPU HDBSCAN jobs can take minutes to return to Python.  A deferred signal
    handler therefore makes Ctrl+C appear ineffective and can strand the peer
    rank in a collective.  ``torchrun`` observes this rank's exit and tears
    down the remaining ranks; the operating system reclaims CUDA/HDF5 handles.
    """

    def exit_now(signum: int, _frame: object) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        os._exit(128 + signum)

    signal.signal(signal.SIGINT, exit_now)
    signal.signal(signal.SIGTERM, exit_now)


def stop_requested() -> bool:
    return _STOP_REQUESTED


def configure_logger(log_dir: Path, rank: int, *, truncate: bool) -> logging.LoggerAdapter:
    """Write concise DDP lifecycle events to one rank-labelled log file."""
    global _FAULT_LOG_FILE
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("transformer_metric")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s rank=%(rank)d pid=%(process)s %(message)s"
    )
    # Rank zero truncates the previous invocation before rank one is allowed
    # to append.  Both processes then write their small set of lifecycle lines
    # to this same file.
    log_path = log_dir / "train.log"
    file_handler = logging.FileHandler(
        log_path, mode="w" if truncate else "a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    _FAULT_LOG_FILE = log_path.open("a", encoding="utf-8")
    faulthandler.enable(_FAULT_LOG_FILE, all_threads=True)

    def log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        rank_logger.critical(
            "uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    class RankAdapter(logging.LoggerAdapter):
        def process(self, message, kwargs):
            kwargs.setdefault("extra", {})["rank"] = rank
            return message, kwargs

    rank_logger = RankAdapter(logger, {})
    sys.excepthook = log_uncaught_exception
    return rank_logger


def parse_devices(requested_device: str | None) -> tuple[torch.device, list[int] | None]:
    """Parse ``cuda:0,1`` for single-process ``torch.nn.DataParallel``."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), None
    if requested_device.startswith("cuda:") and "," in requested_device:
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda:0,1 requires CUDA")
        device_ids = [int(index) for index in requested_device.removeprefix("cuda:").split(",")]
        if len(device_ids) < 2 or len(set(device_ids)) != len(device_ids):
            raise ValueError("multi-GPU --device must list distinct IDs, e.g. cuda:0,1")
        if min(device_ids) < 0 or max(device_ids) >= torch.cuda.device_count():
            raise ValueError("a requested CUDA device is not visible to PyTorch")
        return torch.device(f"cuda:{device_ids[0]}"), device_ids
    return torch.device(requested_device), None


def distributed_context(
    requested_device: str | None,
) -> tuple[torch.device, list[int] | None, int, int, bool]:
    """Initialize DDP when launched by torchrun, otherwise use one process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device, device_ids = parse_devices(requested_device)
        return device, device_ids, 0, 1, False
    if requested_device not in (None, "cuda"):
        raise ValueError("DDP requires --device cuda; torchrun chooses each local GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}"), None, dist.get_rank(), world_size, True


def create_control_group(is_distributed: bool):
    """Create a CPU control channel for validation/early-stop coordination.

    Rank zero can spend far longer than NCCL's watchdog timeout in CPU HDBSCAN
    validation.  This group keeps rank one waiting outside an NCCL collective.
    """
    if not is_distributed:
        return None
    return dist.new_group(backend="gloo", timeout=timedelta(hours=4))


class ExactDistributedSampler(Sampler[int]):
    """Assign every window to exactly one DDP rank, with no padding or repeats."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int, shuffle: bool, seed: int) -> None:
        self.num_samples = len(dataset)
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = list(range(self.num_samples))
        return iter(indices[self.rank :: self.world_size])

    def __len__(self) -> int:
        return max(0, (self.num_samples - self.rank + self.world_size - 1) // self.world_size)


def resolve_h5_files(path: Path, split: str | None = None) -> list[Path]:
    """Resolve the standard TSRD layouts for a requested data split.

    Different releases place scan files under either ``train`` or
    ``train_scan``, sometimes nested inside an additional ``scan`` directory.
    This mirrors the permissive directory handling of the raw-HDBSCAN script.
    """
    names = (split, f"{split}_scan") if split else ()
    if split == "validation":
        names = (*names, "val", "val_scan")
    candidates = (
        tuple(path / name for name in names)
        + tuple(path / "scan" / name for name in names)
        + (path / "scan", path)
        if split
        else (path,)
    )
    for candidate in candidates:
        files = sorted(candidate.glob("*.h5"))
        if files:
            return files
    checked = ", ".join(str(candidate) for candidate in candidates)
    discovered = list(path.rglob("*.h5")) if path.is_dir() else []
    hint = (
        f" Found {len(discovered)} .h5 file(s) elsewhere below the supplied path; "
        "pass the directory directly containing the requested split."
        if discovered
        else " No HDF5 dataset files were found below the supplied path; download or mount TSRD first."
    )
    raise FileNotFoundError(f"No {split!r} .h5 files found in: {checked}.{hint}")


def split_train_files(
    files: list[Path], validation_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Create a reproducible file-level holdout when TSRD has no validation split."""
    if len(files) < 2:
        raise ValueError("at least two train files are required for a train/validation split")
    validation_count = max(1, round(len(files) * validation_fraction))
    validation_count = min(validation_count, len(files) - 1)
    order = np.random.default_rng(seed).permutation(len(files))
    validation_indices = set(order[:validation_count].tolist())
    train_files = [path for index, path in enumerate(files) if index not in validation_indices]
    validation_files = [path for index, path in enumerate(files) if index in validation_indices]
    return train_files, validation_files


class ExhaustivePulseWindowDataset(Dataset):
    """Visit every pulse in every source file once per epoch.

    Windows are non-overlapping. A final partial window is zero-padded by the
    local ``__getitem__`` implementation, so trailing pulses are retained
    without duplicating the preceding complete window. Windows remain in source
    order across epochs.
    """

    def __init__(
        self,
        files: list[Path],
        window_length: int,
        standardizer: PDWStandardizer | None = None,
        hdf5_cache_size: int = 32,
    ) -> None:
        if window_length < 2:
            raise ValueError("window_length must be at least two")
        if hdf5_cache_size <= 0:
            raise ValueError("hdf5_cache_size must be positive")
        self.files = files
        self.window_length = window_length
        self.standardizer = standardizer
        self.hdf5_cache_size = hdf5_cache_size
        self.lengths = []
        self.feature_dim: int | None = None
        for path in files:
            with h5py.File(path, "r") as handle:
                if "data" not in handle or "labels" not in handle:
                    raise ValueError(f"{path} must contain data and labels datasets")
                if handle["data"].ndim != 2:
                    raise ValueError(f"{path} data must have shape (sequence_length, feature_dim)")
                feature_dim = int(handle["data"].shape[1])
                if self.feature_dim is None:
                    self.feature_dim = feature_dim
                elif feature_dim != self.feature_dim:
                    raise ValueError("all HDF5 files must have the same feature dimension")
                self.lengths.append(len(handle["data"]))
        self.windows_per_file = np.asarray(
            [(length + window_length - 1) // window_length for length in self.lengths]
        )
        self.window_offsets = np.cumsum(self.windows_per_file)
        self.windows_per_epoch = int(self.window_offsets[-1]) if len(self.files) else 0
        if self.windows_per_epoch == 0:
            raise ValueError("no non-empty pulse trains were found")
        assert self.feature_dim is not None
        self._hdf5_handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._cache_pid: int | None = None

    def __getstate__(self) -> dict:
        """Ensure spawned workers open their own HDF5 handles."""
        state = self.__dict__.copy()
        state["_hdf5_handles"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def _close_hdf5_handles(self) -> None:
        for handle in self._hdf5_handles.values():
            handle.close()
        self._hdf5_handles.clear()

    def _hdf5_handle(self, file_index: int) -> h5py.File:
        """Return a worker-local LRU-cached read-only HDF5 handle."""
        pid = os.getpid()
        if self._cache_pid != pid:
            self._hdf5_handles = OrderedDict()
            self._cache_pid = pid
        try:
            handle = self._hdf5_handles.pop(file_index)
        except KeyError:
            handle = h5py.File(self.files[file_index], "r")
            if len(self._hdf5_handles) >= self.hdf5_cache_size:
                _, evicted = self._hdf5_handles.popitem(last=False)
                evicted.close()
        self._hdf5_handles[file_index] = handle
        return handle

    def window_location(self, index: int) -> tuple[int, int]:
        file_index = int(np.searchsorted(self.window_offsets, index, side="right"))
        previous_offset = 0 if file_index == 0 else int(self.window_offsets[file_index - 1])
        local_window = index - previous_offset
        return file_index, local_window * self.window_length

    def __len__(self) -> int:
        return self.windows_per_epoch

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        file_index, start = self.window_location(index)
        handle = self._hdf5_handle(file_index)
        end = min(start + self.window_length, self.lengths[file_index])
        data = handle["data"][start:end]
        labels = handle["labels"][start:end].squeeze()
        if self.standardizer is not None:
            data = self.standardizer.transform(data)
        valid_length = len(data)
        padding_mask = np.zeros(self.window_length, dtype=bool)
        if valid_length < self.window_length:
            padded_data = np.zeros((self.window_length, data.shape[1]), dtype=np.float32)
            padded_labels = np.full(self.window_length, -1, dtype=np.int64)
            padded_data[:valid_length] = data
            padded_labels[:valid_length] = labels
            padding_mask[valid_length:] = True
            return padded_data, padded_labels, padding_mask
        return data, labels.astype(np.int64, copy=False), padding_mask

    def __del__(self) -> None:
        try:
            self._close_hdf5_handles()
        except (AttributeError, OSError):
            pass


def validate(
    encoder: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    min_cluster_size: int,
    min_emitters: int,
    n_jobs: int | None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    should_stop: Callable[[], bool] | None = None,
    show_progress: bool = False,
    collect_window_scores: bool = False,
    collect_predictions: bool = False,
    window_indices: Sequence[int] | None = None,
) -> tuple[
    dict[str, float],
    int,
    int,
    int,
    dict[str, list[float]] | None,
    dict[str, np.ndarray] | None,
] | None:
    """Evaluate full, multi-emitter validation windows with HDBSCAN.

    This follows ``evaluate_hdbscan_scan.py``: a window is eligible only when
    it has no tail padding and contains at least ``min_emitters`` labels.
    The returned metrics are sums so DDP can combine disjoint rank-local
    validation shards exactly before calculating the global mean.
    """
    totals: dict[str, float] | None = None
    window_count = 0
    skipped_partial = 0
    skipped_low_emitter = 0
    window_scores = (
        {name: [] for name in METRIC_NAMES} if collect_window_scores else None
    )
    if collect_predictions and window_indices is None:
        raise ValueError("window_indices are required when collecting predictions")
    sampled_window_cursor = 0
    predicted_label_batches: list[np.ndarray] = []
    true_label_batches: list[np.ndarray] = []
    evaluated_window_indices: list[np.ndarray] = []
    encoder.eval()
    # This must be process-level parallelism. The raw HDBSCAN baseline uses a
    # spawn Pool; a thread pool does not reliably scale the Python/sklearn
    # scoring path across CPU cores.  Spawn is CUDA-safe after DDP has started.
    if n_jobs and n_jobs > 1:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        executor = ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context("spawn"),
        )
    else:
        executor = None
    try:
        with torch.no_grad(), tqdm(
            data_loader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
            position=0,
            disable=not show_progress,
        ) as progress:
            for features, labels, padding_mask in progress:
                if should_stop is not None and should_stop():
                    return None
                batch_size = len(features)
                batch_window_indices = None
                if window_indices is not None:
                    batch_window_indices = np.asarray(
                        window_indices[
                            sampled_window_cursor : sampled_window_cursor + batch_size
                        ],
                        dtype=np.int64,
                    )
                    if len(batch_window_indices) != batch_size:
                        raise RuntimeError(
                            "window_indices ended before the evaluation data loader"
                        )
                sampled_window_cursor += batch_size
                eligible, partial_count, low_emitter_count = evaluation_window_mask(
                    labels, padding_mask, min_emitters
                )
                skipped_partial += partial_count
                skipped_low_emitter += low_emitter_count
                if not eligible.any():
                    continue
                features = features[eligible]
                labels = labels[eligible]
                padding_mask = padding_mask[eligible]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    embeddings = encoder(
                        features.to(device, non_blocking=True),
                        padding_mask=padding_mask.to(device, non_blocking=True),
                    )
                embeddings = embeddings.float().cpu().numpy()
                labels_np = labels.numpy()
                masks_np = padding_mask.numpy()
                eligible_window_indices = (
                    batch_window_indices[eligible]
                    if batch_window_indices is not None
                    else None
                )
                score_iterator = (
                    executor.map(
                        score_hdbscan_window,
                        embeddings,
                        labels_np,
                        masks_np,
                        [min_cluster_size] * len(embeddings),
                    )
                    if executor is not None
                    else map(
                        score_hdbscan_window,
                        embeddings,
                        labels_np,
                        masks_np,
                        [min_cluster_size] * len(embeddings),
                    )
                )
                batch_predictions: list[np.ndarray] = []
                for score, predicted in score_iterator:
                    if should_stop is not None and should_stop():
                        return None
                    if totals is None:
                        totals = {key: 0.0 for key in score}
                    for key, value in score.items():
                        totals[key] += float(value)
                        if window_scores is not None:
                            window_scores[key].append(float(value))
                    if collect_predictions:
                        batch_predictions.append(predicted.astype(np.int32, copy=False))
                    window_count += 1
                if collect_predictions:
                    assert eligible_window_indices is not None
                    predicted_label_batches.append(np.stack(batch_predictions))
                    true_label_batches.append(labels_np.astype(np.int64, copy=False))
                    evaluated_window_indices.append(eligible_window_indices)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if window_indices is not None and sampled_window_cursor != len(window_indices):
        raise RuntimeError("window_indices contains entries not consumed by the data loader")
    predictions = None
    if collect_predictions:
        if evaluated_window_indices:
            predictions = {
                "window_indices": np.concatenate(evaluated_window_indices),
                "predicted_labels": np.concatenate(predicted_label_batches),
                "true_labels": np.concatenate(true_label_batches),
            }
        else:
            window_length = int(getattr(data_loader.dataset, "window_length", 0))
            predictions = {
                "window_indices": np.empty(0, dtype=np.int64),
                "predicted_labels": np.empty((0, window_length), dtype=np.int32),
                "true_labels": np.empty((0, window_length), dtype=np.int64),
            }
    return (
        totals or {},
        window_count,
        skipped_partial,
        skipped_low_emitter,
        window_scores,
        predictions,
    )


def summarize_validation(
    result: tuple[
        dict[str, float],
        int,
        int,
        int,
        dict[str, list[float]] | None,
        dict[str, np.ndarray] | None,
    ],
    control_group: object | None = None,
) -> tuple[dict[str, float], int, int, int]:
    """Aggregate disjoint validation shards and calculate global means."""
    totals, window_count, skipped_partial, skipped_low_emitter, _, _ = result
    values = torch.tensor(
        [
            *(totals.get(name, 0.0) for name in METRIC_NAMES),
            window_count,
            skipped_partial,
            skipped_low_emitter,
        ],
        dtype=torch.float64,
    )
    if control_group is not None:
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=control_group)
    total_window_count = int(values[len(METRIC_NAMES)].item())
    if total_window_count == 0:
        raise RuntimeError(
            "validation contained no full windows with the requested minimum emitter count"
        )
    metrics = {
        name: float(values[index].item() / total_window_count)
        for index, name in enumerate(METRIC_NAMES)
    }
    return (
        metrics,
        total_window_count,
        int(values[len(METRIC_NAMES) + 1].item()),
        int(values[len(METRIC_NAMES) + 2].item()),
    )


def evaluation_window_mask(
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    min_emitters: int,
) -> tuple[np.ndarray, int, int]:
    """Select the exact evaluation population used by the raw HDBSCAN script.

    ``DeinterleavingChallengeDataset`` in ``evaluate_hdbscan_scan.py`` only
    creates full windows, then filters by the number of unique true labels.
    The Transformer dataset retains padded tail windows for training, so this
    lightweight selector applies the same policy only at validation/test time.
    """
    if min_emitters <= 0:
        raise ValueError("min_emitters must be positive")
    labels_np = labels.numpy()
    masks_np = padding_mask.numpy()
    complete = ~masks_np.any(axis=1)
    emitter_counts = np.fromiter(
        (np.unique(label[~mask]).size for label, mask in zip(labels_np, masks_np, strict=True)),
        dtype=np.int64,
        count=len(labels_np),
    )
    enough_emitters = emitter_counts >= min_emitters
    eligible = complete & enough_emitters
    return (
        eligible,
        int((~complete).sum()),
        int((complete & ~enough_emitters).sum()),
    )


def score_hdbscan_window(
    embedding: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    min_cluster_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Cluster and score one independent PDW window using one CPU worker."""
    valid = ~mask
    if valid.sum() < min_cluster_size:
        predicted = np.full(valid.sum(), -1, dtype=np.int64)
    else:
        predicted = HDBSCAN(
            min_cluster_size=min_cluster_size,
            n_jobs=1,
            copy=False,
        ).fit_predict(embedding[valid])
    return evaluate_labels(predicted, label[valid]), predicted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Optional validation root. If omitted, hold out files from the train split.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="File-level fraction held out from training when --validation-dir is omitted.",
    )
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--hdf5-cache-size",
        type=int,
        default=32,
        help="Maximum worker-local open HDF5 files (default: 32).",
    )
    parser.add_argument(
        "--shuffle-train-windows",
        action="store_true",
        help="Shuffle the order of all training windows each epoch without sampling them.",
    )
    parser.add_argument(
        "--n-jobs",
        "--num-workers",
        dest="n_jobs",
        type=int,
        default=0,
        help=(
            "Parallel workers for lazy HDF5 training-window reads; also used by "
            "validation to cluster independent windows concurrently."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision for Transformer training and validation.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="AMP floating-point format (default: bf16).",
    )
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument(
        "--min-emitters",
        type=int,
        default=2,
        help=(
            "Minimum unique true emitters in a full validation window (default: 2). "
            "Matches evaluate_hdbscan_scan.py; it does not filter training windows."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many consecutive epochs without V-measure improvement (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("transformer_metric.pt"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the combined training and fatal-signal log. "
            "Defaults to <output-parent>/<output-stem>/logs."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file whose train and model sections provide command defaults.",
    )
    valid_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(
        **load_config_defaults(
            argv,
            sections=("train", "model"),
            valid_destinations=valid_destinations,
        )
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    install_stop_signal_handlers()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.n_jobs < 0
        or args.hdf5_cache_size <= 0
        or args.early_stopping_patience <= 0
        or args.model_dim <= 0
        or args.num_layers <= 0
        or args.num_heads <= 0
        or args.embedding_dim <= 0
        or args.feedforward_dim <= 0
        or args.min_emitters <= 0
        or not 0 <= args.dropout < 1
        or not 0 < args.validation_fraction < 1
    ):
        raise ValueError(
            "positive training/model sizes are required; "
            "--n-jobs cannot be negative; --min-emitters must be positive; "
            "--validation-fraction must be in (0, 1)"
        )
    # Put logs beside the specific model by default, rather than mixing logs
    # from distinct checkpoints in a shared results/logs directory.
    default_log_dir = args.output.parent / args.output.stem / "logs"
    log_dir = (args.log_dir or default_log_dir).expanduser().resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    device, device_ids, rank, world_size, is_distributed = distributed_context(args.device)
    if is_distributed:
        if rank == 0:
            logger = configure_logger(log_dir, rank, truncate=True)
            logger.info("run started config=%s output=%s", args.config, args.output)
        dist.barrier()
        if rank != 0:
            logger = configure_logger(log_dir, rank, truncate=False)
            logger.info("rank joined run")
        dist.barrier()
    else:
        logger = configure_logger(log_dir, rank, truncate=True)
        logger.info("run started config=%s output=%s", args.config, args.output)
    control_group = create_control_group(is_distributed)
    logger.info(
        "runtime implementation=%s distributed=%s rank=%d/%d device=%s data_parallel=%s torch=%s",
        IMPLEMENTATION_TAG,
        is_distributed,
        rank,
        world_size,
        device,
        device_ids,
        torch.__version__,
    )
    if rank == 0:
        print(
            f"Transformer baseline implementation={IMPLEMENTATION_TAG} "
            f"embedding_dim={args.embedding_dim} "
            f"hdbscan_workers_per_rank={args.n_jobs}",
            flush=True,
        )
    if is_distributed and args.batch_size % world_size:
        raise ValueError("--batch-size must be divisible by the DDP world size")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    all_train_files = resolve_h5_files(args.train_dir.expanduser().resolve(), "train")
    if args.validation_dir is None:
        train_files, validation_files = split_train_files(
            all_train_files, args.validation_fraction, args.seed
        )
        if rank == 0:
            print(
                f"No validation directory supplied: holding out {len(validation_files)}/"
                f"{len(all_train_files)} train files for validation."
            )
    else:
        train_files = all_train_files
        validation_files = resolve_h5_files(
            args.validation_dir.expanduser().resolve(), "validation"
        )
    train_dataset: Dataset = ExhaustivePulseWindowDataset(
        train_files, args.window_length, PDWStandardizer(), args.hdf5_cache_size
    )
    if rank == 0:
        print(f"Training on all {len(train_dataset):,} ordered windows per epoch.")
    loader_kwargs: dict[str, object] = {}
    if args.n_jobs:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        )
    train_sampler = (
        ExactDistributedSampler(
            train_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=args.shuffle_train_windows,
            seed=args.seed,
        )
        if is_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size if is_distributed else args.batch_size,
        num_workers=args.n_jobs,
        sampler=train_sampler,
        shuffle=args.shuffle_train_windows if train_sampler is None else False,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    validation_dataset: Dataset = ExhaustivePulseWindowDataset(
        validation_files, args.window_length, PDWStandardizer(), args.hdf5_cache_size
    )
    if rank == 0:
        print(
            f"Validating eligible full windows from {len(validation_dataset):,} ordered candidates "
            f"(min_emitters={args.min_emitters})."
        )
    if rank == 0:
        logger.info(
            "setup train_windows=%d validation_candidates=%d min_emitters=%d window_length=%d global_batch=%d "
            "workers_per_rank=%d amp=%s(%s) model=%dlayer-%ddim-%dhead",
            len(train_dataset),
            len(validation_dataset),
            args.min_emitters,
            args.window_length,
            args.batch_size,
            args.n_jobs,
            amp_enabled,
            args.amp_dtype,
            args.num_layers,
            args.model_dim,
            args.num_heads,
        )
    validation_sampler = (
        ExactDistributedSampler(
            validation_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            seed=args.seed,
        )
        if is_distributed
        else None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size // world_size if is_distributed else args.batch_size,
        sampler=validation_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    input_dim = train_dataset.feature_dim
    model_config = {
        "input_dim": input_dim,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
    }
    encoder = TransformerMetricEncoder(**model_config).to(device)
    training_model: TransformerMetricEncoder | DataParallel | DistributedDataParallel = encoder
    if is_distributed:
        training_model = DistributedDataParallel(encoder, device_ids=[device.index])
    elif device_ids is not None:
        training_model = DataParallel(encoder, device_ids=device_ids, output_device=device_ids[0])
    optimizer = torch.optim.AdamW(
        training_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = None
    if amp_enabled and amp_dtype == torch.float16:
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()
    best_v_measure = float("-inf")
    epochs_without_improvement = 0
    interrupted = False
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        if rank == 0:
            logger.info("epoch=%d/%d training started", epoch + 1, args.epochs)
        losses: list[float] = []
        progress = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stdout,
                position=0,
            )
            if rank == 0
            else None
        )
        train_iterator = progress if progress is not None else train_loader
        should_stop = False
        join_context = training_model.join() if is_distributed else nullcontext()
        with join_context:
            for features, labels, padding_mask in train_iterator:
                padding_mask = padding_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    embeddings = training_model(
                        features.to(device, non_blocking=True), padding_mask=padding_mask
                    )
                    loss = triplet_metric_loss(
                        embeddings, labels, padding_mask=padding_mask, margin=args.margin
                    )
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if rank == 0:
                    losses.append(float(loss.detach().cpu()))
                if stop_requested():
                    interrupted = True
                    break
        if progress is not None:
            progress.close()

        # If torchrun forwarded Ctrl+C to either rank, agree on stopping before
        # rank zero begins the lengthy CPU-side validation phase.
        if is_distributed:
            stop_tensor = torch.tensor(int(interrupted or stop_requested()), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            interrupted = bool(stop_tensor.item())

        validation_result = None
        if not interrupted:
            validation_result = validate(
                encoder,
                validation_loader,
                device,
                args.min_cluster_size,
                args.min_emitters,
                None if args.n_jobs == 0 else args.n_jobs,
                amp_enabled,
                amp_dtype,
                stop_requested,
                show_progress=rank == 0,
            )
            interrupted = validation_result is None

        # Both ranks score disjoint windows. Use Gloo for this long CPU-side
        # phase so HDBSCAN cannot trip NCCL's watchdog.
        if is_distributed:
            interrupt_tensor = torch.tensor(int(interrupted or stop_requested()))
            dist.all_reduce(interrupt_tensor, op=dist.ReduceOp.MAX, group=control_group)
            interrupted = bool(interrupt_tensor.item())

        if not interrupted:
            assert validation_result is not None
            metrics, validation_window_count, skipped_partial, skipped_low_emitter = (
                summarize_validation(
                    validation_result,
                    control_group if is_distributed else None,
                )
            )
            if rank == 0:
                tqdm.write(
                    f"epoch={epoch + 1} triplet_loss={np.mean(losses):.4f} "
                    f"V-measure={metrics['V-measure']:.4f} "
                    f"AMI={metrics['Adjusted Mutual Information']:.4f} "
                    f"eval_windows={validation_window_count}",
                    file=sys.stdout,
                )
                logger.info(
                    "epoch=%d training_loss=%.6f v_measure=%.6f ami=%.6f "
                    "validation_windows=%d skipped_partial=%d skipped_low_emitter=%d",
                    epoch + 1,
                    np.mean(losses),
                    metrics["V-measure"],
                    metrics["Adjusted Mutual Information"],
                    validation_window_count,
                    skipped_partial,
                    skipped_low_emitter,
                )
                if metrics["V-measure"] > best_v_measure:
                    best_v_measure = metrics["V-measure"]
                    epochs_without_improvement = 0
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": encoder.state_dict(),
                            "model_config": model_config,
                            "normalization": "per_window",
                            "amp": args.amp,
                            "evaluation_window_policy": {
                                "full_windows_only": True,
                                "min_emitters": args.min_emitters,
                            },
                            "validation_metrics": metrics,
                        },
                        args.output,
                    )
                    tqdm.write(
                        f"saved best checkpoint to {args.output} (V-measure={best_v_measure:.4f})",
                        file=sys.stdout,
                    )
                    logger.info("saved checkpoint=%s v_measure=%.6f", args.output, best_v_measure)
                else:
                    epochs_without_improvement += 1
                    tqdm.write(
                        "no V-measure improvement for "
                        f"{epochs_without_improvement}/{args.early_stopping_patience} epoch(s)",
                        file=sys.stdout,
                    )
                    should_stop = epochs_without_improvement >= args.early_stopping_patience
                    if should_stop:
                        tqdm.write(
                            "early stopping: V-measure did not improve within the patience limit",
                            file=sys.stdout,
                        )
        if is_distributed:
            # Use the CPU/Gloo control group while rank zero performs its long
            # CPU-side validation. Do not leave rank one in an NCCL collective.
            interrupt_tensor = torch.tensor(int(interrupted or stop_requested()))
            dist.broadcast(interrupt_tensor, src=0, group=control_group)
            interrupted = bool(interrupt_tensor.item())
            stop_tensor = torch.tensor(int(should_stop))
            dist.broadcast(stop_tensor, src=0, group=control_group)
            should_stop = bool(stop_tensor.item())
        if should_stop or interrupted:
            break
    if is_distributed:
        if interrupted:
            if rank == 0:
                logger.info("interrupt received; synchronizing ranks before shutdown")
            try:
                dist.barrier(group=control_group)
            except RuntimeError:
                # One rank may already have failed; destroying the local group
                # is still safer than leaving NCCL heartbeat threads alive.
                pass
        if control_group is not None:
            dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
