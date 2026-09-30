"""Lazy exhaustive HDF5 windows with optional overlap for TSRD datasets."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset, Sampler

try:  # Support both package modules and direct scripts.
    from .model import delta_toa_features, normalize_pdws, normalize_pdws_global
except ImportError:  # pragma: no cover
    from model import delta_toa_features, normalize_pdws, normalize_pdws_global


def filter_empty_h5_files(files: list[Path]) -> list[Path]:
    """Drop valid HDF5 files containing zero pulses and report every omission."""
    retained: list[Path] = []
    skipped: list[Path] = []
    for path in files:
        try:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    raise ValueError(f"{path} is missing dataset 'data'")
                if handle["data"].ndim != 2:
                    raise ValueError(f"{path} dataset 'data' must be 2-D")
                if len(handle["data"]) == 0:
                    skipped.append(path)
                else:
                    retained.append(path)
        except OSError as error:
            raise ValueError(f"Cannot open HDF5 file {path}: {error}") from error
    if skipped:
        names = ", ".join(path.name for path in skipped[:20])
        suffix = "" if len(skipped) <= 20 else f", ... (+{len(skipped) - 20})"
        warnings.warn(
            f"Skipping {len(skipped)} empty HDF5 source file(s): {names}{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not retained:
        raise ValueError("All discovered HDF5 source files contain zero pulses")
    return retained


def resolve_h5_files(path: Path, split: str | None = None) -> list[Path]:
    """Find a direct HDF5 path or a standard TSRD split directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix != ".h5":
            raise ValueError(f"Expected an .h5 file: {path}")
        return filter_empty_h5_files([path])

    names: list[str] = []
    if split:
        names.extend([split, f"{split}_scan"])
        if split in {"validation", "val"}:
            names.extend(["validation", "validation_scan", "val", "val_scan"])
    candidates = [path]
    candidates.extend(path / name for name in names)
    candidates.extend(path / "scan" / name for name in names)
    candidates.append(path / "scan")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        files = sorted(candidate.glob("*.h5"))
        if files:
            return filter_empty_h5_files(files)

    discovered = list(path.rglob("*.h5")) if path.is_dir() else []
    hint = (
        f" Found {len(discovered)} HDF5 file(s) elsewhere below this path; "
        "pass the directory containing the requested split directly."
        if discovered
        else " No HDF5 files were found below the supplied path."
    )
    raise FileNotFoundError(
        f"No .h5 files found for split={split!r} in {path}.{hint}"
    )


def split_train_files(
    files: list[Path], validation_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Create a reproducible file-level validation holdout."""
    if len(files) < 2:
        raise ValueError("At least two train files are required for a holdout")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    validation_count = max(1, round(len(files) * validation_fraction))
    validation_count = min(validation_count, len(files) - 1)
    order = np.random.default_rng(seed).permutation(len(files))
    validation_indices = set(order[:validation_count].tolist())
    train = [path for index, path in enumerate(files) if index not in validation_indices]
    validation = [path for index, path in enumerate(files) if index in validation_indices]
    return train, validation


def fit_global_normalization(
    files: list[Path], max_samples: int = 1_000_000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate one boundary-aware normalization transform from training files."""
    if not files:
        raise ValueError("No files supplied for normalization")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rng = np.random.default_rng(seed)
    samples_per_file = max(2, int(np.ceil(max_samples / len(files))))
    feature_sum: np.ndarray | None = None
    feature_square_sum: np.ndarray | None = None
    sample_count = 0

    for path in files:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            length = len(data)
            if length == 0:
                continue
            count = min(length, samples_per_file)
            start = (
                0
                if length == count
                else int(rng.integers(0, length - count + 1))
            )
            previous_toa = (
                None if start == 0 else float(np.asarray(data[start - 1, 0]))
            )
            raw = np.asarray(data[start : start + count])
            features = delta_toa_features(raw, previous_toa=previous_toa).astype(
                np.float64, copy=False
            )
            current_sum = features.sum(axis=0)
            current_square_sum = np.square(features).sum(axis=0)
            if feature_sum is None:
                feature_sum = current_sum
                feature_square_sum = current_square_sum
            else:
                feature_sum += current_sum
                assert feature_square_sum is not None
                feature_square_sum += current_square_sum
            sample_count += len(features)

    if sample_count == 0 or feature_sum is None or feature_square_sum is None:
        raise ValueError("Training files contain no pulses")
    mean = feature_sum / sample_count
    variance = np.maximum(feature_square_sum / sample_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32), sample_count


def fit_file_normalization(
    path: Path, chunk_size: int = 1_000_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute one boundary-aware normalization transform for a whole file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    feature_sum: np.ndarray | None = None
    feature_square_sum: np.ndarray | None = None
    sample_count = 0
    previous_toa: float | None = None
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        for start in range(0, len(data), chunk_size):
            raw = np.asarray(data[start : start + chunk_size])
            features = delta_toa_features(raw, previous_toa=previous_toa).astype(
                np.float64, copy=False
            )
            if len(raw):
                previous_toa = float(raw[-1, 0])
            current_sum = features.sum(axis=0)
            current_square_sum = np.square(features).sum(axis=0)
            if feature_sum is None:
                feature_sum = current_sum
                feature_square_sum = current_square_sum
            else:
                feature_sum += current_sum
                assert feature_square_sum is not None
                feature_square_sum += current_square_sum
            sample_count += len(features)
    if sample_count == 0 or feature_sum is None or feature_square_sum is None:
        raise ValueError(f"Source file contains no pulses: {path}")
    mean = feature_sum / sample_count
    variance = np.maximum(feature_square_sum / sample_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32), sample_count


def sliding_window_starts(
    length: int, window_length: int, window_stride: int
) -> np.ndarray:
    """Return exhaustive starts separated by one constant stride."""
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    if not 0 < window_stride <= window_length:
        raise ValueError("window_stride must be in [1, window_length]")
    if length <= window_length:
        return np.asarray([0], dtype=np.int64)
    window_count = (length - window_length + window_stride - 1) // window_stride + 1
    return np.arange(window_count, dtype=np.int64) * window_stride


class FileWindowBatchSampler(Sampler[list[int]]):
    """Yield batches containing windows from exactly one source file."""

    def __init__(
        self,
        dataset: "H5WindowDataset",
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        previous = 0
        batches = 0
        for offset in self.dataset.offsets:
            count = int(offset) - previous
            batches += (count + self.batch_size - 1) // self.batch_size
            previous = int(offset)
        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        file_order = np.arange(len(self.dataset.files))
        if self.shuffle:
            rng.shuffle(file_order)
        for file_index in file_order:
            start = 0 if file_index == 0 else int(self.dataset.offsets[file_index - 1])
            stop = int(self.dataset.offsets[file_index])
            indices = np.arange(start, stop)
            if self.shuffle:
                rng.shuffle(indices)
            for batch_start in range(0, len(indices), self.batch_size):
                yield indices[batch_start : batch_start + self.batch_size].tolist()


class H5WindowDataset(Dataset):
    """Visit exhaustive sliding windows and pad only a short final window.

    Returned values are normalized features, labels, padding mask, file index,
    source start, valid length, and global window index. If ``return_raw`` is
    true, a padded copy of the unmodified PDWs is appended for prediction
    export.
    """

    def __init__(
        self,
        files: list[Path],
        window_length: int = 1024,
        window_stride: int | None = None,
        require_labels: bool = True,
        return_raw: bool = False,
        hdf5_cache_size: int = 32,
        normalization: str = "per_window",
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
        normalization_progress: bool = False,
        normalization_description: str = "Fit per-file normalization",
    ) -> None:
        if window_length < 2:
            raise ValueError("window_length must be at least 2")
        if window_stride is None:
            window_stride = window_length
        if not 0 < window_stride <= window_length:
            raise ValueError("window_stride must be in [1, window_length]")
        if not files:
            raise ValueError("No HDF5 files were supplied")
        if hdf5_cache_size <= 0:
            raise ValueError("hdf5_cache_size must be positive")
        if normalization not in {"per_window", "per_file", "global"}:
            raise ValueError(
                "normalization must be 'per_window', 'per_file', or 'global'"
            )
        if normalization == "global" and (
            normalization_mean is None or normalization_std is None
        ):
            raise ValueError("Global normalization requires mean and std")

        self.files = [Path(path).expanduser().resolve() for path in files]
        self.window_length = window_length
        self.window_stride = window_stride
        self.return_raw = return_raw
        self.hdf5_cache_size = hdf5_cache_size
        self.normalization = normalization
        self.normalization_mean = (
            None
            if normalization_mean is None
            else np.asarray(normalization_mean, dtype=np.float32).reshape(-1)
        )
        self.normalization_std = (
            None
            if normalization_std is None
            else np.asarray(normalization_std, dtype=np.float32).reshape(-1)
        )
        self.lengths: list[int] = []
        self.file_normalization_means: list[np.ndarray] = []
        self.file_normalization_stds: list[np.ndarray] = []
        self.feature_dim: int | None = None
        self.raw_dtype: np.dtype | None = None
        label_flags: list[bool] = []

        for path in self.files:
            with h5py.File(path, "r") as handle:
                if "data" not in handle or handle["data"].ndim != 2:
                    raise ValueError(f"{path} must contain 2-D dataset 'data'")
                has_labels = "labels" in handle
                if require_labels and not has_labels:
                    raise ValueError(f"{path} must contain dataset 'labels'")
                if has_labels and len(handle["labels"]) != len(handle["data"]):
                    raise ValueError(f"data/labels length mismatch in {path}")
                current_dim = int(handle["data"].shape[1])
                current_dtype = np.dtype(handle["data"].dtype)
                if self.feature_dim is None:
                    self.feature_dim = current_dim
                    self.raw_dtype = current_dtype
                elif current_dim != self.feature_dim:
                    raise ValueError("All files must have the same feature dimension")
                elif current_dtype != self.raw_dtype:
                    raise ValueError("All files must have the same data dtype")
                self.lengths.append(len(handle["data"]))
                label_flags.append(has_labels)

        if normalization == "global":
            assert self.feature_dim is not None
            if self.normalization_mean is None or self.normalization_std is None:
                raise ValueError("Global normalization statistics are missing")
            if self.normalization_mean.shape != (self.feature_dim,) or (
                self.normalization_std.shape != (self.feature_dim,)
            ):
                raise ValueError("Global normalization statistics have wrong dimension")

        if normalization == "per_file":
            progress_step = max(1, len(self.files) // 20)
            for file_number, path in enumerate(self.files, start=1):
                mean, std, _ = fit_file_normalization(path)
                self.file_normalization_means.append(mean)
                self.file_normalization_stds.append(std)
                if normalization_progress and (
                    file_number == 1
                    or file_number % progress_step == 0
                    or file_number == len(self.files)
                ):
                    print(
                        f"{normalization_description}: "
                        f"{file_number:,}/{len(self.files):,} "
                        f"({100.0 * file_number / len(self.files):.1f}%)",
                        flush=True,
                    )

        if any(label_flags) and not all(label_flags):
            raise ValueError("Either every input file must have labels or none may have labels")
        self.has_labels = all(label_flags)
        self.window_starts = [
            sliding_window_starts(length, window_length, window_stride)
            for length in self.lengths
        ]
        counts = np.asarray([len(starts) for starts in self.window_starts], dtype=np.int64)
        self.offsets = np.cumsum(counts)
        if len(self) == 0:
            raise ValueError("The supplied files contain no pulses")
        self._hdf5_handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._cache_pid: int | None = None

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_hdf5_handles"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def _close_hdf5_handles(self) -> None:
        for handle in self._hdf5_handles.values():
            handle.close()
        self._hdf5_handles.clear()

    def _hdf5_handle(self, file_index: int) -> h5py.File:
        pid = os.getpid()
        if self._cache_pid != pid:
            self._close_hdf5_handles()
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
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self.offsets, index, side="right"))
        previous = 0 if file_index == 0 else int(self.offsets[file_index - 1])
        local_index = index - previous
        return file_index, int(self.window_starts[file_index][local_index])

    def __getitem__(self, index: int) -> tuple:
        file_index, start = self.window_location(index)
        end = min(start + self.window_length, self.lengths[file_index])
        handle = self._hdf5_handle(file_index)
        raw = np.asarray(handle["data"][start:end])
        if self.normalization in {"global", "per_file"}:
            if self.normalization == "global":
                assert self.normalization_mean is not None
                assert self.normalization_std is not None
                mean = self.normalization_mean
                std = self.normalization_std
            else:
                mean = self.file_normalization_means[file_index]
                std = self.file_normalization_stds[file_index]
            previous_toa = (
                None
                if start == 0
                else float(np.asarray(handle["data"][start - 1, 0]))
            )
            features = normalize_pdws_global(
                raw,
                mean,
                std,
                previous_toa=previous_toa,
            )
        else:
            features = normalize_pdws(raw)
        labels = (
            np.asarray(handle["labels"][start:end]).reshape(-1).astype(np.int64)
            if self.has_labels
            else np.full(end - start, -1, dtype=np.int64)
        )

        valid_length = len(raw)
        padded_features = np.zeros(
            (self.window_length, features.shape[1]), dtype=np.float32
        )
        padded_labels = np.full(self.window_length, -1, dtype=np.int64)
        padding_mask = np.ones(self.window_length, dtype=bool)
        padded_features[:valid_length] = features
        padded_labels[:valid_length] = labels
        padding_mask[:valid_length] = False
        result = (
            padded_features,
            padded_labels,
            padding_mask,
            file_index,
            start,
            valid_length,
            index,
        )
        if not self.return_raw:
            return result

        assert self.raw_dtype is not None
        padded_raw = np.zeros(
            (self.window_length, raw.shape[1]), dtype=self.raw_dtype
        )
        padded_raw[:valid_length] = raw
        return (*result, padded_raw)

    def __del__(self) -> None:
        try:
            self._close_hdf5_handles()
        except (AttributeError, OSError):
            pass
