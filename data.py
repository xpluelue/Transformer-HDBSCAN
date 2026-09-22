"""Lazy non-overlapping HDF5 windows for the TSRD dataset."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from model import normalize_pdws


def resolve_h5_files(path: Path, split: str | None = None) -> list[Path]:
    """Find a direct HDF5 directory or a standard TSRD split directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix != ".h5":
            raise ValueError(f"Expected an .h5 file: {path}")
        return [path]

    candidates = [path]
    if split:
        names = [split, f"{split}_scan"]
        if split == "validation":
            names.extend(["val", "val_scan"])
        candidates.extend(path / name for name in names)
        candidates.extend(path / "scan" / name for name in names)

    for candidate in candidates:
        files = sorted(candidate.glob("*.h5"))
        if files:
            return files
    raise FileNotFoundError(
        f"No .h5 files found for split={split!r} below the expected locations in {path}"
    )


class H5WindowDataset(Dataset):
    """Read every non-overlapping window; zero-pad the final partial window."""

    def __init__(
        self,
        files: list[Path],
        window_length: int = 1024,
        require_labels: bool = True,
    ) -> None:
        if window_length < 2:
            raise ValueError("window_length must be at least 2")
        if not files:
            raise ValueError("No HDF5 files were supplied")

        self.files = files
        self.window_length = window_length
        self.lengths: list[int] = []
        self.feature_dim: int | None = None
        label_flags: list[bool] = []

        for path in files:
            with h5py.File(path, "r") as handle:
                if "data" not in handle or handle["data"].ndim != 2:
                    raise ValueError(f"{path} must contain 2-D dataset 'data'")
                has_labels = "labels" in handle
                if require_labels and not has_labels:
                    raise ValueError(f"{path} must contain dataset 'labels'")
                if has_labels and len(handle["labels"]) != len(handle["data"]):
                    raise ValueError(f"data/labels length mismatch in {path}")
                current_dim = int(handle["data"].shape[1])
                if self.feature_dim is None:
                    self.feature_dim = current_dim
                elif current_dim != self.feature_dim:
                    raise ValueError("All files must have the same feature dimension")
                self.lengths.append(len(handle["data"]))
                label_flags.append(has_labels)

        if any(label_flags) and not all(label_flags):
            raise ValueError("Either every input file must have labels or none may have labels")
        self.has_labels = all(label_flags)
        counts = [
            (length + window_length - 1) // window_length for length in self.lengths
        ]
        self.offsets = np.cumsum(counts)
        if len(self) == 0:
            raise ValueError("The supplied files contain no pulses")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def window_location(self, index: int) -> tuple[int, int]:
        file_index = int(np.searchsorted(self.offsets, index, side="right"))
        previous = 0 if file_index == 0 else int(self.offsets[file_index - 1])
        return file_index, (index - previous) * self.window_length

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        file_index, start = self.window_location(index)
        end = min(start + self.window_length, self.lengths[file_index])
        with h5py.File(self.files[file_index], "r") as handle:
            data = normalize_pdws(handle["data"][start:end])
            labels = (
                np.asarray(handle["labels"][start:end]).squeeze().astype(np.int64)
                if self.has_labels
                else np.full(end - start, -1, dtype=np.int64)
            )

        valid_length = len(data)
        padded_data = np.zeros(
            (self.window_length, data.shape[1]), dtype=np.float32
        )
        padded_labels = np.full(self.window_length, -1, dtype=np.int64)
        padding_mask = np.ones(self.window_length, dtype=bool)
        padded_data[:valid_length] = data
        padded_labels[:valid_length] = labels
        padding_mask[:valid_length] = False
        return (
            padded_data,
            padded_labels,
            padding_mask,
            file_index,
            start,
            valid_length,
        )
