from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from turing_deinterleaving_challenge.transformer_hdbscan.repair_dataset import normalize_endpoint, validate_h5


def validate(path: Path):
    return validate_h5(
        path,
        path.name,
        require_labels=True,
        chunk_rows=2,
        full_read=True,
    )


def test_hdf5_audit_detects_empty_and_corrupt_files(tmp_path: Path) -> None:
    valid = tmp_path / "valid.h5"
    with h5py.File(valid, "w") as handle:
        handle.create_dataset("data", data=np.ones((3, 5), dtype=np.float32))
        handle.create_dataset("labels", data=np.arange(3, dtype=np.int64))

    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as handle:
        handle.create_dataset("data", data=np.empty((0, 5), dtype=np.float32))
        handle.create_dataset("labels", data=np.empty(0, dtype=np.int64))

    mismatch = tmp_path / "mismatch.h5"
    with h5py.File(mismatch, "w") as handle:
        handle.create_dataset("data", data=np.ones((3, 5), dtype=np.float32))
        handle.create_dataset("labels", data=np.arange(2, dtype=np.int64))

    corrupt = tmp_path / "corrupt.h5"
    corrupt.write_bytes(b"not an HDF5 file")

    assert validate(valid).valid
    assert not validate(empty).valid
    assert "zero pulses" in validate(empty).reason
    assert not validate(mismatch).valid
    assert "length mismatch" in validate(mismatch).reason
    assert not validate(corrupt).valid


def test_markdown_endpoint_is_normalized() -> None:
    assert normalize_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"
    assert (
        normalize_endpoint("[https://hf-mirror.com](https://hf-mirror.com)")
        == "https://hf-mirror.com"
    )
