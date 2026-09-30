from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

from turing_deinterleaving_challenge.transformer_hdbscan import convert_pdw_studio_format


def test_convert_existing_cluster_without_inference(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "pdw_studio_clusters_dual_gpu_v1"
    config_dir = root / "config_590"
    config_dir.mkdir(parents=True)
    legacy = config_dir / "cluster002.h5"
    pdws = np.arange(30, dtype=np.float32).reshape(6, 5)
    truth = np.asarray([38, 38, 19, 38, 21, 38], dtype=np.int64)
    with h5py.File(legacy, "w") as handle:
        handle.attrs["source_file"] = "/dataset/config_590.h5"
        handle.attrs["output_cluster_id"] = 2
        handle.attrs["original_cluster_label"] = 54
        handle.create_dataset("data", data=pdws)
        handle.create_dataset("labels", data=np.full((6, 1), -1, dtype=np.int32))
        handle.create_dataset("true_label", data=truth.reshape(-1, 1))

    monkeypatch.setattr(sys, "argv", ["convert", str(root)])
    convert_pdw_studio_format.main()
    assert legacy.is_file()
    assert not (config_dir / "config_590_2.h5").exists()

    monkeypatch.setattr(sys, "argv", ["convert", str(root), "--apply"])
    convert_pdw_studio_format.main()

    converted = config_dir / "config_590_2.h5"
    assert converted.is_file()
    assert not legacy.exists()
    with h5py.File(converted, "r") as handle:
        assert not handle.attrs
        assert set(handle) == {"data", "labels"}
        assert handle["data"].shape == (5, 6)
        assert handle["data"].dtype == np.dtype(np.float64)
        assert handle["data"].compression is None
        assert handle["labels"].shape == (6,)
        assert handle["labels"].dtype == np.dtype(np.int32)
        assert handle["labels"].compression is None
        np.testing.assert_array_equal(handle["data"][:], pdws.T)
        np.testing.assert_array_equal(handle["labels"][:], truth)

    with (root / "_clusters.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "cluster_file": "config_590/config_590_2.h5",
            "source_file": "/dataset/config_590.h5",
            "source_stem": "config_590",
            "output_cluster_id": "2",
            "original_cluster_label": "54",
            "pulse_count": "6",
        }
    ]
