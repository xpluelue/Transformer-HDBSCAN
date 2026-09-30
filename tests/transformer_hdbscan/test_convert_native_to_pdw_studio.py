from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from turing_deinterleaving_challenge.transformer_hdbscan.convert_native_to_pdw_studio import convert_native_run
from turing_deinterleaving_challenge.transformer_hdbscan.metrics import evaluate_labels
from turing_deinterleaving_challenge.transformer_hdbscan.output import (
    PredictionRunWriter,
    SourceFileMetadata,
)


def test_native_predictions_convert_without_inference(tmp_path: Path) -> None:
    native = tmp_path / "native"
    metadata = SourceFileMetadata(
        source_file="/dataset/config_7.h5",
        file_index=7,
        pulse_count=5,
        window_count=2,
        window_length=4,
        window_stride=2,
    )
    raw = np.arange(25, dtype=np.float32).reshape(5, 5)
    predicted = np.asarray([-1, -1, 3, 3, 3], dtype=np.int32)
    truth = np.asarray([8, 8, 9, 9, 10], dtype=np.int64)
    metrics = evaluate_labels(predicted, truth)

    writer = PredictionRunWriter(native, {"checkpoint": "model.pt"})
    writer.write(
        metadata,
        raw,
        predicted,
        truth,
        np.asarray([0, 2]),
        np.asarray([4, 3]),
        metrics,
    )
    writer.close()

    output = tmp_path / "pdw"
    summary = convert_native_run(native, output)

    cluster_files = sorted((output / "config_7").glob("config_7_*.h5"))
    assert [path.name for path in cluster_files] == [
        "config_7_0.h5",
        "config_7_1.h5",
    ]
    with h5py.File(cluster_files[0], "r") as handle:
        assert not handle.attrs
        assert set(handle) == {"data", "labels"}
        assert handle["data"].dtype == np.dtype("float64")
        assert handle["data"].shape == (5, 3)
        assert handle["labels"].dtype == np.dtype("int32")
        np.testing.assert_array_equal(handle["labels"][:], [9, 9, 10])

    with (output / "_clusters.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["original_cluster_label"]) for row in rows] == [3, -1]
    assert summary["source_file_count"] == 1
    assert summary["metrics"]["V-measure"]["mean"] == metrics["V-measure"]
    saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["converted_from_native"] == str(native.resolve())
