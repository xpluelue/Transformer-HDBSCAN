from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import v_measure_score

from turing_deinterleaving_challenge.transformer_hdbscan.metrics import evaluate_labels
from turing_deinterleaving_challenge.transformer_hdbscan.output import (
    MetricsOnlyRunWriter,
    PDWStudioRunWriter,
    PredictionRunWriter,
    SourceFileMetadata,
    write_source_file,
)


def metadata() -> SourceFileMetadata:
    return SourceFileMetadata(
        source_file="/dataset/config_1.h5",
        file_index=1,
        pulse_count=4,
        window_count=1,
        window_length=4,
        window_stride=4,
    )


def test_minus_one_is_an_ordinary_cluster_group(tmp_path: Path) -> None:
    pdws = np.arange(20, dtype=np.float32).reshape(4, 5)
    predicted = np.asarray([-1, -1, 0, 0], dtype=np.int32)
    truth = np.asarray([10, 10, 11, 11], dtype=np.int64)
    scores = evaluate_labels(predicted, truth)
    output = tmp_path / "source.h5"
    write_source_file(
        output,
        metadata(),
        pdws,
        predicted,
        truth,
        np.asarray([0]),
        np.asarray([4]),
        scores,
    )

    with h5py.File(output, "r") as handle:
        assert set(handle["predicted_clusters"]) == {"cluster_-1", "cluster_0"}
        assert "noise" not in handle["predicted_clusters"]
        minus_one = handle["predicted_clusters/cluster_-1"]
        np.testing.assert_array_equal(minus_one["member_indices"][:], [0, 1])
        np.testing.assert_array_equal(minus_one["pdws"][:], pdws[:2])
        np.testing.assert_array_equal(minus_one["counterpart_labels"][:], [10, 10])
        assert handle["evaluation"].attrs["V-measure"] == v_measure_score(
            truth, predicted
        )
        assert handle["evaluation"].attrs["metrics_scope"] == "complete_source_file"
        np.testing.assert_array_equal(
            handle["evaluation/contingency_matrix"][:], [[2, 0], [0, 2]]
        )


def test_run_writer_creates_manifest_and_summary(tmp_path: Path) -> None:
    pdws = np.arange(20, dtype=np.float32).reshape(4, 5)
    predicted = np.asarray([-1, -1, 0, 0], dtype=np.int32)
    truth = np.asarray([10, 10, 11, 11], dtype=np.int64)
    scores = evaluate_labels(predicted, truth)
    writer = PredictionRunWriter(tmp_path / "run", {"checkpoint": "model.pt"})
    writer.write(
        metadata(),
        pdws,
        predicted,
        truth,
        np.asarray([0]),
        np.asarray([4]),
        scores,
    )
    summary = writer.close()

    with (tmp_path / "run/manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    saved_summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert len(rows) == 1
    assert rows[0]["predicted_cluster_count"] == "2"
    assert saved_summary["source_file_count"] == 1
    assert saved_summary["total_embedding_window_count"] == 1
    assert saved_summary["evaluated_source_file_count"] == 1
    assert saved_summary["metrics"]["V-measure"]["mean"] == scores["V-measure"]


def test_metrics_only_writer_does_not_duplicate_pulse_hdf5(tmp_path: Path) -> None:
    pdws = np.arange(20, dtype=np.float32).reshape(4, 5)
    predicted = np.asarray([-1, -1, 0, 0], dtype=np.int32)
    truth = np.asarray([10, 10, 11, 11], dtype=np.int64)
    scores = evaluate_labels(predicted, truth)
    output = tmp_path / "metrics_only"
    writer = MetricsOnlyRunWriter(
        output,
        {"checkpoint": "model.pt", "min_cluster_fraction": 0.001},
    )
    writer.write(
        metadata(),
        pdws,
        predicted,
        truth,
        np.asarray([0]),
        np.asarray([4]),
        scores,
    )
    summary = writer.close()

    assert sorted(path.name for path in output.iterdir()) == [
        "metrics.csv",
        "summary.json",
    ]
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["predicted_cluster_count"] == "2"
    assert float(rows[0]["V-measure"]) == scores["V-measure"]
    assert summary["saved_predictions"] is False
    assert summary["metrics"]["V-measure"]["mean"] == scores["V-measure"]


def test_write_without_metrics_keeps_truth_for_later_evaluation(
    tmp_path: Path,
) -> None:
    pdws = np.arange(20, dtype=np.float32).reshape(4, 5)
    predicted = np.asarray([-1, -1, 0, 0], dtype=np.int32)
    truth = np.asarray([10, 10, 11, 11], dtype=np.int64)
    output = tmp_path / "cluster_only.h5"

    write_source_file(
        output,
        metadata(),
        pdws,
        predicted,
        truth,
        np.asarray([0]),
        np.asarray([4]),
        metrics=None,
    )

    with h5py.File(output, "r") as handle:
        assert not bool(handle.attrs["evaluation_computed"])
        assert "evaluation" not in handle
        assert "predicted_clusters" in handle
        assert "true_clusters" in handle
        np.testing.assert_array_equal(handle["pulses/predicted_labels"][:], predicted)
        np.testing.assert_array_equal(handle["pulses/true_labels"][:], truth)


def test_pdw_studio_writer_exports_one_h5_per_cluster(tmp_path: Path) -> None:
    pdws = np.arange(20, dtype=np.float32).reshape(4, 5)
    pdws[:, 0] = [30, 10, 40, 20]
    predicted = np.asarray([-1, 5, 5, 5], dtype=np.int32)
    truth = np.asarray([10, 11, 12, 13], dtype=np.int64)
    writer = PDWStudioRunWriter(tmp_path / "pdw_studio", {"checkpoint": "model.pt"})

    result_dir = writer.write(
        metadata(),
        pdws,
        predicted,
        truth,
        np.asarray([0]),
        np.asarray([4]),
        metrics=None,
    )
    summary = writer.close()

    cluster_files = sorted(result_dir.glob("config_1_*.h5"))
    assert [path.name for path in cluster_files] == [
        "config_1_0.h5",
        "config_1_1.h5",
    ]
    with h5py.File(cluster_files[0], "r") as handle:
        assert not handle.attrs
        assert set(handle) == {"data", "labels"}
        assert handle["data"].shape == (5, 3)
        assert handle["data"].dtype == np.dtype(np.float64)
        assert handle["data"].compression is None
        assert handle["labels"].shape == (3,)
        assert handle["labels"].dtype == np.dtype(np.int32)
        assert handle["labels"].compression is None
        np.testing.assert_array_equal(handle["data"][0], [10, 20, 40])
        np.testing.assert_array_equal(handle["labels"][:], [11, 13, 12])
    with h5py.File(cluster_files[1], "r") as handle:
        assert not handle.attrs
        np.testing.assert_array_equal(handle["data"][0], [30])
        np.testing.assert_array_equal(handle["labels"][:], [10])

    with (tmp_path / "pdw_studio/_clusters.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        cluster_rows = list(csv.DictReader(handle))
    assert [row["cluster_file"] for row in cluster_rows] == [
        "config_1/config_1_0.h5",
        "config_1/config_1_1.h5",
    ]
    assert [row["original_cluster_label"] for row in cluster_rows] == ["5", "-1"]

    with (tmp_path / "pdw_studio/_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["predicted_cluster_count"] == "2"
    assert summary["source_file_count"] == 1
    assert summary["total_predicted_cluster_count"] == 2
    assert summary["minus_one_is_ordinary_cluster"] is True
