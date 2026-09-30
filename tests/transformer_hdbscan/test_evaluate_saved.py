from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from turing_deinterleaving_challenge.transformer_hdbscan.evaluate_saved import evaluate_saved_results
from turing_deinterleaving_challenge.transformer_hdbscan.output import SourceFileMetadata, write_source_file


def test_saved_clustering_can_be_evaluated_separately(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    result_path = prediction_dir / "files" / "file_0000__config_0.h5"
    raw = np.arange(20, dtype=np.float32).reshape(4, 5)
    predicted = np.asarray([-1, -1, 0, 0], dtype=np.int32)
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    write_source_file(
        result_path,
        SourceFileMetadata("config_0.h5", 0, 4, 1, 4, 4),
        raw,
        predicted,
        truth,
        np.asarray([0]),
        np.asarray([4]),
        metrics=None,
    )

    evaluation_dir = tmp_path / "evaluation"
    summary = evaluate_saved_results(prediction_dir, evaluation_dir)

    assert summary["evaluated_source_file_count"] == 1
    assert summary["minus_one_is_ordinary_cluster"] is True
    assert summary["metrics"]["V-measure"]["mean"] == 1.0
    with (evaluation_dir / "evaluation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["minus_one_group_size"] == "2"
    assert rows[0]["predicted_cluster_count"] == "2"
    loaded_summary = json.loads(
        (evaluation_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert loaded_summary == summary
    with h5py.File(result_path, "r") as handle:
        assert "evaluation" not in handle
