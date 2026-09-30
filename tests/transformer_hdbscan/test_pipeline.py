from __future__ import annotations

import csv
import json
import sys
import threading
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import v_measure_score

from turing_deinterleaving_challenge.transformer_hdbscan import predict
from turing_deinterleaving_challenge.transformer_hdbscan.predict import BufferedWindow, aggregate_overlapping_embeddings
from turing_deinterleaving_challenge.transformer_hdbscan.model import TransformerMetricEncoder


def test_prediction_pipeline_exports_every_window(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "test_scan"
    data_dir.mkdir()
    pdws = np.arange(50, dtype=np.float32).reshape(10, 5)
    pdws[:, 0] = np.arange(10, dtype=np.float32) * 10
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    source = data_dir / "config_0.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data", data=pdws)
        handle.create_dataset("labels", data=labels)

    model_config = {
        "input_dim": 5,
        "model_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "embedding_dim": 4,
        "feedforward_dim": 16,
        "dropout": 0.0,
    }
    # Simulate a checkpoint produced before the architecture field existed.
    model = TransformerMetricEncoder(
        **model_config, architecture="legacy_sinusoidal"
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "window_length": 6,
            "window_stride": 3,
            "min_cluster_size": 2,
            "normalization": "per_window",
        },
        checkpoint,
    )
    output = tmp_path / "predictions"
    clustered_lengths: list[int] = []

    def fake_complete_file_hdbscan(
        embedding: np.ndarray,
        _: int,
        min_samples: int | None = None,
        n_jobs: int = 1,
        allow_single_cluster: bool = False,
        backend: str = "sklearn",
        gpu_device: int = 0,
    ) -> np.ndarray:
        assert min_samples is None
        assert n_jobs == 1
        assert not allow_single_cluster
        assert backend == "sklearn"
        assert gpu_device == 0
        clustered_lengths.append(len(embedding))
        return np.where(np.arange(len(embedding)) % 2 == 0, -1, 0).astype(np.int32)

    monkeypatch.setattr(
        predict,
        "cluster_embedding",
        fake_complete_file_hdbscan,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--evaluate",
            "--window-length",
            "6",
            "--batch-size",
            "2",
            "--min-cluster-size",
            "2",
            "--device",
            "cpu",
        ],
    )

    predict.main()

    result_files = sorted((output / "files").glob("*.h5"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        file_rows = list(csv.DictReader(handle))
    complete_prediction = np.where(np.arange(10) % 2 == 0, -1, 0)
    expected_v_measure = v_measure_score(labels, complete_prediction)
    assert len(result_files) == 1
    assert clustered_lengths == [10]
    assert len(file_rows) == 1
    assert float(file_rows[0]["V-measure"]) == expected_v_measure
    assert summary["source_file_count"] == 1
    assert summary["total_embedding_window_count"] == 3
    assert summary["evaluated_source_file_count"] == 1
    assert summary["metrics"]["V-measure"]["mean"] == expected_v_measure
    with h5py.File(result_files[0], "r") as handle:
        assert handle.attrs["pulse_count"] == 10
        assert handle.attrs["window_count"] == 3
        assert handle.attrs["window_stride"] == 3
        assert handle.attrs["clustering_scope"] == "complete_source_file"
        np.testing.assert_array_equal(
            handle["embedding_windows/valid_lengths"][:], [6, 6, 4]
        )
        np.testing.assert_array_equal(handle["embedding_windows/starts"][:], [0, 3, 6])
        assert handle["evaluation"].attrs["V-measure"] == expected_v_measure
        sizes = [
            int(group.attrs["size"]) for group in handle["predicted_clusters"].values()
        ]
        assert sum(sizes) == 10
        assert "cluster_-1" in handle["predicted_clusters"]

    skipped_output = tmp_path / "cluster_only_predictions"

    def fail_if_evaluated(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("evaluation must be skipped")

    monkeypatch.setattr(predict, "evaluate_labels", fail_if_evaluated)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(skipped_output),
            "--window-length",
            "6",
            "--batch-size",
            "2",
            "--min-cluster-size",
            "2",
            "--device",
            "cpu",
        ],
    )

    predict.main()

    skipped_summary = json.loads(
        (skipped_output / "summary.json").read_text(encoding="utf-8")
    )
    skipped_result = next((skipped_output / "files").glob("*.h5"))
    assert skipped_summary["evaluation_enabled"] is False
    assert skipped_summary["evaluated_source_file_count"] == 0
    assert skipped_summary["metrics"] == {}
    with h5py.File(skipped_result, "r") as handle:
        assert not bool(handle.attrs["evaluation_computed"])
        assert "evaluation" not in handle
        assert "true_clusters" in handle
        np.testing.assert_array_equal(handle["pulses/true_labels"][:], labels)

    pdw_output = tmp_path / "pdw_studio_predictions"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(pdw_output),
            "--output-format",
            "pdw_studio",
            "--window-length",
            "6",
            "--batch-size",
            "2",
            "--min-cluster-size",
            "2",
            "--device",
            "cpu",
        ],
    )

    predict.main()

    pdw_cluster_files = sorted((pdw_output / "config_0").glob("config_0_*.h5"))
    assert len(pdw_cluster_files) == 2
    exported_pulses = 0
    for path in pdw_cluster_files:
        with h5py.File(path, "r") as handle:
            assert not handle.attrs
            assert set(handle) == {"data", "labels"}
            assert handle["data"].shape[0] == 5
            assert handle["data"].shape[1] == len(handle["labels"])
            exported_pulses += handle["data"].shape[1]
    assert exported_pulses == 10

    with (pdw_output / "_clusters.csv").open(newline="", encoding="utf-8") as handle:
        original_labels = {
            int(row["original_cluster_label"]) for row in csv.DictReader(handle)
        }
    assert original_labels == {-1, 0}
    assert (pdw_output / "_summary.csv").is_file()
    assert (pdw_output / "summary.json").is_file()
    assert (pdw_output / "run.log").is_file()


def test_overlapping_pulse_embeddings_are_averaged_once() -> None:
    windows = [
        BufferedWindow(
            embedding=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            pulse_start=0,
            valid_length=2,
        ),
        BufferedWindow(
            embedding=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            pulse_start=1,
            valid_length=2,
        ),
    ]

    result = aggregate_overlapping_embeddings(windows, pulse_count=3)

    np.testing.assert_allclose(result[0], [1.0, 0.0])
    np.testing.assert_allclose(result[1], [0.5, 0.5])
    np.testing.assert_allclose(result[2], [0.0, 1.0])


def test_explicit_source_file_subset_preserves_requested_order(tmp_path: Path) -> None:
    files = [tmp_path / name for name in ("config_0.h5", "config_1.h5", "config_10.h5")]

    selected = predict.select_source_files(
        files, ["config_10.h5,config_0", "config_1.h5"]
    )

    assert [path.name for path in selected] == [
        "config_10.h5",
        "config_0.h5",
        "config_1.h5",
    ]


def test_cuml_devices_cluster_different_files_concurrently(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "test_scan"
    data_dir.mkdir()
    for file_index in range(2):
        pdws = np.arange(20, dtype=np.float32).reshape(4, 5)
        pdws[:, 0] = np.arange(4, dtype=np.float32)
        with h5py.File(data_dir / f"config_{file_index}.h5", "w") as handle:
            handle.create_dataset("data", data=pdws)
            handle.create_dataset("labels", data=np.zeros(4, dtype=np.int64))

    model_config = {
        "input_dim": 5,
        "model_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "embedding_dim": 4,
        "feedforward_dim": 16,
        "dropout": 0.0,
    }
    model = TransformerMetricEncoder(**model_config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "window_length": 4,
            "window_stride": 4,
            "min_cluster_size": 2,
            "normalization": "per_window",
        },
        checkpoint,
    )

    barrier = threading.Barrier(2)
    observed_devices: list[int] = []
    observation_lock = threading.Lock()

    def fake_complete_file_hdbscan(
        embedding: np.ndarray,
        _: int,
        min_samples: int | None = None,
        n_jobs: int = 1,
        allow_single_cluster: bool = False,
        backend: str = "sklearn",
        gpu_device: int = 0,
    ) -> np.ndarray:
        del n_jobs, allow_single_cluster
        assert min_samples is None
        assert backend == "cuml"
        with observation_lock:
            observed_devices.append(gpu_device)
        barrier.wait(timeout=5)
        return np.zeros(len(embedding), dtype=np.int32)

    monkeypatch.setattr(predict, "cluster_embedding", fake_complete_file_hdbscan)
    monkeypatch.setattr(predict, "load_cuml_hdbscan", lambda: object)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device_id: f"fake-{device_id}"
    )
    output = tmp_path / "parallel_predictions"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--hdbscan-backend",
            "cuml",
            "--hdbscan-devices",
            "0,1",
            "--hdbscan-parallel-files",
            "2",
        ],
    )

    predict.main()

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert sorted(observed_devices) == [0, 1]
    assert summary["source_file_count"] == 2
    assert summary["hdbscan_devices"] == [0, 1]
    assert summary["hdbscan_parallel_files"] == 2
