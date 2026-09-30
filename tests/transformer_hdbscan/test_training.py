from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from turing_deinterleaving_challenge.transformer_hdbscan import train
from turing_deinterleaving_challenge.transformer_hdbscan.predict import load_checkpoint


def test_stable_progress_uses_complete_lines_without_terminal_controls(capsys) -> None:
    progress = train.StableLineProgress(
        "Train 1/2", total=2, enabled=True, update_seconds=0
    )
    progress.update(1, postfix="L=0.1000")
    progress.update(2, postfix="L=0.0500")

    output = capsys.readouterr().out
    assert "\r" not in output
    assert "\x1b" not in output
    assert output.endswith("\n")
    assert "Train 1/2: 2/2 (100.0%)" in output


def test_validation_v_measure_uses_cuml_complete_file_clustering(
    tmp_path: Path, monkeypatch
) -> None:
    files = []
    complete_embeddings = []
    for file_index in range(2):
        path = tmp_path / f"config_{file_index}.h5"
        truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("labels", data=truth)
        files.append(path)
        complete_embeddings.append(truth[:, None].astype(np.float32))
    dataset = SimpleNamespace(files=files, lengths=[4, 4])
    calls = []

    def fake_cluster(
        embedding,
        min_cluster_size,
        n_jobs=1,
        allow_single_cluster=False,
        backend="sklearn",
        gpu_device=0,
    ):
        calls.append((backend, gpu_device, min_cluster_size, allow_single_cluster))
        return embedding[:, 0].astype(np.int32)

    monkeypatch.setattr(train, "cluster_embedding", fake_cluster)
    macro, weighted, file_count = train.evaluate_validation_v_measure(
        complete_embeddings,
        dataset,
        min_cluster_size=2,
        min_cluster_fraction=0.0,
        allow_single_cluster=True,
        gpu_devices=[0, 1],
        progress_enabled=False,
    )

    assert macro == 1.0
    assert weighted == 1.0
    assert file_count == 2
    assert sorted(calls) == [
        ("cuml", 0, 2, True),
        ("cuml", 1, 2, True),
    ]


def test_training_selects_and_saves_a_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "train_scan"
    data_dir.mkdir()
    for file_index in range(2):
        pdws = np.arange(30, dtype=np.float32).reshape(6, 5) + file_index
        pdws[:, 0] = np.arange(6, dtype=np.float32) * 10
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        with h5py.File(data_dir / f"config_{file_index}.h5", "w") as handle:
            handle.create_dataset("data", data=pdws)
            handle.create_dataset("labels", data=labels)

    checkpoint = tmp_path / "model.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--train-dir",
            str(data_dir),
            "--output",
            str(checkpoint),
            "--window-length",
            "6",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--model-dim",
            "8",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--embedding-dim",
            "4",
            "--feedforward-dim",
            "16",
            "--dropout",
            "0",
            "--min-cluster-size",
            "2",
            "--selection-metric",
            "loss",
            "--device",
            "cpu",
            "--no-progress",
        ],
    )

    train.main()

    saved = load_checkpoint(checkpoint)
    assert saved["epoch"] == 1
    assert saved["window_length"] == 6
    assert saved["window_stride"] == 3
    assert "validation_loss" in saved
    assert saved["training_scope"] == "file"
    assert saved["normalization"] == "per_file"
    assert saved["normalization_mean"] is None
    assert saved["embedding_normalization"] == "none"
    assert saved["model_config"]["normalize_embeddings"] is False
    assert saved["model_config"]["architecture"] == "rope_swiglu_v1"
    assert saved["allow_single_cluster"] is True
    assert saved["selection_metric"] == "loss"

    log_dir = tmp_path / "logs"
    log_text = (log_dir / "train.log").read_text(encoding="utf-8")
    assert "epoch=1/1" in log_text
    assert "\r" not in log_text
    assert "\x1b" not in log_text

    with (log_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["best"] == "True"
    assert rows[0]["selection_metric"] == "loss"
    assert rows[0]["validation_v_measure"] == ""

    config = json.loads((log_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["arguments"]["window_length"] == 6
    assert config["arguments"]["no_progress"] is True


def test_adaptive_rmsnorm_silu_training_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    train_dir = tmp_path / "train_scan"
    validation_dir = tmp_path / "val_scan"
    train_dir.mkdir()
    validation_dir.mkdir()
    for directory in (train_dir, validation_dir):
        pdws = np.arange(40, dtype=np.float32).reshape(8, 5)
        pdws[:, 0] = np.arange(8, dtype=np.float32) * 10
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        with h5py.File(directory / "config_0.h5", "w") as handle:
            handle.create_dataset("data", data=pdws)
            handle.create_dataset("labels", data=labels)

    checkpoint = tmp_path / "adaptive_model.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--train-dir",
            str(train_dir),
            "--validation-dir",
            str(validation_dir),
            "--output",
            str(checkpoint),
            "--window-length",
            "8",
            "--window-stride",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--architecture",
            "rope_swiglu_rmsnorm_silu_v2",
            "--model-dim",
            "8",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--embedding-dim",
            "4",
            "--feedforward-dim",
            "16",
            "--dropout",
            "0",
            "--compactness-weight",
            "0.1",
            "--adaptive-anchors",
            "--anchor-min-per-emitter",
            "2",
            "--anchor-max-per-emitter",
            "4",
            "--anchor-fraction-per-emitter",
            "0.5",
            "--min-cluster-size",
            "2",
            "--selection-metric",
            "loss",
            "--device",
            "cpu",
            "--no-progress",
        ],
    )

    train.main()

    saved = load_checkpoint(checkpoint)
    assert saved["training_design"] == "rope_rmsnorm_silu_adaptive_anchor_v1"
    assert saved["model_config"]["architecture"] == (
        "rope_swiglu_rmsnorm_silu_v2"
    )
    assert saved["compactness_weight"] == 0.1
    assert saved["adaptive_anchors"] is True
    assert saved["anchor_min_per_emitter"] == 2
    assert saved["anchor_max_per_emitter"] == 4
    assert saved["anchor_fraction_per_emitter"] == 0.5

    with (tmp_path / "logs" / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert int(rows[0]["train_anchor_count"]) > 0
    assert rows[0]["train_anchor_count"] == rows[0]["train_unique_anchor_count"]
