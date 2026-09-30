from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from turing_deinterleaving_challenge.transformer_hdbscan import predict
from turing_deinterleaving_challenge.transformer_hdbscan.predict import (
    cluster_embedding,
    effective_min_samples,
    estimate_remaining_seconds,
    format_duration,
    resolve_hdbscan_devices,
    resolve_inference_devices,
)


def test_cuml_hdbscan_uses_selected_gpu_and_returns_numpy_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeCuMLHDBSCAN:
        def __init__(self, **kwargs: object) -> None:
            calls["kwargs"] = kwargs

        def fit_predict(self, embedding: np.ndarray) -> np.ndarray:
            calls["embedding"] = embedding
            return np.asarray([0, -1, 0], dtype=np.int64)

    monkeypatch.setattr(predict, "load_cuml_hdbscan", lambda: FakeCuMLHDBSCAN)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.setdefault("empty", True)
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.setdefault("synchronized", device),
    )

    labels = cluster_embedding(
        np.arange(12, dtype=np.float64).reshape(3, 4),
        min_cluster_size=2,
        min_samples=1,
        allow_single_cluster=True,
        backend="cuml",
        gpu_device=1,
    )

    np.testing.assert_array_equal(labels, [0, -1, 0])
    assert labels.dtype == np.int32
    assert calls["kwargs"] == {
        "min_cluster_size": 2,
        "min_samples": 1,
        "allow_single_cluster": True,
        "output_type": "numpy",
    }
    assert isinstance(calls["embedding"], np.ndarray)
    assert calls["embedding"].dtype == np.float32
    assert calls["empty"] is True
    assert calls["synchronized"] == 1


def test_effective_min_samples_uses_floor_and_adjustable_fraction() -> None:
    assert effective_min_samples(234_273, 5, 0.0005) == 118
    assert effective_min_samples(234_273, 5, 0.001) == 235
    assert effective_min_samples(100, 5, 0.001) == 5


def test_resolve_cpu_device() -> None:
    device, device_ids = resolve_inference_devices("cpu", None)

    assert device == torch.device("cpu")
    assert device_ids == []


def test_resolve_two_visible_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    device, device_ids = resolve_inference_devices(None, "cuda:0,1")

    assert device == torch.device("cuda:0")
    assert device_ids == [0, 1]


def test_reject_device_and_devices_together() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        resolve_inference_devices("cuda:0", "cuda:0,cuda:1")


def test_reject_cuda_device_hidden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="Check CUDA_VISIBLE_DEVICES"):
        resolve_inference_devices(None, "cuda:0,cuda:1")


def test_resolve_two_hdbscan_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_hdbscan_devices("cuml", None, "cuda:0,1", 2) == [0, 1]
    assert resolve_hdbscan_devices("cuml", 1, None, None) == [1]


def test_reject_conflicting_hdbscan_device_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="cannot be used together"):
        resolve_hdbscan_devices("cuml", 0, "0,1", 2)
    with pytest.raises(ValueError, match="between 1"):
        resolve_hdbscan_devices("cuml", None, "0,1", 3)


def test_progress_duration_and_remaining_estimate() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3_661) == "1h01m01s"

    remaining = estimate_remaining_seconds(
        total_windows=10,
        embedded_windows=5,
        embedding_seconds=10.0,
        file_lengths=[10, 20],
        completed_files=1,
        hdbscan_seconds=4.0,
        hdbscan_work=100.0,
        other_seconds=2.0,
        finalized_pulses=10,
    )

    # Embedding: 10 s, HDBSCAN: 16 s, remaining postprocessing: 4 s.
    assert remaining == pytest.approx(30.0)
