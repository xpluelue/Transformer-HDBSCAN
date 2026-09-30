from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from turing_deinterleaving_challenge.transformer_hdbscan.update_pdw_studio_labels import inspect_or_update


def test_update_pdw_studio_labels_from_saved_truth(tmp_path: Path) -> None:
    path = tmp_path / "cluster000.h5"
    truth = np.asarray([[4], [7], [7]], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((3, 5), dtype=np.float32))
        handle.create_dataset("labels", data=-np.ones((3, 1), dtype=np.int32))
        handle.create_dataset("true_label", data=truth)

    assert inspect_or_update(path, apply=False) == "needs_update"
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["labels"][:], -np.ones((3, 1)))

    assert inspect_or_update(path, apply=True) == "updated"
    assert inspect_or_update(path, apply=False) == "already_correct"
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["labels"][:], truth)
        assert handle.attrs["labels_semantics"] == "true_emitter_labels"
