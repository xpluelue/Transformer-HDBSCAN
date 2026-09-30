from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from turing_deinterleaving_challenge.transformer_hdbscan.data import (
    FileWindowBatchSampler,
    H5WindowDataset,
    fit_file_normalization,
    fit_global_normalization,
    resolve_h5_files,
)


def write_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pdws = np.arange(50, dtype=np.float32).reshape(10, 5)
    pdws[:, 0] = np.arange(10, dtype=np.float32) * 10
    labels = np.asarray([0, 0, 1, 1, 1, 0, 2, 2, 3, 3], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=pdws)
        handle.create_dataset("labels", data=labels)
    return pdws, labels


def test_exhaustive_windows_retain_raw_tail(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    pdws, labels = write_source(source)
    dataset = H5WindowDataset([source], 6, return_raw=True)

    assert len(dataset) == 2
    first = dataset[0]
    tail = dataset[1]
    np.testing.assert_array_equal(first[-1][:6], pdws[:6])
    np.testing.assert_array_equal(tail[-1][:4], pdws[6:])
    np.testing.assert_array_equal(tail[1][:4], labels[6:])
    assert tail[5] == 4
    assert tail[2].tolist() == [False, False, False, False, True, True]


def test_global_normalization_preserves_delta_toa_across_window_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    write_source(source)
    mean, std, sample_count = fit_global_normalization([source], max_samples=100)
    dataset = H5WindowDataset(
        [source],
        6,
        normalization="global",
        normalization_mean=mean,
        normalization_std=std,
    )

    tail_features = dataset[1][0]
    recovered_first_delta = tail_features[0, 0] * std[0] + mean[0]

    assert sample_count == 10
    assert recovered_first_delta == pytest.approx(10.0)


def test_overlapping_windows_share_one_file_normalization(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    write_source(source)
    mean, std, sample_count = fit_file_normalization(source, chunk_size=3)
    dataset = H5WindowDataset(
        [source],
        6,
        window_stride=3,
        normalization="per_file",
    )

    assert sample_count == 10
    assert len(dataset) == 3
    assert [dataset.window_location(index)[1] for index in range(3)] == [0, 3, 6]
    np.testing.assert_allclose(dataset.file_normalization_means[0], mean)
    np.testing.assert_allclose(dataset.file_normalization_stds[0], std)
    np.testing.assert_allclose(dataset[0][0][3], dataset[1][0][0])


def test_file_batch_sampler_never_mixes_source_files(tmp_path: Path) -> None:
    sources = [tmp_path / f"source_{index}.h5" for index in range(2)]
    for source in sources:
        write_source(source)
    dataset = H5WindowDataset(sources, 4)
    sampler = FileWindowBatchSampler(dataset, batch_size=2, shuffle=True, seed=7)

    for batch in sampler:
        file_indices = {dataset.window_location(index)[0] for index in batch}
        assert len(file_indices) == 1


def test_resolver_warns_and_skips_empty_source_files(tmp_path: Path) -> None:
    split = tmp_path / "train_scan"
    split.mkdir()
    nonempty = split / "config_0.h5"
    write_source(nonempty)
    empty = split / "config_1.h5"
    with h5py.File(empty, "w") as handle:
        handle.create_dataset("data", data=np.empty((0, 5), dtype=np.float32))
        handle.create_dataset("labels", data=np.empty(0, dtype=np.int64))

    with pytest.warns(RuntimeWarning, match="Skipping 1 empty HDF5"):
        files = resolve_h5_files(tmp_path, split="train")

    assert files == [nonempty.resolve()]
