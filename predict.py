#!/usr/bin/env python3
"""Run a trained Transformer encoder and HDBSCAN on HDF5 pulse windows."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import H5WindowDataset, resolve_h5_files
from model import TransformerMetricEncoder


def cluster_embedding(embedding: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run one independent HDBSCAN window in one CPU process."""
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        n_jobs=1,
        copy=False,
    ).fit_predict(embedding).astype(np.int32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.npz"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument(
        "--min-emitters",
        type=int,
        default=2,
        help="Only evaluate full windows containing at least this many emitters.",
    )
    parser.add_argument(
        "--hdbscan-jobs",
        type=int,
        default=1,
        help="CPU processes used to cluster independent windows (default: 1).",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.min_emitters <= 0
        or args.hdbscan_jobs <= 0
    ):
        raise ValueError(
            "batch-size/min-emitters/hdbscan-jobs must be positive; "
            "num-workers cannot be negative"
        )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TransformerMetricEncoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    window_length = args.window_length or int(checkpoint.get("window_length", 1024))
    min_cluster_size = args.min_cluster_size or int(
        checkpoint.get("min_cluster_size", 5)
    )
    files = resolve_h5_files(args.data_dir, split=args.split)
    dataset = H5WindowDataset(files, window_length, require_labels=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    predicted_flat: list[np.ndarray] = []
    true_flat: list[np.ndarray] = []
    file_indices: list[int] = []
    pulse_starts: list[int] = []
    lengths: list[int] = []
    v_scores: list[float] = []
    ari_scores: list[float] = []
    ami_scores: list[float] = []
    skipped_partial = 0
    skipped_low_emitter = 0

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print(
        f"device={device} ({device_name}), batch_size={args.batch_size}, "
        f"HDBSCAN CPU workers={args.hdbscan_jobs}"
    )

    executor: ProcessPoolExecutor | None = None
    if args.hdbscan_jobs > 1:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        executor = ProcessPoolExecutor(
            max_workers=args.hdbscan_jobs,
            mp_context=mp.get_context("spawn"),
        )

    try:
        with torch.inference_mode():
            for features, labels, padding_mask, file_idx, start, valid_len in tqdm(
                loader, desc="Predict"
            ):
                labels_np = labels.numpy()
                valid_lengths = valid_len.numpy()
                complete = valid_lengths == window_length
                emitter_counts = np.asarray(
                    [
                        np.unique(label[:length]).size
                        for label, length in zip(
                            labels_np, valid_lengths, strict=True
                        )
                    ],
                    dtype=np.int64,
                )
                enough_emitters = emitter_counts >= args.min_emitters
                eligible = complete & enough_emitters
                skipped_partial += int((~complete).sum())
                skipped_low_emitter += int((complete & ~enough_emitters).sum())
                if not eligible.any():
                    continue

                eligible_tensor = torch.from_numpy(eligible)
                features = features[eligible_tensor]
                padding_mask = padding_mask[eligible_tensor]
                labels_np = labels_np[eligible]
                file_indices_np = file_idx.numpy()[eligible]
                pulse_starts_np = start.numpy()[eligible]
                embeddings = model(
                    features.to(device, non_blocking=True),
                    padding_mask=padding_mask.to(device, non_blocking=True),
                ).float().cpu().numpy()

                predictions = (
                    executor.map(
                        cluster_embedding,
                        embeddings,
                        [min_cluster_size] * len(embeddings),
                    )
                    if executor is not None
                    else map(
                        cluster_embedding,
                        embeddings,
                        [min_cluster_size] * len(embeddings),
                    )
                )
                for predicted, truth, current_file, current_start in zip(
                    predictions,
                    labels_np,
                    file_indices_np,
                    pulse_starts_np,
                    strict=True,
                ):
                    predicted_flat.append(predicted)
                    true_flat.append(truth)
                    v_scores.append(v_measure_score(truth, predicted))
                    ari_scores.append(adjusted_rand_score(truth, predicted))
                    ami_scores.append(adjusted_mutual_info_score(truth, predicted))
                    file_indices.append(int(current_file))
                    pulse_starts.append(int(current_start))
                    lengths.append(window_length)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not predicted_flat:
        raise RuntimeError(
            "No full windows contain the requested minimum number of emitters"
        )

    lengths_array = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths_array)))
    arrays: dict[str, np.ndarray] = {
        "predicted_labels": np.concatenate(predicted_flat),
        "window_offsets": offsets,
        "window_lengths": lengths_array,
        "file_indices": np.asarray(file_indices, dtype=np.int64),
        "pulse_starts": np.asarray(pulse_starts, dtype=np.int64),
        "source_files": np.asarray([str(path) for path in files]),
        "noise_label": np.asarray(-1, dtype=np.int64),
    }
    arrays["true_labels"] = np.concatenate(true_flat)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved predictions: {args.output.resolve()}")
    print(
        f"mean V-measure={np.mean(v_scores):.4f}, "
        f"ARI={np.mean(ari_scores):.4f}, AMI={np.mean(ami_scores):.4f}; "
        f"evaluated={len(v_scores)}, skipped_partial={skipped_partial}, "
        f"skipped_low_emitter={skipped_low_emitter}"
    )


if __name__ == "__main__":
    main()
