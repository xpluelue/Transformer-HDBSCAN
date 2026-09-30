#!/usr/bin/env python3
"""Set PDW Studio ``labels`` to saved true labels without reclustering."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="PDW Studio output directory, one config directory, or one HDF5.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag the command only audits files.",
    )
    return parser.parse_args()


def resolve_cluster_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() != ".h5":
            raise ValueError(f"Expected an HDF5 file: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    direct = sorted(root.glob("cluster*.h5"))
    files = direct if direct else sorted(root.rglob("cluster*.h5"))
    if not files:
        raise FileNotFoundError(f"No cluster*.h5 files found below {root}")
    return files


def inspect_or_update(path: Path, apply: bool) -> str:
    mode = "r+" if apply else "r"
    with h5py.File(path, mode) as handle:
        if "data" not in handle or "labels" not in handle:
            raise ValueError(f"{path} is missing data or labels")
        labels = handle["labels"]
        if len(labels) != len(handle["data"]):
            raise ValueError(f"{path} has mismatched data/labels lengths")
        if "true_label" not in handle:
            return "unlabeled"
        truth = np.asarray(handle["true_label"][:], dtype=np.int64).reshape(-1, 1)
        if labels.shape != truth.shape:
            raise ValueError(
                f"{path} labels shape {labels.shape} != true_label shape {truth.shape}"
            )
        current = np.asarray(labels[:], dtype=np.int64)
        if np.array_equal(current, truth):
            if apply:
                handle.attrs["labels_semantics"] = "true_emitter_labels"
            return "already_correct"
        if not apply:
            return "needs_update"
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{path} labels dtype is not integer: {labels.dtype}")
        limits = np.iinfo(labels.dtype)
        if np.any(truth < limits.min) or np.any(truth > limits.max):
            raise OverflowError(
                f"{path} true labels do not fit existing dtype {labels.dtype}"
            )
        labels[...] = truth.astype(labels.dtype, copy=False)
        handle.attrs["labels_semantics"] = "true_emitter_labels"
        handle.flush()
        return "updated"


def main() -> None:
    args = parse_args()
    files = resolve_cluster_files(args.root)
    counts: Counter[str] = Counter()
    errors: list[tuple[Path, Exception]] = []
    action = "APPLY" if args.apply else "DRY-RUN"
    print(f"{action}: scanning {len(files):,} cluster HDF5 files")
    for index, path in enumerate(files, start=1):
        try:
            counts[inspect_or_update(path, args.apply)] += 1
        except (OSError, ValueError, OverflowError) as error:
            errors.append((path, error))
        if index % 500 == 0 or index == len(files):
            print(f"processed={index:,}/{len(files):,}")

    print(
        "summary: "
        f"already_correct={counts['already_correct']:,}, "
        f"needs_update={counts['needs_update']:,}, "
        f"updated={counts['updated']:,}, "
        f"unlabeled={counts['unlabeled']:,}, errors={len(errors):,}"
    )
    for path, error in errors[:20]:
        print(f"ERROR {path}: {error}")
    if errors:
        raise RuntimeError(f"Failed to process {len(errors)} HDF5 file(s)")
    if not args.apply and counts["needs_update"]:
        print("Dry run only. Re-run with --apply to update /labels in place.")


if __name__ == "__main__":
    main()
