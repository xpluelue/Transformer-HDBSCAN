#!/usr/bin/env python3
"""Inspect a complete source-file result and optionally export one cluster."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    from .output import cluster_group_name
except ImportError:  # pragma: no cover
    from output import cluster_group_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--predicted-label", type=int)
    selection.add_argument("--true-label", type=int)
    parser.add_argument(
        "--export-cluster",
        type=Path,
        default=None,
        help="Write the selected cluster's PDWs and indices to an NPZ file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.result_file.expanduser().resolve()
    with h5py.File(path, "r") as handle:
        print(f"Source-file result: {path}")
        print("\nMetadata:")
        for name in sorted(handle.attrs):
            print(f"  {name}: {handle.attrs[name]}")

        for parent_name in ("predicted_clusters", "true_clusters"):
            if parent_name not in handle:
                continue
            print(f"\n{parent_name}:")
            parent = handle[parent_name]
            groups = sorted(parent.values(), key=lambda group: int(group.attrs["label"]))
            for group in groups:
                print(
                    f"  label={int(group.attrs['label']):4d} "
                    f"pulses={int(group.attrs['size']):5d}"
                )

        if "evaluation" in handle:
            evaluation = handle["evaluation"]
            print("\nMetrics:")
            for name in sorted(evaluation.attrs):
                print(f"  {name}: {float(evaluation.attrs[name]):.6f}")
            print("\nContingency matrix (predicted rows x true columns):")
            print("  predicted:", evaluation["predicted_cluster_ids"][:].tolist())
            print("  true     :", evaluation["true_cluster_ids"][:].tolist())
            print(evaluation["contingency_matrix"][:])

        selected_label = (
            args.predicted_label
            if args.predicted_label is not None
            else args.true_label
        )
        if selected_label is None:
            if args.export_cluster is not None:
                raise ValueError(
                    "--export-cluster requires --predicted-label or --true-label"
                )
            return
        parent_name = (
            "predicted_clusters"
            if args.predicted_label is not None
            else "true_clusters"
        )
        group_path = f"{parent_name}/{cluster_group_name(selected_label)}"
        if group_path not in handle:
            raise KeyError(f"Cluster is not present: {group_path}")
        group = handle[group_path]
        print(f"\nSelected: {group_path}, pulses={len(group['pdws'])}")
        if args.export_cluster is not None:
            output = args.export_cluster.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                "label": np.asarray(selected_label, dtype=np.int64),
                "pdws": group["pdws"][:],
                "member_indices": group["member_indices"][:],
                "source_indices": group["source_indices"][:],
                "counterpart_labels": group["counterpart_labels"][:],
            }
            np.savez_compressed(output, **arrays)
            print(f"Exported cluster: {output}")


if __name__ == "__main__":
    main()
