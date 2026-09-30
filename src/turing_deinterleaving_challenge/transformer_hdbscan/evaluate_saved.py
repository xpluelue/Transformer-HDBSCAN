#!/usr/bin/env python3
"""Evaluate already-saved complete-file clustering results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:  # Support package modules and direct script execution.
    from .metrics import METRIC_NAMES, evaluate_labels
except ImportError:  # pragma: no cover
    from metrics import METRIC_NAMES, evaluate_labels


FIELDNAMES = (
    "result_file",
    "source_file",
    "file_index",
    "pulse_count",
    "predicted_cluster_count",
    "true_cluster_count",
    "minus_one_group_size",
    *METRIC_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="One result HDF5 or a prediction directory containing files/*.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for evaluation.csv and summary.json.",
    )
    parser.add_argument(
        "--skip-unlabeled",
        action="store_true",
        help="Skip result files without pulses/true_labels instead of failing.",
    )
    return parser.parse_args()


def resolve_result_files(path: Path) -> list[Path]:
    """Resolve one result file or the files/ directory of a prediction run."""
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Expected an HDF5 result file: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files_dir = resolved / "files"
    search_dir = files_dir if files_dir.is_dir() else resolved
    files = sorted([*search_dir.glob("*.h5"), *search_dir.glob("*.hdf5")])
    if not files:
        raise FileNotFoundError(f"No saved clustering HDF5 files found in {resolved}")
    return files


def _attribute_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def evaluate_result_file(path: Path) -> dict[str, object]:
    """Read saved labels and calculate complete-file metrics without mutation."""
    with h5py.File(path, "r") as handle:
        if "pulses/predicted_labels" not in handle:
            raise ValueError(f"Missing pulses/predicted_labels: {path}")
        if "pulses/true_labels" not in handle:
            raise ValueError(f"Missing pulses/true_labels: {path}")
        predicted = np.asarray(handle["pulses/predicted_labels"][:]).reshape(-1)
        truth = np.asarray(handle["pulses/true_labels"][:]).reshape(-1)
        source_file = _attribute_text(handle.attrs.get("source_file"), path.name)
        file_index = int(handle.attrs.get("file_index", -1))

    if predicted.shape != truth.shape or not len(predicted):
        raise ValueError(f"Saved predicted/true labels are empty or misaligned: {path}")
    scores = evaluate_labels(predicted, truth)
    row: dict[str, object] = {
        "result_file": str(path),
        "source_file": source_file,
        "file_index": file_index,
        "pulse_count": len(predicted),
        "predicted_cluster_count": len(np.unique(predicted)),
        "true_cluster_count": len(np.unique(truth)),
        "minus_one_group_size": int(np.count_nonzero(predicted == -1)),
    }
    row.update(scores)
    return row


def evaluate_saved_results(
    input_path: Path,
    output_dir: Path,
    *,
    skip_unlabeled: bool = False,
) -> dict[str, object]:
    """Evaluate all saved files and atomically write separate reports."""
    files = resolve_result_files(input_path)
    destination = output_dir.expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Evaluation output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(
                f"Evaluation output directory is not empty: {destination}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "evaluation.csv"
    summary_path = destination / "summary.json"
    csv_temporary = destination / ".evaluation.csv.tmp"
    summary_temporary = destination / ".summary.json.tmp"

    rows: list[dict[str, object]] = []
    skipped_files: list[str] = []
    try:
        for path in tqdm(files, desc="Evaluate saved clustering files", unit="file"):
            try:
                rows.append(evaluate_result_file(path))
            except ValueError as error:
                if skip_unlabeled and "pulses/true_labels" in str(error):
                    skipped_files.append(str(path))
                    continue
                raise
        if not rows:
            raise ValueError("No labeled clustering result files were evaluated")

        with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(csv_temporary, csv_path)

        metric_summary = {
            name: {
                "mean": float(np.mean([float(row[name]) for row in rows])),
                "median": float(np.median([float(row[name]) for row in rows])),
                "min": float(np.min([float(row[name]) for row in rows])),
                "max": float(np.max([float(row[name]) for row in rows])),
            }
            for name in METRIC_NAMES
        }
        summary: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(input_path.expanduser().resolve()),
            "evaluated_source_file_count": len(rows),
            "skipped_source_file_count": len(skipped_files),
            "skipped_source_files": skipped_files,
            "total_pulse_count": int(sum(int(row["pulse_count"]) for row in rows)),
            "metrics_scope": "complete_source_file_macro_average",
            "minus_one_is_ordinary_cluster": True,
            "metrics": metric_summary,
        }
        summary_temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(summary_temporary, summary_path)
        return summary
    finally:
        csv_temporary.unlink(missing_ok=True)
        summary_temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    summary = evaluate_saved_results(
        args.input,
        args.output_dir,
        skip_unlabeled=args.skip_unlabeled,
    )
    print(
        f"Evaluated {summary['evaluated_source_file_count']} source file(s); "
        f"reports: {args.output_dir.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
