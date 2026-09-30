#!/usr/bin/env python3
"""Convert saved native predictions to PDW Studio cluster files without inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

try:  # Support both package modules and direct scripts.
    from .metrics import METRIC_NAMES
    from .output import PDWStudioRunWriter, SourceFileMetadata
except ImportError:  # pragma: no cover
    from metrics import METRIC_NAMES
    from output import PDWStudioRunWriter, SourceFileMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Native prediction directory containing manifest.csv and files/.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="New empty PDW Studio output directory.",
    )
    return parser.parse_args()


def _text_attr(handle: h5py.File, name: str) -> str:
    value = handle.attrs[name]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_native_file(
    path: Path,
) -> tuple[
    SourceFileMetadata,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    dict[str, float] | None,
]:
    with h5py.File(path, "r") as handle:
        required = ("pulses/raw_pdws", "pulses/predicted_labels")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path} is not a native result; missing {missing}")

        raw = np.asarray(handle["pulses/raw_pdws"][:])
        predicted = np.asarray(handle["pulses/predicted_labels"][:]).reshape(-1)
        truth = (
            np.asarray(handle["pulses/true_labels"][:]).reshape(-1)
            if "pulses/true_labels" in handle
            else None
        )
        metadata = SourceFileMetadata(
            source_file=_text_attr(handle, "source_file"),
            file_index=int(handle.attrs["file_index"]),
            pulse_count=int(handle.attrs["pulse_count"]),
            window_count=int(handle.attrs["window_count"]),
            window_length=int(handle.attrs["window_length"]),
            window_stride=int(handle.attrs["window_stride"]),
            clustering_scope=_text_attr(handle, "clustering_scope"),
        )
        metrics = None
        if "evaluation" in handle:
            evaluation = handle["evaluation"]
            metrics = {
                name: float(evaluation.attrs[name])
                for name in METRIC_NAMES
                if name in evaluation.attrs
            }
            if not metrics:
                metrics = None
    return metadata, raw, predicted, truth, metrics


def convert_native_run(input_dir: Path, output_dir: Path) -> dict[str, object]:
    source = input_dir.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    manifest_path = source / "manifest.csv"
    files_dir = source / "files"
    if not manifest_path.is_file() or not files_dir.is_dir():
        raise FileNotFoundError(
            f"Expected native manifest.csv and files/ below {source}"
        )
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {target}")

    source_summary_path = source / "summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    run_metadata = {
        "checkpoint": source_summary.get("checkpoint", "<unknown>"),
        "converted_from_native": str(source),
        "evaluation_enabled": source_summary.get("evaluation_enabled", False),
    }

    temporary = target.with_name(f".{target.name}.convert.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary conversion directory already exists: {temporary}")
    writer = PDWStudioRunWriter(temporary, run_metadata)
    converted = 0
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            if "result_file" not in (rows.fieldnames or []):
                raise ValueError(f"Native manifest has no result_file column: {manifest_path}")
            for row in rows:
                result_path = source / row["result_file"]
                metadata, raw, predicted, truth, metrics = _read_native_file(result_path)
                writer.write(
                    metadata,
                    raw,
                    predicted,
                    truth,
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    metrics,
                )
                converted += 1
                if converted == 1 or converted % 25 == 0:
                    print(f"Convert native predictions: {converted} files", flush=True)
        summary = writer.close()
        if target.exists():
            target.rmdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    except BaseException:
        writer.abort()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Converted {converted} source files to {target}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    convert_native_run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
