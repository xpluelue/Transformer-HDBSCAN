#!/usr/bin/env python3
"""Convert existing PDW Studio cluster files without running inference again.

Legacy files named ``clusterNNN.h5`` are replaced in place by files matching
``config_590_2.h5``: ``<source>_<cluster-id>.h5`` containing only ``/data`` as
``float64 [5, N]`` and ``/labels`` as ``int32 [N]``.  The old
``original_cluster_label`` attribute is preserved in the root ``_clusters.csv``
manifest because the target HDF5 format has no attributes.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


LEGACY_NAME = re.compile(r"^cluster(?P<cluster_id>\d+)$")
CLUSTER_FIELDNAMES = (
    "cluster_file",
    "source_file",
    "source_stem",
    "output_cluster_id",
    "original_cluster_label",
    "pulse_count",
)


@dataclass(frozen=True)
class ConversionRecord:
    source: Path
    target: Path
    cluster_file: str
    source_file: str
    source_stem: str
    output_cluster_id: int
    original_cluster_label: int | str
    pulse_count: int
    label_source: str

    def manifest_row(self) -> dict[str, object]:
        row = asdict(self)
        return {name: row[name] for name in CLUSTER_FIELDNAMES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Existing PDW Studio output root, one config directory, or one cluster HDF5.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Convert in place. Without this flag, only validate and report changes.",
    )
    return parser.parse_args()


def resolve_legacy_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() != ".h5" or LEGACY_NAME.fullmatch(root.stem) is None:
            raise ValueError(f"Expected a legacy clusterNNN.h5 file: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    direct = sorted(root.glob("cluster*.h5"))
    files = direct if direct else sorted(root.rglob("cluster*.h5"))
    files = [path for path in files if LEGACY_NAME.fullmatch(path.stem)]
    if not files:
        raise FileNotFoundError(f"No legacy clusterNNN.h5 files found below {root}")
    return files


def output_root_for(requested_root: Path, files: list[Path]) -> Path:
    root = requested_root.expanduser().resolve()
    if root.is_file():
        return root.parent.parent
    if any(path.parent == root for path in files):
        return root.parent
    return root


def _integer_attr(handle: h5py.File, name: str, fallback: int) -> int:
    value = handle.attrs.get(name, fallback)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer attribute {name}={value!r}") from error


def inspect_legacy_file(path: Path, output_root: Path) -> ConversionRecord:
    match = LEGACY_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Not a legacy cluster file name: {path.name}")
    with h5py.File(path, "r") as handle:
        if "data" not in handle or "labels" not in handle:
            raise ValueError(f"{path} must contain /data and /labels")
        label_source = "true_label" if "true_label" in handle else "labels"
        labels = handle[label_source]
        pulse_count = int(labels.size)
        if handle["labels"].size != pulse_count:
            raise ValueError(f"{path} has inconsistent /labels and /{label_source}")
        data = handle["data"]
        if data.ndim != 2:
            raise ValueError(f"{path} /data must be 2-D, got {data.shape}")
        if pulse_count not in data.shape:
            raise ValueError(
                f"{path} /data shape {data.shape} has no pulse axis of {pulse_count}"
            )
        feature_count = data.shape[1] if data.shape[0] == pulse_count else data.shape[0]
        if feature_count != 5:
            raise ValueError(
                f"{path} must contain five PDW features, got shape {data.shape}"
            )

        source_stem_value = handle.attrs.get("source_stem", path.parent.name)
        if isinstance(source_stem_value, bytes):
            source_stem_value = source_stem_value.decode("utf-8")
        source_stem = str(source_stem_value)
        output_cluster_id = _integer_attr(
            handle, "output_cluster_id", int(match.group("cluster_id"))
        )
        original_cluster_label: int | str
        if "original_cluster_label" in handle.attrs:
            original_cluster_label = _integer_attr(
                handle, "original_cluster_label", output_cluster_id
            )
        else:
            original_cluster_label = ""
        source_file_value = handle.attrs.get("source_file", "")
        if isinstance(source_file_value, bytes):
            source_file_value = source_file_value.decode("utf-8")
        source_file = str(source_file_value)

    target = path.parent / f"{source_stem}_{output_cluster_id}.h5"
    try:
        cluster_file = target.relative_to(output_root).as_posix()
    except ValueError:
        cluster_file = target.name
    return ConversionRecord(
        source=path,
        target=target,
        cluster_file=cluster_file,
        source_file=source_file,
        source_stem=source_stem,
        output_cluster_id=output_cluster_id,
        original_cluster_label=original_cluster_label,
        pulse_count=pulse_count,
        label_source=label_source,
    )


def convert_cluster_file(
    record: ConversionRecord, *, remove_source: bool = True
) -> None:
    if record.target.exists():
        raise FileExistsError(f"Target already exists: {record.target}")
    temporary = record.target.with_name(f".{record.target.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary target already exists: {temporary}")
    try:
        with h5py.File(record.source, "r") as source:
            raw = np.asarray(source["data"][:])
            labels = np.asarray(source[record.label_source][:]).reshape(-1)
        if raw.shape[0] == record.pulse_count:
            output_data = raw.T
        elif raw.shape[1] == record.pulse_count:
            output_data = raw
        else:  # Protected by inspection, retained for defensive clarity.
            raise ValueError(f"Cannot find pulse axis in {record.source}: {raw.shape}")

        if not np.issubdtype(labels.dtype, np.integer):
            if not np.all(np.isfinite(labels)) or not np.all(
                labels == np.floor(labels)
            ):
                raise ValueError(f"{record.source} labels are not integer-valued")
        labels64 = labels.astype(np.int64, copy=False)
        limits = np.iinfo(np.int32)
        if np.any(labels64 < limits.min) or np.any(labels64 > limits.max):
            raise OverflowError(f"{record.source} labels do not fit int32")

        with h5py.File(temporary, "w") as target:
            target.create_dataset(
                "data", data=np.asarray(output_data, dtype=np.float64), dtype=np.float64
            )
            target.create_dataset(
                "labels", data=labels64.astype(np.int32), dtype=np.int32
            )
        os.replace(temporary, record.target)
        if remove_source:
            record.source.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def write_cluster_manifest(output_root: Path, records: list[ConversionRecord]) -> Path:
    manifest = output_root / "_clusters.csv"
    temporary = output_root / "._clusters.csv.tmp"
    merged_rows: dict[str, dict[str, object]] = {}
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cluster_file = row.get("cluster_file", "")
                if cluster_file:
                    merged_rows[cluster_file] = {
                        name: row.get(name, "") for name in CLUSTER_FIELDNAMES
                    }
    for record in records:
        merged_rows[record.cluster_file] = record.manifest_row()
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CLUSTER_FIELDNAMES)
            writer.writeheader()
            writer.writerows(merged_rows[key] for key in sorted(merged_rows))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    args = parse_args()
    files = resolve_legacy_files(args.root)
    output_root = output_root_for(args.root, files)
    records: list[ConversionRecord] = []
    errors: list[tuple[Path, Exception]] = []
    counts: Counter[str] = Counter()
    action = "APPLY" if args.apply else "DRY-RUN"
    print(f"{action}: validating {len(files):,} legacy cluster HDF5 files")
    print("No Transformer inference, embedding generation, or HDBSCAN will run.")

    for index, path in enumerate(files, start=1):
        try:
            record = inspect_legacy_file(path, output_root)
            records.append(record)
            counts["ready"] += 1
        except (OSError, ValueError, OverflowError) as error:
            errors.append((path, error))
        if index % 500 == 0 or index == len(files):
            print(f"processed={index:,}/{len(files):,}")

    target_counts = Counter(record.target for record in records)
    duplicate_targets = {target for target, count in target_counts.items() if count > 1}
    for target in sorted(duplicate_targets):
        errors.append((target, ValueError("multiple legacy files map to this target")))
    for record in records:
        if record.target.exists():
            errors.append((record.source, FileExistsError(record.target)))

    if errors:
        for path, error in errors[:20]:
            print(f"ERROR {path}: {error}")
        raise RuntimeError(f"Failed to validate {len(errors)} HDF5 file(s)")

    manifest: Path | None = None
    if args.apply:
        generated: list[Path] = []
        try:
            for index, record in enumerate(records, start=1):
                convert_cluster_file(record, remove_source=False)
                generated.append(record.target)
                counts["converted"] += 1
                if index % 500 == 0 or index == len(records):
                    print(f"converted={index:,}/{len(records):,}")
            manifest = write_cluster_manifest(output_root, records)
        except Exception:
            for target in generated:
                target.unlink(missing_ok=True)
            raise
        cleanup_errors: list[tuple[Path, OSError]] = []
        for record in records:
            try:
                record.source.unlink()
            except OSError as error:
                cleanup_errors.append((record.source, error))
        if cleanup_errors:
            for path, error in cleanup_errors[:20]:
                print(f"WARNING could not remove legacy file {path}: {error}")
            print(
                f"WARNING: {len(cleanup_errors):,} legacy file(s) remain, "
                "but converted files and manifest are complete."
            )
    print(
        "summary: "
        f"ready={counts['ready']:,}, converted={counts['converted']:,}, "
        f"errors={len(errors):,}"
    )
    if args.apply:
        print(f"Cluster manifest: {manifest}")
    else:
        print("Dry run only. Re-run with --apply to convert and rename in place.")


if __name__ == "__main__":
    main()
