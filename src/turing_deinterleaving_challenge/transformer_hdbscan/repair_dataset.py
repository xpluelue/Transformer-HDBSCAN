#!/usr/bin/env python3
"""Audit TSRD HDF5 files and re-download only missing or invalid files."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DEFAULT_REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SPLITS = ("train_scan", "val_scan", "test_scan")
MARKDOWN_URL = re.compile(r"^\[(https?://[^\]]+)\]\((https?://[^)]+)\)$")


@dataclass(frozen=True)
class AuditResult:
    relative_path: str
    valid: bool
    reason: str
    pulse_count: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local download root onto which remote repository paths are appended.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--subdir", default="scan")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=262_144,
        help="Rows read per integrity-check chunk (default: 262144).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only inspect HDF5 metadata; full chunk reads detect more corruption.",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Do not require a labels dataset matching data length.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Download and replace only files that fail the audit.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup directory for replaced files (default: dataset-root/.repair_backups/<time>).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset_repair_report.json"),
    )
    return parser.parse_args()


def _check_finite(dataset: h5py.Dataset, chunk_rows: int) -> None:
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"non-numeric dtype {dataset.dtype}")
    for start in range(0, len(dataset), chunk_rows):
        values = np.asarray(dataset[start : start + chunk_rows])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"NaN/Inf found near row {start}")


def validate_h5(
    path: Path,
    relative_path: str,
    *,
    require_labels: bool,
    chunk_rows: int,
    full_read: bool,
) -> AuditResult:
    """Validate structure and optionally read every HDF5 data chunk."""
    if not path.is_file():
        return AuditResult(relative_path, False, "missing")
    try:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError("missing dataset 'data'")
            data = handle["data"]
            if data.ndim != 2:
                raise ValueError(f"data must be 2-D, got shape {data.shape}")
            pulse_count = len(data)
            if pulse_count == 0:
                raise ValueError("data contains zero pulses")
            labels = handle.get("labels")
            if require_labels and labels is None:
                raise ValueError("missing dataset 'labels'")
            if labels is not None and len(labels) != pulse_count:
                raise ValueError(
                    f"data/labels length mismatch: {pulse_count} != {len(labels)}"
                )
            if full_read:
                try:
                    _check_finite(data, chunk_rows)
                    if labels is not None:
                        _check_finite(labels, chunk_rows)
                except (OSError, RuntimeError) as error:
                    raise ValueError(f"corrupt HDF5 chunk: {error}") from error
        return AuditResult(relative_path, True, "ok", pulse_count)
    except (OSError, RuntimeError, ValueError) as error:
        return AuditResult(relative_path, False, str(error))


def normalize_endpoint(value: str) -> str:
    """Accept a plain endpoint and repair an accidentally pasted Markdown URL."""
    endpoint = value.strip().rstrip("/")
    markdown_match = MARKDOWN_URL.fullmatch(endpoint)
    if markdown_match:
        visible, target = markdown_match.groups()
        if visible != target:
            raise ValueError("Markdown endpoint text and target URL do not match")
        endpoint = target.rstrip("/")
        print(f"Normalized Markdown endpoint to plain URL: {endpoint}")
    if not re.fullmatch(r"https?://[^/]+", endpoint):
        raise ValueError(
            "--endpoint must be a plain origin such as https://hf-mirror.com"
        )
    return endpoint


def remote_h5_files(
    repo_id: str,
    endpoint: str,
    subdir: str,
    splits: list[str],
) -> tuple[list[str], list[str]]:
    """List expected HDF5 paths without downloading dataset contents."""
    api = HfApi(endpoint=endpoint)
    repository_files = api.list_repo_files(
        repo_id,
        repo_type="dataset",
        token=True,
    )
    normalized_subdir = subdir.strip("/")
    if normalized_subdir == ".":
        normalized_subdir = ""
    base_prefix = f"{normalized_subdir}/" if normalized_subdir else ""
    prefixes = tuple(f"{base_prefix}{split.strip('/')}/" for split in splits)
    matches = sorted(
        path
        for path in repository_files
        if path.startswith(prefixes) and Path(path).suffix.lower() in {".h5", ".hdf5"}
    )
    return matches, repository_files


def repair_one_file(
    *,
    relative_path: str,
    dataset_root: Path,
    backup_root: Path,
    repo_id: str,
    endpoint: str,
    require_labels: bool,
    chunk_rows: int,
) -> AuditResult:
    """Download, validate, back up, and atomically replace one result."""
    target = dataset_root / relative_path
    with tempfile.TemporaryDirectory(prefix="tsrd-repair-") as temporary:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=relative_path,
                repo_type="dataset",
                endpoint=endpoint,
                token=True,
                local_dir=temporary,
                local_dir_use_symlinks=False,
                force_download=True,
            )
        )
        downloaded_result = validate_h5(
            downloaded,
            relative_path,
            require_labels=require_labels,
            chunk_rows=chunk_rows,
            full_read=True,
        )
        if not downloaded_result.valid:
            return AuditResult(
                relative_path,
                False,
                f"downloaded copy is invalid: {downloaded_result.reason}",
            )

        if target.exists():
            backup = backup_root / relative_path
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_target = target.with_name(f".{target.name}.repair.tmp")
        try:
            shutil.copy2(downloaded, temporary_target)
            os.replace(temporary_target, target)
        finally:
            temporary_target.unlink(missing_ok=True)
    return validate_h5(
        target,
        relative_path,
        require_labels=require_labels,
        chunk_rows=chunk_rows,
        full_read=True,
    )


def main() -> None:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    endpoint = normalize_endpoint(args.endpoint)

    print(f"Query remote inventory: {endpoint}/{args.repo_id}")
    try:
        expected_files, repository_files = remote_h5_files(
            args.repo_id,
            endpoint,
            args.subdir,
            args.splits,
        )
    except Exception as error:
        raise RuntimeError(
            "Cannot read the gated dataset inventory. Confirm dataset access and "
            "run `hf auth login` (or `huggingface-cli login`) first."
        ) from error
    if not expected_files:
        remote_hdf5 = [
            path
            for path in repository_files
            if Path(path).suffix.lower() in {".h5", ".hdf5"}
        ]
        sample_source = remote_hdf5 if remote_hdf5 else repository_files
        samples = "\n".join(f"  {path}" for path in sample_source[:20])
        normalized_subdir = args.subdir.strip("/")
        if normalized_subdir == ".":
            normalized_subdir = ""
        base_prefix = f"{normalized_subdir}/" if normalized_subdir else ""
        requested = ", ".join(
            f"{base_prefix}{split.strip('/')}/" for split in args.splits
        )
        raise RuntimeError(
            "Remote inventory was accessible, but no HDF5 matched the requested "
            f"prefixes: {requested}. Remote file count={len(repository_files)}, "
            f"HDF5 count={len(remote_hdf5)}. Sample remote paths:\n{samples}\n"
            "Set --subdir to the remote parent of train_scan/val_scan/test_scan. "
            "If HDF5 count is zero, the repository stores archives rather than "
            "individual HDF5 files and per-file Hub download is unavailable."
        )

    require_labels = not args.allow_unlabeled
    audit_results = [
        validate_h5(
            dataset_root / relative_path,
            relative_path,
            require_labels=require_labels,
            chunk_rows=args.chunk_rows,
            full_read=not args.metadata_only,
        )
        for relative_path in tqdm(
            expected_files,
            desc="Audit local HDF5 files",
            unit="file",
        )
    ]
    invalid = [result for result in audit_results if not result.valid]
    for result in invalid:
        print(f"INVALID {result.relative_path}: {result.reason}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        args.backup_dir.expanduser().resolve()
        if args.backup_dir is not None
        else dataset_root / ".repair_backups" / timestamp
    )
    repaired: list[AuditResult] = []
    unresolved = invalid
    if args.repair and invalid:
        repaired = []
        for result in tqdm(invalid, desc="Download invalid HDF5 files", unit="file"):
            try:
                repaired.append(
                    repair_one_file(
                        relative_path=result.relative_path,
                        dataset_root=dataset_root,
                        backup_root=backup_root,
                        repo_id=args.repo_id,
                        endpoint=endpoint,
                        require_labels=require_labels,
                        chunk_rows=args.chunk_rows,
                    )
                )
            except Exception as error:
                repaired.append(
                    AuditResult(result.relative_path, False, f"download failed: {error}")
                )
        unresolved = [result for result in repaired if not result.valid]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "endpoint": endpoint,
        "dataset_root": str(dataset_root),
        "full_chunk_read": not args.metadata_only,
        "expected_file_count": len(expected_files),
        "valid_file_count_before_repair": len(audit_results) - len(invalid),
        "invalid_file_count_before_repair": len(invalid),
        "repair_enabled": args.repair,
        "backup_root": str(backup_root) if args.repair and invalid else None,
        "invalid_before_repair": [asdict(result) for result in invalid],
        "repair_results": [asdict(result) for result in repaired],
        "unresolved": [asdict(result) for result in unresolved],
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Audit complete: expected={len(expected_files)}, invalid={len(invalid)}, "
        f"unresolved={len(unresolved)}, report={report_path}"
    )
    if args.repair and invalid:
        print(f"Backups: {backup_root}")
    if unresolved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
