"""One self-contained prediction HDF5 for each complete source pulse file."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


FORMAT_VERSION = 3


@dataclass(frozen=True)
class SourceFileMetadata:
    source_file: str
    file_index: int
    pulse_count: int
    window_count: int
    window_length: int
    window_stride: int
    clustering_scope: str = "complete_source_file"


def cluster_group_name(label: int) -> str:
    """Map any integer label, including -1, to an ordinary cluster group."""
    return f"cluster_{label}"


def _dataset(group: h5py.Group, name: str, value: np.ndarray) -> h5py.Dataset:
    array = np.asarray(value)
    kwargs = {"compression": "gzip", "compression_opts": 4} if array.size else {}
    return group.create_dataset(name, data=array, **kwargs)


def _write_partition(
    parent: h5py.Group,
    labels: np.ndarray,
    raw_pdws: np.ndarray,
    source_indices: np.ndarray,
    counterpart_labels: np.ndarray | None,
) -> None:
    for label in np.unique(labels):
        integer_label = int(label)
        members = np.flatnonzero(labels == label).astype(np.int64)
        cluster = parent.create_group(cluster_group_name(integer_label))
        cluster.attrs["label"] = integer_label
        cluster.attrs["size"] = len(members)
        _dataset(cluster, "member_indices", members)
        _dataset(cluster, "source_indices", source_indices[members])
        _dataset(cluster, "pdws", raw_pdws[members])
        if counterpart_labels is not None:
            _dataset(cluster, "counterpart_labels", counterpart_labels[members])


def contingency(
    predicted_labels: np.ndarray, true_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted IDs, true IDs, and their complete-file pulse counts."""
    predicted_ids, predicted_inverse = np.unique(predicted_labels, return_inverse=True)
    true_ids, true_inverse = np.unique(true_labels, return_inverse=True)
    table = np.zeros((len(predicted_ids), len(true_ids)), dtype=np.int64)
    np.add.at(table, (predicted_inverse, true_inverse), 1)
    return predicted_ids, true_ids, table


def write_source_file(
    path: Path,
    metadata: SourceFileMetadata,
    raw_pdws: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray | None,
    window_starts: np.ndarray,
    window_valid_lengths: np.ndarray,
    metrics: dict[str, float] | None = None,
) -> None:
    """Atomically write one complete source-file prediction result."""
    raw = np.asarray(raw_pdws)
    predicted = np.asarray(predicted_labels, dtype=np.int32).reshape(-1)
    truth = (
        None
        if true_labels is None
        else np.asarray(true_labels, dtype=np.int64).reshape(-1)
    )
    starts = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    lengths = np.asarray(window_valid_lengths, dtype=np.int64).reshape(-1)
    if raw.ndim != 2 or len(raw) != metadata.pulse_count:
        raise ValueError("raw_pdws must be 2-D and match metadata.pulse_count")
    if len(predicted) != len(raw):
        raise ValueError("predicted labels must match the complete pulse count")
    if truth is not None and len(truth) != len(raw):
        raise ValueError("true labels must match the complete pulse count")
    if len(starts) != metadata.window_count or len(lengths) != metadata.window_count:
        raise ValueError("window metadata must match metadata.window_count")
    ends = starts + lengths
    if (
        np.any(starts < 0)
        or np.any(lengths <= 0)
        or np.any(ends > metadata.pulse_count)
    ):
        raise ValueError("window metadata lies outside the complete pulse file")
    coverage_delta = np.zeros(metadata.pulse_count + 1, dtype=np.int64)
    np.add.at(coverage_delta, starts, 1)
    np.add.at(coverage_delta, ends, -1)
    if np.any(np.cumsum(coverage_delta[:-1]) <= 0):
        raise ValueError("window metadata does not cover every source pulse")

    source_indices = np.arange(metadata.pulse_count, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
            for name, value in asdict(metadata).items():
                handle.attrs[name] = value
            handle.attrs["has_true_labels"] = truth is not None
            handle.attrs["evaluation_computed"] = metrics is not None

            windows = handle.create_group("embedding_windows")
            _dataset(windows, "starts", starts)
            _dataset(windows, "valid_lengths", lengths)

            pulses = handle.create_group("pulses")
            _dataset(pulses, "raw_pdws", raw)
            _dataset(pulses, "source_indices", source_indices)
            _dataset(pulses, "predicted_labels", predicted)
            if truth is not None:
                _dataset(pulses, "true_labels", truth)

            predicted_parent = handle.create_group("predicted_clusters")
            _write_partition(predicted_parent, predicted, raw, source_indices, truth)
            if truth is not None:
                true_parent = handle.create_group("true_clusters")
                _write_partition(true_parent, truth, raw, source_indices, predicted)
            if truth is not None and metrics is not None:
                predicted_ids, true_ids, table = contingency(predicted, truth)
                evaluation = handle.create_group("evaluation")
                evaluation.attrs["metrics_scope"] = "complete_source_file"
                evaluation.attrs["contingency_scope"] = "complete_source_file"
                _dataset(evaluation, "predicted_cluster_ids", predicted_ids)
                _dataset(evaluation, "true_cluster_ids", true_ids)
                _dataset(evaluation, "contingency_matrix", table)
                if metrics:
                    for name, value in metrics.items():
                        evaluation.attrs[name] = float(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PredictionRunWriter:
    """Stream one result file and one manifest row per complete source file."""

    FIELDNAMES = (
        "result_file",
        "source_file",
        "file_index",
        "pulse_count",
        "window_count",
        "window_length",
        "window_stride",
        "predicted_cluster_count",
        "true_cluster_count",
        "Homogeneity",
        "Completeness",
        "V-measure",
        "Adjusted Rand Index",
        "Adjusted Mutual Information",
        "MCC",
        "F1",
        "discount",
    )

    def __init__(self, output_dir: Path, run_metadata: dict[str, object]) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.files_dir = self.output_dir / "files"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {self.output_dir}. Use a new directory."
            )
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_temporary = self.output_dir / ".manifest.csv.tmp"
        self.manifest_path = self.output_dir / "manifest.csv"
        self.summary_path = self.output_dir / "summary.json"
        self._manifest_handle = self.manifest_temporary.open(
            "w", newline="", encoding="utf-8"
        )
        self._manifest = csv.DictWriter(
            self._manifest_handle, fieldnames=self.FIELDNAMES
        )
        self._manifest.writeheader()
        self.run_metadata = run_metadata
        self.source_file_count = 0
        self.total_window_count = 0
        self.evaluated_source_file_count = 0
        self.metric_values: dict[str, list[float]] = {}

    def path_for(self, metadata: SourceFileMetadata) -> Path:
        source_stem = Path(metadata.source_file).stem
        return self.files_dir / f"file_{metadata.file_index:04d}__{source_stem}.h5"

    def write(
        self,
        metadata: SourceFileMetadata,
        raw_pdws: np.ndarray,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray | None,
        window_starts: np.ndarray,
        window_valid_lengths: np.ndarray,
        metrics: dict[str, float] | None,
    ) -> Path:
        path = self.path_for(metadata)
        write_source_file(
            path,
            metadata,
            raw_pdws,
            predicted_labels,
            true_labels,
            window_starts,
            window_valid_lengths,
            metrics,
        )
        row: dict[str, object] = {
            "result_file": str(path.relative_to(self.output_dir)),
            **asdict(metadata),
            "predicted_cluster_count": len(np.unique(predicted_labels)),
            "true_cluster_count": (
                "" if true_labels is None else len(np.unique(true_labels))
            ),
        }
        for name in self.FIELDNAMES[9:]:
            row[name] = "" if metrics is None else metrics[name]
        self._manifest.writerow({key: row.get(key, "") for key in self.FIELDNAMES})
        self._manifest_handle.flush()
        self.source_file_count += 1
        self.total_window_count += metadata.window_count
        if metrics is not None:
            self.evaluated_source_file_count += 1
            for name, value in metrics.items():
                self.metric_values.setdefault(name, []).append(float(value))
        return path

    def close(self) -> dict[str, object]:
        if self._manifest_handle.closed:
            raise RuntimeError("PredictionRunWriter is already closed")
        self._manifest_handle.close()
        os.replace(self.manifest_temporary, self.manifest_path)
        summary: dict[str, object] = {
            **self.run_metadata,
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file_count": self.source_file_count,
            "total_embedding_window_count": self.total_window_count,
            "evaluated_source_file_count": self.evaluated_source_file_count,
            "metrics_scope": "complete_source_file_macro_average",
            "metrics": {
                name: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for name, values in self.metric_values.items()
                if values
            },
        }
        temporary = self.summary_path.with_name(".summary.json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.summary_path)
        return summary

    def abort(self) -> None:
        if not self._manifest_handle.closed:
            self._manifest_handle.close()
        self.manifest_temporary.unlink(missing_ok=True)


class MetricsOnlyRunWriter:
    """Write per-file metrics and a run summary without duplicating pulse HDF5."""

    FIELDNAMES = (
        "source_file",
        "file_index",
        "pulse_count",
        "window_count",
        "predicted_cluster_count",
        "true_cluster_count",
        *(
            "Homogeneity",
            "Completeness",
            "V-measure",
            "Adjusted Rand Index",
            "Adjusted Mutual Information",
            "MCC",
            "F1",
            "discount",
        ),
    )

    def __init__(self, output_dir: Path, run_metadata: dict[str, object]) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {self.output_dir}. Use a new directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_temporary = self.output_dir / ".metrics.csv.tmp"
        self.metrics_path = self.output_dir / "metrics.csv"
        self.summary_path = self.output_dir / "summary.json"
        self._metrics_handle = self.metrics_temporary.open(
            "w", newline="", encoding="utf-8"
        )
        self._metrics = csv.DictWriter(
            self._metrics_handle, fieldnames=self.FIELDNAMES
        )
        self._metrics.writeheader()
        self.run_metadata = run_metadata
        self.source_file_count = 0
        self.total_window_count = 0
        self.evaluated_source_file_count = 0
        self.metric_values: dict[str, list[float]] = {}

    def write(
        self,
        metadata: SourceFileMetadata,
        raw_pdws: np.ndarray,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray | None,
        window_starts: np.ndarray,
        window_valid_lengths: np.ndarray,
        metrics: dict[str, float] | None,
    ) -> Path:
        del raw_pdws, window_starts, window_valid_lengths
        predicted = np.asarray(predicted_labels).reshape(-1)
        truth = (
            None
            if true_labels is None
            else np.asarray(true_labels).reshape(-1)
        )
        if len(predicted) != metadata.pulse_count:
            raise ValueError("predicted labels must match metadata.pulse_count")
        if truth is not None and len(truth) != metadata.pulse_count:
            raise ValueError("true labels must match metadata.pulse_count")
        row: dict[str, object] = {
            "source_file": metadata.source_file,
            "file_index": metadata.file_index,
            "pulse_count": metadata.pulse_count,
            "window_count": metadata.window_count,
            "predicted_cluster_count": len(np.unique(predicted)),
            "true_cluster_count": "" if truth is None else len(np.unique(truth)),
        }
        for name in self.FIELDNAMES[6:]:
            row[name] = "" if metrics is None else metrics[name]
        self._metrics.writerow(row)
        self._metrics_handle.flush()
        self.source_file_count += 1
        self.total_window_count += metadata.window_count
        if metrics is not None:
            self.evaluated_source_file_count += 1
            for name, value in metrics.items():
                self.metric_values.setdefault(name, []).append(float(value))
        return self.output_dir

    def close(self) -> dict[str, object]:
        if self._metrics_handle.closed:
            raise RuntimeError("MetricsOnlyRunWriter is already closed")
        self._metrics_handle.close()
        os.replace(self.metrics_temporary, self.metrics_path)
        summary: dict[str, object] = {
            **self.run_metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file_count": self.source_file_count,
            "total_embedding_window_count": self.total_window_count,
            "evaluated_source_file_count": self.evaluated_source_file_count,
            "metrics_scope": "complete_source_file_macro_average",
            "saved_predictions": False,
            "metrics": {
                name: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for name, values in self.metric_values.items()
                if values
            },
        }
        temporary = self.summary_path.with_name(".summary.json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.summary_path)
        return summary

    def abort(self) -> None:
        if not self._metrics_handle.closed:
            self._metrics_handle.close()
        self.metrics_temporary.unlink(missing_ok=True)


class PDWStudioRunWriter:
    """Write ``<source>/<source>_<cluster>.h5`` files for PDW Studio."""

    FIELDNAMES = (
        "source_file",
        "source_stem",
        "pulse_count",
        "window_count",
        "predicted_cluster_count",
        "true_cluster_count",
        "status",
    )
    CLUSTER_FIELDNAMES = (
        "cluster_file",
        "source_file",
        "source_stem",
        "output_cluster_id",
        "original_cluster_label",
        "pulse_count",
    )

    def __init__(self, output_dir: Path, run_metadata: dict[str, object]) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            message = (
                f"Output directory is not empty: {self.output_dir}. "
                "Use a new directory."
            )
            raise FileExistsError(message)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_temporary = self.output_dir / "._summary.csv.tmp"
        self.manifest_path = self.output_dir / "_summary.csv"
        self.cluster_manifest_temporary = self.output_dir / "._clusters.csv.tmp"
        self.cluster_manifest_path = self.output_dir / "_clusters.csv"
        self.summary_path = self.output_dir / "summary.json"
        self.log_path = self.output_dir / "run.log"
        self._manifest_handle = self.manifest_temporary.open(
            "w", newline="", encoding="utf-8"
        )
        self._manifest = csv.DictWriter(
            self._manifest_handle, fieldnames=self.FIELDNAMES
        )
        self._manifest.writeheader()
        self._manifest_handle.flush()
        self._cluster_manifest_handle = self.cluster_manifest_temporary.open(
            "w", newline="", encoding="utf-8"
        )
        self._cluster_manifest = csv.DictWriter(
            self._cluster_manifest_handle, fieldnames=self.CLUSTER_FIELDNAMES
        )
        self._cluster_manifest.writeheader()
        self._cluster_manifest_handle.flush()
        self.run_metadata = run_metadata
        self.source_file_count = 0
        self.total_window_count = 0
        self.total_cluster_count = 0
        self.evaluated_source_file_count = 0
        self.metric_values: dict[str, list[float]] = {}
        self.log_path.write_text(
            f"{datetime.now(timezone.utc).isoformat()} | START | "
            f"checkpoint={run_metadata.get('checkpoint', '<unknown>')}\n",
            encoding="utf-8",
        )

    def _log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")

    @staticmethod
    def _write_cluster_file(
        path: Path,
        raw_pdws: np.ndarray,
        true_labels: np.ndarray | None,
    ) -> None:
        pulse_count = len(raw_pdws)
        if true_labels is None:
            studio_labels = np.full(pulse_count, -1, dtype=np.int32)
        else:
            truth = np.asarray(true_labels, dtype=np.int64).reshape(-1)
            limits = np.iinfo(np.int32)
            if np.any(truth < limits.min) or np.any(truth > limits.max):
                raise OverflowError("true labels do not fit PDW Studio int32 labels")
            studio_labels = truth.astype(np.int32, copy=False)
        if len(studio_labels) != pulse_count:
            raise ValueError("PDW Studio labels must match the pulse count")
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "data",
                data=np.asarray(raw_pdws, dtype=np.float64).T,
                dtype=np.float64,
            )
            handle.create_dataset(
                "labels",
                data=studio_labels,
                dtype=np.int32,
            )

    def write(
        self,
        metadata: SourceFileMetadata,
        raw_pdws: np.ndarray,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray | None,
        window_starts: np.ndarray,
        window_valid_lengths: np.ndarray,
        metrics: dict[str, float] | None,
    ) -> Path:
        del window_starts, window_valid_lengths
        raw = np.asarray(raw_pdws)
        predicted = np.asarray(predicted_labels, dtype=np.int32).reshape(-1)
        truth = (
            None
            if true_labels is None
            else np.asarray(true_labels, dtype=np.int64).reshape(-1)
        )
        if raw.ndim != 2 or len(raw) != metadata.pulse_count:
            raise ValueError("raw_pdws must be 2-D and match metadata.pulse_count")
        if len(predicted) != len(raw):
            raise ValueError("predicted labels must match the complete pulse count")
        if truth is not None and len(truth) != len(raw):
            raise ValueError("true labels must match the complete pulse count")

        source_stem = Path(metadata.source_file).stem
        target = self.output_dir / source_stem
        temporary = self.output_dir / f".{source_stem}.tmp"
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Cluster output already exists for {source_stem}")
        temporary.mkdir(parents=False)

        cluster_ids, cluster_counts = np.unique(predicted, return_counts=True)
        ranked_clusters = sorted(
            zip(cluster_ids.tolist(), cluster_counts.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        cluster_rows: list[dict[str, object]] = []
        try:
            for output_cluster_id, (cluster_id, _) in enumerate(ranked_clusters):
                members = np.flatnonzero(predicted == cluster_id)
                if raw.shape[1] > 0:
                    members = members[np.argsort(raw[members, 0], kind="stable")]
                self._write_cluster_file(
                    temporary / f"{source_stem}_{output_cluster_id}.h5",
                    raw[members],
                    None if truth is None else truth[members],
                )
                cluster_rows.append(
                    {
                        "cluster_file": (
                            f"{source_stem}/{source_stem}_{output_cluster_id}.h5"
                        ),
                        "source_file": metadata.source_file,
                        "source_stem": source_stem,
                        "output_cluster_id": output_cluster_id,
                        "original_cluster_label": int(cluster_id),
                        "pulse_count": int(len(members)),
                    }
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        self._cluster_manifest.writerows(cluster_rows)

        row = {
            "source_file": metadata.source_file,
            "source_stem": source_stem,
            "pulse_count": metadata.pulse_count,
            "window_count": metadata.window_count,
            "predicted_cluster_count": len(ranked_clusters),
            "true_cluster_count": ("" if truth is None else len(np.unique(truth))),
            "status": "ok",
        }
        self._manifest.writerow(row)
        self._manifest_handle.flush()
        self._cluster_manifest_handle.flush()
        self._log(
            f"{source_stem} | pulses={metadata.pulse_count} | "
            f"clusters={len(ranked_clusters)} | ok"
        )

        self.source_file_count += 1
        self.total_window_count += metadata.window_count
        self.total_cluster_count += len(ranked_clusters)
        if metrics is not None:
            self.evaluated_source_file_count += 1
            for name, value in metrics.items():
                self.metric_values.setdefault(name, []).append(float(value))
        return target

    def close(self) -> dict[str, object]:
        if self._manifest_handle.closed:
            raise RuntimeError("PDWStudioRunWriter is already closed")
        self._manifest_handle.close()
        self._cluster_manifest_handle.close()
        os.replace(self.manifest_temporary, self.manifest_path)
        os.replace(self.cluster_manifest_temporary, self.cluster_manifest_path)
        summary: dict[str, object] = {
            **self.run_metadata,
            "output_format": "pdw_studio_cluster_files",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file_count": self.source_file_count,
            "total_embedding_window_count": self.total_window_count,
            "total_predicted_cluster_count": self.total_cluster_count,
            "evaluated_source_file_count": self.evaluated_source_file_count,
            "minus_one_is_ordinary_cluster": True,
            "cluster_order": "descending_size_then_original_label",
            "metrics": {
                name: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for name, values in self.metric_values.items()
                if values
            },
        }
        temporary = self.summary_path.with_name(".summary.json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.summary_path)
        self._log(
            f"DONE | source_files={self.source_file_count} | "
            f"clusters={self.total_cluster_count}"
        )
        return summary

    def abort(self) -> None:
        if not self._manifest_handle.closed:
            self._manifest_handle.close()
        if not self._cluster_manifest_handle.closed:
            self._cluster_manifest_handle.close()
        self.manifest_temporary.unlink(missing_ok=True)
        self.cluster_manifest_temporary.unlink(missing_ok=True)
        self._log("ABORTED")
