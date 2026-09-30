#!/usr/bin/env python3
"""Evaluate the repository's raw-PDW HDBSCAN baseline on TSRD scan data.

The default settings reproduce the parameter sweep in
``examples/identity_model.ipynb``.  Results are persisted as JSON so that a
run can be compared or plotted later without re-evaluating the dataset.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from turing_deinterleaving_challenge import DeinterleavingChallengeDataset
from turing_deinterleaving_challenge.models.evaluate import evaluate_model_on_dataset
from turing_deinterleaving_challenge.models.model import IdentityModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing scan data. It may be the scan root "
            "(with test/train/validation subdirectories) or a directory "
            "containing the .h5 files directly."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=1024,
        help="Number of pulses per evaluation window (default: 1024).",
    )
    parser.add_argument(
        "--min-emitters",
        type=int,
        default=2,
        help="Exclude windows with fewer emitters (default: 2).",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.5),
        help=(
            "HDBSCAN cluster_selection_epsilon values. The defaults reproduce "
            "the repository notebook. Use --epsilons 0 for one baseline run."
        ),
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=1000,
        help="Maximum number of windows per epsilon; use 0 for all windows (default: 1000).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes used during evaluation (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hdbscan_scan_results.json"),
        help="JSON result path (default: hdbscan_scan_results.json).",
    )
    return parser.parse_args()


def resolve_data_dir(scan_dir: Path, split: str) -> Path:
    """Find the directory that directly contains the requested split's H5 files."""
    candidates = (
        scan_dir / split,
        scan_dir / f"{split}_scan",
        scan_dir / "scan" / split,
        scan_dir / "scan" / f"{split}_scan",
        scan_dir / "scan",
        scan_dir,
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.h5")):
            return candidate
    locations = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No .h5 files for the '{split}' split were found. Checked:\n  {locations}"
    )


def summarise(scores: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Produce JSON-serialisable summary statistics for evaluator output."""
    return {
        name: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in scores.items()
        if values
    }


def to_builtin_floats(scores: dict[str, list[float]]) -> dict[str, list[float]]:
    """Convert NumPy scalar metric values so the full result can be JSON encoded."""
    return {name: [float(value) for value in values] for name, values in scores.items()}


def main() -> None:
    args = parse_args()
    if args.window_length <= 0:
        raise ValueError("--window-length must be positive")
    if args.min_emitters <= 0:
        raise ValueError("--min-emitters must be positive")
    if args.max_eval < 0:
        raise ValueError("--max-eval must be zero or positive")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")

    data_dir = resolve_data_dir(args.scan_dir.expanduser().resolve(), args.split)
    print(f"Loading scan data from: {data_dir}")
    dataset = DeinterleavingChallengeDataset(
        local_path=data_dir,
        window_length=args.window_length,
        min_emitters=args.min_emitters,
    )
    print(f"Usable windows: {len(dataset)}")
    if not len(dataset):
        raise RuntimeError("No windows matched the chosen filters.")

    max_eval = None if args.max_eval == 0 else args.max_eval
    runs: list[dict[str, Any]] = []
    for epsilon in args.epsilons:
        print(f"\nEvaluating HDBSCAN with cluster_selection_epsilon={epsilon:g}")
        model = IdentityModel(
            clusterer="hdbscan",
            cl_params={"cluster_selection_epsilon": epsilon, "copy": True},
            default_label=-1,
        )
        scores = to_builtin_floats(
            evaluate_model_on_dataset(
                model=model,
                dataloader=dataset,
                n_jobs=args.n_jobs,
                max_eval=max_eval,
                return_average=False,
            )
        )
        summary = summarise(scores)
        runs.append({"epsilon": epsilon, "summary": summary, "scores": scores})
        print(
            "V-measure: "
            f"mean={summary['V-measure']['mean']:.4f}, "
            f"median={summary['V-measure']['median']:.4f}"
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir),
        "split": args.split,
        "window_length": args.window_length,
        "min_emitters": args.min_emitters,
        "max_eval_per_epsilon": max_eval,
        "n_jobs": args.n_jobs,
        "runs": runs,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved full scores and summaries to: {output}")


if __name__ == "__main__":
    main()
