"""Window-level clustering metrics; every label, including -1, is a group."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)


METRIC_NAMES = (
    "Homogeneity",
    "Completeness",
    "V-measure",
    "Adjusted Rand Index",
    "Adjusted Mutual Information",
    "MCC",
    "F1",
    "discount",
)


def cluster_wise_score(
    labels_true: np.ndarray, labels_pred: np.ndarray, score: str
) -> float:
    """Return the original cluster-wise score using a vectorized contingency."""
    score = score.lower()
    if score not in {"mcc", "f1"}:
        raise ValueError(f"Unsupported cluster-wise score: {score}")
    truth = np.asarray(labels_true).reshape(-1)
    predicted = np.asarray(labels_pred).reshape(-1)
    if truth.shape != predicted.shape or not len(truth):
        raise ValueError("labels must be equal-length non-empty arrays")
    _, true_inverse = np.unique(truth, return_inverse=True)
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    table = np.zeros(
        (int(true_inverse.max()) + 1, int(predicted_inverse.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(table, (true_inverse, predicted_inverse), 1)
    true_sizes = table.sum(axis=1, keepdims=True).astype(np.float64)
    predicted_sizes = table.sum(axis=0, keepdims=True).astype(np.float64)
    true_positive = table.astype(np.float64)

    if score == "f1":
        denominator = true_sizes + predicted_sizes
        pair_scores = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
    else:
        false_negative = true_sizes - true_positive
        false_positive = predicted_sizes - true_positive
        true_negative = len(truth) - true_positive - false_negative - false_positive
        numerator = true_positive * true_negative - false_positive * false_negative
        denominator = np.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        pair_scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    return float(pair_scores.max(axis=1).min())


def evaluate_labels(
    labels_pred: np.ndarray, labels_true: np.ndarray
) -> dict[str, float]:
    """Evaluate one complete source file without special-casing label -1."""
    predicted = np.asarray(labels_pred).reshape(-1)
    truth = np.asarray(labels_true).reshape(-1)
    if predicted.shape != truth.shape or not len(truth):
        raise ValueError("predicted and true labels must be equal-length non-empty arrays")
    return {
        "Homogeneity": float(homogeneity_score(truth, predicted)),
        "Completeness": float(completeness_score(truth, predicted)),
        "V-measure": float(v_measure_score(truth, predicted)),
        "Adjusted Rand Index": float(adjusted_rand_score(truth, predicted)),
        "Adjusted Mutual Information": float(
            adjusted_mutual_info_score(truth, predicted)
        ),
        "MCC": cluster_wise_score(truth, predicted, "mcc"),
        "F1": cluster_wise_score(truth, predicted, "f1"),
        "discount": 1.0,
    }
