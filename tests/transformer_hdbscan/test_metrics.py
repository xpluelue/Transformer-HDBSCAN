from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from turing_deinterleaving_challenge.transformer_hdbscan.metrics import cluster_wise_score


def slow_reference(truth: np.ndarray, predicted: np.ndarray, score: str) -> float:
    values: list[float] = []
    for true_label in np.unique(truth):
        target = (truth == true_label).astype(np.int8)
        candidates: list[float] = []
        for predicted_label in np.unique(predicted):
            candidate = (predicted == predicted_label).astype(np.int8)
            if score == "mcc":
                value = (
                    0.0
                    if np.unique(target).size < 2 or np.unique(candidate).size < 2
                    else float(matthews_corrcoef(candidate, target))
                )
            else:
                value = float(f1_score(candidate, target, zero_division=0))
            candidates.append(value)
        values.append(max(candidates))
    return min(values)


def test_vectorized_cluster_scores_match_original_definition() -> None:
    rng = np.random.default_rng(9)
    truth = rng.integers(0, 7, size=500)
    predicted = rng.integers(-1, 15, size=500)

    for score in ("mcc", "f1"):
        np.testing.assert_allclose(
            cluster_wise_score(truth, predicted, score),
            slow_reference(truth, predicted, score),
            rtol=1e-12,
            atol=1e-12,
        )
