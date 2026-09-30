from __future__ import annotations

import numpy as np
import pytest
import torch

from turing_deinterleaving_challenge.models.eend_eda import (
    EncoderDecoderAttractor, RadarEENDEDA, leading_existence_counts,
    radar_eda_loss)


def test_hungarian_assignment_aligns_arbitrary_attractor_order() -> None:
    probabilities = torch.tensor(
        [
            [0.05, 0.90, 0.05],
            [0.10, 0.80, 0.10],
            [0.85, 0.10, 0.05],
            [0.75, 0.15, 0.10],
            [0.10, 0.05, 0.85],
            [0.05, 0.10, 0.85],
        ],
        dtype=torch.float32,
    )
    assignment_logits = probabilities.log().unsqueeze(0).requires_grad_()
    existence_logits = torch.tensor(
        [[10.0, 10.0, 10.0, -10.0]], requires_grad=True
    )
    labels = torch.tensor([[10, 10, 20, 20, 30, 30]])

    loss = radar_eda_loss(
        assignment_logits,
        existence_logits,
        labels,
        alpha=0.0,
    )

    expected = -float(
        np.mean(np.log([0.90, 0.80, 0.85, 0.75, 0.85, 0.85]))
    )
    assert loss.assignment.item() == pytest.approx(expected, abs=1e-6)
    loss.total.backward()
    assert assignment_logits.grad is not None


def test_hungarian_loss_is_invariant_to_true_label_values() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    existence = torch.tensor([[5.0, 5.0, 5.0, -5.0]])
    first_labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
    renamed_labels = torch.tensor([[91, 91, 7, 7, 42, 42]])

    first = radar_eda_loss(logits, existence, first_labels)
    renamed = radar_eda_loss(logits, existence, renamed_labels)

    assert first.total.item() == pytest.approx(renamed.total.item())
    assert first.assignment.item() == pytest.approx(renamed.assignment.item())
    assert first.existence.item() == pytest.approx(renamed.existence.item())


def test_eda_encoder_ignores_padded_embedding_values() -> None:
    torch.manual_seed(3)
    module = EncoderDecoderAttractor(embedding_dim=4)
    embeddings = torch.randn(1, 5, 4)
    changed_padding = embeddings.clone()
    changed_padding[:, 3:] = 1000.0
    padding_mask = torch.tensor([[False, False, False, True, True]])

    first_attractors, first_existence = module(
        embeddings, padding_mask=padding_mask, num_steps=3
    )
    second_attractors, second_existence = module(
        changed_padding, padding_mask=padding_mask, num_steps=3
    )

    torch.testing.assert_close(first_attractors, second_attractors)
    torch.testing.assert_close(first_existence, second_existence)


def test_leading_existence_count_stops_at_first_failure() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.8, 0.2, 0.9],
            [0.1, 0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9],
        ]
    )

    counts = leading_existence_counts(
        probabilities, threshold=0.5, max_attractors=4
    )

    assert counts.tolist() == [2, 1, 4]


@pytest.mark.filterwarnings("ignore:enable_nested_tensor.*:UserWarning")
def test_full_model_uses_dynamic_training_steps() -> None:
    torch.manual_seed(4)
    model = RadarEENDEDA(
        input_dim=5,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        embedding_dim=4,
        feedforward_dim=16,
        dropout=0.0,
        max_attractors=5,
    )
    features = torch.randn(2, 6, 5)
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
        ]
    )

    assignment, existence = model(features, padding_mask, num_steps=4)

    assert assignment.shape == (2, 6, 3)
    assert existence.shape == (2, 4)


@pytest.mark.filterwarnings("ignore:enable_nested_tensor.*:UserWarning")
def test_predict_assigns_one_label_and_masks_padding() -> None:
    torch.manual_seed(5)
    model = RadarEENDEDA(
        input_dim=5,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        embedding_dim=4,
        feedforward_dim=16,
        dropout=0.0,
        max_attractors=6,
    )
    with torch.no_grad():
        model.eda.existence_head.weight.zero_()
        model.eda.existence_head.bias.fill_(-10.0)
    features = torch.randn(1, 6, 5)
    padding_mask = torch.tensor(
        [[False, False, False, False, True, True]]
    )

    labels, counts, probabilities = model.predict(features, padding_mask)

    assert counts.tolist() == [1]
    assert labels.tolist() == [[0, 0, 0, 0, -1, -1]]
    assert probabilities.shape == (1, 1)
