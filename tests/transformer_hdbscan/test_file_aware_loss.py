from __future__ import annotations

import torch

from turing_deinterleaving_challenge.transformer_hdbscan.model import (
    RMSNorm,
    RoPESwiGLUEncoderLayer,
    TransformerMetricEncoder,
    effective_anchor_count,
    emitter_compactness_loss,
    file_aware_triplet_metric_loss,
)


def test_encoder_uses_rope_swiglu_pre_norm_architecture() -> None:
    model = TransformerMetricEncoder(
        input_dim=5,
        model_dim=64,
        num_layers=4,
        num_heads=4,
        embedding_dim=8,
        feedforward_dim=128,
        dropout=0.1,
    )

    assert model.architecture == "rope_swiglu_v1"
    assert len(model.blocks) == 4
    assert all(isinstance(block, RoPESwiGLUEncoderLayer) for block in model.blocks)
    assert model.blocks[0].attention.head_dim == 16
    assert model.blocks[0].feedforward.output_projection.in_features == 128
    assert not hasattr(model, "position_encoding")

    model.eval()
    features = torch.randn(2, 7, 5)
    padding_mask = torch.tensor(
        [[False] * 7, [False] * 5 + [True] * 2], dtype=torch.bool
    )
    output = model(features, padding_mask=padding_mask)
    assert output.shape == (2, 7, 8)

    changed_padding = features.clone()
    changed_padding[1, 5:] = 10_000
    changed_output = model(changed_padding, padding_mask=padding_mask)
    torch.testing.assert_close(output[1, :5], changed_output[1, :5])


def test_encoder_can_preserve_embedding_magnitude() -> None:
    torch.manual_seed(5)
    raw_model = TransformerMetricEncoder(
        input_dim=5,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        embedding_dim=4,
        feedforward_dim=16,
        dropout=0.0,
        normalize_embeddings=False,
    ).eval()
    normalized_model = TransformerMetricEncoder(
        input_dim=5,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        embedding_dim=4,
        feedforward_dim=16,
        dropout=0.0,
        normalize_embeddings=True,
    ).eval()
    normalized_model.load_state_dict(raw_model.state_dict())
    features = torch.randn(2, 6, 5)

    with torch.inference_mode():
        raw = raw_model(features)
        normalized = normalized_model(features)

    assert not torch.allclose(torch.linalg.vector_norm(raw, dim=-1), torch.ones(2, 6))
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized, dim=-1),
        torch.ones(2, 6),
    )


def test_rmsnorm_silu_encoder_uses_requested_components() -> None:
    model = TransformerMetricEncoder(
        input_dim=5,
        model_dim=8,
        num_layers=2,
        num_heads=2,
        embedding_dim=4,
        feedforward_dim=16,
        dropout=0.0,
        normalize_embeddings=False,
        architecture="rope_swiglu_rmsnorm_silu_v2",
    ).eval()

    assert isinstance(model.input_projection[1], RMSNorm)
    assert isinstance(model.input_projection[2], torch.nn.SiLU)
    assert all(
        isinstance(block.attention_norm, RMSNorm)
        and isinstance(block.feedforward_norm, RMSNorm)
        for block in model.blocks
    )
    output = model(torch.randn(2, 6, 5))
    assert output.shape == (2, 6, 4)
    assert not torch.allclose(
        torch.linalg.vector_norm(output, dim=-1), torch.ones(2, 6)
    )


def test_adaptive_anchor_count_bounds() -> None:
    assert effective_anchor_count(50) == 50
    assert effective_anchor_count(500) == 128
    assert effective_anchor_count(2_000) == 128
    assert effective_anchor_count(3_000) == 150
    assert effective_anchor_count(5_000) == 250
    assert effective_anchor_count(10_000) == 500
    assert effective_anchor_count(20_000) == 512


def test_adaptive_anchors_deduplicate_overlapping_physical_pulses() -> None:
    torch.manual_seed(11)
    embeddings = torch.randn(2, 6, 4, requires_grad=True)
    labels = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ]
    )
    source_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],
        ]
    )
    file_indices = torch.tensor([0, 0])

    loss, stats = file_aware_triplet_metric_loss(
        embeddings,
        labels,
        file_indices,
        source_indices=source_indices,
        adaptive_anchors=True,
        anchor_min_per_emitter=2,
        anchor_max_per_emitter=4,
        anchor_fraction_per_emitter=0.5,
        return_sampling_stats=True,
    )

    assert torch.isfinite(loss)
    assert stats.candidate_occurrence_count == 12
    assert stats.candidate_unique_pulse_count == 9
    assert stats.anchor_count == 5
    assert stats.unique_anchor_count == 5
    assert stats.emitter_count == 2
    loss.backward()


def test_file_aware_losses_are_finite_and_differentiable() -> None:
    torch.manual_seed(3)
    embeddings = torch.randn(2, 6, 4, requires_grad=True)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    labels = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
        ]
    )
    file_indices = torch.tensor([0, 0])
    padding_mask = torch.zeros_like(labels, dtype=torch.bool)

    triplet = file_aware_triplet_metric_loss(
        embeddings,
        labels,
        file_indices,
        padding_mask=padding_mask,
        max_anchors_per_emitter=4,
    )
    compactness = emitter_compactness_loss(
        embeddings,
        labels,
        file_indices,
        padding_mask=padding_mask,
    )
    loss = triplet + 0.05 * compactness

    assert torch.isfinite(loss)
    assert triplet.item() >= 0
    assert compactness.item() >= 0
    loss.backward()


def test_single_emitter_still_has_compactness_signal() -> None:
    embeddings = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]],
        requires_grad=True,
    )
    labels = torch.zeros((2, 2), dtype=torch.long)
    file_indices = torch.zeros(2, dtype=torch.long)

    compactness = emitter_compactness_loss(embeddings, labels, file_indices)

    assert compactness.item() > 0
    compactness.backward()
