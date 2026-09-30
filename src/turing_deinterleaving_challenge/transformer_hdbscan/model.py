"""Transformer metric encoder and reusable HDBSCAN inference components."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from torch import Tensor, nn
from torch.nn import functional as F


def delta_toa_features(
    pdws: np.ndarray, previous_toa: float | None = None
) -> np.ndarray:
    """Replace absolute ToA with a boundary-aware inter-pulse interval."""
    features = np.asarray(pdws, dtype=np.float32).copy()
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("PDWs must have shape (num_pulses, num_features)")
    if len(features) == 0:
        return features
    features[1:, 0] = np.diff(features[:, 0])
    features[0, 0] = (
        0.0 if previous_toa is None else float(pdws[0, 0]) - float(previous_toa)
    )
    return features


def normalize_pdws(pdws: np.ndarray) -> np.ndarray:
    """Convert absolute ToA to delta-ToA and standardize one legacy window."""
    features = delta_toa_features(pdws)
    if len(features) == 0:
        return features
    mean = features.mean(axis=0, dtype=np.float64)
    std = np.maximum(features.std(axis=0, dtype=np.float64), 1e-6)
    return ((features - mean) / std).astype(np.float32)


def normalize_pdws_global(
    pdws: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    previous_toa: float | None = None,
) -> np.ndarray:
    """Apply one training-set transform consistently across all file windows."""
    features = delta_toa_features(pdws, previous_toa=previous_toa)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_array.shape != (features.shape[1],) or std_array.shape != mean_array.shape:
        raise ValueError("Global normalization statistics have incompatible shapes")
    if np.any(std_array <= 0):
        raise ValueError("Global normalization standard deviations must be positive")
    return ((features - mean_array) / std_array).astype(np.float32)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.model_dim = model_dim

    def forward(self, x: Tensor) -> Tensor:
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        frequencies = torch.exp(
            torch.arange(0, self.model_dim, 2, device=x.device, dtype=x.dtype)
            * (-np.log(10000.0) / self.model_dim)
        )
        encoding = torch.zeros(
            x.shape[1], self.model_dim, device=x.device, dtype=x.dtype
        )
        encoding[:, 0::2] = torch.sin(position[:, None] * frequencies)
        encoding[:, 1::2] = torch.cos(position[:, None] * frequencies)
        return x + encoding.unsqueeze(0)


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to queries and keys."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.qkv_projection = nn.Linear(model_dim, 3 * model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.attention_dropout = float(dropout)
        inverse_frequency = 1.0 / (
            10000
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer(
            "inverse_frequency", inverse_frequency, persistent=False
        )

    def _apply_rope(self, values: Tensor) -> Tensor:
        sequence_length = values.shape[-2]
        positions = torch.arange(
            sequence_length, device=values.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.inverse_frequency.float())
        cosine = angles.cos().to(dtype=values.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=values.dtype)[None, None, :, :]
        pairs = values.reshape(*values.shape[:-1], self.head_dim // 2, 2)
        even, odd = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(start_dim=-2)

    def forward(self, hidden: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        batch_size, sequence_length, model_dim = hidden.shape
        qkv = self.qkv_projection(hidden).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        attention_mask = (
            None
            if padding_mask is None
            else (~padding_mask)[:, None, None, :]
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, model_dim
        )
        return self.output_projection(attended)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network with an explicit hidden dimension."""

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_value_projection = nn.Linear(model_dim, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, model_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.gate_value_projection(hidden).chunk(2, dim=-1)
        return self.output_projection(F.silu(gate) * value)


class RMSNorm(nn.Module):
    """PyTorch-version-independent RMSNorm with stable FP32 accumulation."""

    def __init__(self, model_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(model_dim))

    def forward(self, hidden: Tensor) -> Tensor:
        input_dtype = hidden.dtype
        hidden_float = hidden.float()
        normalized = hidden_float * torch.rsqrt(
            hidden_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=input_dtype) * self.weight.to(dtype=input_dtype)


class RoPESwiGLUEncoderLayer(nn.Module):
    """Pre-norm Transformer block using RoPE attention and SwiGLU."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        normalization: str = "layernorm",
    ) -> None:
        super().__init__()
        if normalization == "layernorm":
            norm_factory = lambda: nn.LayerNorm(model_dim)
        elif normalization == "rmsnorm":
            norm_factory = lambda: RMSNorm(model_dim, eps=1e-6)
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")
        self.attention_norm = norm_factory()
        self.attention = RotarySelfAttention(model_dim, num_heads, dropout)
        self.feedforward_norm = norm_factory()
        self.feedforward = SwiGLUFeedForward(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        hidden = hidden + self.dropout(
            self.attention(self.attention_norm(hidden), padding_mask=padding_mask)
        )
        hidden = hidden + self.dropout(
            self.feedforward(self.feedforward_norm(hidden))
        )
        return hidden


class TransformerMetricEncoder(nn.Module):
    """Produce one magnitude-preserving embedding for every input pulse."""

    def __init__(
        self,
        input_dim: int = 5,
        model_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        embedding_dim: int = 8,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        normalize_embeddings: bool = False,
        architecture: str = "rope_swiglu_v1",
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if architecture not in {
            "rope_swiglu_v1",
            "rope_swiglu_rmsnorm_silu_v2",
            "legacy_sinusoidal",
        }:
            raise ValueError(f"Unsupported Transformer architecture: {architecture}")
        self.architecture = architecture
        rmsnorm_silu = architecture == "rope_swiglu_rmsnorm_silu_v2"
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            (
                RMSNorm(model_dim, eps=1e-6)
                if rmsnorm_silu
                else nn.LayerNorm(model_dim)
            ),
            nn.SiLU() if rmsnorm_silu else nn.GELU(),
        )
        if architecture == "legacy_sinusoidal":
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.position_encoding = SinusoidalPositionEncoding(model_dim)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    RoPESwiGLUEncoderLayer(
                        model_dim,
                        num_heads,
                        feedforward_dim,
                        dropout,
                        normalization="rmsnorm" if rmsnorm_silu else "layernorm",
                    )
                    for _ in range(num_layers)
                ]
            )
        self.embedding_projection = nn.Linear(model_dim, embedding_dim)
        self.normalize_embeddings = normalize_embeddings

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, length, feature_dim)")
        if padding_mask is not None and padding_mask.shape != features.shape[:2]:
            raise ValueError("padding_mask must match the batch and sequence dimensions")
        hidden = self.input_projection(features)
        if self.architecture == "legacy_sinusoidal":
            hidden = self.position_encoding(hidden)
            # PyTorch 2.0's fused inference path converts this bool padding mask
            # internally and emits a spurious performance warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Converting mask without torch.bool dtype to bool.*",
                    category=UserWarning,
                )
                hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        else:
            for block in self.blocks:
                hidden = block(hidden, padding_mask=padding_mask)
        embeddings = self.embedding_projection(hidden)
        if self.normalize_embeddings:
            return F.normalize(embeddings, p=2, dim=-1)
        return embeddings


def triplet_metric_loss(
    embeddings: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    margin: float = 0.2,
) -> Tensor:
    """Build triplets only within a window because emitter IDs are local."""
    if embeddings.ndim != 3 or labels.shape != embeddings.shape[:2]:
        raise ValueError("embeddings and labels have incompatible shapes")
    if padding_mask is not None and padding_mask.shape != labels.shape:
        raise ValueError("padding_mask and labels must have identical shapes")
    anchors: list[Tensor] = []
    positives: list[Tensor] = []
    negatives: list[Tensor] = []

    for batch_index in range(embeddings.shape[0]):
        valid = (
            ~padding_mask[batch_index]
            if padding_mask is not None
            else torch.ones_like(labels[batch_index], dtype=torch.bool)
        )
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        window_labels = labels[batch_index, indices]
        for label in torch.unique(window_labels):
            members = indices[window_labels == label]
            others = indices[window_labels != label]
            if len(members) < 2 or len(others) == 0:
                continue
            anchors.append(embeddings[batch_index, members])
            positives.append(embeddings[batch_index, members.roll(-1)])
            sampled = torch.randint(len(others), (len(members),), device=embeddings.device)
            negatives.append(embeddings[batch_index, others[sampled]])

    if not anchors:
        return embeddings.sum() * 0.0
    return F.triplet_margin_loss(
        torch.cat(anchors),
        torch.cat(positives),
        torch.cat(negatives),
        margin=margin,
    )


@dataclass
class AnchorSamplingStats:
    """Counters describing file-aware triplet anchor sampling."""

    anchor_count: int = 0
    unique_anchor_count: int = 0
    emitter_count: int = 0
    candidate_occurrence_count: int = 0
    candidate_unique_pulse_count: int = 0


def effective_anchor_count(
    emitter_pulse_count: int,
    minimum: int = 128,
    maximum: int = 512,
    fraction: float = 0.05,
) -> int:
    """Return the bounded adaptive anchor count for one file-local emitter."""
    if emitter_pulse_count < 0:
        raise ValueError("emitter_pulse_count cannot be negative")
    if minimum <= 0 or maximum <= 0 or minimum > maximum or fraction < 0:
        raise ValueError("Invalid adaptive anchor configuration")
    target = min(maximum, max(minimum, math.ceil(emitter_pulse_count * fraction)))
    return min(emitter_pulse_count, target)


def _random_source_representatives(
    members: Tensor, member_sources: Tensor
) -> Tensor:
    """Choose one random window occurrence for every physical pulse."""
    order = torch.argsort(member_sources)
    sorted_sources = member_sources[order]
    _, counts = torch.unique_consecutive(sorted_sources, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    offsets = torch.floor(
        torch.rand(counts.shape, device=counts.device) * counts
    ).to(dtype=torch.long)
    return members[order[starts + offsets]]


def file_aware_triplet_metric_loss(
    embeddings: Tensor,
    labels: Tensor,
    file_indices: Tensor,
    source_indices: Tensor | None = None,
    padding_mask: Tensor | None = None,
    margin: float = 0.2,
    max_anchors_per_emitter: int = 64,
    adaptive_anchors: bool = False,
    anchor_min_per_emitter: int = 128,
    anchor_max_per_emitter: int = 512,
    anchor_fraction_per_emitter: float = 0.05,
    return_sampling_stats: bool = False,
) -> Tensor | tuple[Tensor, AnchorSamplingStats]:
    """Sample triplets across windows while keeping emitter IDs file-local.

    Positives are preferentially drawn from a different window of the same
    source file. Negatives always come from another emitter in that file.
    This keeps memory linear in the pulse count and avoids a full pairwise
    distance matrix.
    """
    if embeddings.ndim != 3 or labels.shape != embeddings.shape[:2]:
        raise ValueError("embeddings and labels have incompatible shapes")
    if file_indices.shape != (embeddings.shape[0],):
        raise ValueError("file_indices must contain one value per window")
    if source_indices is not None and source_indices.shape != labels.shape:
        raise ValueError("source_indices and labels must have identical shapes")
    if padding_mask is not None and padding_mask.shape != labels.shape:
        raise ValueError("padding_mask and labels must have identical shapes")
    if max_anchors_per_emitter <= 0:
        raise ValueError("max_anchors_per_emitter must be positive")
    if adaptive_anchors:
        if source_indices is None:
            raise ValueError("adaptive anchors require source_indices")
        effective_anchor_count(
            0,
            anchor_min_per_emitter,
            anchor_max_per_emitter,
            anchor_fraction_per_emitter,
        )

    valid = (
        ~padding_mask
        if padding_mask is not None
        else torch.ones_like(labels, dtype=torch.bool)
    )
    flat_embeddings = embeddings[valid]
    flat_labels = labels[valid]
    flat_files = file_indices[:, None].expand_as(labels)[valid]
    flat_windows = (
        torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        .expand_as(labels)[valid]
    )
    flat_sources = (
        source_indices[valid]
        if source_indices is not None
        else torch.arange(labels.numel(), device=embeddings.device)
        .reshape_as(labels)[valid]
    )
    anchors: list[Tensor] = []
    positives: list[Tensor] = []
    negatives: list[Tensor] = []
    sampling_stats = AnchorSamplingStats()

    for file_index in torch.unique(flat_files):
        file_members = torch.nonzero(flat_files == file_index, as_tuple=False).flatten()
        file_labels = flat_labels[file_members]
        for label in torch.unique(file_labels):
            members = file_members[file_labels == label]
            others = file_members[file_labels != label]
            if len(members) < 2 or len(others) == 0:
                continue
            member_sources = flat_sources[members]
            unique_pulse_count = int(torch.unique(member_sources).numel())
            sampling_stats.candidate_occurrence_count += len(members)
            sampling_stats.candidate_unique_pulse_count += unique_pulse_count
            if adaptive_anchors:
                representatives = _random_source_representatives(
                    members, member_sources
                )
                anchor_count = effective_anchor_count(
                    len(representatives),
                    anchor_min_per_emitter,
                    anchor_max_per_emitter,
                    anchor_fraction_per_emitter,
                )
                order = torch.randperm(len(representatives), device=embeddings.device)
                selected = representatives[order[:anchor_count]]
            elif len(members) > max_anchors_per_emitter:
                order = torch.randperm(len(members), device=embeddings.device)
                selected = members[order[:max_anchors_per_emitter]]
            else:
                selected = members

            anchor_windows = flat_windows[selected]
            anchor_sources = flat_sources[selected]
            candidate_positions = torch.randint(
                len(members),
                (8, len(selected)),
                device=embeddings.device,
            )
            candidate_indices = members[candidate_positions]
            candidate_windows = flat_windows[candidate_indices]
            candidate_sources = flat_sources[candidate_indices]
            different_pulse = candidate_sources != anchor_sources
            cross_window = different_pulse & (
                candidate_windows != anchor_windows
            )
            positive_indices = candidate_indices[-1]
            has_positive = torch.zeros(
                len(selected), dtype=torch.bool, device=embeddings.device
            )
            for attempt in range(candidate_indices.shape[0]):
                positive_indices = torch.where(
                    different_pulse[attempt],
                    candidate_indices[attempt],
                    positive_indices,
                )
                has_positive |= different_pulse[attempt]
            for attempt in range(candidate_indices.shape[0]):
                positive_indices = torch.where(
                    cross_window[attempt],
                    candidate_indices[attempt],
                    positive_indices,
                )
            selected = selected[has_positive]
            positive_indices = positive_indices[has_positive]
            if len(selected) == 0:
                continue

            sampled_negatives = torch.randint(
                len(others), (len(selected),), device=embeddings.device
            )
            sampling_stats.anchor_count += len(selected)
            sampling_stats.unique_anchor_count += int(
                torch.unique(flat_sources[selected]).numel()
            )
            sampling_stats.emitter_count += 1
            anchors.append(flat_embeddings[selected])
            positives.append(flat_embeddings[positive_indices])
            negatives.append(flat_embeddings[others[sampled_negatives]])

    loss = (
        embeddings.sum() * 0.0
        if not anchors
        else F.triplet_margin_loss(
            torch.cat(anchors),
            torch.cat(positives),
            torch.cat(negatives),
            margin=margin,
        )
    )
    if return_sampling_stats:
        return loss, sampling_stats
    return loss


def emitter_compactness_loss(
    embeddings: Tensor,
    labels: Tensor,
    file_indices: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Pull each file-local emitter toward its raw Euclidean centroid."""
    if embeddings.ndim != 3 or labels.shape != embeddings.shape[:2]:
        raise ValueError("embeddings and labels have incompatible shapes")
    if file_indices.shape != (embeddings.shape[0],):
        raise ValueError("file_indices must contain one value per window")
    valid = (
        ~padding_mask
        if padding_mask is not None
        else torch.ones_like(labels, dtype=torch.bool)
    )
    flat_embeddings = embeddings[valid]
    flat_labels = labels[valid]
    flat_files = file_indices[:, None].expand_as(labels)[valid]
    losses: list[Tensor] = []
    for file_index in torch.unique(flat_files):
        file_members = flat_files == file_index
        file_embeddings = flat_embeddings[file_members]
        file_labels = flat_labels[file_members]
        for label in torch.unique(file_labels):
            members = file_embeddings[file_labels == label]
            if not len(members):
                continue
            centroid = members.mean(dim=0)
            losses.append(torch.square(members - centroid).sum(dim=-1).mean())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


class TransformerHDBSCAN:
    """Small inference wrapper for a trained encoder."""

    def __init__(
        self,
        encoder: TransformerMetricEncoder,
        device: str | torch.device,
        min_cluster_size: int = 5,
        hdbscan_params: dict[str, Any] | None = None,
    ) -> None:
        self.encoder = encoder.to(device).eval()
        self.device = torch.device(device)
        self.cluster_params = {
            "min_cluster_size": min_cluster_size,
            "copy": False,
        }
        if hdbscan_params:
            self.cluster_params.update(hdbscan_params)

    def embed(self, pdws: np.ndarray) -> np.ndarray:
        features = torch.from_numpy(normalize_pdws(pdws)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embeddings = self.encoder(features).squeeze(0)
        return embeddings.cpu().numpy()

    def predict(self, pdws: np.ndarray) -> np.ndarray:
        """Return local cluster IDs; -1 is retained as an ordinary group ID."""
        if len(pdws) < self.cluster_params["min_cluster_size"]:
            return np.full(len(pdws), -1, dtype=np.int32)
        labels = HDBSCAN(**self.cluster_params).fit_predict(self.embed(pdws))
        return labels.astype(np.int32, copy=False)
