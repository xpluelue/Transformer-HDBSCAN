"""Transformer metric encoder followed by per-window HDBSCAN clustering."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from torch import Tensor, nn
from torch.nn import functional as F


def normalize_pdws(pdws: np.ndarray) -> np.ndarray:
    """Convert absolute ToA to delta-ToA and standardize one window."""
    features = np.asarray(pdws, dtype=np.float32).copy()
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("PDWs must have shape (num_pulses, num_features)")
    if len(features) == 0:
        return features
    features[1:, 0] = np.diff(features[:, 0])
    features[0, 0] = 0.0
    mean = features.mean(axis=0, dtype=np.float64)
    std = np.maximum(features.std(axis=0, dtype=np.float64), 1e-6)
    return ((features - mean) / std).astype(np.float32)


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


class TransformerMetricEncoder(nn.Module):
    """Produce one L2-normalized embedding for every input pulse."""

    def __init__(
        self,
        input_dim: int = 5,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        embedding_dim: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.embedding_projection = nn.Linear(model_dim, embedding_dim)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        hidden = self.position_encoding(self.input_projection(features))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return F.normalize(self.embedding_projection(hidden), p=2, dim=-1)


def triplet_metric_loss(
    embeddings: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    margin: float = 0.2,
) -> Tensor:
    """Build triplets only within a window because emitter IDs are local."""
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
        if len(pdws) < self.cluster_params["min_cluster_size"]:
            return np.full(len(pdws), -1, dtype=np.int32)
        labels = HDBSCAN(**self.cluster_params).fit_predict(self.embed(pdws))
        return labels.astype(np.int32, copy=False)
