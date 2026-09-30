"""Transformer metric-learning baseline for radar pulse deinterleaving.

The baseline has an encoder only: it produces one contextual embedding for
each observed PDW, then delegates the unknown-number-of-emitters decision to a
density clusterer such as HDBSCAN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import Deinterleaver


def delta_toa_features(pdws: np.ndarray) -> np.ndarray:
    """Replace the absolute ToA column with the preceding inter-pulse interval."""
    features = np.asarray(pdws, dtype=np.float32).copy()
    if features.ndim != 2:
        raise ValueError("pdws must have shape (sequence_length, feature_dim)")
    if features.shape[1] == 0:
        raise ValueError("pdws must contain ToA in column zero")
    if len(features):
        features[1:, 0] = np.diff(features[:, 0])
        features[0, 0] = 0.0
    return features


class PDWStandardizer:
    """Normalize every model-input window independently."""

    def __init__(self) -> None:
        self.feature_dim_: int | None = None

    def transform(self, pdws: np.ndarray) -> np.ndarray:
        """Return delta-ToA PDWs standardized within this window alone."""
        features = delta_toa_features(pdws)
        if self.feature_dim_ is None:
            self.feature_dim_ = features.shape[1]
        elif features.shape[1] != self.feature_dim_:
            raise ValueError("PDW feature dimension differs from earlier windows")
        if not len(features):
            return features
        mean = features.mean(axis=0, dtype=np.float64)
        std = np.maximum(features.std(axis=0, dtype=np.float64), 1e-6)
        return ((features - mean) / std).astype(np.float32)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed positional encoding for a batch-first Transformer."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.model_dim = model_dim

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        position = torch.arange(length, device=x.device, dtype=x.dtype)
        frequencies = torch.exp(
            torch.arange(0, self.model_dim, 2, device=x.device, dtype=x.dtype)
            * (-np.log(10000.0) / self.model_dim)
        )
        encoding = torch.zeros(length, self.model_dim, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(position[:, None] * frequencies)
        encoding[:, 1::2] = torch.cos(position[:, None] * frequencies)
        return x + encoding.unsqueeze(0)


class TransformerMetricEncoder(nn.Module):
    """Four-layer Transformer encoder that outputs one unit embedding per PDW."""

    def __init__(
        self,
        input_dim: int = 5,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        embedding_dim: int = 64,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.position_encoding = SinusoidalPositionEncoding(model_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding_projection = nn.Linear(model_dim, embedding_dim)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Encode ``(batch, length, feature_dim)`` into normalized embeddings."""
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, length, feature_dim)")
        hidden = self.position_encoding(self.input_projection(features))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return F.normalize(self.embedding_projection(hidden), p=2, dim=-1)


def triplet_metric_loss(
    embeddings: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    margin: float = 0.2,
) -> Tensor:
    """Sample same-window triplets and optimize their embedding distances.

    Emitter labels are only meaningful inside an individual pulse train.  This
    function consequently never creates positives across batch elements.
    """
    if embeddings.ndim != 3 or labels.shape != embeddings.shape[:2]:
        raise ValueError("embeddings must be (batch, length, dim) and labels (batch, length)")
    if padding_mask is not None and padding_mask.shape != labels.shape:
        raise ValueError("padding_mask must have the same shape as labels")

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
            if len(members) < 2 or not len(others):
                continue
            anchors.append(embeddings[batch_index, members])
            positives.append(embeddings[batch_index, members.roll(shifts=-1)])
            sampled_negatives = torch.randint(
                len(others), (len(members),), device=embeddings.device
            )
            negatives.append(embeddings[batch_index, others[sampled_negatives]])

    if not anchors:
        return embeddings.sum() * 0.0
    return F.triplet_margin_loss(
        torch.cat(anchors), torch.cat(positives), torch.cat(negatives), margin=margin
    )


class TransformerMetricDeinterleaver(Deinterleaver):
    """Inference wrapper: trained encoder followed by HDBSCAN per pulse window."""

    def __init__(
        self,
        encoder: TransformerMetricEncoder,
        standardizer: PDWStandardizer,
        cluster_params: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        default_label: int = -1,
    ) -> None:
        super().__init__(default_label=default_label)
        self.encoder = encoder
        self.standardizer = standardizer
        self.cluster_params = {"min_cluster_size": 5, "copy": False}
        if cluster_params is not None:
            self.cluster_params.update(cluster_params)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder.to(self.device)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Return normalized embeddings for one raw, ToA-sorted pulse window."""
        features = self.standardizer.transform(data)
        self.encoder.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            embeddings = self.encoder(tensor).squeeze(0)
        return embeddings.cpu().numpy()

    def __call__(self, data: np.ndarray) -> np.ndarray:
        embeddings = self.embed(data)
        return sklearn.cluster.HDBSCAN(**self.cluster_params).fit_predict(embeddings)
