"""Chronological EEND-EDA adaptation for fixed radar-pulse windows.

The model keeps the existing Transformer PDW encoder and replaces HDBSCAN
with an encoder-decoder attractor (EDA) head.  Unlike speaker diarization,
every observed radar pulse belongs to exactly one emitter, so assignments use
a softmax over attractors.  Training aligns arbitrary emitter IDs to attractor
slots with a Hungarian assignment before calculating cross entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .model import Deinterleaver
from .transformer import PDWStandardizer, TransformerMetricEncoder


class EncoderDecoderAttractor(nn.Module):
    """Generate an ordered sequence of attractors from chronological embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.LSTMCell(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
        )
        self.existence_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        embeddings: Tensor,
        padding_mask: Tensor | None,
        num_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """Return attractors and existence logits for ``num_steps`` decoder steps."""
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape (batch, length, embedding_dim)"
            )
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("embedding dimension does not match the EDA module")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if padding_mask is not None and padding_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "padding_mask must match the first two embedding dimensions"
            )

        if padding_mask is None:
            _, (hidden, cell) = self.encoder(embeddings)
        else:
            lengths = (~padding_mask).sum(dim=1)
            if bool((lengths <= 0).any()):
                raise ValueError(
                    "every EDA input must contain at least one valid pulse"
                )
            packed = pack_padded_sequence(
                embeddings,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, cell) = self.encoder(packed)

        hidden_state = hidden[-1]
        cell_state = cell[-1]
        decoder_input = embeddings.new_zeros(
            embeddings.shape[0], self.embedding_dim
        )
        attractors: list[Tensor] = []
        existence_logits: list[Tensor] = []
        for _ in range(num_steps):
            hidden_state, cell_state = self.decoder(
                decoder_input, (hidden_state, cell_state)
            )
            attractors.append(hidden_state)
            existence_logits.append(
                self.existence_head(hidden_state).squeeze(-1)
            )
        return (
            torch.stack(attractors, dim=1),
            torch.stack(existence_logits, dim=1),
        )


def leading_existence_counts(
    existence_probabilities: Tensor,
    threshold: float,
    max_attractors: int,
) -> Tensor:
    """Count consecutive existing attractors, forcing one for non-empty windows."""
    if existence_probabilities.ndim != 2:
        raise ValueError("existence probabilities must have shape (batch, steps)")
    if not 0 < threshold < 1:
        raise ValueError("existence threshold must be in (0, 1)")
    if max_attractors <= 0:
        raise ValueError("max_attractors must be positive")
    usable = existence_probabilities[:, :max_attractors] >= threshold
    # cumprod changes every position after the first failed existence decision
    # to zero, so a later non-monotonic positive cannot reopen the sequence.
    consecutive = usable.to(torch.int64).cumprod(dim=1)
    return consecutive.sum(dim=1).clamp(min=1, max=max_attractors)


class RadarEENDEDA(nn.Module):
    """Transformer encoder followed by chronological encoder-decoder attractors."""

    def __init__(
        self,
        input_dim: int = 5,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        embedding_dim: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        max_attractors: int = 96,
        existence_threshold: float = 0.5,
        assignment_logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if max_attractors <= 0:
            raise ValueError("max_attractors must be positive")
        if not 0 < existence_threshold < 1:
            raise ValueError("existence_threshold must be in (0, 1)")
        if assignment_logit_scale <= 0:
            raise ValueError("assignment_logit_scale must be positive")
        self.max_attractors = max_attractors
        self.existence_threshold = existence_threshold
        self.encoder = TransformerMetricEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.eda = EncoderDecoderAttractor(embedding_dim)
        self.log_assignment_scale = nn.Parameter(
            torch.tensor(float(np.log(assignment_logit_scale)), dtype=torch.float32)
        )

    def assignment_logits(self, embeddings: Tensor, attractors: Tensor) -> Tensor:
        """Return scaled cosine similarities for every pulse-attractor pair."""
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_attractors = F.normalize(attractors, p=2, dim=-1)
        scale = self.log_assignment_scale.exp().clamp(max=100.0)
        return scale * torch.einsum(
            "btd,bsd->bts", normalized_embeddings, normalized_attractors
        )

    def forward(
        self,
        features: Tensor,
        padding_mask: Tensor | None,
        num_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """Return assignment logits and existence logits for training.

        ``num_steps`` includes the final stop step.  Assignment logits are only
        needed for the preceding attractor steps, while existence logits retain
        all steps including the stop decision.
        """
        if num_steps <= 1:
            raise ValueError(
                "num_steps must include at least one attractor and one stop"
            )
        if num_steps > self.max_attractors + 1:
            raise ValueError("num_steps exceeds max_attractors plus the stop step")
        model_mask = (
            padding_mask
            if padding_mask is not None and bool(padding_mask.any())
            else None
        )
        embeddings = self.encoder(features, padding_mask=model_mask)
        attractors, existence_logits = self.eda(
            embeddings, padding_mask=model_mask, num_steps=num_steps
        )
        assignment_logits = self.assignment_logits(
            embeddings, attractors[:, : num_steps - 1]
        )
        return assignment_logits, existence_logits

    @torch.no_grad()
    def predict(
        self,
        features: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict one mutually exclusive emitter label per valid pulse.

        Decoding stops once every batch item has produced its first existence
        probability below the configured threshold.  ``max_attractors`` is
        therefore a safety cap, not a fixed allocation.
        """
        model_mask = (
            padding_mask
            if padding_mask is not None and bool(padding_mask.any())
            else None
        )
        embeddings = self.encoder(features, padding_mask=model_mask)
        if padding_mask is None:
            effective_mask = torch.zeros(
                embeddings.shape[:2], dtype=torch.bool, device=embeddings.device
            )
        else:
            effective_mask = padding_mask

        # Encode the chronological sequence once, then reproduce the decoder
        # loop here so inference can stop early without allocating 1024 slots.
        if bool(effective_mask.any()):
            lengths = (~effective_mask).sum(dim=1)
            if bool((lengths <= 0).any()):
                raise ValueError("every prediction window must contain a valid pulse")
            packed = pack_padded_sequence(
                embeddings,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, cell) = self.eda.encoder(packed)
        else:
            _, (hidden, cell) = self.eda.encoder(embeddings)

        hidden_state = hidden[-1]
        cell_state = cell[-1]
        decoder_input = embeddings.new_zeros(
            embeddings.shape[0], embeddings.shape[-1]
        )
        active = torch.ones(
            embeddings.shape[0], dtype=torch.bool, device=embeddings.device
        )
        predicted_counts = torch.zeros(
            embeddings.shape[0], dtype=torch.long, device=embeddings.device
        )
        attractors: list[Tensor] = []
        existence_probabilities: list[Tensor] = []
        for _ in range(self.max_attractors):
            hidden_state, cell_state = self.eda.decoder(
                decoder_input, (hidden_state, cell_state)
            )
            attractors.append(hidden_state)
            probability = torch.sigmoid(
                self.eda.existence_head(hidden_state).squeeze(-1)
            )
            existence_probabilities.append(probability)
            exists = probability >= self.existence_threshold
            predicted_counts += (active & exists).to(torch.long)
            active &= exists
            if not bool(active.any()):
                break

        predicted_counts.clamp_(min=1, max=self.max_attractors)
        attractor_tensor = torch.stack(attractors, dim=1)
        probability_tensor = torch.stack(existence_probabilities, dim=1)
        max_count = int(predicted_counts.max().item())
        logits = self.assignment_logits(
            embeddings, attractor_tensor[:, :max_count]
        )
        slot_indices = torch.arange(max_count, device=logits.device)
        inactive_slots = slot_indices.unsqueeze(0) >= predicted_counts.unsqueeze(1)
        logits = logits.masked_fill(inactive_slots.unsqueeze(1), float("-inf"))
        labels = logits.softmax(dim=-1).argmax(dim=-1)
        labels = labels.masked_fill(effective_mask, -1)
        return labels, predicted_counts, probability_tensor


@dataclass(frozen=True)
class RadarEDALoss:
    """The two losses used by the radar EEND-EDA adaptation."""

    total: Tensor
    assignment: Tensor
    existence: Tensor


def radar_eda_loss(
    assignment_logits: Tensor,
    existence_logits: Tensor,
    labels: Tensor,
    padding_mask: Tensor | None = None,
    alpha: float = 1.0,
) -> RadarEDALoss:
    """Calculate Hungarian-matched assignment CE plus attractor existence BCE."""
    if assignment_logits.ndim != 3 or existence_logits.ndim != 2:
        raise ValueError("assignment logits must be 3D and existence logits must be 2D")
    if labels.shape != assignment_logits.shape[:2]:
        raise ValueError(
            "labels must match the assignment batch and sequence dimensions"
        )
    if existence_logits.shape[0] != labels.shape[0]:
        raise ValueError("existence logits must have the same batch size as labels")
    if padding_mask is not None and padding_mask.shape != labels.shape:
        raise ValueError("padding_mask must have the same shape as labels")
    if alpha < 0:
        raise ValueError("alpha cannot be negative")

    assignment_losses: list[Tensor] = []
    existence_losses: list[Tensor] = []
    for batch_index in range(labels.shape[0]):
        valid = (
            ~padding_mask[batch_index]
            if padding_mask is not None
            else torch.ones_like(labels[batch_index], dtype=torch.bool)
        )
        valid_labels = labels[batch_index, valid]
        if not len(valid_labels):
            raise ValueError("every loss item must contain at least one valid pulse")
        _, local_labels = torch.unique(
            valid_labels, sorted=True, return_inverse=True
        )
        emitter_count = int(local_labels.max().item()) + 1
        if emitter_count > assignment_logits.shape[2]:
            raise ValueError(
                "assignment logits contain fewer attractors than true emitters"
            )
        if emitter_count + 1 > existence_logits.shape[1]:
            raise ValueError("existence logits do not contain the required stop step")

        item_logits = assignment_logits[
            batch_index, valid, :emitter_count
        ]
        log_probabilities = F.log_softmax(item_logits, dim=-1)
        membership = F.one_hot(
            local_labels, num_classes=emitter_count
        ).to(log_probabilities.dtype)
        emitter_sizes = membership.sum(dim=0).clamp_min(1.0)
        cost = -(
            membership.transpose(0, 1) @ log_probabilities
        ) / emitter_sizes.unsqueeze(1)
        true_indices, attractor_indices = linear_sum_assignment(
            cost.detach().float().cpu().numpy()
        )
        true_to_attractor = torch.empty(
            emitter_count, dtype=torch.long, device=labels.device
        )
        true_to_attractor[
            torch.as_tensor(true_indices, device=labels.device)
        ] = torch.as_tensor(attractor_indices, device=labels.device)
        matched_targets = true_to_attractor[local_labels]
        assignment_losses.append(F.cross_entropy(item_logits, matched_targets))

        existence_target = existence_logits.new_zeros(emitter_count + 1)
        existence_target[:emitter_count] = 1.0
        existence_losses.append(
            F.binary_cross_entropy_with_logits(
                existence_logits[batch_index, : emitter_count + 1],
                existence_target,
            )
        )

    assignment_loss = torch.stack(assignment_losses).mean()
    existence_loss = torch.stack(existence_losses).mean()
    total_loss = assignment_loss + alpha * existence_loss
    return RadarEDALoss(
        total=total_loss,
        assignment=assignment_loss,
        existence=existence_loss,
    )


class EENDEDADeinterleaver(Deinterleaver):
    """NumPy inference wrapper compatible with the challenge model interface."""

    def __init__(
        self,
        model: RadarEENDEDA,
        standardizer: PDWStandardizer,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(default_label=None)
        self.model = model
        self.standardizer = standardizer
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        features = self.standardizer.transform(data)
        self.model.eval()
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        labels, _, _ = self.model.predict(tensor)
        return labels.squeeze(0).cpu().numpy()
