#!/usr/bin/env python3
"""Train the standalone Transformer metric encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import H5WindowDataset, resolve_h5_files
from model import TransformerMetricEncoder, triplet_metric_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive; num-workers cannot be negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    files = resolve_h5_files(args.train_dir, split="train")
    dataset = H5WindowDataset(files, args.window_length, require_labels=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model_config = {
        "input_dim": dataset.feature_dim,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
    }
    model = TransformerMetricEncoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for features, labels, padding_mask, *_ in progress:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(features, padding_mask=padding_mask)
            loss = triplet_metric_loss(
                embeddings, labels, padding_mask=padding_mask, margin=args.margin
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch={epoch} mean_loss={total_loss / len(loader):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "window_length": args.window_length,
            "min_cluster_size": args.min_cluster_size,
            "normalization": "per_window",
        },
        args.output,
    )
    print(f"Saved checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
