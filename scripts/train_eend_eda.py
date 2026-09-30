#!/usr/bin/env python3
"""Train chronological Transformer EEND-EDA on exhaustive radar windows."""

from __future__ import annotations

import argparse
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from experiment_config import load_config_defaults
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from train_transformer_metric import (METRIC_NAMES, ExactDistributedSampler,
                                      ExhaustivePulseWindowDataset,
                                      configure_logger, create_control_group,
                                      distributed_context,
                                      evaluation_window_mask,
                                      install_stop_signal_handlers,
                                      resolve_h5_files, split_train_files,
                                      stop_requested)

from turing_deinterleaving_challenge.models import (PDWStandardizer,
                                                    RadarEENDEDA,
                                                    evaluate_labels,
                                                    radar_eda_loss)

IMPLEMENTATION_TAG = "chronological-radar-eend-eda-v1"
COUNT_METRIC_NAMES = (
    "Count Accuracy",
    "Count MAE",
    "Count Under",
    "Count Over",
    "Predicted Emitters",
    "True Emitters",
)
EDA_METRIC_NAMES = (*METRIC_NAMES, *COUNT_METRIC_NAMES)


def count_emitters(labels: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Return the number of unique non-padding labels in every batch item."""
    if labels.shape != padding_mask.shape:
        raise ValueError("labels and padding_mask must have identical shapes")
    counts = [
        torch.unique(item_labels[~item_mask]).numel()
        for item_labels, item_mask in zip(labels, padding_mask, strict=True)
    ]
    if any(count <= 0 for count in counts):
        raise ValueError("every training item must contain at least one emitter")
    return torch.as_tensor(counts, dtype=torch.long, device=labels.device)


def evaluate_eda_labels(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    predicted_count: int,
) -> dict[str, float]:
    """Calculate clustering and local emitter-count metrics for one window."""
    score = evaluate_labels(labels_pred, labels_true)
    true_count = int(np.unique(labels_true).size)
    score.update(
        {
            "Count Accuracy": float(predicted_count == true_count),
            "Count MAE": float(abs(predicted_count - true_count)),
            "Count Under": float(predicted_count < true_count),
            "Count Over": float(predicted_count > true_count),
            "Predicted Emitters": float(predicted_count),
            "True Emitters": float(true_count),
        }
    )
    return score


def validate_eda(
    model: RadarEENDEDA,
    data_loader: DataLoader,
    device: torch.device,
    min_emitters: int,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    should_stop: Callable[[], bool] | None = None,
    show_progress: bool = False,
    collect_window_scores: bool = False,
) -> tuple[
    dict[str, float],
    int,
    int,
    int,
    dict[str, list[float]] | None,
] | None:
    """Evaluate full multi-emitter windows using direct EDA assignments."""
    totals = {name: 0.0 for name in EDA_METRIC_NAMES}
    window_count = 0
    skipped_partial = 0
    skipped_low_emitter = 0
    window_scores = (
        {name: [] for name in EDA_METRIC_NAMES}
        if collect_window_scores
        else None
    )
    model.eval()
    with torch.no_grad(), tqdm(
        data_loader,
        desc="Validation",
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
        file=sys.stdout,
        position=0,
        disable=not show_progress,
    ) as progress:
        for features, labels, padding_mask in progress:
            if should_stop is not None and should_stop():
                return None
            eligible, partial_count, low_emitter_count = evaluation_window_mask(
                labels, padding_mask, min_emitters
            )
            skipped_partial += partial_count
            skipped_low_emitter += low_emitter_count
            if not eligible.any():
                continue
            features = features[eligible]
            labels = labels[eligible]
            padding_mask = padding_mask[eligible]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predicted, predicted_counts, _ = model.predict(
                    features.to(device, non_blocking=True),
                    padding_mask=padding_mask.to(device, non_blocking=True),
                )
            predicted_np = predicted.cpu().numpy()
            counts_np = predicted_counts.cpu().numpy()
            labels_np = labels.numpy()
            masks_np = padding_mask.numpy()
            for item_predicted, item_labels, item_mask, item_count in zip(
                predicted_np,
                labels_np,
                masks_np,
                counts_np,
                strict=True,
            ):
                valid = ~item_mask
                score = evaluate_eda_labels(
                    item_predicted[valid], item_labels[valid], int(item_count)
                )
                for name, value in score.items():
                    totals[name] += float(value)
                    if window_scores is not None:
                        window_scores[name].append(float(value))
                window_count += 1
    return (
        totals,
        window_count,
        skipped_partial,
        skipped_low_emitter,
        window_scores,
    )


def summarize_eda_validation(
    result: tuple[
        dict[str, float],
        int,
        int,
        int,
        dict[str, list[float]] | None,
    ],
    control_group: object | None = None,
) -> tuple[dict[str, float], int, int, int]:
    """Aggregate EDA metrics from disjoint validation or test shards."""
    totals, window_count, skipped_partial, skipped_low_emitter, _ = result
    values = torch.tensor(
        [
            *(totals.get(name, 0.0) for name in EDA_METRIC_NAMES),
            window_count,
            skipped_partial,
            skipped_low_emitter,
        ],
        dtype=torch.float64,
    )
    if control_group is not None:
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=control_group)
    total_window_count = int(values[len(EDA_METRIC_NAMES)].item())
    if total_window_count == 0:
        raise RuntimeError(
            "validation contained no full windows with the requested emitter count"
        )
    metrics = {
        name: float(values[index].item() / total_window_count)
        for index, name in enumerate(EDA_METRIC_NAMES)
    }
    return (
        metrics,
        total_window_count,
        int(values[len(EDA_METRIC_NAMES) + 1].item()),
        int(values[len(EDA_METRIC_NAMES) + 2].item()),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hdf5-cache-size", type=int, default=32)
    parser.add_argument("--shuffle-train-windows", action="store_true")
    parser.add_argument(
        "--n-jobs", "--num-workers", dest="n_jobs", type=int, default=0
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-attractors", type=int, default=96)
    parser.add_argument("--existence-threshold", type=float, default=0.5)
    parser.add_argument("--assignment-logit-scale", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--min-emitters", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("transformer_eda.pt"))
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    valid_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(
        **load_config_defaults(
            argv,
            sections=("train", "model"),
            valid_destinations=valid_destinations,
        )
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    install_stop_signal_handlers()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.n_jobs < 0
        or args.hdf5_cache_size <= 0
        or args.early_stopping_patience <= 0
        or args.model_dim <= 0
        or args.num_layers <= 0
        or args.num_heads <= 0
        or args.embedding_dim <= 0
        or args.feedforward_dim <= 0
        or args.max_attractors <= 0
        or args.min_emitters <= 0
        or args.assignment_logit_scale <= 0
        or args.alpha < 0
        or not 0 <= args.dropout < 1
        or not 0 < args.existence_threshold < 1
        or not 0 < args.validation_fraction < 1
    ):
        raise ValueError("invalid EEND-EDA training or model configuration")
    if args.max_attractors > args.window_length:
        raise ValueError("max_attractors cannot exceed the pulse-window length")

    args.output = args.output.expanduser().resolve()
    resume_checkpoint = args.output if args.output.is_file() else None
    default_log_dir = args.output.parent / args.output.stem / "logs"
    log_dir = (args.log_dir or default_log_dir).expanduser().resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    device, device_ids, rank, world_size, is_distributed = distributed_context(
        args.device
    )
    if is_distributed:
        if rank == 0:
            logger = configure_logger(log_dir, rank, truncate=False)
            logger.info(
                "%s config=%s output=%s",
                "resuming run from" if resume_checkpoint is not None else "run started",
                args.config,
                args.output,
            )
        dist.barrier()
        if rank != 0:
            logger = configure_logger(log_dir, rank, truncate=False)
            logger.info("rank joined run")
        dist.barrier()
    else:
        logger = configure_logger(log_dir, rank, truncate=False)
        logger.info(
            "%s config=%s output=%s",
            "resuming run from" if resume_checkpoint is not None else "run started",
            args.config,
            args.output,
        )
    control_group = create_control_group(is_distributed)
    if is_distributed and args.batch_size % world_size:
        raise ValueError("--batch-size must be divisible by the DDP world size")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    logger.info(
        "runtime implementation=%s distributed=%s rank=%d/%d device=%s torch=%s",
        IMPLEMENTATION_TAG,
        is_distributed,
        rank,
        world_size,
        device,
        torch.__version__,
    )
    if rank == 0:
        print(
            f"Radar EEND-EDA implementation={IMPLEMENTATION_TAG} "
            f"embedding_dim={args.embedding_dim} max_attractors={args.max_attractors}",
            flush=True,
        )

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    all_train_files = resolve_h5_files(args.train_dir.expanduser().resolve(), "train")
    if args.validation_dir is None:
        train_files, validation_files = split_train_files(
            all_train_files, args.validation_fraction, args.seed
        )
        if rank == 0:
            print(
                f"Holding out {len(validation_files)}/{len(all_train_files)} "
                "train files for validation."
            )
    else:
        train_files = all_train_files
        validation_files = resolve_h5_files(
            args.validation_dir.expanduser().resolve(), "validation"
        )

    train_dataset: Dataset = ExhaustivePulseWindowDataset(
        train_files,
        args.window_length,
        PDWStandardizer(),
        args.hdf5_cache_size,
    )
    validation_dataset: Dataset = ExhaustivePulseWindowDataset(
        validation_files,
        args.window_length,
        PDWStandardizer(),
        args.hdf5_cache_size,
    )
    loader_kwargs: dict[str, object] = {}
    if args.n_jobs:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        )
    train_sampler = (
        ExactDistributedSampler(
            train_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=args.shuffle_train_windows,
            seed=args.seed,
        )
        if is_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size if is_distributed else args.batch_size,
        num_workers=args.n_jobs,
        sampler=train_sampler,
        shuffle=args.shuffle_train_windows if train_sampler is None else False,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    validation_sampler = (
        ExactDistributedSampler(
            validation_dataset,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            seed=args.seed,
        )
        if is_distributed
        else None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size // world_size if is_distributed else args.batch_size,
        sampler=validation_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    input_dim = train_dataset.feature_dim
    model_config = {
        "input_dim": input_dim,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "max_attractors": args.max_attractors,
        "existence_threshold": args.existence_threshold,
        "assignment_logit_scale": args.assignment_logit_scale,
    }
    model = RadarEENDEDA(**model_config).to(device)
    training_model: RadarEENDEDA | DataParallel | DistributedDataParallel = model
    if is_distributed:
        training_model = DistributedDataParallel(model, device_ids=[device.index])
    elif device_ids is not None:
        training_model = DataParallel(
            model, device_ids=device_ids, output_device=device_ids[0]
        )
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = None
    if amp_enabled and amp_dtype == torch.float16:
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_v_measure = float("-inf")
    epochs_without_improvement = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        if checkpoint.get("normalization") != "per_window":
            raise ValueError("resume checkpoint does not use per-window normalization")
        if checkpoint.get("assignment") != "hungarian_softmax":
            raise ValueError("resume checkpoint is not a radar EEND-EDA model")
        checkpoint_config = checkpoint.get("model_config")
        if checkpoint_config != model_config:
            raise ValueError(
                "resume checkpoint model_config differs from the requested model configuration"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        training_state = checkpoint.get("training_state", {})
        if training_state:
            optimizer_state = training_state.get("optimizer_state_dict")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            else:
                logger.warning("resume checkpoint has no optimizer state; optimizer restarted")
            scaler_state = training_state.get("scaler_state_dict")
            if scaler is not None and scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            start_epoch = int(training_state.get("completed_epochs", 0))
            best_v_measure = float(
                training_state.get(
                    "best_v_measure",
                    checkpoint.get("validation_metrics", {}).get(
                        "V-measure", float("-inf")
                    ),
                )
            )
            epochs_without_improvement = int(
                training_state.get("epochs_without_improvement", 0)
            )
        else:
            log_path = log_dir / "train.log"
            try:
                completed_epochs = [
                    int(match)
                    for match in re.findall(
                        r"\bepoch=(\d+)\b",
                        log_path.read_text(encoding="utf-8"),
                    )
                ]
            except OSError:
                completed_epochs = []
            start_epoch = max(completed_epochs, default=0)
            logger.warning(
                "legacy checkpoint has no training state; resuming model weights "
                "with completed_epochs=%d inferred from %s",
                start_epoch,
                log_path,
            )
        if start_epoch < 0:
            raise ValueError("resume checkpoint has a negative completed epoch count")
        logger.info(
            "resume loaded checkpoint=%s completed_epochs=%d best_v_measure=%.6f "
            "epochs_without_improvement=%d",
            resume_checkpoint,
            start_epoch,
            best_v_measure,
            epochs_without_improvement,
        )

    if rank == 0:
        logger.info(
            "setup train_windows=%d validation_candidates=%d min_emitters=%d "
            "window_length=%d global_batch=%d workers_per_rank=%d amp=%s(%s)",
            len(train_dataset),
            len(validation_dataset),
            args.min_emitters,
            args.window_length,
            args.batch_size,
            args.n_jobs,
            amp_enabled,
            args.amp_dtype,
        )

    interrupted = False
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        total_losses: list[float] = []
        assignment_losses: list[float] = []
        existence_losses: list[float] = []
        progress = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stdout,
            )
            if rank == 0
            else None
        )
        train_iterator = progress if progress is not None else train_loader
        should_early_stop = False
        join_context = training_model.join() if is_distributed else nullcontext()
        with join_context:
            for features, labels, padding_mask in train_iterator:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                padding_mask = padding_mask.to(device, non_blocking=True)
                emitter_counts = count_emitters(labels, padding_mask)
                max_emitter_count = int(emitter_counts.max().item())
                if max_emitter_count > args.max_attractors:
                    raise ValueError(
                        "a training window contains more emitters than max_attractors"
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    assignment_logits, existence_logits = training_model(
                        features,
                        padding_mask=padding_mask,
                        num_steps=max_emitter_count + 1,
                    )
                    losses = radar_eda_loss(
                        assignment_logits,
                        existence_logits,
                        labels,
                        padding_mask=padding_mask,
                        alpha=args.alpha,
                    )
                if scaler is not None:
                    scaler.scale(losses.total).backward()
                    scaler.unscale_(optimizer)
                else:
                    losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if rank == 0:
                    total_losses.append(float(losses.total.detach().cpu()))
                    assignment_losses.append(float(losses.assignment.detach().cpu()))
                    existence_losses.append(float(losses.existence.detach().cpu()))
                if stop_requested():
                    interrupted = True
                    break
        if progress is not None:
            progress.close()

        if is_distributed:
            stop_tensor = torch.tensor(
                int(interrupted or stop_requested()), device=device
            )
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            interrupted = bool(stop_tensor.item())

        validation_result = None
        if not interrupted:
            validation_result = validate_eda(
                model,
                validation_loader,
                device,
                args.min_emitters,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                should_stop=stop_requested,
                show_progress=rank == 0,
            )
            interrupted = validation_result is None
        if is_distributed:
            interrupt_tensor = torch.tensor(int(interrupted or stop_requested()))
            dist.all_reduce(
                interrupt_tensor, op=dist.ReduceOp.MAX, group=control_group
            )
            interrupted = bool(interrupt_tensor.item())

        if not interrupted:
            assert validation_result is not None
            metrics, window_count, skipped_partial, skipped_low_emitter = (
                summarize_eda_validation(
                    validation_result,
                    control_group if is_distributed else None,
                )
            )
            if rank == 0:
                mean_total = float(np.mean(total_losses))
                mean_assignment = float(np.mean(assignment_losses))
                mean_existence = float(np.mean(existence_losses))
                tqdm.write(
                    f"epoch={epoch + 1} total_loss={mean_total:.4f} "
                    f"assignment_loss={mean_assignment:.4f} "
                    f"existence_loss={mean_existence:.4f} "
                    f"V-measure={metrics['V-measure']:.4f} "
                    f"count_MAE={metrics['Count MAE']:.3f} "
                    f"eval_windows={window_count}",
                    file=sys.stdout,
                )
                logger.info(
                    "epoch=%d total_loss=%.6f assignment_loss=%.6f "
                    "existence_loss=%.6f v_measure=%.6f count_mae=%.6f "
                    "count_accuracy=%.6f validation_windows=%d",
                    epoch + 1,
                    mean_total,
                    mean_assignment,
                    mean_existence,
                    metrics["V-measure"],
                    metrics["Count MAE"],
                    metrics["Count Accuracy"],
                    window_count,
                )
                if metrics["V-measure"] > best_v_measure:
                    best_v_measure = metrics["V-measure"]
                    epochs_without_improvement = 0
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": model_config,
                            "normalization": "per_window",
                            "assignment": "hungarian_softmax",
                            "eda_embedding_order": "chronological",
                            "loss_config": {"alpha": args.alpha},
                            "evaluation_window_policy": {
                                "full_windows_only": True,
                                "min_emitters": args.min_emitters,
                            },
                            "validation_metrics": metrics,
                            "training_state": {
                                "completed_epochs": epoch + 1,
                                "best_v_measure": best_v_measure,
                                "epochs_without_improvement": epochs_without_improvement,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scaler_state_dict": (
                                    scaler.state_dict() if scaler is not None else None
                                ),
                            },
                        },
                        args.output,
                    )
                    tqdm.write(
                        f"saved best checkpoint to {args.output} "
                        f"(V-measure={best_v_measure:.4f})",
                        file=sys.stdout,
                    )
                else:
                    epochs_without_improvement += 1
                    should_early_stop = (
                        epochs_without_improvement >= args.early_stopping_patience
                    )
                logger.info(
                    "validation skipped_partial=%d skipped_low_emitter=%d",
                    skipped_partial,
                    skipped_low_emitter,
                )
        if is_distributed:
            interrupt_tensor = torch.tensor(int(interrupted or stop_requested()))
            dist.broadcast(interrupt_tensor, src=0, group=control_group)
            interrupted = bool(interrupt_tensor.item())
            early_stop_tensor = torch.tensor(int(should_early_stop))
            dist.broadcast(early_stop_tensor, src=0, group=control_group)
            should_early_stop = bool(early_stop_tensor.item())
        if interrupted or should_early_stop:
            break

    if is_distributed:
        if control_group is not None:
            dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
