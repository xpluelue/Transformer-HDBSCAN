#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# Joint experiment: SiLU input activation, RMSNorm pre-norm blocks,
# compactness weight 0.1, and file-level adaptive unique-pulse anchors.
CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.train \
  --train-dir turing-synthetic-radar-dataset/scan/scan/train_scan \
  --validation-dir turing-synthetic-radar-dataset/scan/scan/val_scan \
  --output experiments/transformer_metric_scan_rope_rmsnorm_silu_adaptive/model.pt \
  --log-dir experiments/transformer_metric_scan_rope_rmsnorm_silu_adaptive/logs \
  --window-length 4096 \
  --window-stride 2048 \
  --batch-size 1024 \
  --num-workers 8 \
  --epochs 50 \
  --shuffle-train-windows \
  --normalization per_file \
  --training-scope file \
  --architecture rope_swiglu_rmsnorm_silu_v2 \
  --model-dim 64 \
  --num-layers 4 \
  --num-heads 4 \
  --feedforward-dim 128 \
  --embedding-dim 8 \
  --dropout 0.1 \
  --margin 0.2 \
  --compactness-weight 0.1 \
  --adaptive-anchors \
  --anchor-min-per-emitter 128 \
  --anchor-max-per-emitter 512 \
  --anchor-fraction-per-emitter 0.05 \
  --min-cluster-size 5 \
  --min-cluster-fraction 0.0005 \
  --allow-single-cluster \
  --validation-max-files 128 \
  --selection-metric v_measure \
  --validation-hdbscan-devices 0,1 \
  --validation-hdbscan-parallel-files 2 \
  --early-stopping-patience 5 \
  --devices cuda:0,cuda:1 \
  --amp \
  --amp-dtype bf16
