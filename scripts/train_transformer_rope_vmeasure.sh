#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# RoPE-QK + SwiGLU experiment. train_scan is used only for optimization;
# val_scan is used for checkpoint selection and early stopping. test_scan is
# intentionally reserved for scripts/test_transformer_rope_vmeasure.sh.
CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.train \
  --train-dir turing-synthetic-radar-dataset/scan/scan/train_scan \
  --validation-dir turing-synthetic-radar-dataset/scan/scan/val_scan \
  --output experiments/transformer_metric_scan_rope_vmeasure/model.pt \
  --log-dir experiments/transformer_metric_scan_rope_vmeasure/logs \
  --window-length 4096 \
  --window-stride 2048 \
  --batch-size 1024 \
  --num-workers 8 \
  --epochs 50 \
  --shuffle-train-windows \
  --normalization per_file \
  --training-scope file \
  --model-dim 64 \
  --num-layers 4 \
  --num-heads 4 \
  --feedforward-dim 128 \
  --embedding-dim 8 \
  --dropout 0.1 \
  --margin 0.2 \
  --compactness-weight 0.05 \
  --max-anchors-per-emitter 64 \
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
