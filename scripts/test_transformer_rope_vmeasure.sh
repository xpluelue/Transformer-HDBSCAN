#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# Run once after training. test_scan never participates in optimization,
# checkpoint selection, or early stopping. Output strictly follows the existing
# PDW Studio layout: one directory per source file and one
# config_<id>_<cluster-index>.h5 file per predicted cluster.
CHECKPOINT="${CHECKPOINT:-experiments/transformer_metric_scan_rope_vmeasure/model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/transformer_metric_scan_rope_vmeasure/pdw_studio_clusters_dual_gpu_v1}"
MIN_CLUSTER_FRACTION="${MIN_CLUSTER_FRACTION:-0.0005}"

CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.predict \
  --data-dir turing-synthetic-radar-dataset/scan/scan/test_scan \
  --split test \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --output-format pdw_studio \
  --devices cuda:0,cuda:1 \
  --batch-size 512 \
  --num-workers 8 \
  --min-cluster-fraction "$MIN_CLUSTER_FRACTION" \
  --min-samples 5 \
  --min-samples-fraction 0.0005 \
  --hdbscan-backend cuml \
  --hdbscan-devices 0,1 \
  --hdbscan-parallel-files 2 \
  --evaluate \
  --amp \
  --amp-dtype bf16
