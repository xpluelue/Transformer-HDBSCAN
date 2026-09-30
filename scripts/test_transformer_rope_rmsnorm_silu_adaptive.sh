#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# Temporary Stare smoke evaluation. By default it evaluates three fixed source
# files and writes only aggregate/per-file metrics, not millions of cluster H5s.
# Override the subset with TEST_FILES=config_0.h5,config_1.h5,...
CHECKPOINT="${CHECKPOINT:-experiments/transformer_metric_scan_rope_rmsnorm_silu_adaptive/model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/transformer_metric_scan_rope_rmsnorm_silu_adaptive/stare_test_subset_metrics_v1}"
MIN_CLUSTER_FRACTION="${MIN_CLUSTER_FRACTION:-0.0005}"
TEST_FILES="${TEST_FILES:-config_0.h5,config_1.h5,config_108.h5}"
IFS=',' read -r -a TEST_FILE_ARGS <<< "$TEST_FILES"

CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.predict \
  --data-dir turing-synthetic-radar-dataset/stare/stare/test_stare \
  --split test \
  --source-files "${TEST_FILE_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --output-format metrics_only \
  --devices cuda:0 \
  --batch-size 256 \
  --num-workers 8 \
  --min-cluster-fraction "$MIN_CLUSTER_FRACTION" \
  --min-samples 5 \
  --min-samples-fraction 0.0005 \
  --hdbscan-backend cuml \
  --hdbscan-devices 1 \
  --hdbscan-parallel-files 1 \
  --evaluate \
  --amp \
  --amp-dtype bf16
