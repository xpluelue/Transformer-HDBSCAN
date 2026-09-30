#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

# Export all 250 test_scan source files in PDW Studio layout.  The clustering
# settings intentionally match test_scan_all_metrics/summary.json so that the
# generated clusters correspond to the already reported test metrics.
CHECKPOINT="${CHECKPOINT:-experiments/transformer_metric_scan_rope_rmsnorm_silu_adaptive/model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/transformer_metric_scan_rope_rmsnorm_silu_adaptive/test_scan_all_clusters}"

CUDA_VISIBLE_DEVICES=0,1 python -m turing_deinterleaving_challenge.transformer_hdbscan.predict \
  --data-dir turing-synthetic-radar-dataset/scan/scan/test_scan \
  --split test \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --output-format pdw_studio \
  --devices cuda:0 \
  --batch-size 256 \
  --num-workers 8 \
  --min-cluster-size 5 \
  --min-cluster-fraction 0.0005 \
  --min-samples 5 \
  --min-samples-fraction 0.0005 \
  --hdbscan-backend cuml \
  --hdbscan-devices 1 \
  --hdbscan-parallel-files 1 \
  --skip-evaluation \
  --amp \
  --amp-dtype bf16
