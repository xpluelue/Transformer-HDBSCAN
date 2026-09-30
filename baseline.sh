#!/usr/bin/env bash
# Run one or more YAML-defined experiments.
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/桌面/xpluelue/turing-deinterleaving-challenge/turing-synthetic-radar-dataset/scan}"
DEVICE="${DEVICE:-cuda:0,1}"
N_JOBS="${N_JOBS:-48}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RUN_ROOT="${RUN_ROOT:-experiments}"
ONLY="${ONLY:-}"
TRAIN_N_JOBS="${TRAIN_N_JOBS:-$(( N_JOBS / NPROC_PER_NODE ))}"
if (( TRAIN_N_JOBS < 1 )); then
  TRAIN_N_JOBS=1
fi

# Add name:YAML pairs here. The YAML file contains all experiment/model
# parameters; each name gets its own model, evaluation result, and logs.
EXPERIMENTS=(
  "transformer_metric_scan:configs/transformer_metric_scan.yaml"
  "transformer_eda_scan:configs/transformer_eda_scan.yaml"
)

should_run() {
  local name="$1"
  [[ -z "$ONLY" || ",$ONLY," == *",$name,"* ]]
}

run_one() {
  local name="$1"
  local config="$2"
  local run_dir="${RUN_ROOT}/${name}"

  echo
  echo "===== ${name} ====="
  echo "run_dir=${run_dir}"
  echo "config=${config}"
  mkdir -p "$run_dir"
  cp "$config" "${run_dir}/config.yaml"

  python scripts/run_experiment.py \
    --config "${run_dir}/config.yaml" \
    --run-dir "$run_dir" \
    --data-dir "$DATA_DIR" \
    --device "$DEVICE" \
    --n-jobs "$N_JOBS" \
    --batch-size "$BATCH_SIZE" \
    --train-n-jobs "$TRAIN_N_JOBS" \
    --nproc-per-node "$NPROC_PER_NODE"
}

for experiment in "${EXPERIMENTS[@]}"; do
  IFS=: read -r name config <<< "$experiment"
  if [[ -z "$name" || -z "$config" ]]; then
    echo "Invalid experiment entry: $experiment (expected name:config.yaml)" >&2
    exit 2
  fi
  if should_run "$name"; then
    run_one "$name" "$config"
  fi
done
