#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IGNORE_FILE="$ROOT_DIR/resync.ignore"
if [[ -f "$ROOT_DIR/.sync.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.sync.env"
  set +a
fi
usage() {
  cat <<'EOF'
Usage: scripts/sync_workspace.sh {push-code|pull-code|pull-metrics} [--dry-run]

Run this script on the local workstation, not inside the remote server shell.

  push-code     Make server code match local code; preserve excluded server data/results.
  pull-code     Make local code match server code; preserve excluded local data/results.
  pull-metrics  Download only compact experiment metrics and run configuration files.

Required local configuration (set in the ignored .sync.env file):
  TDC_REMOTE_HOST   SSH target, for example your-user@your-server
  TDC_REMOTE_DIR    Absolute remote project directory
EOF
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

ACTION="$1"
case "$ACTION" in
  push-code|pull-code|pull-metrics) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ -z "${TDC_REMOTE_HOST:-}" || -z "${TDC_REMOTE_DIR:-}" ]]; then
  echo "Missing local synchronization configuration: $ROOT_DIR/.sync.env" >&2
  echo "Run this command from the local workstation where .sync.env is stored." >&2
  echo "The file is intentionally excluded from Git and rsync and will not exist on the server." >&2
  exit 2
fi
REMOTE_HOST="$TDC_REMOTE_HOST"
REMOTE_DIR="${TDC_REMOTE_DIR%/}"

if [[ "$REMOTE_HOST" != *@* || "$REMOTE_HOST" == @* || "$REMOTE_HOST" == *@ ]]; then
  echo "TDC_REMOTE_HOST must have the form <ssh-user>@<server-host>." >&2
  exit 2
fi
REMOTE_TARGET="${REMOTE_HOST#*@}"
case "$REMOTE_TARGET" in
  user|server|host|hostname|your-server|example.com)
    echo "TDC_REMOTE_HOST still contains a placeholder server host." >&2
    echo "Replace the text after @ in .sync.env with the real IP, DNS name, or SSH alias." >&2
    exit 2
    ;;
esac
if [[ "$REMOTE_DIR" != /* || "$REMOTE_DIR" == *:* || "$REMOTE_DIR" == *@* ]]; then
  echo "TDC_REMOTE_DIR must be an absolute server path without user@host:." >&2
  exit 2
fi

RSYNC_EXTRA=()
if [[ $# -eq 2 ]]; then
  if [[ "$2" != "--dry-run" ]]; then
    usage >&2
    exit 2
  fi
  RSYNC_EXTRA=(--dry-run --itemize-changes)
fi

case "$ACTION" in
  push-code)
    rsync -az --delete-delay \
      "${RSYNC_EXTRA[@]}" \
      --exclude-from="$IGNORE_FILE" \
      "$ROOT_DIR/" \
      "$REMOTE_HOST:$REMOTE_DIR/"
    ;;
  pull-code)
    rsync -az --delete-delay \
      "${RSYNC_EXTRA[@]}" \
      --exclude-from="$IGNORE_FILE" \
      "$REMOTE_HOST:$REMOTE_DIR/" \
      "$ROOT_DIR/"
    ;;
  pull-metrics)
    mkdir -p "$ROOT_DIR/experiments"
    rsync -az --prune-empty-dirs \
      "${RSYNC_EXTRA[@]}" \
      --include='*/' \
      --include='metrics.csv' \
      --include='_summary.csv' \
      --include='comparison.csv' \
      --include='summary.json' \
      --include='run_config.json' \
      --include='config.yaml' \
      --exclude='*' \
      "$REMOTE_HOST:$REMOTE_DIR/experiments/" \
      "$ROOT_DIR/experiments/"
    ;;
esac
