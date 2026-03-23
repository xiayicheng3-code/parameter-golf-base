#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCH_SCRIPT="$SCRIPT_DIR/fetch_runpod_log_snapshot.sh"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
HOST="${HOST:-lh89r0qpwpmohi-64411d16@ssh.runpod.io}"
REMOTE_LOG="${REMOTE_LOG:-/workspace/parameter-golf/experiments/causal_deltanet/dual_causal_ttt/logs/sliding_only_smoke_4090_v2.txt}"
LOCAL_LOG="${LOCAL_LOG:-./experiments/causal_deltanet/dual_causal_ttt/logs/sliding_only_smoke_4090_v2.remote.txt}"

while true; do
  "$FETCH_SCRIPT" "$HOST" "$REMOTE_LOG" "$LOCAL_LOG" || true
  date +"[%Y-%m-%d %H:%M:%S] snapshot updated"
  sleep "$INTERVAL_SECONDS"
done
