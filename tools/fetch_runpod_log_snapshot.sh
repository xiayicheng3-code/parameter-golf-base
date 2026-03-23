#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-lh89r0qpwpmohi-64411d16@ssh.runpod.io}"
REMOTE_LOG="${2:-/workspace/parameter-golf/experiments/causal_deltanet/dual_causal_ttt/logs/sliding_only_smoke_4090_v2.txt}"
LOCAL_LOG="${3:-./experiments/causal_deltanet/dual_causal_ttt/logs/sliding_only_smoke_4090_v2.remote.txt}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

mkdir -p "$(dirname "$LOCAL_LOG")"

RAW_OUTPUT="$(mktemp)"
cleanup() {
  rm -f "$RAW_OUTPUT"
}
trap cleanup EXIT

ssh -tt \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -i "$SSH_KEY" \
  "$HOST" \
  >"$RAW_OUTPUT" 2>/dev/null <<EOF || true
printf '__CODEX_BEGIN__\n'
if [ -f '$REMOTE_LOG' ]; then
  cat '$REMOTE_LOG'
else
  echo 'REMOTE_LOG_MISSING'
fi
printf '\n__CODEX_END__\n'
exit
EOF

awk '
  /__CODEX_BEGIN__/ {capture=1; next}
  /__CODEX_END__/ {capture=0; exit}
  capture {print}
' "$RAW_OUTPUT" >"$LOCAL_LOG"

if grep -qx 'REMOTE_LOG_MISSING' "$LOCAL_LOG"; then
  echo "Remote log not found: $REMOTE_LOG" >&2
  exit 1
fi

echo "Saved snapshot to $LOCAL_LOG"
