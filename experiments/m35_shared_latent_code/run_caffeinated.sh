#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 1
fi

if ! command -v caffeinate >/dev/null 2>&1; then
    exec "$@"
fi

# Keep the machine awake for the lifetime of the command without forcing the
# display to stay on overnight.
exec caffeinate -i -m -s "$@"
