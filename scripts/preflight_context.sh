#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CTX_DIR="$ROOT_DIR/context"
MAX_FILE_MB="${MAX_FILE_MB:-95}"
MAX_CTX_GB="${MAX_CTX_GB:-8}"

if [ ! -d "$CTX_DIR" ]; then
  echo "ERROR: Missing context directory: $CTX_DIR"
  echo "Run ./prepare_context.sh first."
  exit 1
fi

ctx_kb=$(du -sk "$CTX_DIR" | awk '{print $1}')
ctx_gb=$(awk -v kb="$ctx_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')

if awk -v gb="$ctx_gb" -v max="$MAX_CTX_GB" 'BEGIN { exit !(gb > max) }'; then
  echo "ERROR: context size is ${ctx_gb}GB, exceeds MAX_CTX_GB=${MAX_CTX_GB}"
  echo "Trim context exclusions in prepare_context.sh before pushing to GitHub."
  exit 1
fi

oversized="$(find "$CTX_DIR" -type f -size +"${MAX_FILE_MB}"M -print | sort || true)"
if [ -n "$oversized" ]; then
  echo "ERROR: One or more files exceed ${MAX_FILE_MB}MB in context (likely to break GitHub push limits):"
  echo "$oversized"
  exit 1
fi

echo "Preflight OK"
echo "- context dir: $CTX_DIR"
echo "- context size: ${ctx_gb}GB"
echo "- max file size check: <= ${MAX_FILE_MB}MB"
