#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
EPHEM_ROOT="${EPHEM_ROOT:-/opt/ephemeral}"
COMFY_PORT="${COMFY_PORT:-8188}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
JUPYTER_ROOT_DIR="${JUPYTER_ROOT_DIR:-/workspace}"
COMFY_ARGS="${COMFY_ARGS:-}"

MODELS_DIR="$EPHEM_ROOT/models"
INPUT_DIR="$EPHEM_ROOT/input"
OUTPUT_DIR="$EPHEM_ROOT/output"
TEMP_DIR="$EPHEM_ROOT/temp"
HF_CACHE_DIR="$EPHEM_ROOT/cache/hf"
TORCH_CACHE_DIR="$EPHEM_ROOT/cache/torch"

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TORCH_HOME="$TORCH_CACHE_DIR"

mkdir -p "$MODELS_DIR" "$INPUT_DIR" "$OUTPUT_DIR" "$TEMP_DIR" "$HF_CACHE_DIR" "$TORCH_CACHE_DIR"
mkdir -p /workspace

if [ ! -d "$COMFY_DIR" ]; then
  echo "ERROR: COMFY_DIR not found: $COMFY_DIR"
  exit 1
fi

ensure_link() {
  local target="$1"
  local link_path="$2"
  mkdir -p "$(dirname "$link_path")"

  if [ -L "$link_path" ]; then
    return
  fi

  if [ -e "$link_path" ]; then
    if [ -d "$link_path" ] && [ "$(ls -A "$link_path" 2>/dev/null || true)" != "" ]; then
      echo "Syncing existing data from $link_path to $target"
      cp -an "$link_path"/. "$target"/ || true
    fi
    rm -rf "$link_path"
  fi

  ln -s "$target" "$link_path"
}

ensure_link "$MODELS_DIR" "$COMFY_DIR/models"
ensure_link "$INPUT_DIR" "$COMFY_DIR/input"
ensure_link "$OUTPUT_DIR" "$COMFY_DIR/output"
ensure_link "$TEMP_DIR" "$COMFY_DIR/temp"

cd "$COMFY_DIR"
if [ ! -f main.py ]; then
  echo "ERROR: main.py not found in COMFY_DIR: $COMFY_DIR"
  exit 1
fi

jupyter lab \
  --ip=0.0.0.0 \
  --port="$JUPYTER_PORT" \
  --no-browser \
  --allow-root \
  --ServerApp.token="$JUPYTER_TOKEN" \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir="$JUPYTER_ROOT_DIR" \
  > /workspace/jupyter.log 2>&1 &
JUPYTER_PID=$!

python main.py --listen 0.0.0.0 --port "$COMFY_PORT" $COMFY_ARGS > /workspace/comfy.log 2>&1 &
COMFY_PID=$!

cleanup() {
  kill "$COMFY_PID" "$JUPYTER_PID" 2>/dev/null || true
  wait "$COMFY_PID" "$JUPYTER_PID" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM

wait -n "$COMFY_PID" "$JUPYTER_PID"
exit $?
