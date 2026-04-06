#!/usr/bin/env bash
set -e

PERSIST=/workspace
EPHEM=/opt/ephemeral

export HF_HOME=$EPHEM/cache/hf
export TORCH_HOME=$EPHEM/cache/torch
export COMFY_MODELS_DIR=$EPHEM/models
export MODEL_MANIFEST=$PERSIST/config/model_manifest.yaml

WORKFLOW_JSON="${WORKFLOW_JSON:-}"

mkdir -p $EPHEM/models $EPHEM/cache/hf $EPHEM/cache/torch
mkdir -p $PERSIST/config $PERSIST/workflows

cd $PERSIST/ComfyUI

# Ensure models are ephemeral
if [ ! -L "models" ]; then
  rm -rf models
  ln -s $EPHEM/models models
fi

# Prefetch required models if a workflow is provided
if [ -n "$WORKFLOW_JSON" ]; then
  echo "Prefetching models for workflow:"
  echo "  $WORKFLOW_JSON"
  python3 $PERSIST/config/ensure_models.py "$WORKFLOW_JSON" || true
else
  echo "No WORKFLOW_JSON set. Starting ComfyUI without prefetch."
fi

python3 main.py --listen 0.0.0.0 --port 8188
