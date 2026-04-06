#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/workspace"
EXPORT_ROOT="/workspace/comfy_docker_export"
CTX_DIR="$EXPORT_ROOT/context"

echo "[1/5] Rebuilding context directory: $CTX_DIR"
rm -rf "$CTX_DIR"
mkdir -p "$CTX_DIR/ComfyUI" "$CTX_DIR/workspace_files"

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required to prepare the build context."
  exit 1
fi

echo "[2/5] Copying ComfyUI with heavy mutable data excluded"
rsync -a "$SRC_ROOT/ComfyUI/" "$CTX_DIR/ComfyUI/" \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'models/' \
  --exclude 'input/' \
  --exclude 'output/' \
  --exclude 'temp/' \
  --exclude 'user/comfyui*.log' \
  --exclude 'user/__manager/cache/' \
  --exclude 'custom_nodes/*/.git/' \
  --exclude 'custom_nodes/*/__pycache__/' \
  --exclude 'custom_nodes/comfyui_controlnet_aux/ckpts/' \
  --exclude 'custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache/'

echo "[3/5] Copying selected /workspace workflow files"
mkdir -p "$CTX_DIR/workspace_files/config" "$CTX_DIR/workspace_files/scripts" "$CTX_DIR/workspace_files/workflows"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    cp -a "$src" "$dst"
  fi
}

copy_if_exists "$SRC_ROOT/comfy_workflow_model_sync.py" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/comfy_workflow_model_sync_ORIGINAL.py" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/comfy_workflow_model_sync_ORIGINAL_v02.py" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/comfy_workflow_model_sync_ORIGINAL_V03.py" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/workflow_model_report.json" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/start-comfy.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/setup_comfy_accel.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/download_wan_models.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/install_accel.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/install_ltx2_models.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/install_wan21_infinitetalk_workflow_models.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/install_wan22_i2v_ttm_workflow_models.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/install_wan22_i2v_workflow_models.sh" "$CTX_DIR/workspace_files/"
copy_if_exists "$SRC_ROOT/install_wan_acceleration.sh" "$CTX_DIR/workspace_files/"

copy_if_exists "$SRC_ROOT/config/ensure_models.py" "$CTX_DIR/workspace_files/config/"
copy_if_exists "$SRC_ROOT/config/model_manifest.yaml" "$CTX_DIR/workspace_files/config/"
copy_if_exists "$SRC_ROOT/config/start_comfy.sh" "$CTX_DIR/workspace_files/config/"

copy_if_exists "$SRC_ROOT/scripts/check_ttm_requirements.sh" "$CTX_DIR/workspace_files/scripts/"
copy_if_exists "$SRC_ROOT/scripts/workflow-models" "$CTX_DIR/workspace_files/scripts/"
copy_if_exists "$SRC_ROOT/scripts/workflow_model_manager.py" "$CTX_DIR/workspace_files/scripts/"

find "$SRC_ROOT/workflows" -maxdepth 1 -type f -name '*.json' -exec cp -a {} "$CTX_DIR/workspace_files/workflows/" \; 2>/dev/null || true

# Preserve ComfyUI user workflows/settings while skipping manager cache/logs
mkdir -p "$CTX_DIR/ComfyUI/user"
rsync -a "$SRC_ROOT/ComfyUI/user/" "$CTX_DIR/ComfyUI/user/" \
  --exclude '__manager/cache/' \
  --exclude 'comfyui*.log' \
  --exclude '*.lock' \
  --exclude '*.db*' \
  --exclude '.ipynb_checkpoints/'

echo "[4/5] Capturing environment snapshots"
if [ -x "$SRC_ROOT/comfy-venv/bin/pip" ]; then
  "$SRC_ROOT/comfy-venv/bin/pip" freeze > "$CTX_DIR/requirements_current_freeze.txt" || true
fi
if [ -x "$SRC_ROOT/comfy-venv/bin/python" ]; then
  "$SRC_ROOT/comfy-venv/bin/python" -V > "$CTX_DIR/python_version.txt" || true
fi

find "$CTX_DIR/ComfyUI/custom_nodes" -maxdepth 2 -type f -iname 'requirements*.txt' | sort > "$CTX_DIR/custom_node_requirement_files.txt" || true

{
  echo "Prepared at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Source root: $SRC_ROOT"
  echo "Excluded heavy mutable paths: ComfyUI/models, input, output, temp, custom_nodes/*/.git, comfyui_controlnet_aux/ckpts"
} > "$CTX_DIR/context_manifest.txt"

echo "[5/5] Context ready"
du -h --max-depth=2 "$CTX_DIR" | sort -hr | head -n 40
