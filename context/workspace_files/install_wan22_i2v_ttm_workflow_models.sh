#!/usr/bin/env bash
set -euo pipefail

# ====== Config ======
EPHEMERAL_ROOT="/opt/ephemeral/models"
COMFY_DIR="/workspace/ComfyUI"
COMFY_MODELS_DIR="$COMFY_DIR/models"

# --- HF token handling (universal) ---
# If HF_TOKEN is set, we'll use it. If not, we'll still try (public repos work).
if [ -n "${HF_TOKEN:-}" ]; then
  echo "🔐 HF_TOKEN detected (will use authenticated downloads)."
else
  echo "⚠️  HF_TOKEN is not set. Public downloads should still work, but private repos / rate limits may fail."
  echo "   Tip: export HF_TOKEN=xxxx (or set it in ~/.bashrc) for smoother downloads."
fi

# Hugging Face download helper (uses python + huggingface_hub for reliability)
py_download_hf () {
  local repo="$1"
  local file="$2"
  local dest_dir="$3"

  mkdir -p "$dest_dir"

  python3 - "$repo" "$file" "$dest_dir" <<'PY'
import os, sys

repo = sys.argv[1]
file = sys.argv[2]
dest_dir = sys.argv[3]

try:
    from huggingface_hub import hf_hub_download
except Exception:
    print("Installing huggingface_hub ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "hf_transfer"])
    from huggingface_hub import hf_hub_download

# Enable faster transfers when available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Pull token from environment (set via export HF_TOKEN=...)
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# Download (HF will resume automatically when possible)
path = hf_hub_download(
    repo_id=repo,
    filename=file,
    local_dir=dest_dir,
    token=token,  # None is fine for public repos
)

print(f"✅ {repo}/{file} -> {path}")
PY
}

# Create symlinks from ComfyUI/models -> /opt/ephemeral/models (per-category)
link_models_tree () {
  mkdir -p "$EPHEMERAL_ROOT"

  local subs=(
    "diffusion_models"
    "loras"
    "vae"
    "text_encoders"
    "clip"
  )

  mkdir -p "$COMFY_MODELS_DIR"
  for sub in "${subs[@]}"; do
    mkdir -p "$EPHEMERAL_ROOT/$sub"
    if [ -e "$COMFY_MODELS_DIR/$sub" ] && [ ! -L "$COMFY_MODELS_DIR/$sub" ]; then
      echo "⚠️  $COMFY_MODELS_DIR/$sub exists and is not a symlink. Leaving it alone."
      continue
    fi
    ln -sfn "$EPHEMERAL_ROOT/$sub" "$COMFY_MODELS_DIR/$sub"
  done
}

echo "🔧 Linking ComfyUI models -> ephemeral storage..."
link_models_tree

# ====== Downloads (from workflow) ======
echo "⬇️  Downloading Wan 2.2 I2V models (Kijai fp8 scaled)..."
py_download_hf "Kijai/WanVideo_comfy_fp8_scaled" \
  "I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
  "$EPHEMERAL_ROOT/diffusion_models/WanVideo/2_2"

py_download_hf "Kijai/WanVideo_comfy_fp8_scaled" \
  "I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
  "$EPHEMERAL_ROOT/diffusion_models/WanVideo/2_2"

echo "⬇️  Downloading Wan VAE + UMT5 encoder (Kijai/WanVideo_comfy)..."
py_download_hf "Kijai/WanVideo_comfy" \
  "Wan2_1_VAE_bf16.safetensors" \
  "$EPHEMERAL_ROOT/vae/wanvideo"

py_download_hf "Kijai/WanVideo_comfy" \
  "umt5-xxl-enc-bf16.safetensors" \
  "$EPHEMERAL_ROOT/text_encoders"

echo "⬇️  Downloading Lightx2v I2V LoRA (Kijai/WanVideo_comfy)..."
py_download_hf "Kijai/WanVideo_comfy" \
  "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
  "$EPHEMERAL_ROOT/loras/WanVideo/Lightx2v"

# Some node setups look for it in diffusion_models path too; we mirror it for compatibility.
mkdir -p "$EPHEMERAL_ROOT/diffusion_models/WanVideo/Lightx2v"
cp -n \
  "$EPHEMERAL_ROOT/loras/WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
  "$EPHEMERAL_ROOT/diffusion_models/WanVideo/Lightx2v/" 2>/dev/null || true

echo "⬇️  Downloading umt5_xxl_fp16.safetensors (Comfy-Org repackaged)..."
py_download_hf "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
  "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
  "$EPHEMERAL_ROOT/text_encoders"

# Some loaders treat this as "clip" as well; copy for max compatibility.
mkdir -p "$EPHEMERAL_ROOT/clip"
cp -n "$EPHEMERAL_ROOT/text_encoders/umt5_xxl_fp16.safetensors" "$EPHEMERAL_ROOT/clip/" 2>/dev/null || true

echo ""
echo "✅ Done."
echo "   Ephemeral models: $EPHEMERAL_ROOT"
echo "   ComfyUI models symlinked under: $COMFY_MODELS_DIR"
