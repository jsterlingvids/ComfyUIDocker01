#!/usr/bin/env bash
set -Eeuo pipefail

: "${DIFFUSION_DIR:?Set DIFFUSION_DIR first}"
: "${VAE_DIR:?Set VAE_DIR first}"
: "${TEXT_ENCODER_DIR:?Set TEXT_ENCODER_DIR first}"

echo "Using model paths:"
echo "  diffusion:    $DIFFUSION_DIR"
echo "  vae:          $VAE_DIR"
echo "  text encoder: $TEXT_ENCODER_DIR"

mkdir -p "$DIFFUSION_DIR" "$VAE_DIR" "$TEXT_ENCODER_DIR"

python -m pip install -U huggingface_hub

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

downloads = [
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "WanVideo/wan2.1_t2v_1.3B_fp16.safetensors",
        "local_dir": os.environ["DIFFUSION_DIR"],
        "expected_name": "wan2.1_t2v_1.3B_fp16.safetensors",
    },
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "Wan2_1_VAE_bf16.safetensors",
        "local_dir": os.environ["VAE_DIR"],
        "expected_name": "Wan2_1_VAE_bf16.safetensors",
    },
    {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "umt5-xxl-enc-bf16.safetensors",
    },
]

for item in downloads:
    hf_hub_download(
        repo_id=item["repo_id"],
        filename=item["filename"],
        local_dir=item["local_dir"],
        local_dir_use_symlinks=False,
    )
    expected_path = os.path.join(item["local_dir"], item["expected_name"])
    if not os.path.exists(expected_path):
        raise RuntimeError(f"Expected file missing: {expected_path}")
    print(f"OK: {expected_path}")
PY

echo
echo "Final files:"
find "$DIFFUSION_DIR" "$VAE_DIR" "$TEXT_ENCODER_DIR" -maxdepth 2 -type f | sort
