#!/usr/bin/env bash
set -e

MODELS="/opt/ephemeral/models"

mkdir -p \
  "$MODELS/diffusion_models" \
  "$MODELS/unet" \
  "$MODELS/vae" \
  "$MODELS/loras" \
  "$MODELS/text_encoders" \
  "$MODELS/clip" \
  "$MODELS/clip/wan"

echo "Downloading Wan 2.2 I2V A14B (HIGH)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
  -O "$MODELS/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"

echo "Downloading Wan 2.2 I2V A14B (LOW)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
  -O "$MODELS/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"

echo "Linking Wan diffusion models into unet folder (compat)..."
ln -sf "$MODELS/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
       "$MODELS/unet/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
ln -sf "$MODELS/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
       "$MODELS/unet/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"

echo "Downloading Wan VAE..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
  -O "$MODELS/vae/Wan2_1_VAE_bf16.safetensors"

echo "Downloading Lightx2v I2V LoRA..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
  -O "$MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

echo "Downloading Wan text encoder (umt5-xxl-enc-bf16)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" \
  -O "$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors"

echo "Downloading CLIPLoader reference (umt5_xxl_fp16) into clip/wan/..."
wget -c "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors" \
  -O "$MODELS/clip/wan/umt5_xxl_fp16.safetensors"

echo ""
echo "✅ Wan 2.2 model installation complete."
echo ""
echo "Sanity check:"
ls -lah \
  "$MODELS/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
  "$MODELS/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
  "$MODELS/unet/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
  "$MODELS/unet/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
  "$MODELS/vae/Wan2_1_VAE_bf16.safetensors" \
  "$MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
  "$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors" \
  "$MODELS/clip/wan/umt5_xxl_fp16.safetensors" \
  2>/dev/null || true
