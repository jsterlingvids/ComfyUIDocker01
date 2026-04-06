#!/usr/bin/env bash
set -e

MODELS="/opt/ephemeral/models"

mkdir -p \
  "$MODELS/diffusion_models" \
  "$MODELS/clip" \
  "$MODELS/vae" \
  "$MODELS/loras" \
  "$MODELS/latent_upscale_models" \
  "$MODELS/unet"

echo "Downloading LTX2 GGUF..."
wget -c https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf \
  -O "$MODELS/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf"

echo "Linking GGUF into unet folder..."
ln -sf "$MODELS/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf" \
       "$MODELS/unet/ltx-2-19b-distilled_Q4_K_M.gguf"

echo "Downloading text encoder..."
wget -c https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors \
  -O "$MODELS/clip/gemma_3_12B_it_fp8_e4m3fn.safetensors"

echo "Downloading embeddings connector..."
wget -c https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors \
  -O "$MODELS/clip/ltx-2-19b-embeddings_connector_dev_bf16.safetensors"

echo "Downloading VAEs..."
wget -c https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_audio_vae_bf16.safetensors \
  -O "$MODELS/vae/LTX2_audio_vae_bf16.safetensors"

wget -c https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/VAE/LTX2_video_vae_bf16.safetensors \
  -O "$MODELS/vae/LTX2_video_vae_bf16.safetensors"

echo "Downloading LoRA..."
wget -c https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors \
  -O "$MODELS/loras/ltx-2-19b-distilled-lora-384.safetensors"

echo "Downloading spatial upscaler..."
wget -c https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors \
  -O "$MODELS/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"

echo ""
echo "✅ LTX2 model installation complete."
