#!/usr/bin/env bash
set -e

MODELS="/opt/ephemeral/models"

mkdir -p \
  "$MODELS/diffusion_models" \
  "$MODELS/unet" \
  "$MODELS/vae" \
  "$MODELS/loras" \
  "$MODELS/text_encoders" \
  "$MODELS/clip_vision" \
  "$MODELS/wav2vec2"

echo "Downloading Wan 2.1 I2V 14B 480p (GGUF Q8_0)..."
wget -c "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf" \
  -O "$MODELS/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf"

echo "Linking Wan GGUF into unet folder (compat)..."
ln -sf "$MODELS/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf" \
       "$MODELS/unet/wan2.1-i2v-14b-480p-Q8_0.gguf"

echo "Downloading InfiniteTalk (GGUF Single_Q8)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
  -O "$MODELS/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf"

echo "Linking InfiniteTalk GGUF into unet folder (compat)..."
ln -sf "$MODELS/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
       "$MODELS/unet/Wan2_1-InfiniteTalk_Single_Q8.gguf"

echo "Downloading Wan 2.1 VAE (bf16)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
  -O "$MODELS/vae/Wan2_1_VAE_bf16.safetensors"

echo "Downloading Lightx2v I2V LoRA (bf16)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
  -O "$MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

echo "Downloading UMT5 XXL encoder (bf16)..."
wget -c "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" \
  -O "$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors"

echo "Downloading CLIP Vision H..."
wget -c "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
  -O "$MODELS/clip_vision/clip_vision_h.safetensors"

echo "Downloading Wav2Vec2 Chinese (fp16 safetensors)..."
wget -c "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors" \
  -O "$MODELS/wav2vec2/wav2vec2-chinese-base_fp16.safetensors"

# MelBandRoFormer model
# (Some nodes look in diffusion_models; others look in a dedicated folder. We'll keep it in diffusion_models
#  and also add a compat symlink folder name.)
mkdir -p "$MODELS/diffusion_models/MelBandRoFormer"

echo "Downloading MelBandRoFormer (vocal separator)..."
wget -c "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors" \
  -O "$MODELS/diffusion_models/MelBandRoFormer/MelBandRoformer_fp16.safetensors"

echo ""
echo "✅ InfiniteTalk workflow model installation complete."
echo ""
echo "Sanity check:"
ls -lah \
  "$MODELS/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf" \
  "$MODELS/unet/wan2.1-i2v-14b-480p-Q8_0.gguf" \
  "$MODELS/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
  "$MODELS/unet/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
  "$MODELS/vae/Wan2_1_VAE_bf16.safetensors" \
  "$MODELS/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
  "$MODELS/text_encoders/umt5-xxl-enc-bf16.safetensors" \
  "$MODELS/clip_vision/clip_vision_h.safetensors" \
  "$MODELS/wav2vec2/wav2vec2-chinese-base_fp16.safetensors" \
  "$MODELS/diffusion_models/MelBandRoFormer/MelBandRoformer_fp16.safetensors" \
  2>/dev/null || true

echo ""
echo "🔁 If ComfyUI is running: rescan models (ComfyUI-Manager) or restart ComfyUI."
