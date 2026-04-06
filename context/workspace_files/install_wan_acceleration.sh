#!/usr/bin/env bash
set -e

echo "=============================="
echo "Wan Acceleration Installer"
echo "=============================="
echo ""

echo "Checking Python version..."
python --version
echo ""

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Installing Triton..."
pip install --upgrade triton

echo ""
echo "Installing Flash Attention (optional but recommended)..."
pip install --upgrade flash-attn --no-build-isolation || echo "Flash-Attn skipped (may require matching CUDA)."

echo ""
echo "Installing SageAttention..."
pip install --upgrade sageattention || echo "SageAttention failed (often CUDA mismatch, safe to ignore)."

echo ""
echo "Installing xformers (fallback acceleration)..."
pip install --upgrade xformers || echo "xformers skipped."

echo ""
echo "================================"
echo "Acceleration install complete."
echo "Restart ComfyUI to apply changes."
echo "================================"
