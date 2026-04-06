#!/usr/bin/env bash
set -e

echo "🔍 Checking for persistent venv..."

if [ ! -d "/workspace/venv" ]; then
  echo "📦 Creating persistent venv at /workspace/venv"
  python3 -m venv /workspace/venv
fi

source /workspace/venv/bin/activate

echo "⬆️ Upgrading pip..."
python -m pip install -U pip setuptools wheel

echo "📦 Installing SageAttention..."
python -m pip install -U sageattention

echo "🔎 Verifying torch / triton / sageattention..."

python - <<'PY'
import importlib

modules = ["torch", "triton", "sageattention"]

for m in modules:
    try:
        importlib.import_module(m)
        print(f"✅ {m} import OK")
    except Exception as e:
        print(f"❌ {m} import FAILED:", e)

import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "🎉 Setup complete. Make sure ComfyUI uses /workspace/venv."
