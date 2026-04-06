#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"

# Try common venv locations (adjust if yours is different)
if [ -d "/workspace/venv" ]; then
  VENV="/workspace/venv"
elif [ -d "$COMFY_DIR/venv" ]; then
  VENV="$COMFY_DIR/venv"
else
  echo "❌ Couldn't find a venv in /workspace/venv or $COMFY_DIR/venv"
  echo "   Create one on /workspace to make installs persistent:"
  echo "   python3 -m venv /workspace/venv"
  exit 1
fi

echo "✅ Using venv: $VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel

echo "📦 Installing/Updating SageAttention (PyPI: sageattention)"
python -m pip install -U sageattention

echo "📦 Ensuring Triton is available (often comes with torch on Linux)"
python - <<'PY'
import importlib, sys
ok = True
for m in ["torch", "triton", "sageattention"]:
    try:
        importlib.import_module(m)
        print(f"✅ import {m} OK")
    except Exception as e:
        ok = False
        print(f"❌ import {m} FAILED: {e}")
if not ok:
    sys.exit(2)

import torch
print("---- Versions ----")
print("torch:", torch.__version__)
try:
    import triton
    print("triton:", triton.__version__)
except Exception as e:
    print("triton: (import failed at version print)", e)
try:
    import sageattention
    print("sageattention:", getattr(sageattention, "__version__", "unknown"))
except Exception as e:
    print("sageattention: (import failed at version print)", e)

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "✅ Done. If ComfyUI uses this venv, it will see these packages."
echo "   Next: start ComfyUI from the same venv (or ensure your comfy start activates it)."
