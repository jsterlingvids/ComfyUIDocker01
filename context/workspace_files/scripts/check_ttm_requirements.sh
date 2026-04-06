#!/usr/bin/env bash
set -euo pipefail

WF_DIR="/workspace/ComfyUI/user/default/workflows"
WF="${1:-}"
if [[ -z "$WF" ]]; then
  shopt -s nullglob
  matches=("$WF_DIR"/*ttm*.json)
  shopt -u nullglob
  if [[ ${#matches[@]} -gt 0 ]]; then
    WF=$(ls -1t "${matches[@]}" | head -n1)
  fi
fi
if [[ -z "$WF" || ! -f "$WF" ]]; then
  echo "No ttm workflow JSON found."
  exit 1
fi

MODELS_BASE="/workspace/ComfyUI/models"
INPUT_BASE="/workspace/ComfyUI/input"

echo "Workflow: $WF"

mapfile -t dm_models < <(jq -r '.nodes[] | select(.type=="WanVideoModelLoader") | .widgets_values[0]' "$WF" | sed '/^null$/d' | sort -u)
mapfile -t vae_models < <(jq -r '.nodes[] | select(.type=="WanVideoVAELoader") | .widgets_values[0]' "$WF" | sed '/^null$/d' | sort -u)
mapfile -t lora_models < <(jq -r '.nodes[] | select(.type=="WanVideoLoraSelect") | .widgets_values[0]' "$WF" | sed '/^null$/d' | sort -u)
mapfile -t te_models < <(jq -r '.nodes[] | select(.type=="WanVideoTextEncodeCached") | .widgets_values[0]' "$WF" | sed '/^null$/d' | sort -u)
mapfile -t input_images < <(jq -r '.nodes[] | select(.type=="LoadImage") | .widgets_values[0]' "$WF" | sed '/^null$/d' | sort -u)

missing=0
check_list() {
  local prefix="$1"; shift
  local arr=("$@")
  for rel in "${arr[@]}"; do
    [[ -n "$rel" ]] || continue
    local full="$MODELS_BASE/$prefix/$rel"
    if [[ -f "$full" ]]; then
      echo "OK      $full"
    else
      echo "MISSING $full"
      missing=1
    fi
  done
}

check_list diffusion_models "${dm_models[@]}"
check_list vae "${vae_models[@]}"
check_list loras "${lora_models[@]}"
check_list text_encoders "${te_models[@]}"

for f in "${input_images[@]}"; do
  [[ -n "$f" ]] || continue
  if [[ -f "$INPUT_BASE/$f" ]]; then
    echo "OK      $INPUT_BASE/$f"
  else
    echo "MISSING $INPUT_BASE/$f"
    missing=1
  fi
done

if [[ "$missing" -eq 0 ]]; then
  echo "All checked requirements are present."
else
  echo "One or more requirements are missing."
fi
