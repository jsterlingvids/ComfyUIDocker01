# ComfyUI Docker Export Manifest

Generated from live server audit on 2026-04-06 (UTC).

## Included
- `/workspace/ComfyUI` codebase and custom nodes (with selective pruning during context prep).
- Selected workflow/helper files from `/workspace`:
  - `comfy_workflow_model_sync*.py`
  - `workflow_model_report.json`
  - `/workspace/config/*` (model manifest + startup helper + model ensure script)
  - `/workspace/scripts/*` (workflow helpers)
  - install/setup shell scripts
  - workflow JSON files under `/workspace/workflows`
- Startup logic to launch ComfyUI and JupyterLab together.
- OpenAI Codex CLI (`@openai/codex`) plus `/workspace/bin/codexp` wrapper.
- Dependency install logic:
  - ComfyUI `requirements.txt`
  - custom node `requirements*.txt` (top-level node directories)
  - extra packages observed from current runtime usage

## Excluded (or redirected to runtime ephemeral storage)
- Model and checkpoint payloads, e.g. `/workspace/ComfyUI/models`.
- Mutable runtime data:
  - `/workspace/ComfyUI/input`
  - `/workspace/ComfyUI/output`
  - `/workspace/ComfyUI/temp`
- Heavy caches and non-reproducible local artifacts:
  - `.cache`, HF/pip caches, logs, `.ipynb_checkpoints`
  - `custom_nodes/*/.git`
  - `custom_nodes/comfyui_controlnet_aux/ckpts`
  - `custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache`
  - ComfyUI Manager cache/log/DB lock files under `user`

## Runtime symlink policy
At container startup (`start.sh`), these links are enforced:
- `/workspace/ComfyUI/models` -> `/opt/ephemeral/models`
- `/workspace/ComfyUI/input` -> `/opt/ephemeral/input`
- `/workspace/ComfyUI/output` -> `/opt/ephemeral/output`
- `/workspace/ComfyUI/temp` -> `/opt/ephemeral/temp`

`/opt/ephemeral/{models,input,output,temp,cache/hf,cache/torch}` are auto-created if missing.
