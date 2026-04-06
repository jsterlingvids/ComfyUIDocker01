# ComfyUI Docker Export (RunPod/VastAI)

This export packages your current ComfyUI setup into a rebuildable image while keeping large mutable data outside the image.

## What this export does
- Uses a GPU-friendly CUDA base image.
- Rebuilds Python env in `/workspace/comfy-venv`.
- Includes your current ComfyUI repo + custom nodes + selected workflow scripts from `/workspace`.
- Installs OpenAI Codex CLI (`@openai/codex`) and provides `codexp` wrapper.
- Starts both services:
  - ComfyUI on `0.0.0.0:8188`
  - JupyterLab on `0.0.0.0:8888`
- Automatically symlinks runtime mutable paths to `/opt/ephemeral`.

## Files in this folder
- `prepare_context.sh`: stages a filtered build context from your live filesystem.
- `Dockerfile`: image build recipe.
- `start.sh`: runtime startup + symlink logic.
- `.dockerignore`: keeps build context lean.
- `MANIFEST.md`: included/excluded notes from audit.

## Build flow
1. Prepare context from your current machine state:
```bash
cd /workspace/comfy_docker_export
./prepare_context.sh
```

2. Build image:
```bash
docker build -t <dockerhub_user>/<image_name>:<tag> /workspace/comfy_docker_export
```

3. Run test locally:
```bash
docker run --gpus all --rm -it \
  -p 8188:8188 -p 8888:8888 \
  -v /opt/ephemeral:/opt/ephemeral \
  <dockerhub_user>/<image_name>:<tag>
```

## Runtime env vars
- `COMFY_DIR` (default `/opt/ComfyUI`)
- `COMFY_PORT` (default `8188`)
- `JUPYTER_PORT` (default `8888`)
- `JUPYTER_TOKEN` (default empty)
- `COMFY_ARGS` (extra args appended to `python main.py ...`)
- `EPHEM_ROOT` (default `/opt/ephemeral`)

## Codex in the container
- Binary paths:
  - `codex` from `/workspace/.npm-global/bin/codex`
  - `codexp` wrapper at `/workspace/bin/codexp`
- `codexp` sets:
  - `HOME=/workspace/home`
  - npm prefix/path for global CLI access
- Auth is not baked into the image. After container start, authenticate once:
```bash
codexp login
```

## Notes
- If any custom node dependency install fails during build, default behavior is warn-and-continue.
- To enforce strict dependency installs, build with:
```bash
docker build --build-arg CUSTOM_NODE_REQ_STRICT=1 -t <dockerhub_user>/<image_name>:<tag> /workspace/comfy_docker_export
```

## Build from GitHub Actions (recommended)
This avoids RunPod privilege limits and pushes to Docker Hub automatically.

1. Put this folder in a GitHub repo root and commit:
- `Dockerfile`, `start.sh`, `prepare_context.sh`
- `.github/workflows/dockerhub-build.yml`
- `scripts/preflight_context.sh`
- `context/` (generated via `./prepare_context.sh`)

2. In GitHub repo settings, add secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub access token)

3. Optional GitHub variable:
- `DOCKERHUB_REPO` (default fallback is `comfyui-runpod`)

4. Push to `main` or run the workflow manually (`workflow_dispatch`).

Preflight checks:
- Workflow runs `scripts/preflight_context.sh`.
- It fails if `context/` is missing, too large, or contains very large files likely to break GitHub limits.

## How To Update This Image Later
1. Update your RunPod working setup (ComfyUI/custom nodes/scripts).
2. Regenerate build context:
```bash
cd /workspace/comfy_docker_export
./prepare_context.sh
./scripts/preflight_context.sh
```
3. Commit and push changed files (`Dockerfile`, `start.sh`, `context/`, etc.) to GitHub.
4. GitHub Actions rebuilds and pushes a new tag automatically.
5. Launch new RunPod/VastAI instances using the new Docker Hub tag.
