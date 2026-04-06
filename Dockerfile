FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/workspace/comfy-venv \
    PATH=/workspace/comfy-venv/bin:/workspace/.npm-global/bin:/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    nodejs \
    npm \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    portaudio19-dev \
    imagemagick \
    libmagickwand-dev \
    poppler-utils \
    rsync \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VENV_PATH" && \
    pip install --upgrade pip setuptools wheel uv

RUN mkdir -p /workspace/.npm-global /workspace/home && \
    npm config set prefix /workspace/.npm-global && \
    npm install -g @openai/codex@0.116.0

WORKDIR /workspace

COPY context/ComfyUI /workspace/ComfyUI
COPY context/workspace_files /workspace/workspace_files
COPY context/custom_node_requirement_files.txt /workspace/custom_node_requirement_files.txt
COPY context/requirements_current_freeze.txt /workspace/requirements_current_freeze.txt
COPY context/context_manifest.txt /workspace/context_manifest.txt
COPY start.sh /start.sh

RUN chmod +x /start.sh

RUN mkdir -p /workspace/bin && cat > /workspace/bin/codexp <<'EOS' && chmod +x /workspace/bin/codexp
#!/usr/bin/env bash
export HOME=/workspace/home
export NPM_CONFIG_PREFIX=/workspace/.npm-global
export PATH=/workspace/.npm-global/bin:/workspace/bin:$PATH
exec codex "$@"
EOS

# Install core ComfyUI dependencies from the audited setup snapshot.
RUN pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt

# Install top-level custom node dependencies. Failures can be tolerated for optional nodes.
ARG CUSTOM_NODE_REQ_STRICT=0
RUN bash -lc 'set -euo pipefail; \
  shopt -s nullglob; \
  for req in /workspace/ComfyUI/custom_nodes/*/requirements*.txt; do \
    echo "Installing $req"; \
    if ! pip install --no-cache-dir -r "$req"; then \
      if [ "$CUSTOM_NODE_REQ_STRICT" = "1" ]; then \
        echo "Failed requirement file: $req"; exit 1; \
      else \
        echo "WARN: skipped failing requirement file: $req"; \
      fi; \
    fi; \
  done'

# Extra packages observed in the live environment via ComfyUI-Manager actions + Jupyter.
RUN pip install --no-cache-dir \
  jupyterlab notebook ipykernel \
  librosa sounddevice glitch-this PyOpenGL glfw moviepy matplotlib reportlab openai PyPDF2 pdf2image Wand

# Restore selected workflow/helper files into /workspace.
RUN cp -a /workspace/workspace_files/. /workspace/

EXPOSE 8188 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
