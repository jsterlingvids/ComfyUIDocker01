#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

log = logging.getLogger("ensure_models")


# ---------------- Logging ----------------
def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------- Workflow Parsing (best-effort) ----------------
# Standard ComfyUI loaders often use these keys.
MODEL_FIELDS = {
    "ckpt_name",
    "checkpoint",
    "checkpoint_name",
    "vae_name",
    "lora_name",
    "unet_name",
    "clip_name",
    "control_net_name",
    "controlnet_name",
    "model_name",
    "upscale_model",
    "embedding",
    "embedding_name",
}

# Map manifest logical subdirs -> ComfyUI "models/" folders
SUBDIR_MAP = {
    "checkpoints": "checkpoints",
    "loras": "loras",
    "vae": "vae",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "text_encoders": "text_encoders",
    "diffusion_models": "diffusion_models",
    "unet": "unet",
    "controlnet": "controlnet",
    "upscale_models": "upscale_models",
    "embeddings": "embeddings",
    "audio": "audio",
}


def _iter_nodes(workflow: dict):
    """
    Supports common ComfyUI export formats:
    - {"nodes":[...]}
    - {"prompt": {...}}  (API prompt format)
    - older exports where top-level dict values are nodes
    """
    if isinstance(workflow, dict) and "nodes" in workflow and isinstance(workflow["nodes"], list):
        for n in workflow["nodes"]:
            if isinstance(n, dict):
                yield n
        return

    if isinstance(workflow, dict) and "prompt" in workflow and isinstance(workflow["prompt"], dict):
        for n in workflow["prompt"].values():
            if isinstance(n, dict):
                yield n
        return

    if isinstance(workflow, dict):
        for v in workflow.values():
            if isinstance(v, dict) and ("inputs" in v or "class_type" in v or "type" in v):
                yield v


def extract_model_filenames(workflow_path: Path) -> set[str]:
    """
    Returns strings that look like model dropdown selections (often filenames).
    Note: many custom Wan/MultiTalk nodes DO NOT expose filenames in the workflow JSON.
    For those, use --preset instead.
    """
    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    needed: set[str] = set()

    for node in _iter_nodes(wf):
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        for k, v in inputs.items():
            if k in MODEL_FIELDS and isinstance(v, str) and v.strip():
                needed.add(v.strip())

    return needed


# ---------------- Manifest ----------------
def load_manifest(manifest_path: Path) -> dict:
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    models = data.get("models", {}) or {}
    if not isinstance(models, dict):
        raise ValueError("Manifest must have a top-level 'models:' mapping (dict).")
    return models


# ---------------- Download Helpers ----------------
def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], check=True):
    log.debug("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=check)


def aria2c_download(url: str, out_path: Path, dry_run=False):
    ensure_dir(out_path.parent)
    if dry_run:
        log.info("[DRY] aria2c -> %s", out_path)
        return
    run([
        "aria2c",
        "-x", "16", "-s", "16", "-k", "1M",
        "-c",
        "-d", str(out_path.parent),
        "-o", out_path.name,
        url
    ])


def wget_download(url: str, out_path: Path, dry_run=False):
    ensure_dir(out_path.parent)
    if dry_run:
        log.info("[DRY] wget -> %s", out_path)
        return
    run(["bash", "-lc", f"wget -q --show-progress -O '{out_path}' '{url}'"])


def download_hf(entry: dict, out_path: Path, dry_run=False):
    """
    Hugging Face download with "flattening":
    - download to HF cache (supports filename paths like split_files/...)
    - copy the resolved cached file to out_path (so Comfy sees a flat file)
    This prevents nested split_files folders inside your models directory.
    """
    ensure_dir(out_path.parent)
    if dry_run:
        log.info("[DRY] HF %s/%s -> %s", entry["repo"], entry["filename"], out_path)
        return

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not installed or broken: {e}")

    # Force anonymous unless you explicitly add tokens later
    # (prevents weirdness with cached/invalid auth)
    token = None

    cached_path = Path(hf_hub_download(
        repo_id=entry["repo"],
        filename=entry["filename"],
        token=token,
    ))

    # Copy to exact ComfyUI destination
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    shutil.copy2(cached_path, tmp_path)
    tmp_path.replace(out_path)


def download_civitai(entry: dict, out_path: Path, dry_run=False):
    """
    CivitAI download via model_version_id.
    Token env: CIVITAI_TOKEN (optional, recommended for rate limits/private)
    """
    ensure_dir(out_path.parent)

    v_id = int(entry["model_version_id"])
    url = f"https://civitai.com/api/download/models/{v_id}"
    token = os.environ.get("CIVITAI_TOKEN")
    if token:
        url = f"{url}?token={token}"

    if which("aria2c"):
        aria2c_download(url, out_path, dry_run=dry_run)
    else:
        wget_download(url, out_path, dry_run=dry_run)


def resolve_target_path(model_name: str, entry: dict, models_dir: Path) -> Path:
    subdir_key = entry.get("target_subdir", "checkpoints")
    subdir = SUBDIR_MAP.get(subdir_key, subdir_key)
    return models_dir / subdir / model_name


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow_json", help="Path to workflow JSON (ignored if --preset is used)")
    ap.add_argument(
        "--preset",
        default="",
        help="Preset name (e.g. wan). If set, skip workflow parsing and download all models tagged with this preset.",
    )
    ap.add_argument("--manifest", default=os.environ.get("MODEL_MANIFEST", "/workspace/config/model_manifest.yaml"))
    ap.add_argument("--models-dir", default=os.environ.get("COMFY_MODELS_DIR", "/opt/ephemeral/models"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    workflow_path = Path(args.workflow_json)
    manifest_path = Path(args.manifest)
    models_dir = Path(args.models_dir)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest = load_manifest(manifest_path)

    # Determine needed models
    if args.preset.strip():
        preset = args.preset.strip().lower()
        needed = {
            name for name, entry in manifest.items()
            if str(entry.get("preset", "")).strip().lower() == preset
        }
        log.info("Preset '%s' selected: %d model file(s).", preset, len(needed))
        if not needed:
            log.error("No models found for preset '%s'. Add 'preset: %s' to manifest entries.", preset, preset)
            sys.exit(2)
    else:
        if not workflow_path.exists():
            log.error("Workflow not found: %s", workflow_path)
            sys.exit(1)
        needed = extract_model_filenames(workflow_path)
        log.info("Workflow references %d model file(s).", len(needed))

        missing = sorted([m for m in needed if m not in manifest])
        if missing:
            log.error("Missing in manifest (%d):", len(missing))
            for m in missing:
                log.error("  - %s", m)
            sys.exit(3)

    downloaded = 0
    skipped = 0
    t0 = time.time()

    for model_name in sorted(needed):
        entry = manifest.get(model_name)
        if entry is None:
            log.error("Internal error: %s not found in manifest.", model_name)
            sys.exit(4)

        out_path = resolve_target_path(model_name, entry, models_dir)

        if out_path.exists():
            log.info("[OK] %s", model_name)
            skipped += 1
            continue

        log.info("[DL] %s -> %s", model_name, out_path)

        t = entry.get("type")
        try:
            if t == "huggingface":
                download_hf(entry, out_path, dry_run=args.dry_run)
            elif t == "civitai":
                download_civitai(entry, out_path, dry_run=args.dry_run)
            else:
                raise ValueError(f"Unknown type for {model_name}: {t}")
        except Exception as e:
            log.error("Failed downloading %s: %s", model_name, e)
            sys.exit(5)

        downloaded += 1

    log.info(
        "Summary: %d downloaded, %d already present. Total time: %.1fs",
        downloaded, skipped, time.time() - t0
    )


if __name__ == "__main__":
    main()
