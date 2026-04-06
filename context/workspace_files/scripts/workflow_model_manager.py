#!/usr/bin/env python3
"""
Workflow model manager for ComfyUI on RunPod.

Goals:
- List available workflow JSON files.
- Inspect model requirements for a workflow and show present/missing status.
- Download missing models to ephemeral storage (/opt/ephemeral/models by default)
  and place them under the correct ComfyUI model subfolders.

This script is intentionally pragmatic for WanVideo-style workflows and can be
extended with more node mappings over time.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_WORKFLOW_DIR = Path("/workspace/ComfyUI/user/default/workflows")
DEFAULT_COMFY_MODELS = Path("/workspace/ComfyUI/models")
DEFAULT_EPHEMERAL_MODELS = Path("/opt/ephemeral/models")

# Keep HF cache off the network volume by default.
os.environ.setdefault("HF_HOME", "/opt/ephemeral/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/opt/ephemeral/hf_home/hub")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


NODE_MODEL_MAP: Dict[str, str] = {
    "WanVideoModelLoader": "diffusion_models",
    "WanVideoVAELoader": "vae",
    "WanVideoLoraSelect": "loras",
    "WanVideoTextEncodeCached": "text_encoders",
    "LoadWanVideoT5TextEncoder": "text_encoders",
    "CLIPLoader": "text_encoders",
    "WanVideoControlnetLoader": "controlnet",
}

# Known candidate model repos to search by category.
REPO_CANDIDATES: Dict[str, List[str]] = {
    "diffusion_models": [
        "Kijai/WanVideo_comfy_fp8_scaled",
        "Kijai/WanVideo_comfy",
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    ],
    "loras": [
        "Kijai/WanVideo_comfy",
        "Kijai/WanVideo_comfy_fp8_scaled",
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    ],
    "vae": [
        "Kijai/WanVideo_comfy",
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    ],
    "text_encoders": [
        "Kijai/WanVideo_comfy",
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    ],
    "controlnet": [
        "Kijai/WanVideo_comfy",
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    ],
}


@dataclass(frozen=True)
class ModelRef:
    category: str
    relpath: str
    source_node: str

    @property
    def basename(self) -> str:
        return Path(self.relpath).name


def list_workflows(workflow_dir: Path) -> List[Path]:
    if not workflow_dir.exists():
        return []
    return sorted(workflow_dir.glob("*.json"), key=lambda p: p.name.lower())


def load_workflow(workflow_path: Path) -> dict:
    with workflow_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_str(v) -> Optional[str]:
    if isinstance(v, str):
        vv = v.strip()
        return vv if vv else None
    return None


def extract_model_refs(workflow_json: dict) -> List[ModelRef]:
    refs: Set[Tuple[str, str, str]] = set()
    nodes = workflow_json.get("nodes", [])
    if not isinstance(nodes, list):
        return []

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if not isinstance(node_type, str):
            continue

        category = NODE_MODEL_MAP.get(node_type)
        if not category:
            continue

        widgets = node.get("widgets_values")
        if not isinstance(widgets, list) or not widgets:
            continue

        model_name = _as_str(widgets[0])
        if not model_name:
            continue

        # Ignore non-model strings from CLIPLoader slot-0 in other contexts.
        if node_type == "CLIPLoader" and not (
            model_name.endswith(".safetensors")
            or model_name.endswith(".gguf")
            or model_name.endswith(".bin")
            or model_name.endswith(".ckpt")
            or model_name.endswith(".pt")
            or model_name.endswith(".pth")
        ):
            continue

        norm = model_name.replace("\\", "/").lstrip("/")
        refs.add((category, norm, node_type))

    out = [ModelRef(*x) for x in sorted(refs, key=lambda t: (t[0], t[1], t[2]))]
    return out


def ensure_models_symlink(comfy_models: Path, ephemeral_models: Path) -> None:
    # If comfy_models already points to ephemeral, nothing to do.
    if comfy_models.is_symlink():
        try:
            if comfy_models.resolve() == ephemeral_models.resolve():
                return
        except FileNotFoundError:
            pass

    # If it's a real directory, do not replace automatically.
    if comfy_models.exists() and not comfy_models.is_symlink():
        return

    comfy_models.parent.mkdir(parents=True, exist_ok=True)
    ephemeral_models.mkdir(parents=True, exist_ok=True)

    if comfy_models.exists() or comfy_models.is_symlink():
        comfy_models.unlink()
    comfy_models.symlink_to(ephemeral_models)


def model_exists(base_models_dir: Path, ref: ModelRef) -> bool:
    return (base_models_dir / ref.category / ref.relpath).is_file()


def install_missing_models(
    refs: Iterable[ModelRef],
    models_dir: Path,
    dry_run: bool = False,
) -> Tuple[List[ModelRef], List[Tuple[ModelRef, str]]]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required. Run:\n"
            "  /workspace/comfy-venv/bin/python -m pip install -U huggingface_hub hf_transfer"
        ) from e

    repo_file_cache: Dict[str, List[str]] = {}

    def get_repo_files(repo: str) -> List[str]:
        if repo not in repo_file_cache:
            repo_file_cache[repo] = list_repo_files(repo_id=repo, repo_type="model")
        return repo_file_cache[repo]

    installed: List[ModelRef] = []
    failed: List[Tuple[ModelRef, str]] = []

    for ref in refs:
        target = models_dir / ref.category / ref.relpath
        if target.is_file():
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        candidates = REPO_CANDIDATES.get(ref.category, [])
        found = False
        last_err = ""

        for repo in candidates:
            try:
                repo_files = get_repo_files(repo)
                match = next(
                    (
                        f
                        for f in repo_files
                        if f == ref.relpath
                        or f.endswith("/" + ref.relpath)
                        or Path(f).name == ref.basename
                    ),
                    None,
                )
                if not match:
                    continue

                if dry_run:
                    print(f"[DRY-RUN] {ref.relpath} <= {repo}:{match}")
                    found = True
                    break

                downloaded = hf_hub_download(
                    repo_id=repo,
                    repo_type="model",
                    filename=match,
                )
                # Copy into exact workflow-expected location.
                with open(downloaded, "rb") as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

                print(f"[OK] {target}  <=  {repo}:{match}")
                installed.append(ref)
                found = True
                break
            except Exception as ex:
                last_err = str(ex)
                continue

        if not found:
            failed.append((ref, last_err or "not found in candidate repos"))
            print(
                f"[MISS] {ref.category}/{ref.relpath} "
                f"(node={ref.source_node}) :: {last_err or 'not found'}"
            )

    return installed, failed


def pick_workflow(workflow_dir: Path, selector: str) -> Path:
    p = Path(selector)
    if p.is_file():
        return p

    if not selector.lower().endswith(".json"):
        selector_json = selector + ".json"
    else:
        selector_json = selector

    candidate = workflow_dir / selector_json
    if candidate.is_file():
        return candidate

    matches = [x for x in list_workflows(workflow_dir) if selector.lower() in x.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(f"  - {m.name}" for m in matches)
        raise ValueError(f"Ambiguous workflow selector '{selector}'. Matches:\n{names}")

    raise FileNotFoundError(f"Workflow not found: {selector}")


def cmd_list(args: argparse.Namespace) -> int:
    workflows = list_workflows(args.workflow_dir)
    if not workflows:
        print(f"No workflows found in {args.workflow_dir}")
        return 1

    print(f"Workflows in {args.workflow_dir}:")
    for i, wf in enumerate(workflows, 1):
        print(f"{i:3d}. {wf.name}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    workflow = pick_workflow(args.workflow_dir, args.workflow)
    data = load_workflow(workflow)
    refs = extract_model_refs(data)
    print(f"Workflow: {workflow}")
    if not refs:
        print("No model references found from mapped nodes.")
        return 0

    grouped: Dict[str, List[ModelRef]] = {}
    for r in refs:
        grouped.setdefault(r.category, []).append(r)

    missing_count = 0
    for cat in sorted(grouped.keys()):
        print(f"\n[{cat}]")
        for r in grouped[cat]:
            exists = model_exists(args.models_dir, r)
            mark = "OK   " if exists else "MISS "
            if not exists:
                missing_count += 1
            print(f"{mark} {r.relpath}   (from {r.source_node})")

    print(f"\nSummary: total={len(refs)} missing={missing_count} models_dir={args.models_dir}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    workflow = pick_workflow(args.workflow_dir, args.workflow)
    ensure_models_symlink(args.comfy_models_dir, args.models_dir)

    data = load_workflow(workflow)
    refs = extract_model_refs(data)
    if not refs:
        print("No model references found from mapped nodes.")
        return 0

    missing = [r for r in refs if not model_exists(args.models_dir, r)]
    print(f"Workflow: {workflow}")
    print(f"Models dir: {args.models_dir}")
    print(f"Required models: {len(refs)}, missing: {len(missing)}")
    if not missing:
        print("Nothing to download.")
        return 0

    installed, failed = install_missing_models(missing, args.models_dir, dry_run=args.dry_run)
    print("\nInstall summary:")
    print(f"- installed: {len(installed)}")
    print(f"- failed:    {len(failed)}")
    if failed:
        print("Failed items:")
        for ref, err in failed:
            print(f"  - {ref.category}/{ref.relpath} (node={ref.source_node}) :: {err}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Comfy workflow model installer (ephemeral-aware)")
    p.add_argument(
        "--workflow-dir",
        type=Path,
        default=DEFAULT_WORKFLOW_DIR,
        help=f"Workflow directory (default: {DEFAULT_WORKFLOW_DIR})",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_EPHEMERAL_MODELS,
        help=f"Target models dir (default: {DEFAULT_EPHEMERAL_MODELS})",
    )
    p.add_argument(
        "--comfy-models-dir",
        type=Path,
        default=DEFAULT_COMFY_MODELS,
        help=f"ComfyUI models path to validate/link (default: {DEFAULT_COMFY_MODELS})",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-workflows", help="List workflows in --workflow-dir")
    p_list.set_defaults(func=cmd_list)

    p_inspect = sub.add_parser("inspect-workflow", help="Show required models and missing status")
    p_inspect.add_argument("workflow", help="Workflow name, partial name, or full path")
    p_inspect.set_defaults(func=cmd_inspect)

    p_install = sub.add_parser("install-workflow", help="Download missing models for a workflow")
    p_install.add_argument("workflow", help="Workflow name, partial name, or full path")
    p_install.add_argument("--dry-run", action="store_true", help="Resolve matches without downloading")
    p_install.set_defaults(func=cmd_install)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

