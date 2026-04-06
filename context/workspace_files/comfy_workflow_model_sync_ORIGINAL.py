#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

COMMON_COMFY_DIRS = [
    "/workspace/ComfyUI",
    "/root/ComfyUI",
    "/workspace/comfyui",
    "/root/comfyui",
]

COMMON_MODEL_ROOTS = [
    "/opt/ephemeral/models",
    "/workspace/models",
]

COMMON_WORKFLOW_DIRS = [
    "/workspace/ComfyUI/user/default/workflows",
    "/workspace/ComfyUI/user/workflows",
    "/workspace/ComfyUI/workflows",
    "/workspace/workflows",
    "/root/workflows",
    "/root/ComfyUI/user/default/workflows",
    "/root/ComfyUI/user/workflows",
    "/root/ComfyUI/workflows",
]

KNOWN_MODEL_SUBDIRS = {
    "checkpoints": "checkpoints",
    "diffusion_models": "diffusion_models",
    "unet": "unet",
    "unets": "unet",
    "vae": "vae",
    "loras": "loras",
    "lora": "loras",
    "text_encoders": "text_encoders",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "controlnet": "controlnet",
    "embeddings": "embeddings",
    "upscale_models": "upscale_models",
    "vae_approx": "vae_approx",
    "gligen": "gligen",
    "photomaker": "photomaker",
    "style_models": "style_models",
    "hypernetworks": "hypernetworks",
}

LOADER_TYPE_TO_DIR_HINT = {
    "UNETLoader": "diffusion_models",
    "CheckpointLoaderSimple": "checkpoints",
    "CheckpointLoader": "checkpoints",
    "VAELoader": "vae",
    "LoraLoader": "loras",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "CLIPVisionLoader": "clip_vision",
    "ControlNetLoader": "controlnet",
}

WORKFLOW_EXTENSIONS = {".json"}

USER_AGENT = "Mozilla/5.0 (compatible; ComfyWorkflowModelSync/1.0)"
REPORT_NAME = "workflow_model_report.json"


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class ModelRef:
    node_id: int
    node_type: str
    node_mode: int
    active: bool
    name: str
    url: Optional[str]
    directory: str
    source: str  # properties.models, widgets_values, markdown, etc.

@dataclass
class ModelResult:
    name: str
    directory: str
    url: Optional[str]
    node_type: str
    node_id: int
    active: bool
    source: str
    target_path: str
    status: str
    message: str
    matched_path: Optional[str] = None
    symlink_path: Optional[str] = None


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def prompt_yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{msg} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y or n.")

def prompt_choice(msg: str, min_val: int, max_val: int, default: Optional[int] = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{msg}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if raw.isdigit():
            val = int(raw)
            if min_val <= val <= max_val:
                return val
        print(f"Please enter a number between {min_val} and {max_val}.")

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def human_bytes(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if num < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"

def find_comfy_dir() -> Optional[Path]:
    for candidate in COMMON_COMFY_DIRS:
        p = Path(candidate)
        if (p / "main.py").exists() and (p / "models").exists():
            return p

    # fallback scan
    for root in [Path("/workspace"), Path("/root")]:
        if not root.exists():
            continue
        try:
            for p in root.rglob("main.py"):
                parent = p.parent
                if (parent / "models").exists():
                    return parent
        except Exception:
            pass
    return None

def find_model_root(comfy_dir: Path) -> Path:
    for candidate in COMMON_MODEL_ROOTS:
        p = Path(candidate)
        if p.exists():
            return p
    return comfy_dir / "models"

def gather_workflow_candidates(comfy_dir: Optional[Path]) -> List[Path]:
    candidates = []

    dirs = list(COMMON_WORKFLOW_DIRS)
    if comfy_dir:
        dirs.extend([
            str(comfy_dir / "user/default/workflows"),
            str(comfy_dir / "user/workflows"),
            str(comfy_dir / "workflows"),
        ])

    seen = set()
    for d in dirs:
        p = Path(d)
        if not p.exists() or not p.is_dir():
            continue
        try:
            for file in p.rglob("*"):
                if file.is_file() and file.suffix.lower() in WORKFLOW_EXTENSIONS:
                    rp = file.resolve()
                    if str(rp) not in seen:
                        seen.add(str(rp))
                        candidates.append(rp)
        except Exception:
            pass

    candidates.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    return candidates

def choose_workflow(candidates: List[Path], explicit_workflow: Optional[str] = None) -> Path:
    if explicit_workflow:
        wf = Path(explicit_workflow).expanduser().resolve()
        if not wf.exists():
            raise FileNotFoundError(f"Workflow not found: {wf}")
        return wf

    if not candidates:
        raise FileNotFoundError("No workflow JSON files were found in the usual workflow folders.")

    most_recent = candidates[0]
    print("\nWorkflow candidates found:")
    for idx, wf in enumerate(candidates, start=1):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wf.stat().st_mtime))
        print(f"  {idx}. {wf}   (modified: {ts})")

    print(f"\nMost recent workflow:\n  1. {most_recent.name}")
    if prompt_yes_no("Use this workflow?", default=True):
        return most_recent

    choice = prompt_choice("Choose a workflow number", 1, len(candidates))
    return candidates[choice - 1]

def load_workflow_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def node_is_active(node: dict) -> bool:
    # In Comfy workflow JSON, mode=4 is commonly bypassed/muted.
    # mode=0 is active. We treat everything except 4 as active-ish.
    return int(node.get("mode", 0)) != 4

def infer_dir_from_node(node_type: str, model_name: str) -> str:
    if node_type in LOADER_TYPE_TO_DIR_HINT:
        return LOADER_TYPE_TO_DIR_HINT[node_type]

    lower = model_name.lower()
    if "lora" in lower:
        return "loras"
    if "vae" in lower:
        return "vae"
    if "clip" in lower or "umt5" in lower or "text" in lower:
        return "text_encoders"
    return "checkpoints"

def extract_models_from_properties(node: dict) -> List[ModelRef]:
    results = []
    properties = node.get("properties", {}) or {}
    models = properties.get("models", [])
    node_type = node.get("type", "Unknown")
    node_id = int(node.get("id", -1))
    mode = int(node.get("mode", 0))
    active = node_is_active(node)

    if isinstance(models, list):
        for m in models:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not name:
                continue
            directory = m.get("directory") or infer_dir_from_node(node_type, name)
            url = m.get("url")
            results.append(ModelRef(
                node_id=node_id,
                node_type=node_type,
                node_mode=mode,
                active=active,
                name=name,
                url=url,
                directory=directory,
                source="properties.models",
            ))
    return results

def extract_model_from_widgets(node: dict) -> List[ModelRef]:
    """
    Fallback for workflows that don't populate properties.models.
    This is intentionally conservative to avoid false positives.
    """
    node_type = node.get("type", "Unknown")
    node_id = int(node.get("id", -1))
    mode = int(node.get("mode", 0))
    active = node_is_active(node)
    widgets = node.get("widgets_values", [])

    if not isinstance(widgets, list) or not widgets:
        return []

    first = widgets[0]
    if not isinstance(first, str):
        return []

    first_lower = first.lower()
    if not (first_lower.endswith(".safetensors") or first_lower.endswith(".ckpt") or first_lower.endswith(".pt") or first_lower.endswith(".bin")):
        return []

    directory = infer_dir_from_node(node_type, first)
    return [ModelRef(
        node_id=node_id,
        node_type=node_type,
        node_mode=mode,
        active=active,
        name=first,
        url=None,
        directory=directory,
        source="widgets_values",
    )]

def extract_models_from_markdown(node: dict) -> List[ModelRef]:
    """
    Optional fallback: parse MarkdownNote links like:
    - [file.safetensors](https://...)
    Used carefully; marks them as inactive unless the node itself is active.
    """
    node_type = node.get("type", "")
    if node_type != "MarkdownNote":
        return []

    widgets = node.get("widgets_values", [])
    if not widgets:
        return []

    text = "\n".join(str(x) for x in widgets if isinstance(x, str))
    pattern = re.compile(
        r"\[([^\]]+\.(?:safetensors|ckpt|pt|bin))\]\((https?://[^)]+)\)",
        re.IGNORECASE
    )
    results = []
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        url = match.group(2).strip()
        directory = infer_directory_from_url_or_name(url, name)
        results.append(ModelRef(
            node_id=int(node.get("id", -1)),
            node_type=node_type,
            node_mode=int(node.get("mode", 0)),
            active=False,  # markdown notes are reference-only
            name=name,
            url=url,
            directory=directory,
            source="markdown_note",
        ))
    return results

def infer_directory_from_url_or_name(url: Optional[str], name: str) -> str:
    if url:
        for key in KNOWN_MODEL_SUBDIRS.keys():
            if f"/{key}/" in url:
                return KNOWN_MODEL_SUBDIRS[key]
    return infer_dir_from_node("Unknown", name)

def dedupe_models(model_refs: List[ModelRef], include_inactive: bool) -> List[ModelRef]:
    """
    Deduplicate by (name, directory), preferring:
    1. active node over inactive node
    2. direct properties.models over markdown/widgets
    3. entry with URL over no URL
    """
    score_source = {
        "properties.models": 3,
        "widgets_values": 2,
        "markdown_note": 1,
    }

    best: Dict[Tuple[str, str], ModelRef] = {}
    for m in model_refs:
        if not include_inactive and not m.active:
            continue

        key = (m.name, m.directory)
        current = best.get(key)
        if current is None:
            best[key] = m
            continue

        cur_score = (
            1 if current.active else 0,
            score_source.get(current.source, 0),
            1 if current.url else 0,
        )
        new_score = (
            1 if m.active else 0,
            score_source.get(m.source, 0),
            1 if m.url else 0,
        )
        if new_score > cur_score:
            best[key] = m

    return list(best.values())

def parse_workflow_models(workflow: dict, include_inactive: bool) -> List[ModelRef]:
    nodes = workflow.get("nodes", [])
    all_refs: List[ModelRef] = []
    for node in nodes:
        all_refs.extend(extract_models_from_properties(node))
        all_refs.extend(extract_model_from_widgets(node))
        all_refs.extend(extract_models_from_markdown(node))
    return dedupe_models(all_refs, include_inactive=include_inactive)

def find_existing_model(model_root: Path, comfy_dir: Optional[Path], model_ref: ModelRef) -> Tuple[Optional[Path], str]:
    """
    Search order:
    1. exact path in preferred model root
    2. exact path in Comfy models
    3. exact filename anywhere under preferred model root
    4. exact filename anywhere under Comfy models
    5. normalized filename match
    """
    exact_candidates = []
    preferred_subdir = model_root / KNOWN_MODEL_SUBDIRS.get(model_ref.directory, model_ref.directory)
    exact_candidates.append(preferred_subdir / model_ref.name)

    if comfy_dir:
        comfy_models = comfy_dir / "models" / KNOWN_MODEL_SUBDIRS.get(model_ref.directory, model_ref.directory)
        exact_candidates.append(comfy_models / model_ref.name)

    for candidate in exact_candidates:
        if candidate.exists():
            return candidate, "exact"

    search_roots = [model_root]
    if comfy_dir:
        search_roots.append(comfy_dir / "models")

    # exact filename anywhere
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob(model_ref.name):
                if p.is_file():
                    return p, "filename"
        except Exception:
            pass

    # normalized filename match
    target_norm = normalize_name(model_ref.name)
    best_partial = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".safetensors", ".ckpt", ".pt", ".bin"}:
                    continue
                candidate_norm = normalize_name(p.name)
                if candidate_norm == target_norm:
                    return p, "normalized"
                if target_norm in candidate_norm or candidate_norm in target_norm:
                    best_partial.append(p)
        except Exception:
            pass

    if len(best_partial) == 1:
        return best_partial[0], "partial"
    if len(best_partial) > 1:
        return None, f"ambiguous:{len(best_partial)}"

    return None, "missing"

def build_target_path(model_root: Path, model_ref: ModelRef) -> Path:
    subdir = KNOWN_MODEL_SUBDIRS.get(model_ref.directory, model_ref.directory)
    return model_root / subdir / model_ref.name

def create_symlink_if_needed(real_file: Path, comfy_dir: Optional[Path], model_ref: ModelRef) -> Optional[Path]:
    if comfy_dir is None:
        return None

    subdir = KNOWN_MODEL_SUBDIRS.get(model_ref.directory, model_ref.directory)
    link_path = comfy_dir / "models" / subdir / model_ref.name

    try:
        safe_mkdir(link_path.parent)
        if link_path.exists() or link_path.is_symlink():
            try:
                if link_path.resolve() == real_file.resolve():
                    return link_path
            except Exception:
                pass
            # If a different file already exists, leave it alone.
            return None
        os.symlink(real_file, link_path)
        return link_path
    except Exception:
        return None

def download_file(url: str, dest: Path, hf_token: Optional[str] = None) -> Tuple[bool, str]:
    safe_mkdir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")

    headers = {"User-Agent": USER_AGENT}
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=60) as resp:
            total = resp.headers.get("Content-Length")
            total_int = int(total) if total and total.isdigit() else None

            downloaded = 0
            chunk_size = 1024 * 1024 * 4  # 4MB

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_int:
                        pct = downloaded * 100 / total_int
                        print(f"    ... {human_bytes(downloaded)} / {human_bytes(total_int)} ({pct:.1f}%)", end="\r", flush=True)
                    else:
                        print(f"    ... {human_bytes(downloaded)}", end="\r", flush=True)

        print(" " * 100, end="\r")
        tmp.replace(dest)
        return True, "downloaded"
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False, str(e)

def summarize_results(results: List[ModelResult]):
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print("\nSummary")
    print("-------")
    for key in sorted(counts.keys()):
        print(f"{key}: {counts[key]}")

def print_model_plan(model_refs: List[ModelRef], model_root: Path):
    print("\nDetected models")
    print("---------------")
    for idx, m in enumerate(model_refs, start=1):
        target = build_target_path(model_root, m)
        active_label = "ACTIVE" if m.active else "INACTIVE"
        print(f"{idx}. {m.name}")
        print(f"   type/node: {m.node_type} (node {m.node_id})")
        print(f"   status: {active_label}")
        print(f"   directory: {m.directory}")
        print(f"   target: {target}")
        print(f"   url: {m.url or 'none'}")
        print(f"   source: {m.source}")

def save_report(report_path: Path, workflow_path: Path, comfy_dir: Optional[Path], model_root: Path, results: List[ModelResult]):
    payload = {
        "workflow": str(workflow_path),
        "comfy_dir": str(comfy_dir) if comfy_dir else None,
        "model_root": str(model_root),
        "results": [asdict(r) for r in results],
        "generated_at_epoch": int(time.time()),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def choose_mode(noninteractive: bool, cli_mode: Optional[str]) -> str:
    valid = {"active", "all", "dry-run"}
    if cli_mode:
        if cli_mode not in valid:
            raise ValueError(f"Invalid mode: {cli_mode}")
        return cli_mode

    if noninteractive:
        return "active"

    print("\nDownload mode:")
    print("  1. Active models only")
    print("  2. All referenced models")
    print("  3. Dry run only")
    choice = prompt_choice("Choose [1-3]", 1, 3, default=1)
    return {1: "active", 2: "all", 3: "dry-run"}[choice]

def resolve_and_sync_models(
    workflow_path: Path,
    workflow: dict,
    comfy_dir: Optional[Path],
    model_root: Path,
    mode: str,
    create_symlinks: bool,
    hf_token: Optional[str],
) -> List[ModelResult]:
    include_inactive = mode in {"all"}
    model_refs = parse_workflow_models(workflow, include_inactive=include_inactive)

    # For dry-run, still default to active unless explicitly requested otherwise.
    if mode == "dry-run":
        active_only = prompt_yes_no("Dry run for active models only?", default=True)
        model_refs = parse_workflow_models(workflow, include_inactive=not active_only)

    if not model_refs:
        print("No model references found in the workflow.")
        return []

    print_model_plan(model_refs, model_root)

    if mode != "dry-run":
        if not prompt_yes_no("\nProceed with this plan?", default=True):
            print("Aborted.")
            sys.exit(0)

    results: List[ModelResult] = []
    for m in model_refs:
        target = build_target_path(model_root, m)
        existing, match_kind = find_existing_model(model_root, comfy_dir, m)

        if existing:
            symlink_path = None
            if create_symlinks and comfy_dir:
                symlink = create_symlink_if_needed(existing, comfy_dir, m)
                symlink_path = str(symlink) if symlink else None

            status = "OK"
            message = f"already present ({match_kind})"
            print(f"[OK] {m.name} -> {existing} ({match_kind})")
            results.append(ModelResult(
                name=m.name,
                directory=m.directory,
                url=m.url,
                node_type=m.node_type,
                node_id=m.node_id,
                active=m.active,
                source=m.source,
                target_path=str(target),
                status=status,
                message=message,
                matched_path=str(existing),
                symlink_path=symlink_path,
            ))
            continue

        if match_kind.startswith("ambiguous:"):
            count = match_kind.split(":")[-1]
            print(f"[AMBIGUOUS] {m.name} -> multiple possible matches found ({count})")
            results.append(ModelResult(
                name=m.name,
                directory=m.directory,
                url=m.url,
                node_type=m.node_type,
                node_id=m.node_id,
                active=m.active,
                source=m.source,
                target_path=str(target),
                status="AMBIGUOUS",
                message=f"multiple possible local matches found ({count})",
            ))
            continue

        if mode == "dry-run":
            print(f"[MISSING] {m.name} -> would go to {target}")
            results.append(ModelResult(
                name=m.name,
                directory=m.directory,
                url=m.url,
                node_type=m.node_type,
                node_id=m.node_id,
                active=m.active,
                source=m.source,
                target_path=str(target),
                status="MISSING",
                message="not found locally",
            ))
            continue

        if not m.url:
            print(f"[UNRESOLVED] {m.name} -> no download URL known")
            results.append(ModelResult(
                name=m.name,
                directory=m.directory,
                url=m.url,
                node_type=m.node_type,
                node_id=m.node_id,
                active=m.active,
                source=m.source,
                target_path=str(target),
                status="UNRESOLVED",
                message="not found locally and no URL was present in workflow",
            ))
            continue

        print(f"[DOWNLOADING] {m.name}")
        ok, msg = download_file(m.url, target, hf_token=hf_token)
        if ok:
            symlink_path = None
            if create_symlinks and comfy_dir:
                symlink = create_symlink_if_needed(target, comfy_dir, m)
                symlink_path = str(symlink) if symlink else None

            print(f"[DOWNLOADED] {m.name} -> {target}")
            results.append(ModelResult(
                name=m.name,
                directory=m.directory,
                url=m.url,
                node_type=m.node_type,
                node_id=m.node_id,
                active=m.active,
                source=m.source,
                target_path=str(target),
                status="DOWNLOADED",
                message="downloaded successfully",
                matched_path=str(target),
                symlink_path=symlink_path,
            ))
        else:
            print(f"[UNRESOLVED] {m.name} -> download failed: {msg}")
            results.append(ModelResult(
                name=m.name,
                directory=m.directory,
                url=m.url,
                node_type=m.node_type,
                node_id=m.node_id,
                active=m.active,
                source=m.source,
                target_path=str(target),
                status="UNRESOLVED",
                message=f"download failed: {msg}",
            ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Detect models needed by a ComfyUI workflow and sync them to local storage.")
    parser.add_argument("--workflow", help="Path to a workflow JSON file.")
    parser.add_argument("--comfy-dir", help="Explicit ComfyUI directory.")
    parser.add_argument("--model-root", help="Explicit model root (e.g. /opt/ephemeral/models).")
    parser.add_argument("--mode", choices=["active", "all", "dry-run"], help="Download mode.")
    parser.add_argument("--yes", action="store_true", help="Non-interactive mode.")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks into ComfyUI/models for files stored elsewhere.")
    parser.add_argument("--report", help="Path to save the JSON report.")
    parser.add_argument("--hf-token", help="Hugging Face token. Falls back to HF_TOKEN env var.")
    args = parser.parse_args()

    comfy_dir = Path(args.comfy_dir).expanduser().resolve() if args.comfy_dir else find_comfy_dir()
    if comfy_dir:
        print(f"[1] Found ComfyUI at: {comfy_dir}")
    else:
        print("[1] Could not auto-detect ComfyUI. The script can still continue.")

    model_root = Path(args.model_root).expanduser().resolve() if args.model_root else find_model_root(comfy_dir) if comfy_dir else Path("/opt/ephemeral/models")
    print(f"[2] Preferred model root: {model_root}")

    candidates = gather_workflow_candidates(comfy_dir)
    workflow_path = choose_workflow(candidates, explicit_workflow=args.workflow)
    print(f"[3] Selected workflow: {workflow_path}")

    workflow = load_workflow_json(workflow_path)

    mode = choose_mode(noninteractive=args.yes, cli_mode=args.mode)
    print(f"[4] Mode: {mode}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.symlink:
        print("[5] Symlink mode: ON")
    else:
        print("[5] Symlink mode: OFF")

    safe_mkdir(model_root)

    results = resolve_and_sync_models(
        workflow_path=workflow_path,
        workflow=workflow,
        comfy_dir=comfy_dir,
        model_root=model_root,
        mode=mode,
        create_symlinks=args.symlink,
        hf_token=hf_token,
    )

    report_path = Path(args.report).expanduser().resolve() if args.report else Path.cwd() / REPORT_NAME
    save_report(report_path, workflow_path, comfy_dir, model_root, results)
    summarize_results(results)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()