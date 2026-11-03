import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# #89: local model discovery over the shared Fusion-MLX cache
# (~/.fusion-mlx/models). Two layouts live side by side there:
#   - HF cache: models--org--repo/snapshots/<commit>/... (downloaded by MLX)
#   - plain model dirs: Qwen3.8-27B-DFlash2/ etc. (converted / trained output)
# The scanner is pure filesystem + name heuristics — no MLX import, no HTTP.
# DB registration and status decisions live in server/app.py (_sync_local_models).

_HF_DIR_RE = re.compile(r"^models--(?P<org>[^/]+)--(?P<repo>[^/]+)$")

_SKIP_DIRS = {".locks", "blobs", "xet", "snapshots", "refs", ".cache"}

_MODEL_MARKERS = (
    "config.json",
    "adapter_config.json",
    "tokenizer.json",
    "model_index.json",
)

_AUDIO_HINTS = ("kokoro", "tts", "spark-tts", "musetalk", "ace-step", "whisper")
_EMBED_HINTS = ("embedding", "bge-", "clip-", "clip_", "umt5", "e5-", "face-parsing")
_IMAGE_HINTS = (
    "qwen-image",
    "flux",
    "stable-diffusion",
    "sd-vae",
    "ltx",
    "wan2",
    "catvton",
    "diffusion",
    "qwen-image-edit",
    "qwen-image-2512",
)


def _looks_like_model_dir(d: Path) -> bool:
    for marker in _MODEL_MARKERS:
        if (d / marker).is_file():
            return True
    try:
        for child in os.scandir(d):
            name = child.name.lower()
            if child.is_file() and name.endswith((".safetensors", ".gguf")):
                return True
    except OSError:
        return False
    return False


def _snapshot_dir(model_dir: Path) -> Path | None:
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    try:
        entries = sorted(os.listdir(snapshots))
    except OSError:
        return None
    for e in entries:
        snap = snapshots / e
        if snap.is_dir():
            return snap
    return None


def guess_model_type(name: str, path: Path) -> str:
    # Name+path heuristics only; the DB default for an unclassified dir is llm.
    low = f"{name} {path.name}".lower()
    if any(h in low for h in _EMBED_HINTS):
        return "embedding"
    if any(h in low for h in _AUDIO_HINTS):
        return "audio"
    if any(h in low for h in _IMAGE_HINTS):
        return "image"
    return "llm"


def discover_local_models(models_dir: str | os.PathLike) -> list[dict]:
    root = Path(models_dir).expanduser()
    if not root.is_dir():
        logger.warning("models_dir does not exist, nothing to discover: %s", root)
        return []

    found: list[dict] = []
    seen_names: set[str] = set()
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        logger.exception("models_dir unreadable: %s", root)
        return []

    for entry in entries:
        if entry.startswith("."):
            continue
        model_dir = root / entry
        if not model_dir.is_dir() or model_dir.is_symlink():
            continue

        hf = _HF_DIR_RE.match(entry)
        if hf:
            repo = f"{hf.group('org')}/{hf.group('repo')}"
            snap = _snapshot_dir(model_dir)
            if snap is None or not _looks_like_model_dir(snap):
                continue
            found.append({"name": repo, "hf_repo": repo, "path": str(snap), "kind": "hf"})
            seen_names.add(repo)
            continue

        if entry in _SKIP_DIRS:
            continue
        if not _looks_like_model_dir(model_dir):
            continue
        if entry in seen_names:
            continue
        found.append({"name": entry, "hf_repo": "", "path": str(model_dir), "kind": "local"})
        seen_names.add(entry)

    logger.info("Discovered %d local model(s) under %s", len(found), root)
    return found
