#!/usr/bin/env python3
"""
download_models.py — Download all GGUF model files for the Local AI Server stack.

Downloads models from HuggingFace Hub into:
  /srv/models/<publisher>/<repo>/filename.gguf

The swappable-slot download list is derived from models.ini — every active
(un-commented) [section]'s `model = ` path is parsed into a repo_id, filename,
and local_dir. To add or remove a swappable model, edit models.ini only; this
script needs no matching edit. A section's HuggingFace repo id is assumed to be
its on-disk <publisher>/<repo> path, with any exception listed in
HF_REPO_OVERRIDES below.

The image server's files are named in docker-compose.yml (the `imagegen`
command:), not models.ini. IMAGEGEN_MODELS lists the defaults; any of them set
in .env (IMAGEGEN_DIFFUSION_MODEL, IMAGEGEN_TEXT_ENCODER, IMAGEGEN_VAE) is
downloaded instead.

Usage:
  python scripts/download_models.py [--models-dir /path/to/models] [--dry-run]

Requirements:
  pip install huggingface_hub

Environment variables:
  HF_TOKEN       — HuggingFace token (required for gated/private repos)
  MODELS_DIR     — Override default model storage directory (default: /srv/models)

To add or change a model:
  1. Add/edit a [section] in models.ini (image server: the IMAGEGEN_* paths
     in .env).
  2. Run: make models-download
  3. Run: make restart-llama-swappable
"""

import argparse
import configparser
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("  pip install huggingface_hub")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_INI_PATH = REPO_ROOT / "models.ini"

# Image generation (compose service `imagegen`, sd-server) — the defaults of
# docker-compose.yml's imagegen command:, not models.ini.
IMAGEGEN_MODELS = [
    {
        "repo_id":   "leejet/Z-Image-Turbo-GGUF",
        "filename":  "z_image_turbo-Q8_0.gguf",
        "local_dir": "leejet/Z-Image-Turbo-GGUF",
        "slot":      "imagegen",
        "notes":     "Z-Image-Turbo diffusion model — ~6.3 GB VRAM",
    },
    {
        "repo_id":   "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "filename":  "Qwen3-4B-Instruct-2507-Q8_0.gguf",
        "local_dir": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "slot":      "imagegen",
        "notes":     "Z-Image text encoder — ~4.1 GB VRAM",
    },
    {
        "repo_id":   "Comfy-Org/z_image_turbo",
        "filename":  "split_files/vae/ae.safetensors",
        "local_dir": "Comfy-Org/z_image_turbo",
        "slot":      "imagegen",
        "notes":     "FLUX VAE used by Z-Image",
        # hf_hub_download keeps the repo's subdirectories under local_dir
        "target":    "ae.safetensors",
    },
]

# The .env keys that override IMAGEGEN_MODELS, in the same order.
IMAGEGEN_ENV_KEYS = ["IMAGEGEN_DIFFUSION_MODEL", "IMAGEGEN_TEXT_ENCODER", "IMAGEGEN_VAE"]


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a compose .env file; comments and blanks skipped."""
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'\"")
    return env


def imagegen_models(env: dict[str, str]) -> list[dict]:
    """
    IMAGEGEN_MODELS with any .env override applied. An override is a path
    under MODELS_DIR of the form <publisher>/<repo>/<file in repo>, so it is
    also the download spec: hf_hub_download keeps the file's subdirectories
    under local_dir, which lands it exactly where compose looks.
    """
    models = []
    for key, default in zip(IMAGEGEN_ENV_KEYS, IMAGEGEN_MODELS):
        path = env.get(key)
        if not path:
            models.append(default)
            continue
        publisher, repo, filename = path.split("/", 2)
        models.append({
            "repo_id":   f"{publisher}/{repo}",
            "filename":  filename,
            "local_dir": f"{publisher}/{repo}",
            "slot":      "imagegen",
            "notes":     f"{key} (set in .env)",
        })
    return models


# Cases where a GGUF's on-disk <publisher>/<repo> path (as used in models.ini
# and MODELS_DIR) doesn't match its actual HuggingFace repo id.
HF_REPO_OVERRIDES: dict[str, str] = {
}


def load_swappable_models(models_ini_path: Path) -> list[dict]:
    """
    Derive the swappable-slot download list from models.ini's active sections.
    Commented-out (`;`-prefixed) sections are invisible to configparser and are
    correctly skipped. Includes vision projectors (`mmproj =`). De-duplicates by resolved GGUF path — several aliases
    share a file (e.g. brain/qwen3.6-35b-a3b-chat, lore/gemma4-26b-a4b-chat).
    """
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(models_ini_path)

    seen_paths: dict[str, dict] = {}
    for section in parser.sections():
        # `mmproj` is a vision preset's projector. It lives in the same repo as
        # its model, so it resolves to a repo id the same way.
        for key in ("model", "mmproj"):
            path = parser[section].get(key)
            if not path or path in seen_paths:
                continue

            # /srv/models/<publisher>/<repo>/<filename>.gguf
            rel = Path(path).relative_to("/srv/models")
            local_dir = str(rel.parent)
            repo_id = HF_REPO_OVERRIDES.get(local_dir, local_dir)

            seen_paths[path] = {
                "repo_id":   repo_id,
                "filename":  rel.name,
                "local_dir": local_dir,
                "slot":      "swappable",
                "notes":     f"[{section}] {key} and any alias sharing this file",
            }

    return list(seen_paths.values())


def sizeof_fmt(num_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def download_model(entry: dict, models_dir: Path, dry_run: bool, hf_token: str | None) -> bool:
    """
    Download a single GGUF file. Returns True on success, False on failure.
    Skips download if the file already exists at the target path.
    """
    target_dir = models_dir / entry["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / entry.get("target", entry["filename"])

    print(f"\n{'─' * 60}")
    print(f"  Model : {entry['notes']}")
    print(f"  Repo  : {entry['repo_id']}")
    print(f"  File  : {entry['filename']}")
    print(f"  Dest  : {target_path}")

    if target_path.exists():
        size = target_path.stat().st_size
        print(f"  ✓ Already downloaded ({sizeof_fmt(size)}) — skipping")
        return True

    if dry_run:
        print(f"  [DRY RUN] Would download to {target_path}")
        return True


    try:
        print("  ↓ Downloading...")
        downloaded = hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            local_dir=str(target_dir),
            token=hf_token,
        )
        # hf_hub_download may save to a cache path — move to target if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != target_path and downloaded_path.exists():
            downloaded_path.rename(target_path)

        size = target_path.stat().st_size
        print(f"  ✓ Downloaded ({sizeof_fmt(size)})")
        return True

    except RepositoryNotFoundError:
        print(f"  ✗ ERROR: Repository '{entry['repo_id']}' not found.")
        print("         Check HF_REPO_OVERRIDES in scripts/download_models.py")
        return False
    except EntryNotFoundError:
        print(f"  ✗ ERROR: File '{entry['filename']}' not found in '{entry['repo_id']}'.")
        print("         Check the model= path in models.ini")
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all GGUF model files for the Local AI Server stack."
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("MODELS_DIR", "/srv/models"),
        help="Root directory for model storage (default: /srv/models or $MODELS_DIR)",
    )
    parser.add_argument(
        "--models-ini",
        default=os.environ.get("MODELS_INI", str(MODELS_INI_PATH)),
        help="Path to models.ini (default: repo root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--slot",
        choices=["swappable", "imagegen", "all"],
        default="all",
        help="Download only models for a specific slot (default: all)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_ini_path = Path(args.models_ini).resolve()
    hf_token = os.environ.get("HF_TOKEN")

    print(f"\n{'=' * 60}")
    print("  Local AI Server — Model Downloader")
    print(f"{'=' * 60}")
    print(f"  Models dir : {models_dir}")
    print(f"  models.ini : {models_ini_path}")
    print(f"  HF token   : {'set' if hf_token else 'not set (public repos only)'}")
    if args.dry_run:
        print("  Mode       : DRY RUN — no files will be downloaded")
    if args.slot != "all":
        print(f"  Slot filter: {args.slot} only")

    env = {**read_env_file(REPO_ROOT / ".env"), **os.environ}
    all_models = load_swappable_models(models_ini_path) + imagegen_models(env)

    # Filter by slot if requested
    models_to_download = [
        m for m in all_models
        if args.slot == "all" or m["slot"] == args.slot
    ]

    print(f"\n  {len(models_to_download)} model(s) to process")

    successes = 0
    failures = 0

    for entry in models_to_download:
        ok = download_model(entry, models_dir, args.dry_run, hf_token)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"\n{'=' * 60}")
    print(f"  Done: {successes} succeeded, {failures} failed")
    print(f"{'=' * 60}\n")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
