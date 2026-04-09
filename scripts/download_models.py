#!/usr/bin/env python3
"""
download_models.py — Download all GGUF model files for the Local AI Server stack.

Downloads models from HuggingFace Hub into:
  /srv/models/<publisher>/<repo>/filename.gguf

Usage:
  python scripts/download_models.py [--models-dir /path/to/models] [--dry-run]

Requirements:
  pip install huggingface_hub

Environment variables:
  HF_TOKEN       — HuggingFace token (required for gated/private repos)
  MODELS_DIR     — Override default model storage directory (default: /srv/models)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("  pip install huggingface_hub")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Model manifest
# Each entry defines one GGUF file to download.
#
# Fields:
#   repo_id    — HuggingFace repo (owner/repo-name)
#   filename   — exact filename within the repo
#   local_dir  — subdirectory under MODELS_DIR to save into
#                format: <publisher>/<model-name>
#   slot       — "permanent" or "swappable" (informational only)
#   notes      — human-readable description
# ──────────────────────────────────────────────────────────────────────────────
MODELS = [
    # ── Permanent slot (:11435) ──────────────────────────────────────────────
    {
        "repo_id":   "unsloth/Qwen3.5-2B-GGUF",
        "filename":  "Qwen3.5-2B-IQ4_NL.gguf",
        "local_dir": "unsloth/Qwen3.5-2B-GGUF",
        "slot":      "permanent",
        "notes":     "Autocomplete model — ~1.21 GB VRAM",
    },
    {
        "repo_id":   "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF",
        "filename":  "qwen2.5-coder-1.5b-q8_0.gguf",
        "local_dir": "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF",
        "slot":      "permanent",
        "notes":     "Autocomplete model — ~1.65 GB VRAM",
    },

    # ── Swappable slot (:11434) ──────────────────────────────────────────────
    {
        "repo_id":   "unsloth/Qwen3.5-35B-A3B-GGUF",
        "filename":  "Qwen3.5-35B-A3B-UD-IQ4_NL.gguf",
        "local_dir": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "slot":      "swappable",
        "notes":     "Brain (coding assistant) + openwebui chat (shared GGUF, different params) — ~17.8 GB VRAM",
    },
    {
        "repo_id":   "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "filename":  "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "local_dir": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "slot":      "swappable",
        "notes":     "Agent (coding assistant)— ~17.8 GB VRAM",
    },
    {
        "repo_id":   "HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive",
        "filename":  "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
        "local_dir": "HauhauCS/Qwen3.5-35B-A3B-Uncensored",
        "slot":      "swappable",
        "notes":     "Mimic personas + image captioning — ~18 GB VRAM (shared GGUF)",
    },
    {
        "repo_id":   "unsloth/gemma-3-12b-it-GGUF",
        "filename":  "gemma-3-12b-it-UD-Q8_K_XL.gguf",
        "local_dir": "unsloth/gemma-3-12b-it-GGUF",
        "slot":      "swappable",
        "notes":     "Lore assistant — ~14.4 GB VRAM",
    },
]


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
    target_path = target_dir / entry["filename"]

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
        print(f"  ↓ Downloading...")
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
        print(f"         Check the repo_id in scripts/download_models.py")
        return False
    except EntryNotFoundError:
        print(f"  ✗ ERROR: File '{entry['filename']}' not found in '{entry['repo_id']}'.")
        print(f"         Check the filename in scripts/download_models.py")
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
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--slot",
        choices=["permanent", "swappable", "all"],
        default="all",
        help="Download only models for a specific slot (default: all)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    hf_token = os.environ.get("HF_TOKEN")

    print(f"\n{'=' * 60}")
    print(f"  Local AI Server — Model Downloader")
    print(f"{'=' * 60}")
    print(f"  Models dir : {models_dir}")
    print(f"  HF token   : {'set' if hf_token else 'not set (public repos only)'}")
    if args.dry_run:
        print(f"  Mode       : DRY RUN — no files will be downloaded")
    if args.slot != "all":
        print(f"  Slot filter: {args.slot} only")

    # Filter by slot if requested
    models_to_download = [
        m for m in MODELS
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
