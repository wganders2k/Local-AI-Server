"""
Finished adapter -> GGUF LoRA that llama-server loads over the shared base.

Not a merge. A merged GGUF per persona would be a full re-quantised model each
(~7 GB at Q4_K_M for the 12B, ~18 GB for the 35B) and a rebuild on every
retrain. A GGUF adapter is tens of MB and sits on top of the one base GGUF
every persona shares — so a persona is a models.ini section, not a model file.

    python export_adapter.py --config configs/mimic_gemma4_12b.yaml --name mimic_user1

reads <output_dir>/final (only if it carries a COMPLETE marker), converts it with
llama.cpp's convert_lora_to_gguf.py at the tag pinned in the Dockerfile, and
writes /srv/models/lora/<name>-lora.gguf. CPU only — no GPU lease, safe to run
beside live inference.
"""

import argparse
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checkpoints
from train_worker import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("export-adapter")

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp")
LORA_DIR = os.environ.get("LORA_GGUF_DIR", "/srv/models/lora")


def base_config_dir(model_id: str) -> str:
    """
    The base's config files from the local HF cache — the converter reads only
    config, never weights. Cache-only on purpose: the training run already
    fetched exactly this base, and resolving it again from the Hub could pick
    up a re-upload that is not what the adapter was trained against.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        local_files_only=True,
        allow_patterns=["config.json", "generation_config.json", "tokenizer*", "*.jinja"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a finished adapter to a GGUF LoRA")
    ap.add_argument("--config", required=True, help="The training config the run used")
    ap.add_argument("--name", required=True, help="Alias it will be served as, e.g. mimic_user1")
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32", "q8_0"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    adapter = os.path.join(cfg["output_dir"], "final")
    if not checkpoints.is_complete(adapter):
        logger.error(f"{adapter} has no COMPLETE marker — the run has not finished")
        return 1

    os.makedirs(LORA_DIR, exist_ok=True)
    outfile = os.path.join(LORA_DIR, f"{args.name}-lora.gguf")
    cmd = [
        sys.executable, os.path.join(LLAMA_CPP_DIR, "convert_lora_to_gguf.py"),
        "--base", base_config_dir(cfg["model"]),
        "--outtype", args.outtype,
        "--outfile", outfile,
        adapter,
    ]
    logger.info(f"Converting {adapter} -> {outfile}")
    # cwd so the converter's own `from conversion import ...` resolves.
    result = subprocess.run(cmd, cwd=LLAMA_CPP_DIR)
    if result.returncode != 0:
        logger.error(f"Conversion failed (exit {result.returncode})")
        return result.returncode

    size_mb = os.path.getsize(outfile) / 1e6
    logger.info(f"Wrote {outfile} ({size_mb:.0f} MB). Point a models.ini section's `lora` at it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
