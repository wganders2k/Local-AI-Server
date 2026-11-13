"""
Configuration hub for the history-service.

Loads environment variables (from .env file or system env), provides
defaults, and defines the single source of truth for all service settings.
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Required — no defaults
# ──────────────────────────────────────────────────────────────

DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
DISCORD_GUILD_ID: str = os.environ.get("DISCORD_GUILD_ID", "")
PROXY_URL: str = os.environ.get("PROXY_URL", "http://proxy:11436")


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

# Cold storage paths — bind-mounted from host
ARCHIVE_RAW_DIR: str = "/archive/raw"
ARCHIVE_DIR: str = "/archive/archive"
ARCHIVE_STATE_DIR: str = "/archive/state"


# ──────────────────────────────────────────────────────────────
# Channel filtering
# ──────────────────────────────────────────────────────────────

# Comma-separated list of channel IDs *or names* to exclude from DCE export.
# Example: EXCLUDED_CHANNELS=123456789012345678,bot-zone
EXCLUDED_CHANNELS: List[str] = [
    x.strip() for x in os.environ.get("EXCLUDED_CHANNELS", "").split(",") if x.strip()
]


# ──────────────────────────────────────────────────────────────
# Archive media (media_store.py) — network and disk, no GPU
# ──────────────────────────────────────────────────────────────

ARCHIVE_MEDIA_DIR: str = "/archive/media"
MEDIA_FETCH_ENABLED: bool = os.environ.get("MEDIA_FETCH_ENABLED", "false").lower() == "true"
# 200 per run every 2 min clears the 43.2k-image backlog in about a day and a
# half even if every download were slow, and keeps one run's refresh-urls
# traffic at 4 calls.
MEDIA_FETCH_BATCH_SIZE: int = int(os.environ.get("MEDIA_FETCH_BATCH_SIZE", "200"))
MEDIA_FETCH_WORKERS: int = int(os.environ.get("MEDIA_FETCH_WORKERS", "4"))
# Discord's own upload cap for a non-boosted server is 25 MB (as of 2026); the
# largest archived image is under that, so this only guards against surprises.
MEDIA_FETCH_MAX_FILE_SIZE_MB: int = int(os.environ.get("MEDIA_FETCH_MAX_FILE_SIZE_MB", "50"))
# Transient failures are retried on later runs; past this an image is left.
MEDIA_FETCH_MAX_ATTEMPTS: int = int(os.environ.get("MEDIA_FETCH_MAX_ATTEMPTS", "3"))


# ──────────────────────────────────────────────────────────────
# Image captioning (image_captioner.py) — GPU, through the proxy
# ──────────────────────────────────────────────────────────────

IMAGE_CAPTION_ENABLED: bool = os.environ.get("IMAGE_CAPTION_ENABLED", "false").lower() == "true"
# Must be an alias in models.ini AND listed in the proxy's BATCH_MODELS, so the
# backfill ranks as llm-batch and always yields to a person.
IMAGE_CAPTION_MODEL: str = os.environ.get("IMAGE_CAPTION_MODEL", "caption")
# Results are written once per batch, per user file. The proxy may hold each
# batch request until interactive traffic has been quiet for 30s, so a batch
# can take minutes; 20 keeps one batch's unwritten work small.
IMAGE_CAPTION_BATCH_SIZE: int = int(os.environ.get("IMAGE_CAPTION_BATCH_SIZE", "20"))
IMAGE_CAPTION_MAX_ATTEMPTS: int = int(os.environ.get("IMAGE_CAPTION_MAX_ATTEMPTS", "3"))
# Longest side sent to the model. Gemma 4's vision encoder resizes anyway, so
# a 4K screenshot sent whole is only a bigger request body for no gain.
IMAGE_CAPTION_MAX_SIDE: int = int(os.environ.get("IMAGE_CAPTION_MAX_SIDE", "1536"))
# How long one caption request may wait for its answer. Hours, not minutes: the
# proxy holds batch requests for as long as people are using the GPU, and a
# client timeout does not withdraw a request — it only abandons one the GPU
# will still run later (lora-training/cut_context.py's CLIENT_TIMEOUT, same
# reason). Waiting is free; the job's max_instances=1 skips runs meanwhile.
IMAGE_CAPTION_TIMEOUT_S: float = float(os.environ.get("IMAGE_CAPTION_TIMEOUT_S", str(6 * 3600)))


# ──────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────

HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "11437"))


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────


def validate_required() -> None:
    """Check required environment variables at startup."""
    missing = []
    if not DISCORD_TOKEN:
        missing.append("DISCORD_TOKEN")
    if not DISCORD_GUILD_ID:
        missing.append("DISCORD_GUILD_ID")
    if not PROXY_URL:
        missing.append("PROXY_URL")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or docker-compose.yml environment section."
        )
