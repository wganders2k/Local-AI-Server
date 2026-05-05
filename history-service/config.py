"""
Configuration hub for the history-service.

Loads environment variables (from .env file or system env), provides
defaults, and defines the single source of truth for all service settings.
"""

import os

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
# Image captioning
# ──────────────────────────────────────────────────────────────

IMAGE_CAPTION_ENABLED: bool = os.environ.get("IMAGE_CAPTION_ENABLED", "false").lower() == "true"
IMAGE_CAPTION_MODEL: str = os.environ.get("IMAGE_CAPTION_MODEL", "image-caption")
IMAGE_CAPTION_BATCH_SIZE: int = int(os.environ.get("IMAGE_CAPTION_BATCH_SIZE", "10"))
IMAGE_CAPTION_WINDOW_START: int = int(os.environ.get("IMAGE_CAPTION_WINDOW_START", "3"))
IMAGE_CAPTION_WINDOW_END: int = int(os.environ.get("IMAGE_CAPTION_WINDOW_END", "6"))
IMAGE_CAPTION_MAX_FILE_SIZE_MB: int = int(os.environ.get("IMAGE_CAPTION_MAX_FILE_SIZE_MB", "10"))


# ──────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────

HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "11437"))
