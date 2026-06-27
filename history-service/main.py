"""
History Service — Main entry point.

FastAPI application exposing internal HTTP endpoints on :11437.
Registers APScheduler jobs for:
  (1) Image captioning batch (every 5 min, caption window)
"""

import asyncio
import glob
import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import (
    DISCORD_GUILD_ID,
    DISCORD_TOKEN,
    HOST,
    IMAGE_CAPTION_ENABLED,
    IMAGE_CAPTION_WINDOW_END,
    IMAGE_CAPTION_WINDOW_START,
    PORT,
    PROXY_URL,
)
from channel_state import load_state, reset_state
from dce_orchestrator import evaluate_and_export
from image_captioner import process_pending_batch
from jsonl_store import get_message_count, get_all_user_ids, clear_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# APScheduler instance
scheduler = AsyncIOScheduler()


# ──────────────────────────────────────────────────────────────
# Background scheduler jobs
# ──────────────────────────────────────────────────────────────


def _is_within_window(window_start: int, window_end: int) -> bool:
    """Check if current UTC hour is within the given window."""
    now = datetime.now(timezone.utc)
    hour = now.hour

    if window_start <= window_end:
        return window_start <= hour < window_end
    else:
        return hour >= window_start or hour < window_end


async def caption_batch_job() -> None:
    """
    Scheduled job: process pending image captions.

    Runs every 300 seconds but only
    executes within the caption window.
    """
    if not IMAGE_CAPTION_ENABLED:
        return

    if not _is_within_window(IMAGE_CAPTION_WINDOW_START, IMAGE_CAPTION_WINDOW_END):
        return

    try:
        count = process_pending_batch()
        if count > 0:
            logger.info(f"Caption batch: {count} images processed")
    except Exception as e:
        logger.error(f"Caption batch job failed: {e}")


# ──────────────────────────────────────────────────────────────
# FastAPI lifespan
# ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    from config import validate_required
    validate_required()
    logger.info("History service starting up")

    if IMAGE_CAPTION_ENABLED:
        scheduler.add_job(
            caption_batch_job,
            "interval",
            seconds=300,
            id="caption_batch",
            name="Image captioning batch",
            max_instances=1,
        )
        logger.info(
            f"Caption batch job registered: every 300s, "
            f"window {IMAGE_CAPTION_WINDOW_START}:00–{IMAGE_CAPTION_WINDOW_END}:00 UTC"
        )

    scheduler.start()
    logger.info("Scheduler started")

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")
    logger.info("History service shut down")


app = FastAPI(
    title="History Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────
# HTTP Endpoints
# ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
async def status() -> dict:
    """Service status and archive summary."""
    user_ids = get_all_user_ids()
    user_counts = {uid: get_message_count(uid) for uid in user_ids}

    return {
        "users_in_archive": len(user_ids),
        "user_counts": user_counts,
    }


@app.post("/evaluate")
async def evaluate() -> dict:
    """
    Trigger channel evaluation + targeted DCE exports.

    Called by host cron (monthly). Evaluates all channels in the
    guild, invokes DCE for channels with activity since last export,
    merges raw exports into per-user JSONL archive, and notifies
    lora-training service of new data.
    """
    logger.info("Channel evaluation triggered via POST /evaluate")

    try:
        # Fetch channels (blocking HTTP call) in thread pool
        channels = await asyncio.to_thread(_fetch_discord_channels)

        if not channels:
            return {
                "status": "error",
                "message": "Failed to fetch Discord channels",
            }

        logger.info(f"Fetched {len(channels)} channels from guild {DISCORD_GUILD_ID}")

        # Run export pipeline (blocking subprocess calls) in thread pool
        new_message_counts = await asyncio.to_thread(evaluate_and_export, channels)

        # Notify lora-training (blocking HTTP call) in thread pool
        await asyncio.to_thread(_notify_lora_training, new_message_counts)

        return {
            "status": "complete",
            "channels_evaluated": len(channels),
            "users_updated": len(new_message_counts),
            "total_new_messages": sum(new_message_counts.values()),
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/reparse")
async def reparse() -> dict:
    """
    Re-parse all existing raw DCE export files into the JSONL archive.

    Scans /archive/raw/ for all timestamped export directories, parses
    every JSON file found using the DCE parser, and appends new records
    to the per-user JSONL archive. Skips duplicates via message_id.

    This endpoint does NOT call the Discord API or invoke DCE subprocesses.
    Use it to iterate on parser changes without waiting for fresh exports.
    """
    logger.info("Re-parse triggered via POST /reparse")

    try:
        from dce_parser import parse_dce_export_directory
        from jsonl_store import append_messages as _append_messages

        raw_dir = os.path.join("/archive", "raw")
        if not os.path.isdir(raw_dir):
            return {
                "status": "error",
                "message": f"Raw exports directory does not exist: {raw_dir}",
            }

        # Collect all timestamped export directories
        export_dirs = sorted([
            d for d in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, d))
        ])

        if not export_dirs:
            return {
                "status": "complete",
                "message": "No export directories found in raw/",
                "directories_scanned": 0,
                "total_records_parsed": 0,
                "total_records_appended": 0,
                "users_updated": 0,
            }

        total_parsed = 0
        total_appended = 0
        users_touched: set = set()

        for dirname in export_dirs:
            dirpath = os.path.join(raw_dir, dirname)
            logger.info(f"Parsing export directory: {dirpath}")

            # Parse returns {user_id: [records]}
            user_records = await asyncio.to_thread(parse_dce_export_directory, dirpath)
            total_parsed += sum(len(recs) for recs in user_records.values())

            for uid, records in user_records.items():
                if records:
                    appended = await asyncio.to_thread(_append_messages, uid, records)
                    total_appended += appended
                    users_touched.add(uid)

        return {
            "status": "complete",
            "directories_scanned": len(export_dirs),
            "total_records_parsed": total_parsed,
            "total_records_appended": total_appended,
            "users_updated": len(users_touched),
        }

    except Exception as e:
        logger.error(f"Re-parse failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/archive/{user_id}/count")
async def archive_count(user_id: str) -> dict:
    """Message count for a user in the archive."""
    count = get_message_count(user_id)
    return {
        "user_id": user_id,
        "message_count": count,
    }


@app.post("/clear")
async def clear_history() -> dict:
    """
    Clear all retrieved history data.

    Deletes:
      - All per-user JSONL archive files
      - All raw DCE export files
      - Channel state tracking data
    Resets:
      - In-memory message ID cache
    """
    from config import ARCHIVE_RAW_DIR

    logger.info("Clear history triggered via POST /clear")

    # 1. Clear user archives
    archive_count = clear_all()

    # 2. Clear raw DCE exports
    raw_deleted = 0
    if os.path.exists(ARCHIVE_RAW_DIR):
        pattern = os.path.join(ARCHIVE_RAW_DIR, "*")
        for f in glob.glob(pattern):
            try:
                if os.path.isfile(f):
                    os.unlink(f)
                    raw_deleted += 1
                elif os.path.isdir(f):
                    shutil.rmtree(f)
                    raw_deleted += 1
            except PermissionError:
                logger.warning(
                    f"Permission denied skipping {f} — "
                    "may need matching file ownership"
                )
            except OSError as e:
                logger.warning(f"Failed to delete raw export {f}: {e}")

    # 3. Reset channel state tracking
    reset_state()

    logger.info(
        f"History cleared: {archive_count} archives, "
        f"{raw_deleted} raw exports, channel state reset"
    )

    return {
        "status": "cleared",
        "archives_deleted": archive_count,
        "raw_exports_deleted": raw_deleted,
        "channel_state": "reset",
    }


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────


def _snowflake_to_timestamp(snowflake_id):
    """Extract UTC timestamp from a Discord Snowflake ID."""
    if not snowflake_id:
        return None
    epoch = 1420070400000  # Discord epoch (ms)
    timestamp_ms = (int(snowflake_id) >> 22) + epoch
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return dt.isoformat()


def _fetch_discord_channels() -> list:
    """
    Fetch channels from the Discord guild via the Discord API.

    Returns list of channel dicts with 'id', 'name', and
    'last_message_at' (ISO 8601 or None, converted from snowflake ID).
    """
    token = DISCORD_TOKEN
    guild_id = DISCORD_GUILD_ID

    if not token or not guild_id:
        logger.error("DISCORD_TOKEN or DISCORD_GUILD_ID not configured")
        return []

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"https://discord.com/api/v10/guilds/{guild_id}/channels",
                headers={"Authorization": f"Bot {token}"},
            )

            if resp.status_code != 200:
                logger.error(
                    f"Failed to fetch channels: HTTP {resp.status_code} — {resp.text[:200]}"
                )
                return []

            raw_channels = resp.json()
            channels = []

            for ch in raw_channels:
                # Only include text channels (type 0) and announcement channels (type 5)
                ch_type = ch.get("type", 0)
                if ch_type not in (0, 5, 15):  # text, announcement, forum
                    continue

                channels.append({
                    "id": ch["id"],
                    "name": ch.get("name", ""),
                    "last_message_at": _snowflake_to_timestamp(ch.get("last_message_id")),
                })

            return channels

    except Exception as e:
        logger.error(f"Failed to fetch Discord channels: {e}")
        return []


def _notify_lora_training(new_message_counts: dict) -> None:
    """
    Notify lora-training service that new data is available.

    Args:
        new_message_counts: Mapping of user_id → new message count.
    """
    if not new_message_counts:
        return

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                "http://lora-training:11438/notify",
                json={"users": list(new_message_counts.keys())},
            )
            if resp.status_code == 200:
                logger.info(f"Notified lora-training of {len(new_message_counts)} users with new data")
            else:
                logger.warning(f"lora-training notify returned HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to notify lora-training: {e}")


# ──────────────────────────────────────────────────────────────
# Run with uvicorn
# ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
