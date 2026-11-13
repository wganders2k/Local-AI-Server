"""
History Service — Main entry point.

FastAPI application exposing internal HTTP endpoints on :11437.
Registers APScheduler jobs for:
  (1) Archive media fetch (every 2 min, MEDIA_FETCH_ENABLED)
  (2) Image captioning batch (every minute, IMAGE_CAPTION_ENABLED)
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
from fastapi import FastAPI

from config import (
    DISCORD_GUILD_ID,
    DISCORD_TOKEN,
    EXCLUDED_CHANNELS,
    HOST,
    IMAGE_CAPTION_ENABLED,
    MEDIA_FETCH_ENABLED,
    PORT,
)
from channel_state import reset_state
from dce_orchestrator import evaluate_and_export
from image_captioner import process_pending_batch
from media_store import fetch_batch
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


async def media_fetch_job() -> None:
    """Scheduled job: keep a local copy of image attachments (media_store)."""
    try:
        await asyncio.to_thread(fetch_batch)
    except Exception as e:
        logger.error(f"Media fetch job failed: {e}", exc_info=True)


async def caption_batch_job() -> None:
    """
    Scheduled job: caption stored images (image_captioner).

    No time window: the proxy ranks these requests as batch work and holds
    them while anyone is using the GPU, so the job can run around the clock.
    In a worker thread because a batch blocks for minutes on the proxy, and
    on the event loop that would stall every HTTP endpoint with it.
    """
    try:
        await asyncio.to_thread(process_pending_batch)
    except Exception as e:
        logger.error(f"Caption batch job failed: {e}", exc_info=True)


# ──────────────────────────────────────────────────────────────
# FastAPI lifespan
# ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    from config import validate_required
    validate_required()
    logger.info("History service starting up")

    if MEDIA_FETCH_ENABLED:
        scheduler.add_job(
            media_fetch_job, "interval", seconds=120,
            id="media_fetch", name="Archive media fetch", max_instances=1,
        )
        logger.info("Media fetch job registered: every 120s")

    if IMAGE_CAPTION_ENABLED:
        scheduler.add_job(
            caption_batch_job, "interval", seconds=60,
            id="caption_batch", name="Image captioning batch", max_instances=1,
        )
        logger.info("Caption batch job registered: every 60s (proxy-ranked batch work)")

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
    and merges raw exports into per-user JSONL archive.
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

    Messages already archived get the parser-owned fields (type, reference,
    is_pinned, mentions) refreshed from the parse, so a parser change reaches
    old records too — re-running this after one is the backfill. Attachments
    get their parser-owned fields (id, url, name, size, type) refreshed the
    same way; their captions and downloaded media are never touched. Nothing
    else on an archived record is. Where a message appears in several
    exports, the latest export wins.

    This endpoint does NOT call the Discord API or invoke DCE subprocesses.
    Use it to iterate on parser changes without waiting for fresh exports.
    """
    logger.info("Re-parse triggered via POST /reparse")

    try:
        from dce_parser import parse_dce_export_directory
        from dce_parser import ATTACHMENT_PARSER_FIELDS, PARSER_OWNED_FIELDS
        from jsonl_store import refresh_attachments as _refresh_attachments
        from jsonl_store import append_messages as _append_messages
        from jsonl_store import refresh_fields as _refresh_fields

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
                "total_records_refreshed": 0,
                "users_updated": 0,
            }

        total_parsed = 0
        excluded = 0
        # user_id -> message_id -> record. Gathered across every export before
        # writing, so each user's file is rewritten at most once.
        latest: dict = {}

        for dirname in export_dirs:
            dirpath = os.path.join(raw_dir, dirname)
            logger.info(f"Parsing export directory: {dirpath}")

            # Parse returns {user_id: [records]}
            user_records = await asyncio.to_thread(parse_dce_export_directory, dirpath)
            total_parsed += sum(len(recs) for recs in user_records.values())
            for uid, records in user_records.items():
                by_id = latest.setdefault(uid, {})
                for rec in records:
                    # The raw exports predate some exclusions, so the rule
                    # /evaluate applies before exporting is applied here too.
                    if rec.get("channel_id") in EXCLUDED_CHANNELS or rec.get("channel_name") in EXCLUDED_CHANNELS:
                        excluded += 1
                        continue
                    by_id[rec["message_id"]] = rec

        total_appended = 0
        total_refreshed = 0
        total_att_refreshed = 0
        users_touched: set = set()
        for uid, by_id in latest.items():
            records = sorted(by_id.values(), key=lambda r: (r.get("timestamp", ""), r["message_id"]))
            appended = await asyncio.to_thread(_append_messages, uid, records)
            refreshed = await asyncio.to_thread(_refresh_fields, uid, records, PARSER_OWNED_FIELDS)
            att_refreshed = await asyncio.to_thread(_refresh_attachments, uid, records, ATTACHMENT_PARSER_FIELDS)
            total_appended += appended
            total_refreshed += refreshed
            total_att_refreshed += att_refreshed
            if appended or refreshed or att_refreshed:
                users_touched.add(uid)

        return {
            "status": "complete",
            "directories_scanned": len(export_dirs),
            "total_records_parsed": total_parsed,
            "total_records_appended": total_appended,
            "total_records_refreshed": total_refreshed,
            "total_attachment_records_refreshed": total_att_refreshed,
            "total_records_excluded": excluded,
            "users_updated": len(users_touched),
        }

    except Exception as e:
        logger.error(f"Re-parse failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/media/status")
async def media_status() -> dict:
    """
    Progress of the media fetch and caption backfill: attachment counts by
    media_status and by caption_status, across the whole archive.
    """
    def count() -> dict:
        from collections import Counter
        from jsonl_store import load_all_records
        media, caption = Counter(), Counter()
        for uid in get_all_user_ids():
            for rec in load_all_records(uid):
                for att in rec.get("attachments") or []:
                    caption[att.get("caption_status", "none")] += 1
                    if att.get("caption_status") != "skipped":
                        media[att.get("media_status", "none")] += 1
        return {"media_status": dict(media), "caption_status": dict(caption)}

    return await asyncio.to_thread(count)


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
