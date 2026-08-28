"""
DCE (DiscordChatExporter) orchestrator.

Invokes DiscordChatExporter as a one-shot Docker container via
`docker compose run`. Handles channel evaluation, date-range
calculation, and post-export merge into per-user JSONL archive.
"""

import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    ARCHIVE_RAW_DIR,
    DISCORD_GUILD_ID,
    DISCORD_TOKEN,
    EXCLUDED_CHANNELS,
)
from channel_state import (
    get_channel,
    load_state,
    save_state,
    update_channel,
    channels_needing_export,
)
from dce_parser import parse_dce_export_directory
from jsonl_store import append_messages

DCE_OUTPUT_ROOT = "/out"       # Path as seen by DCE container (mounted cold storage)
HOST_RAW_DIR = "/archive/raw"  # Path as seen by history-service (bind mount)

logger = logging.getLogger(__name__)

# Dedicated logger for DCE output — allows selective filtering of DCE logs.
# stdout lines → INFO, stderr lines → WARNING.
dce_logger = logging.getLogger(__name__ + ".dce")


def _stream_pipe(pipe, logger_method, prefix: str) -> None:
    """
    Read lines from a subprocess pipe and log each one in real-time.

    Runs in a background thread so the main thread can wait() on the process.
    """
    try:
        for line in pipe:
            stripped = line.rstrip("\n\r")
            if stripped:
                logger_method(f"[DCE] {stripped}")
    except Exception as e:
        logger_method(f"[DCE] pipe read error: {e}")


def _current_user_string() -> str:
    """
    Return the current process UID:GID as a string (e.g. "1000:1000").

    Used to pass ``--user`` to ``docker compose run`` so that DCE writes
    files owned by the same UID/GID as the history-service process
    (appuser).  This avoids permission issues when the service later tries
    to delete or rewrite those files.
    """
    return f"{os.getuid()}:{os.getgid()}"


def _timestamp_dir() -> str:
    """Generate a unique timestamp-based directory name for this export run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def export_channel(
    channel_id: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Invoke DCE via docker compose run to export a single channel.
    
    Args:
        channel_id: Discord channel ID to export.
        after: Optional start date (inclusive), ISO format YYYY-MM-DD.
        before: Optional end date (inclusive), ISO format YYYY-MM-DD.
        output_dir: Optional override for output directory (DCE container path).
                    Defaults to /out/<timestamp> (DCE container sees /out).
        
    Returns:
        Path to the output directory as seen by the DCE container on success,
        None on failure.
    """
    ts = _timestamp_dir()
    # DCE container writes to /out/<ts>  (DCE's bind-mount root)
    dce_output_dir = output_dir if output_dir else os.path.join(DCE_OUTPUT_ROOT, ts)
    # history-service sees the same storage at /archive/raw/<ts>
    host_output_dir = os.path.join(HOST_RAW_DIR, os.path.basename(dce_output_dir))

    # Pre-create directory using the history-service-visible path so DCE can
    # write into it. Both paths point to the same bind-mounted cold storage.
    os.makedirs(host_output_dir, exist_ok=True)
    
    # Build docker compose run command.
    # The compose file is mounted at /etc/dce-compose.yml and .env at /etc/dce.env.
    user_str = _current_user_string()

    # Build docker compose run command.
    # The compose file is mounted at /etc/dce-compose.yml and .env at /etc/dce.env.
    # --user ensures DCE writes files owned by the same UID/GID as appuser
    # so the history-service process can later manage (delete/rewrite) them.
    cmd = [
        "docker", "compose",
        "-f", "/etc/dce-compose.yml",
        "--env-file", "/etc/dce.env",
        "run", "--rm", "--user", user_str,
        "discord-chat-exporter",
        "export",
        "--channel", channel_id,
        "--format", "Json",
        "--output", dce_output_dir,
    ]
    
    if after:
        cmd.extend(["--after", after])
    if before:
        cmd.extend(["--before", before])
    
    logger.info(f"Running DCE export: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Start background threads to stream stdout and stderr to the logger.
        stdout_thread = threading.Thread(
            target=_stream_pipe,
            args=(process.stdout, dce_logger.info, ""),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_pipe,
            args=(process.stderr, dce_logger.warning, ""),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete with a timeout.
        try:
            process.wait(timeout=3600)  # 1 hour max per channel export
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            logger.error(f"DCE export timed out for channel {channel_id}")
            return None

        # Ensure threads have finished reading any remaining pipe data.
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        # Close pipes to free resources.
        process.stdout.close()
        process.stderr.close()

        if process.returncode != 0:
            logger.error(
                f"DCE export failed for channel {channel_id} (exit {process.returncode})"
            )
            return None

        logger.info(f"DCE export completed for channel {channel_id} → {dce_output_dir}")
        return dce_output_dir

    except Exception as e:
        logger.error(f"DCE export failed for channel {channel_id}: {e}")
        return None


def merge_export(
    output_dir: str,
) -> Dict[str, int]:
    """
    Parse DCE export output and merge into per-user JSONL archives.
    
    Args:
        output_dir: Path to the DCE export output directory.
        
    Returns:
        Dictionary mapping user_id to number of new messages appended.
    """
    user_records = parse_dce_export_directory(output_dir)
    
    results = {}
    for user_id, records in user_records.items():
        count = append_messages(user_id, records)
        results[user_id] = count
        logger.info(
            f"Merged user {user_id}: {count} new / {len(records)} total from export"
        )
    
    return results


def evaluate_and_export(
    channels: List[dict],
) -> Dict[str, int]:
    """
    Full evaluation and export pipeline.
    
    1. Load channel state
    2. Determine which channels need export
    3. For each channel, invoke DCE and merge results
    4. Update channel state
    
    Args:
        channels: List of channel dicts from Discord API.
                  Each must have 'id' and 'name'.
                  
    Returns:
        Dictionary mapping user_id to total new messages appended.
    """
    state = load_state()
    needs_export = channels_needing_export(state, channels)

    # Filter out excluded channels
    if EXCLUDED_CHANNELS:
        needs_export = [
            ch for ch in needs_export
            if ch["channel_id"] not in EXCLUDED_CHANNELS
        ]
        logger.info(f"Filtered out {len(EXCLUDED_CHANNELS)} excluded channels")
    
    total_results: Dict[str, int] = {}
    
    for ch in needs_export:
        channel_id = ch["channel_id"]
        channel_name = ch["channel_name"]
        last_export_at = ch["last_export_at"]
        
        logger.info(
            f"Evaluating channel {channel_id} ({channel_name}): "
            f"{ch['reason']}"
        )
        
        # Calculate date range
        after = None
        before = None
        
        if last_export_at:
            # Targeted export: since last export
            after = datetime.fromisoformat(last_export_at).strftime("%Y-%m-%d")
            before = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Run export
        output_dir = export_channel(channel_id, after=after, before=before)
        
        if output_dir:
            # Build host-side path for reading DCE output (DCE writes to /out, which is mounted cold storage)
            host_output_dir = os.path.join(HOST_RAW_DIR, os.path.basename(output_dir))
            # Merge results
            merge_results = merge_export(host_output_dir)
            
            for user_id, count in merge_results.items():
                total_results[user_id] = total_results.get(user_id, 0) + count
            
            # Update channel state
            now = datetime.now(timezone.utc).isoformat()
            total_exported = sum(merge_results.values())
            existing_record = get_channel(state, channel_id)
            previous_total = existing_record.get("total_messages_exported", 0) if existing_record else 0
            new_total = previous_total + total_exported
            update_channel(
                state,
                channel_id=channel_id,
                channel_name=channel_name,
                last_export_at=now,
                last_message_at=ch.get("last_message_at"),
                total_messages_exported=new_total,
            )
        else:
            logger.warning(f"Skipping state update for channel {channel_id} (export failed)")
    
    # Save updated channel state
    save_state(state)
    
    logger.info(
        f"Export pipeline complete: "
        f"{len(total_results)} users, "
        f"{sum(total_results.values())} total new messages"
    )
    
    return total_results