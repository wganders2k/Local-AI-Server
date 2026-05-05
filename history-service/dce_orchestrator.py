"""
DCE (DiscordChatExporter) orchestrator.

Invokes DiscordChatExporter as a one-shot Docker container via
`docker compose run`. Handles channel evaluation, date-range
calculation, and post-export merge into per-user JSONL archive.
"""

import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    ARCHIVE_RAW_DIR,
    DISCORD_GUILD_ID,
    DISCORD_TOKEN,
)
from channel_state import (
    load_state,
    save_state,
    update_channel,
    channels_needing_export,
)
from dce_parser import parse_dce_export_directory
from jsonl_store import append_messages

logger = logging.getLogger(__name__)


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
        output_dir: Optional override for output directory.
                    Defaults to /archive/raw/<timestamp>.
        
    Returns:
        Path to the output directory on success, None on failure.
    """
    if output_dir is None:
        output_dir = os.path.join(ARCHIVE_RAW_DIR, _timestamp_dir())
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build docker compose run command
    cmd = [
        "docker", "compose",
        "--profile", "manual",
        "run", "--rm", "discord-chat-exporter",
        "export",
        "--channel", channel_id,
        "--format", "Json",
        "--output", output_dir,
    ]
    
    if after:
        cmd.extend(["--after", after])
    if before:
        cmd.extend(["--before", before])
    
    logger.info(f"Running DCE export: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per channel export
        )
        
        if result.returncode != 0:
            logger.error(
                f"DCE export failed for channel {channel_id} (exit {result.returncode}):\n"
                f"STDERR: {result.stderr[:500]}"
            )
            return None
        
        logger.info(f"DCE export completed for channel {channel_id} → {output_dir}")
        return output_dir
    
    except subprocess.TimeoutExpired:
        logger.error(f"DCE export timed out for channel {channel_id}")
        return None
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
            # Merge results
            merge_results = merge_export(output_dir)
            
            for user_id, count in merge_results.items():
                total_results[user_id] = total_results.get(user_id, 0) + count
            
            # Update channel state
            now = datetime.now(timezone.utc).isoformat()
            total_exported = sum(merge_results.values())
            update_channel(
                state,
                channel_id=channel_id,
                channel_name=channel_name,
                last_export_at=now,
                last_message_at=ch.get("last_message_at"),
                total_messages_exported=total_exported,
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
