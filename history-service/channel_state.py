"""
Channel state management.

Tracks last_export_at and last_message_at per channel.
State persisted to /archive/state/channel_state.json.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import ARCHIVE_STATE_DIR

logger = logging.getLogger(__name__)

CHANNEL_STATE_FILE = os.path.join(ARCHIVE_STATE_DIR, "channel_state.json")


def _ensure_state_dir() -> None:
    """Create state directory if it doesn't exist."""
    os.makedirs(ARCHIVE_STATE_DIR, exist_ok=True)


def load_state() -> Dict[str, dict]:
    """
    Load channel state from disk.
    
    Returns:
        Dictionary mapping channel_id to channel state record.
        Empty dict if file doesn't exist or is invalid.
    """
    if not os.path.exists(CHANNEL_STATE_FILE):
        logger.info("No channel state file found — starting fresh")
        return {}
    
    try:
        with open(CHANNEL_STATE_FILE, "r") as f:
            state = json.load(f)
        logger.info(f"Loaded channel state: {len(state)} channels")
        return state
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load channel state: {e}")
        return {}


def save_state(state: Dict[str, dict]) -> None:
    """
    Persist channel state to disk.
    
    Args:
        state: Dictionary mapping channel_id to channel state record.
    """
    _ensure_state_dir()
    try:
        with open(CHANNEL_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved channel state: {len(state)} channels")
    except IOError as e:
        logger.error(f"Failed to save channel state: {e}")


def get_channel(state: Dict[str, dict], channel_id: str) -> Optional[dict]:
    """
    Get state record for a specific channel.
    
    Args:
        state: Full channel state dictionary.
        channel_id: Discord channel ID.
        
    Returns:
        Channel state record or None if not found.
    """
    return state.get(channel_id)


def update_channel(
    state: Dict[str, dict],
    channel_id: str,
    channel_name: str,
    last_export_at: str,
    last_message_at: Optional[str] = None,
    total_messages_exported: int = 0,
) -> dict:
    """
    Upsert channel state record.
    
    Args:
        state: Full channel state dictionary (modified in place).
        channel_id: Discord channel ID.
        channel_name: Human-readable channel name.
        last_export_at: ISO 8601 timestamp of last export.
        last_message_at: ISO 8601 timestamp of most recent message seen.
        total_messages_exported: Cumulative message count from exports.
        
    Returns:
        Updated channel state record.
    """
    record = {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "last_export_at": last_export_at,
        "last_message_at": last_message_at or last_export_at,
        "total_messages_exported": total_messages_exported,
    }
    state[channel_id] = record
    return record


def channels_needing_export(
    state: Dict[str, dict],
    channels: List[dict],
) -> List[dict]:
    """
    Evaluate which channels need exporting.
    
    Compares each channel's last_message_at against last_export_at.
    Channels with activity since last export (or no prior export) are returned.
    
    Args:
        state: Full channel state dictionary.
        channels: List of channel records from Discord API, each with
                  at least 'id', 'name', and optionally 'last_message_timestamp'.
                  
    Returns:
        List of channels needing export, each with:
        - channel_id, channel_name, last_export_at (None if first time),
          last_message_at, should_export (bool), reason (str).
    """
    needs_export = []
    
    for channel in channels:
        channel_id = channel["id"]
        channel_name = channel.get("name", channel_id)
        last_message_at = channel.get("last_message_timestamp")
        
        record = get_channel(state, channel_id)
        
        if record is None:
            # New channel — full export needed
            needs_export.append({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "last_export_at": None,
                "last_message_at": last_message_at,
                "should_export": True,
                "reason": "new channel (no prior export)",
            })
        elif last_message_at and last_message_at > record.get("last_export_at", ""):
            # Activity since last export
            needs_export.append({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "last_export_at": record["last_export_at"],
                "last_message_at": last_message_at,
                "should_export": True,
                "reason": f"activity since {record['last_export_at']}",
            })
        else:
            logger.debug(
                f"Channel {channel_id} ({channel_name}): "
                f"no new activity since {record['last_export_at']}"
            )
    
    logger.info(f"Channel evaluation: {len(needs_export)} of {len(channels)} channels need export")
    return needs_export


def reset_state() -> None:
    """
    Delete the channel state file and write an empty state.

    This resets all tracking data so that the next evaluate run
    will treat every channel as needing a full export.
    """
    _ensure_state_dir()
    try:
        if os.path.exists(CHANNEL_STATE_FILE):
            os.unlink(CHANNEL_STATE_FILE)
        save_state({})
        logger.info("Channel state reset: all tracking data cleared")
    except IOError as e:
        logger.error(f"Failed to reset channel state: {e}")
