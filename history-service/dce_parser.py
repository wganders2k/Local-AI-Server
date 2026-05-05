"""
DCE (DiscordChatExporter) JSON parser.

Maps DCE native JSON output fields to the internal archive schema.
DCE uses PascalCase fields (Id, Author, Content, Timestamp, etc.)
while the internal schema uses snake_case.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_dce_message(raw: dict) -> Optional[dict]:
    """
    Parse a single DCE message record into the internal archive schema.
    
    DCE message fields:
        Id, Author, Content, Timestamp, Attachments, Embeds,
        Reactions, ReferencedMessage, EditedTimestamp, ChannelId
    
    Internal schema fields:
        message_id, user_id, username, channel_id, channel_name,
        timestamp, content, attachments
    
    Args:
        raw: Raw DCE message dict.
        
    Returns:
        Internal archive record dict, or None if the message should be skipped.
    """
    msg_id = raw.get("Id")
    if not msg_id:
        return None
    
    author = raw.get("Author", {})
    user_id = author.get("Id", "")
    username = author.get("Name", "") or author.get("Discriminator", "")
    
    # DCE Timestamp is ISO 8601 string
    timestamp = raw.get("Timestamp", "")
    
    # Content — DCE stores as string (may be null/None for embed-only messages)
    content = raw.get("Content", "") or ""
    
    # Channel — DCE may include ChannelId in the message or at file level
    channel_id = raw.get("ChannelId", "")
    channel_name = raw.get("ChannelName", "")
    
    # Parse attachments
    attachments = _parse_attachments(raw.get("Attachments", []))
    
    record = {
        "message_id": str(msg_id),
        "user_id": str(user_id),
        "username": username,
        "channel_id": str(channel_id),
        "channel_name": channel_name,
        "timestamp": timestamp,
        "content": content,
    }
    
    # Only include attachments field if there are any (backward compatible)
    if attachments:
        record["attachments"] = attachments
    
    return record


def _parse_attachments(raw_attachments: list) -> List[dict]:
    """
    Parse DCE attachment objects into internal attachment schema.
    
    DCE attachment fields: Url, ContentType, Filename, Size
    
    Internal schema:
        url, content_type, filename, file_size_bytes,
        caption_status, caption_excluded_from_training
    
    Args:
        raw_attachments: List of raw DCE attachment dicts.
        
    Returns:
        List of internal attachment records.
    """
    attachments = []
    for att in raw_attachments:
        url = att.get("Url", "") or att.get("URL", "")
        content_type = att.get("ContentType", "") or att.get("Content-Type", "")
        filename = att.get("Filename", "")
        size = att.get("Size", 0) or 0
        
        internal = {
            "url": url,
            "content_type": content_type,
            "filename": filename,
            "file_size_bytes": int(size),
            "caption_status": "pending" if _is_image(content_type) else "skipped",
            "caption_excluded_from_training": True,
        }
        attachments.append(internal)
    
    return attachments


def _is_image(content_type: str) -> bool:
    """Check if a content type is an image."""
    return content_type.startswith("image/") if content_type else False


def parse_dce_export_file(filepath: str) -> List[dict]:
    """
    Parse a DCE export JSON file into internal archive records.
    
    DCE JSON output is a list of message objects at the top level.
    
    Args:
        filepath: Path to the DCE JSON export file.
        
    Returns:
        List of internal archive records.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_messages = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to parse DCE export file {filepath}: {e}")
        return []
    
    if isinstance(raw_messages, dict):
        # Some DCE versions wrap messages in a dict
        raw_messages = raw_messages.get("Messages", [raw_messages])
    
    records = []
    skipped = 0
    for raw in raw_messages:
        record = parse_dce_message(raw)
        if record:
            records.append(record)
        else:
            skipped += 1
    
    if skipped:
        logger.info(f"Parsed {filepath}: {len(records)} records, {skipped} skipped")
    else:
        logger.info(f"Parsed {filepath}: {len(records)} records")
    
    return records


def parse_dce_export_directory(directory: str) -> Dict[str, List[dict]]:
    """
    Parse all DCE JSON files in an export directory.
    
    DCE outputs one JSON file per channel. This function scans the
    directory for JSON files, parses them, and groups records by user_id.
    
    Args:
        directory: Path to the DCE export output directory.
        
    Returns:
        Dictionary mapping user_id to list of archive records.
    """
    user_records: Dict[str, List[dict]] = {}
    
    if not os.path.isdir(directory):
        logger.error(f"Export directory does not exist: {directory}")
        return user_records
    
    json_files = sorted([
        f for f in os.listdir(directory)
        if f.endswith(".json")
    ])
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory}")
        return user_records
    
    total_records = 0
    for filename in json_files:
        filepath = os.path.join(directory, filename)
        records = parse_dce_export_file(filepath)
        
        for record in records:
            uid = record["user_id"]
            if uid not in user_records:
                user_records[uid] = []
            user_records[uid].append(record)
            total_records += 1
    
    logger.info(
        f"Parsed export directory {directory}: "
        f"{total_records} records across {len(user_records)} users"
    )
    
    return user_records
