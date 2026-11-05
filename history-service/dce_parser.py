"""
DCE (DiscordChatExporter) JSON parser.

Maps DCE native JSON output fields to the internal archive schema.
DCE uses lowercase fields (id, author, content, timestamp, etc.)
while the internal schema uses snake_case.

Raw DCE file structure:
{
  "guild": { "id": "...", "name": "..." },
  "channel": { "id": "...", "name": "..." },
  "messages": [
    {
      "id": "...",
      "type": "Default",
      "timestamp": "2026-05-14T03:51:27.802+00:00",
      "content": "...",
      "author": { "id": "...", "name": "...", "isBot": false, ... },
      "attachments": [...],
      "embeds": [],
      "reactions": []
    }
  ]
}
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Message types are recorded, never filtered. Tier 2 is the unfiltered archive,
# and the system messages carry structure other consumers want: a pin event
# names the message that was pinned, a thread creation names the thread. Each
# consumer drops the types it has no use for (lora-training/build_dataset.py
# keeps Default and Reply).
#
# DCE writes a type it has a name for as that name ("Default", "Reply",
# "ChannelPinnedMessage", "ThreadCreated") and only an unnamed one as Discord's
# number ("20", "46"), so a consumer matching types must expect both forms.
# An earlier skip list here held only the numbers, and so let through nearly
# everything it meant to drop.


def _extract_channel_info(data: dict) -> Tuple[str, str]:
    """
    Extract channel id and name from the top-level DCE object.

    Args:
        data: Parsed DCE JSON root object.

    Returns:
        (channel_id, channel_name) tuple.
    """
    channel = data.get("channel", {})
    channel_id = str(channel.get("id", ""))
    channel_name = channel.get("name", "")
    return channel_id, channel_name



def parse_dce_message(
    raw: dict,
    channel_id: str = "",
    channel_name: str = "",
) -> Optional[dict]:
    """
    Parse a single DCE message record into the internal archive schema.

    DCE message fields (lowercase):
        id, author, content, timestamp, attachments, embeds,
        reactions, referenced_message, edited_timestamp, type

    Internal schema fields:
        message_id, user_id, username, channel_id, channel_name,
        timestamp, content, is_bot, type
        and when present: attachments, reference, is_pinned, mentions

    Args:
        raw: Raw DCE message dict.
        channel_id: Channel ID from top-level DCE object (passed by caller).
        channel_name: Channel name from top-level DCE object (passed by caller).

    Returns:
        Internal archive record dict, or None if the message should be skipped.
    """
    msg_id = raw.get("id")
    if not msg_id:
        return None

    author = raw.get("author", {}) or {}
    user_id = author.get("id", "")
    is_bot = author.get("isBot", False)

    # Username: prefer global @username. Fall back to user_id for stability.
    # Deliberately skipping server nicknames (they change over time).
    username = author.get("name", "") or str(user_id)

    # Skip bot messages entirely — they add noise to the lore corpus
    if is_bot:
        return None

    # DCE timestamp is ISO 8601 string
    timestamp = raw.get("timestamp", "")

    # Content — DCE stores as string (may be null/None for embed-only messages)
    content = raw.get("content", "") or ""

    # Parse attachments
    attachments = _parse_attachments(raw.get("attachments", []))

    record = {
        "message_id": str(msg_id),
        "user_id": str(user_id),
        "username": username,
        "channel_id": str(channel_id),
        "channel_name": channel_name,
        "timestamp": timestamp,
        "content": content,
        "is_bot": is_bot,
        "type": "Default" if raw.get("type") is None else str(raw["type"]),
    }

    # Only include attachments field if there are any (backward compatible)
    if attachments:
        record["attachments"] = attachments

    # Optional fields, present only when they say something (like attachments)
    reference = parse_reference(raw.get("reference"))
    if reference:
        record["reference"] = reference

    if raw.get("isPinned"):
        record["is_pinned"] = True

    mentions = parse_mentions(raw.get("mentions"), content)
    if mentions:
        record["mentions"] = mentions

    return record


# What the parser owns on a record beyond its identity. POST /reparse refreshes
# these on messages already archived, so a parser change reaches old records;
# everything else (attachments, and the captions on them) is left alone.
PARSER_OWNED_FIELDS = ("type", "reference", "is_pinned", "mentions")


def parse_reference(raw_reference: Optional[dict]) -> Optional[dict]:
    """
    The message this one points at: the message a reply answers, a forward's
    source, the message a pin event pinned. `type` is DCE's ("Default", or
    "Forward" for a forward). Fields DCE leaves null are omitted.
    """
    if not raw_reference:
        return None
    ref = {
        "message_id": raw_reference.get("messageId"),
        "channel_id": raw_reference.get("channelId"),
        "type": raw_reference.get("type"),
    }
    ref = {k: str(v) for k, v in ref.items() if v}
    return ref if ("message_id" in ref or "channel_id" in ref) else None


def parse_mentions(raw_mentions: Optional[list], content: str) -> List[dict]:
    """
    Who each "@Name" in `content` refers to.

    DCE renders a user mention as "@" + the server nickname (or global name)
    at export time, and nicknames hold spaces and punctuation, so the text
    alone cannot say where a mention ends or whose it is. DCE's `mentions`
    list can: `display` is the exact text after "@" in `content`, `user_id`
    is who it was. Users DCE lists but `content` never names — a reply's
    implicit ping — are left out.
    """
    out = []
    for m in raw_mentions or []:
        display = m.get("nickname") or m.get("name") or ""
        if display and f"@{display}" in content:
            out.append({
                "user_id": str(m.get("id", "")),
                "username": m.get("name", ""),
                "display": display,
            })
    return out


def _parse_attachments(raw_attachments: list) -> List[dict]:
    """
    Parse DCE attachment objects into internal attachment schema.

    DCE attachment fields (lowercase): url, content_type, filename, size

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
        url = att.get("url", "") or att.get("URL", "")
        content_type = att.get("content_type", "") or att.get("ContentType", "")
        filename = att.get("filename", "")
        size = att.get("size", 0) or 0

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

    DCE JSON output is a single object with top-level metadata and a
    "messages" array:
        {
          "guild": {...},
          "channel": {"id": "...", "name": "..."},
          "messages": [...]
        }

    Args:
        filepath: Path to the DCE JSON export file.

    Returns:
        List of internal archive records.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to parse DCE export file {filepath}: {e}")
        return []

    # Handle both formats:
    # 1. Object with "messages" key (standard DCE output)
    # 2. Plain list of messages (older/alternative format)
    if isinstance(data, dict):
        channel_id, channel_name = _extract_channel_info(data)
        raw_messages = data.get("messages", [])
        if not raw_messages:
            # Fallback: try "Messages" (capital M) for older DCE versions
            raw_messages = data.get("Messages", [])
    elif isinstance(data, list):
        channel_id, channel_name = "", ""
        raw_messages = data
    else:
        logger.warning(f"Unexpected JSON structure in {filepath}")
        return []

    records = []
    skipped = 0
    for raw in raw_messages:
        record = parse_dce_message(raw, channel_id=channel_id, channel_name=channel_name)
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
