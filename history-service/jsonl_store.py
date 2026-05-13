"""
JSONL archive management.

Handles per-user JSONL file read/write, deduplication by message_id,
and attachment caption status tracking.

Archive files live at /archive/archive/<user_id>.jsonl.
"""

import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Set

from config import ARCHIVE_DIR

logger = logging.getLogger(__name__)

# Module-level cache: user_id -> set of known message_ids
_user_id_cache: Dict[str, Set[str]] = {}


def _ensure_archive_dir() -> None:
    """Create archive directory if it doesn't exist."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)


def _user_file_path(user_id: str) -> str:
    """Get the file path for a user's archive."""
    return os.path.join(ARCHIVE_DIR, f"{user_id}.jsonl")


def _load_existing_ids(user_id: str) -> Set[str]:
    """Load message IDs for a user, using in-memory cache after first load."""
    if user_id in _user_id_cache:
        return _user_id_cache[user_id]

    # Load from disk and cache
    ids: Set[str] = set()
    path = _user_file_path(user_id)
    if not os.path.exists(path):
        _user_id_cache[user_id] = ids
        return ids

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    msg_id = record.get("message_id")
                    if msg_id:
                        ids.add(str(msg_id))
                except json.JSONDecodeError:
                    continue
    except IOError as e:
        logger.error(f"Failed to read existing IDs for user {user_id}: {e}")

    _user_id_cache[user_id] = ids
    return ids


def _add_to_cache(user_id: str, msg_ids: Set[str]) -> None:
    """Update in-memory cache after successful append."""
    if user_id not in _user_id_cache:
        _user_id_cache[user_id] = set()
    _user_id_cache[user_id].update(msg_ids)


def append_message(user_id: str, record: dict) -> bool:
    """
    Append a message record to the user's archive if not already present.
    
    Args:
        user_id: Discord user ID.
        record: Message record dict (must contain 'message_id').
        
    Returns:
        True if message was appended, False if duplicate.
    """
    msg_id = record.get("message_id")
    if not msg_id:
        logger.warning(f"Skipping record without message_id for user {user_id}")
        return False
    
    path = _user_file_path(user_id)
    _ensure_archive_dir()
    
    # For large files, avoid loading all IDs into memory.
    # Instead, check the last N lines or use a simple append with
    # a periodic dedup pass. For now, load IDs for correctness.
    existing_ids = _load_existing_ids(user_id)
    
    if msg_id in existing_ids:
        return False
    
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _add_to_cache(user_id, {str(msg_id)})
        return True
    except IOError as e:
        logger.error(f"Failed to append message for user {user_id}: {e}")
        return False


def append_messages(user_id: str, records: List[dict]) -> int:
    """
    Append multiple message records to the user's archive.
    
    Args:
        user_id: Discord user ID.
        records: List of message record dicts.
        
    Returns:
        Number of messages actually appended (excluding duplicates).
    """
    if not records:
        return 0
    
    existing_ids = _load_existing_ids(user_id)
    path = _user_file_path(user_id)
    _ensure_archive_dir()
    
    appended = 0
    new_ids: Set[str] = set()
    try:
        with open(path, "a") as f:
            for record in records:
                msg_id = record.get("message_id")
                if not msg_id:
                    continue
                if msg_id in existing_ids:
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                str_msg_id = str(msg_id)
                existing_ids.add(str_msg_id)
                new_ids.add(str_msg_id)
                appended += 1
        _add_to_cache(user_id, new_ids)
    except IOError as e:
        logger.error(f"Failed to append messages for user {user_id}: {e}")

    return appended


def get_message_count(user_id: str) -> int:
    """
    Count messages in a user's archive.
    
    Args:
        user_id: Discord user ID.
        
    Returns:
        Number of messages in the archive.
    """
    path = _user_file_path(user_id)
    if not os.path.exists(path):
        return 0
    
    count = 0
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
    except IOError as e:
        logger.error(f"Failed to count messages for user {user_id}: {e}")
    return count


def get_all_user_ids() -> List[str]:
    """
    List all user IDs with archive files.
    
    Returns:
        List of user IDs (as strings).
    """
    _ensure_archive_dir()
    users = []
    try:
        for filename in os.listdir(ARCHIVE_DIR):
            if filename.endswith(".jsonl"):
                users.append(filename[:-6])  # Strip .jsonl
    except IOError as e:
        logger.error(f"Failed to list archive users: {e}")
    return users


def load_all_records(user_id: str) -> List[dict]:
    """
    Load all records from a user's archive.
    
    Args:
        user_id: Discord user ID.
        
    Returns:
        List of message record dicts.
    """
    path = _user_file_path(user_id)
    if not os.path.exists(path):
        return []
    
    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except IOError as e:
        logger.error(f"Failed to load records for user {user_id}: {e}")
    return records


def get_pending_captions(limit: int = 100) -> List[tuple]:
    """
    Scan all user archives for attachments with caption_status == "pending".
    
    Args:
        limit: Maximum number of pending captions to return.
        
    Returns:
        List of tuples: (user_id, record, attachment_index, attachment)
    """
    pending = []
    for user_id in get_all_user_ids():
        records = load_all_records(user_id)
        for record in records:
            attachments = record.get("attachments", [])
            for idx, att in enumerate(attachments):
                if att.get("caption_status") == "pending":
                    pending.append((user_id, record, idx, att))
                    if len(pending) >= limit:
                        return pending
    return pending


def rewrite_archive(user_id: str, records: List[dict]) -> None:
    """
    Rewrite a user's entire archive file atomically using write-then-rename.

    Use this when updating records in place (e.g., after captioning).

    Args:
        user_id: Discord user ID.
        records: Complete list of message records to write.
    """
    path = _user_file_path(user_id)
    _ensure_archive_dir()

    try:
        # Write to temp file in same directory (ensures same filesystem for atomic rename)
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix=os.path.basename(path))
        try:
            with os.fdopen(fd, "w") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Atomic rename on POSIX
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(f"Rewrote archive for user {user_id}: {len(records)} records")
    except IOError as e:
        logger.error(f"Failed to rewrite archive for user {user_id}: {e}")


def clear_all() -> int:
    """
    Delete all user archive files and reset the in-memory ID cache.

    Returns:
        Number of archive files deleted.
    """
    import glob

    _ensure_archive_dir()
    pattern = os.path.join(ARCHIVE_DIR, "*.jsonl")
    files = glob.glob(pattern)
    deleted = 0
    for f in files:
        try:
            os.unlink(f)
            deleted += 1
        except OSError as e:
            logger.error(f"Failed to delete {f}: {e}")

    # Reset in-memory cache
    _user_id_cache.clear()
    logger.info(f"Cleared {deleted} user archive files and reset ID cache")
    return deleted
