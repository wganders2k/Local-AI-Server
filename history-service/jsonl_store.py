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
import threading
from typing import Callable, Dict, Iterable, List, Set, Union

from config import ARCHIVE_DIR

logger = logging.getLogger(__name__)

# Module-level cache: user_id -> set of known message_ids
_user_id_cache: Dict[str, Set[str]] = {}

# Held by every write. A rewrite replaces a user's whole file, so anything
# written between reading that file and renaming over it — an appended message,
# a caption, a refreshed field — would be silently lost. Writers come from the
# caption job and from request handlers running in worker threads. Reentrant so
# update_records can call rewrite_archive.
_archive_lock = threading.RLock()


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
    with _archive_lock:
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
    
    path = _user_file_path(user_id)
    _ensure_archive_dir()

    appended = 0
    new_ids: Set[str] = set()
    with _archive_lock:
        existing_ids = _load_existing_ids(user_id)
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


def _read_entries(path: str) -> List[Union[dict, str]]:
    """
    Every non-blank line of an archive file, in order: the record it holds, or
    the line itself where it is not a JSON object. For a rewrite, which must put
    back what it cannot read rather than drop it. Raises OSError: a file read in
    part is no basis for writing the whole of it back.
    """
    entries: List[Union[dict, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            entries.append(record if isinstance(record, dict) else line)
    return entries


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

    with _archive_lock:
        _write_atomically(path, user_id, records)


def _write_atomically(path: str, user_id: str, records: List[Union[dict, str]]) -> None:
    """`records` may hold raw lines (str), written back exactly as they were."""
    try:
        # Write to temp file in same directory (ensures same filesystem for atomic rename)
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix=os.path.basename(path))
        try:
            with os.fdopen(fd, "w") as f:
                for record in records:
                    line = record if isinstance(record, str) else json.dumps(record, ensure_ascii=False)
                    f.write(line + "\n")
            # mkstemp creates 0600, and the rename would hand that to the
            # archive, locking out every other reader (rag, the trainer, the
            # host). Keep the file's existing mode; a new file gets 0644.
            try:
                mode = os.stat(path).st_mode & 0o777
            except FileNotFoundError:
                mode = 0o644
            os.chmod(tmp_path, mode)
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


def update_records(user_id: str, apply: Callable[[List[dict]], bool]) -> bool:
    """
    Read-modify-write a user's archive under the archive lock.

    `apply` changes the freshly loaded records in place and returns whether it
    changed anything; the file is rewritten only if it did. Load inside the
    lock, never earlier: a copy held across slow work (a caption batch takes
    minutes) is stale by the time it is written back.

    Stricter than load_all_records, because the whole file is written back: a
    line that is not a JSON object is kept exactly where it was, never dropped,
    and a file that cannot be read in full is not rewritten at all.
    """
    path = _user_file_path(user_id)
    with _archive_lock:
        if not os.path.exists(path):
            return False
        try:
            entries = _read_entries(path)
        except OSError as e:
            logger.error(f"Not updating archive for user {user_id}: could not read it in full: {e}")
            return False
        records = [e for e in entries if isinstance(e, dict)]
        if not records or not apply(records):
            return False
        unreadable = len(entries) - len(records)
        if unreadable:
            logger.warning(f"Archive for user {user_id} has {unreadable} unreadable line(s); kept as they were")
        _write_atomically(path, user_id, entries)
        return True


def refresh_fields(user_id: str, parsed: Iterable[dict], fields: Iterable[str]) -> int:
    """
    Bring `fields` on already-archived records in line with freshly parsed ones.

    A field the parser now emits is set; one it no longer emits is removed.
    Fields outside `fields` — attachments and their captions above all — are
    never touched. Returns the number of records changed.
    """
    fresh = {str(r["message_id"]): r for r in parsed if r.get("message_id")}
    fields = tuple(fields)
    changed = 0

    def apply(records: List[dict]) -> bool:
        nonlocal changed
        for rec in records:
            new = fresh.get(str(rec.get("message_id")))
            if new is None:
                continue
            before = {f: rec.get(f) for f in fields}
            for f in fields:
                if f in new:
                    rec[f] = new[f]
                else:
                    rec.pop(f, None)
            if before != {f: rec.get(f) for f in fields}:
                changed += 1
        return changed > 0

    update_records(user_id, apply)
    return changed


def refresh_attachments(user_id: str, parsed: Iterable[dict], fields: Iterable[str]) -> int:
    """
    Bring the parser-owned `fields` of archived attachments in line with a
    fresh parse, leaving captions and media alone.

    Attachments pair up by `attachment_id` where the archive has one, and by
    position otherwise — records archived before the id was parsed carry none,
    and DCE lists a message's attachments in a stable order. A message whose
    attachment count changed is left as it is rather than guessed at.

    caption_status is the one non-parser field this moves, and only between
    the two values the parser itself decides — "skipped" and "pending" — so an
    image the old parser could not recognise becomes captionable, while
    "done", "error" and anything else the captioner wrote stays put.
    Returns the number of records changed.
    """
    fresh = {str(r["message_id"]): r.get("attachments") or [] for r in parsed if r.get("message_id")}
    fields = tuple(fields)
    changed = 0

    def pair(old: List[dict], new: List[dict]) -> List[tuple]:
        by_id = {a.get("attachment_id"): a for a in new if a.get("attachment_id")}
        if old and all(a.get("attachment_id") in by_id for a in old):
            return [(a, by_id[a["attachment_id"]]) for a in old]
        if len(old) == len(new):
            return list(zip(old, new))
        return []

    def apply(records: List[dict]) -> bool:
        nonlocal changed
        for rec in records:
            new_atts = fresh.get(str(rec.get("message_id")))
            if not new_atts or not rec.get("attachments"):
                continue
            touched = False
            for old, new in pair(rec["attachments"], new_atts):
                for f in fields:
                    if f in new and old.get(f) != new[f]:
                        old[f] = new[f]
                        touched = True
                status = old.get("caption_status")
                if status in ("skipped", "pending", None) and status != new.get("caption_status"):
                    old["caption_status"] = new.get("caption_status")
                    touched = True
            changed += touched
        return changed > 0

    update_records(user_id, apply)
    return changed


def iter_attachments(predicate: Callable[[dict], bool], limit: int) -> List[tuple]:
    """
    Up to `limit` attachments across the archive matching `predicate`, as
    (user_id, message_id, attachment_id, attachment) tuples — a snapshot for
    picking work, never for writing back. Writers go through update_records.
    """
    found: List[tuple] = []
    for user_id in get_all_user_ids():
        for record in load_all_records(user_id):
            for att in record.get("attachments") or []:
                if att.get("attachment_id") and predicate(att):
                    found.append((user_id, str(record.get("message_id")), att["attachment_id"], dict(att, channel_id=record.get("channel_id"))))
                    if len(found) >= limit:
                        return found
    return found


def update_attachments(user_id: str, changes: Dict[tuple, dict]) -> int:
    """
    Merge changes into archived attachments, keyed (message_id, attachment_id),
    in one read-modify-write of the user's file under the archive lock.
    Batched because a rewrite is the whole file: per-attachment writes would
    rewrite a large user's archive once per image across a 43k-image backfill.
    Returns how many attachments were found and changed.
    """
    done = 0

    def apply(records: List[dict]) -> bool:
        nonlocal done
        for rec in records:
            mid = str(rec.get("message_id"))
            for att in rec.get("attachments") or []:
                c = changes.get((mid, att.get("attachment_id")))
                if c:
                    att.update(c)
                    done += 1
        return done > 0

    update_records(user_id, apply)
    return done


def carry_attachment_state(records: List[dict], old_by_id: Dict[str, dict], parser_fields: Iterable[str]) -> int:
    """
    Give freshly parsed attachments what the archive already knew about them:
    caption, downloaded copy, attempt counts — every field outside
    `parser_fields`. Paired by attachment_id, which Discord never changes, so
    a full re-export keeps three days of captioning. caption_status follows
    the rule in refresh_attachments: the parser's "skipped"/"pending" yields
    to whatever the captioner decided. Returns the attachments carried over.
    """
    parser_fields = set(parser_fields)
    carried = 0
    for rec in records:
        for att in rec.get("attachments") or []:
            old = old_by_id.get(att.get("attachment_id"))
            if old is None:
                continue
            for k, v in old.items():
                if k not in parser_fields and k != "caption_status":
                    att[k] = v
            if old.get("caption_status") not in ("skipped", "pending", None):
                att["caption_status"] = old["caption_status"]
            carried += 1
    return carried


def replace_archive(fresh: Dict[str, List[dict]], prev_dir: str, parser_fields: Iterable[str]) -> dict:
    """
    Replace the whole archive with `fresh` ({user_id: records}, a full
    re-export), keeping one copy of the archive it replaces in `prev_dir`.

    Captions and media carry over by attachment_id (carry_attachment_state).
    Messages missing from `fresh` were deleted on Discord and are dropped,
    as is the file of a user left with none. Holds the archive lock
    throughout, so a caption written meanwhile cannot land in the old copy
    and be lost.
    """
    import shutil

    _ensure_archive_dir()
    with _archive_lock:
        if os.path.isdir(prev_dir):
            shutil.rmtree(prev_dir)
        os.makedirs(prev_dir)
        old_by_id: Dict[str, dict] = {}
        old_users = get_all_user_ids()
        for uid in old_users:
            shutil.copy2(_user_file_path(uid), prev_dir)
            for rec in load_all_records(uid):
                for att in rec.get("attachments") or []:
                    if att.get("attachment_id"):
                        old_by_id[att["attachment_id"]] = att
        carried = messages = 0
        for uid, records in fresh.items():
            records = sorted(records, key=lambda r: (r.get("timestamp", ""), r["message_id"]))
            carried += carry_attachment_state(records, old_by_id, parser_fields)
            messages += len(records)
            _write_atomically(_user_file_path(uid), uid, records)
        for uid in set(old_users) - set(fresh):
            os.unlink(_user_file_path(uid))
        _user_id_cache.clear()
    logger.info(f"Archive replaced: {messages} messages, {carried} attachments carried over, previous copy in {prev_dir}")
    return {"messages": messages, "users": len(fresh), "attachments_carried": carried}


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
