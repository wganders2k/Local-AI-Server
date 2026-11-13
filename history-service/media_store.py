"""
Archive media store: keeps a local copy of every image attachment.

The archive holds signed Discord CDN links that expire (`ex=` in the query),
so a link is usable for about a day after export and the image is gone from
our reach after that unless it was fetched. This module fetches each image
once, to /archive/media/<channel_id>/<attachment_id><ext>, and records where
on the attachment. The captioner reads from there, and a later re-caption
(new model, new prompt) costs no network at all.

An expired link is re-signed through Discord's attachments/refresh-urls
endpoint, which takes up to 50 links per call and answers with fresh ones for
attachments that still exist. A deleted message's attachment gets no fresh
link and is marked "gone" — that image cannot be recovered, which is exactly
why fetching early matters for new exports.

Network and disk only; no GPU. Measured 2026-11-12 from the raw exports:
43.2k images, ~24 GiB.
"""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import httpx

from config import (
    ARCHIVE_MEDIA_DIR,
    DISCORD_TOKEN,
    MEDIA_FETCH_BATCH_SIZE,
    MEDIA_FETCH_MAX_ATTEMPTS,
    MEDIA_FETCH_MAX_FILE_SIZE_MB,
    MEDIA_FETCH_WORKERS,
)
from dce_parser import _is_image
from jsonl_store import iter_attachments, update_attachments

logger = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"
# Discord's documented ceiling for one refresh-urls call.
REFRESH_BATCH = 50


def needs_fetch(att: dict) -> bool:
    """An image with no stored copy that has not been given up on."""
    return (
        _is_image(att.get("content_type", ""))
        and att.get("media_status") not in ("stored", "gone", "too_large")
        and att.get("media_attempts", 0) < MEDIA_FETCH_MAX_ATTEMPTS
    )


def media_relpath(channel_id: str, attachment_id: str, filename: str) -> str:
    """Where an attachment is kept, relative to the archive root (/archive)."""
    ext = os.path.splitext(filename)[1].lower() or ".bin"
    return os.path.join("media", str(channel_id or "unknown"), f"{attachment_id}{ext}")


def refresh_urls(client: httpx.Client, urls: List[str]) -> Dict[str, Optional[str]]:
    """
    Fresh signed links for `urls`, keyed by the original. A link absent from
    the result belongs to an attachment Discord no longer has; one mapped to
    None was in a call that failed, so nothing is known about it yet.
    """
    out: Dict[str, Optional[str]] = {}
    for i in range(0, len(urls), REFRESH_BATCH):
        chunk = urls[i:i + REFRESH_BATCH]
        resp = _post_with_ratelimit(client, f"{DISCORD_API}/attachments/refresh-urls", {"attachment_urls": chunk})
        if resp is None:
            out.update(dict.fromkeys(chunk))
            continue
        for item in resp.json().get("refreshed_urls", []):
            if item.get("original") and item.get("refreshed"):
                out[item["original"]] = item["refreshed"]
    return out


def _post_with_ratelimit(client: httpx.Client, url: str, body: dict, tries: int = 3) -> Optional[httpx.Response]:
    import time

    for _ in range(tries):
        resp = client.post(url, json=body, headers={"Authorization": f"Bot {DISCORD_TOKEN}"})
        if resp.status_code == 429:
            wait = float(resp.json().get("retry_after", 5))
            logger.info(f"refresh-urls rate limited; waiting {wait:.1f}s")
            time.sleep(min(wait, 60))
            continue
        if resp.status_code != 200:
            logger.warning(f"refresh-urls failed: HTTP {resp.status_code} {resp.text[:200]}")
            return None
        return resp
    return None


def download(client: httpx.Client, url: str, dest: str, max_bytes: int) -> str:
    """
    Fetch `url` to `dest` atomically. Returns a media_status: "stored",
    "too_large", "gone" (403/404 — the link is dead) or "error" (transient).
    """
    try:
        with client.stream("GET", url) as resp:
            if resp.status_code in (403, 404):
                return "gone"
            if resp.status_code != 200:
                logger.warning(f"Download failed: HTTP {resp.status_code}")
                return "error"
            if int(resp.headers.get("content-length") or 0) > max_bytes:
                return "too_large"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
            size = 0
            try:
                with os.fdopen(fd, "wb") as f:
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise _TooLarge()
                        f.write(chunk)
                os.chmod(tmp, 0o644)
                os.replace(tmp, dest)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        return "stored"
    except _TooLarge:
        return "too_large"
    except (httpx.HTTPError, OSError) as e:
        # OSError: the media directory is unwritable or the disk is full —
        # retried on later runs, and logged loudly enough to be noticed.
        logger.warning(f"Download failed: {e}")
        return "error"


class _TooLarge(Exception):
    pass


def fetch_batch(limit: int = MEDIA_FETCH_BATCH_SIZE) -> int:
    """
    Fetch up to `limit` images that have no stored copy yet. Returns how many
    attachments had their media state settled or retried.
    """
    work = iter_attachments(needs_fetch, limit)
    if not work:
        return 0

    archive_root = os.path.dirname(ARCHIVE_MEDIA_DIR.rstrip("/"))
    max_bytes = MEDIA_FETCH_MAX_FILE_SIZE_MB * 1024 * 1024

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        fresh = refresh_urls(client, [att["url"] for *_, att in work if att.get("url")])

        def one(item: Tuple[str, str, str, dict]) -> Tuple[str, str, str, dict]:
            user_id, message_id, attachment_id, att = item
            rel = media_relpath(att.get("channel_id"), attachment_id, att.get("filename", ""))
            dest = os.path.join(archive_root, rel)
            original = att.get("url", "")
            url = fresh.get(original)
            if os.path.exists(dest):
                status = "stored"
            elif original in fresh and url is None:
                status = "error"  # the refresh call failed; try again later
            elif not url:
                status = "gone"
            else:
                status = download(client, url, dest, max_bytes)
            changes = {"media_status": status, "media_attempts": att.get("media_attempts", 0) + 1}
            if status == "stored":
                changes["media_path"] = rel
            return user_id, message_id, attachment_id, changes

        with ThreadPoolExecutor(MEDIA_FETCH_WORKERS) as pool:
            results = list(pool.map(one, work))

    by_user: Dict[str, Dict[tuple, dict]] = {}
    counts: Dict[str, int] = {}
    for user_id, message_id, attachment_id, changes in results:
        by_user.setdefault(user_id, {})[(message_id, attachment_id)] = changes
        counts[changes["media_status"]] = counts.get(changes["media_status"], 0) + 1
    for user_id, changes in by_user.items():
        update_attachments(user_id, changes)

    logger.info(f"Media batch: {len(results)} attachments — {counts}")
    return len(results)
