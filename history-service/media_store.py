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

Two kinds of failure, kept apart so neither costs what it should not:

- The refresh service itself fails (bad token, Discord 5xx, rate limited out,
  network down). That says nothing about any image, so the run stops with
  nothing written and no attempt counted, and later runs back off (2 min
  doubling to an hour) so a dead token costs one call an hour, not every run.
- Discord rejects a call (400) because of one link in it. The chunk is halved
  until the bad link is alone; it alone is marked "error" and counted toward
  MEDIA_FETCH_MAX_ATTEMPTS, and the rest go through.

Network and disk only; no GPU. Measured 2026-11-12 from the raw exports:
43.2k images, ~24 GiB.
"""

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

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
# Backoff after the refresh service fails, doubling per consecutive failure.
BACKOFF_FIRST_S = 120.0
BACKOFF_MAX_S = 3600.0
# A link Discord refused on its own (a 400 for that single URL): the image's
# own failure, counted like a failed download.
REJECTED = object()


class RefreshUnavailable(Exception):
    """The refresh-urls service failed as a whole: nothing is known about any link."""


class _Backoff:
    """Skip runs for a while after the refresh service fails; reset on success."""

    def __init__(self) -> None:
        self.failures = 0
        self.until = 0.0

    def active(self) -> bool:
        return time.monotonic() < self.until

    def failed(self) -> float:
        self.failures += 1
        delay = min(BACKOFF_FIRST_S * 2 ** (self.failures - 1), BACKOFF_MAX_S)
        self.until = time.monotonic() + delay
        return delay

    def succeeded(self) -> None:
        self.failures = 0
        self.until = 0.0


_backoff = _Backoff()


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


def refresh_urls(client: httpx.Client, urls: List[str]) -> Dict[str, object]:
    """
    Fresh signed links for `urls`, keyed by the original. A link absent from
    the result belongs to an attachment Discord no longer has; one mapped to
    REJECTED was refused by Discord on its own. Raises RefreshUnavailable when
    the service fails as a whole.
    """
    out: Dict[str, object] = {}
    for i in range(0, len(urls), REFRESH_BATCH):
        _refresh_chunk(client, urls[i:i + REFRESH_BATCH], out)
    return out


def _refresh_chunk(client: httpx.Client, chunk: List[str], out: Dict[str, object]) -> None:
    resp = _post_with_ratelimit(client, f"{DISCORD_API}/attachments/refresh-urls", {"attachment_urls": chunk})
    if resp.status_code == 400:
        if len(chunk) == 1:
            logger.info(f"refresh-urls rejected one link: {resp.text[:200]}")
            out[chunk[0]] = REJECTED
            return
        # One bad link fails the whole call: halve until it is alone.
        mid = len(chunk) // 2
        _refresh_chunk(client, chunk[:mid], out)
        _refresh_chunk(client, chunk[mid:], out)
        return
    for item in resp.json().get("refreshed_urls", []):
        if item.get("original") and item.get("refreshed"):
            out[item["original"]] = item["refreshed"]


def _post_with_ratelimit(client: httpx.Client, url: str, body: dict, tries: int = 3) -> httpx.Response:
    """A 200 or a 400 response; anything else is RefreshUnavailable."""
    for _ in range(tries):
        try:
            resp = client.post(url, json=body, headers={"Authorization": f"Bot {DISCORD_TOKEN}"})
        except httpx.HTTPError as e:
            raise RefreshUnavailable(f"{type(e).__name__}: {e}") from e
        if resp.status_code == 429:
            try:
                wait = float(resp.json().get("retry_after", 5))
            except (ValueError, AttributeError, TypeError):
                wait = 5.0  # a 429 from something in front of Discord, not JSON
            logger.info(f"refresh-urls rate limited; waiting {wait:.1f}s")
            time.sleep(min(wait, 60))
            continue
        if resp.status_code in (200, 400):
            return resp
        raise RefreshUnavailable(f"HTTP {resp.status_code} {resp.text[:200]}")
    raise RefreshUnavailable(f"still rate limited after {tries} tries")


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
    if _backoff.active():
        return 0
    work = iter_attachments(needs_fetch, limit)
    if not work:
        return 0

    archive_root = os.path.dirname(ARCHIVE_MEDIA_DIR.rstrip("/"))
    max_bytes = MEDIA_FETCH_MAX_FILE_SIZE_MB * 1024 * 1024

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        try:
            fresh = refresh_urls(client, [att["url"] for *_, att in work if att.get("url")])
        except RefreshUnavailable as e:
            delay = _backoff.failed()
            logger.warning(
                f"refresh-urls unavailable ({e}); nothing recorded, next try in {delay / 60:.0f} min"
            )
            return 0
        _backoff.succeeded()

        def one(item: Tuple[str, str, str, dict]) -> Tuple[str, str, str, dict]:
            user_id, message_id, attachment_id, att = item
            rel = media_relpath(att.get("channel_id"), attachment_id, att.get("filename", ""))
            dest = os.path.join(archive_root, rel)
            original = att.get("url", "")
            url = fresh.get(original)
            if os.path.exists(dest):
                status = "stored"
            elif url is REJECTED:
                status = "error"  # Discord refused this link on its own; counted
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
