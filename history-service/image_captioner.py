"""
Image captioning: archive backfill for Discord image attachments.

Picks images that media_store has a local copy of and that have no caption
yet, sends each to the proxy's `caption` preset (models.ini), and writes the
result onto the attachment. rag/ingest.py folds captions into a message's
text, which is what makes image content searchable.

GPU contention is not this module's problem. `caption` is one of the proxy's
BATCH_MODELS, so every request is ranked as arbiter job llm-batch (below any
person, above the video job) and held until interactive traffic has been quiet
for a while. That replaces the 3–6 AM window and the queue-depth check this
module used to carry: the proxy knows when the card is wanted, this does not.

Each caption records the model and a hash of the prompt that produced it, so a
re-caption after either changes can find exactly the stale ones.

History: until 2026-11-12 this never captioned anything — the parser marked
every attachment "skipped", the stored links had expired, the preset was
commented out, and the write-back clobbered concurrent writes. See git log.
"""

import base64
import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import httpx
from PIL import Image

from config import (
    ARCHIVE_MEDIA_DIR,
    IMAGE_CAPTION_BATCH_SIZE,
    IMAGE_CAPTION_MAX_ATTEMPTS,
    IMAGE_CAPTION_MAX_SIDE,
    IMAGE_CAPTION_MODEL,
    IMAGE_CAPTION_TIMEOUT_S,
    PROXY_URL,
)
from jsonl_store import iter_attachments, update_attachments

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "caption.md"
CAPTION_PROMPT = PROMPT_PATH.read_text().strip()
PROMPT_VERSION = hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest()[:12]

# Ceiling on the answer. The prompt asks for 1–3 sentences plus any quoted
# text; a long chat screenshot's transcription is the only thing that nears it.
MAX_TOKENS = 200


def needs_caption(att: dict) -> bool:
    return (
        att.get("caption_status") == "pending"
        and att.get("media_status") == "stored"
        and att.get("caption_attempts", 0) < IMAGE_CAPTION_MAX_ATTEMPTS
    )


def encode_image(path: str, max_side: int = IMAGE_CAPTION_MAX_SIDE) -> str:
    """
    A data: URL the model can read. Everything is re-encoded as JPEG: the
    projector's loader (stb_image in llama.cpp) reads no WebP, an animated GIF
    is captioned by its first frame, and alpha has nothing to say to a caption.
    """
    with Image.open(path) as im:
        im.seek(0)
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def caption_image(client: httpx.Client, data_url: str) -> Optional[str]:
    payload = {
        "model": IMAGE_CAPTION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }],
        "max_tokens": MAX_TOKENS,
    }
    try:
        resp = client.post(f"{PROXY_URL}/v1/chat/completions", json=payload)
    except httpx.HTTPError as e:
        logger.warning(f"Caption request failed: {e}")
        return None
    if resp.status_code != 200:
        logger.warning(f"Caption request failed: HTTP {resp.status_code} — {resp.text[:200]}")
        return None
    choices = resp.json().get("choices") or []
    text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    return text or None


def process_pending_batch(limit: int = IMAGE_CAPTION_BATCH_SIZE) -> int:
    """Caption up to `limit` stored images. Returns how many were attempted."""
    work = iter_attachments(needs_caption, limit)
    if not work:
        return 0

    archive_root = os.path.dirname(ARCHIVE_MEDIA_DIR.rstrip("/"))
    by_user: Dict[str, Dict[tuple, dict]] = {}
    done = 0

    with httpx.Client(timeout=IMAGE_CAPTION_TIMEOUT_S) as client:
        for user_id, message_id, attachment_id, att in work:
            attempts = att.get("caption_attempts", 0) + 1
            changes: dict = {"caption_attempts": attempts}
            try:
                data_url = encode_image(os.path.join(archive_root, att["media_path"]))
            except Exception as e:  # unreadable or not really an image
                logger.info(f"Unreadable image {att.get('media_path')}: {e}")
                changes["caption_status"] = "unreadable"
            else:
                caption = caption_image(client, data_url)
                if caption:
                    changes.update(
                        caption=caption,
                        caption_status="done",
                        caption_model=IMAGE_CAPTION_MODEL,
                        caption_prompt_version=PROMPT_VERSION,
                    )
                    done += 1
                elif attempts >= IMAGE_CAPTION_MAX_ATTEMPTS:
                    changes["caption_status"] = "error"
            by_user.setdefault(user_id, {})[(message_id, attachment_id)] = changes

    for user_id, changes in by_user.items():
        update_attachments(user_id, changes)

    logger.info(f"Caption batch: {done}/{len(work)} captioned")
    return len(work)
