"""
Describing images posted to the bot, as text the models already handle.

A caption is made by whichever model is already loaded for the feature —
[gemma4-12b-mimic-u] with no adapter applied for mimic chats, the /lore agent model
for lore threads — so describing an image never costs a model swap. Both
presets carry a vision projector in models.ini.

Mimic: captions reach the chat controller only (mimic.controller renders
"[image: ...]"). Persona prompts still show "[attachment]", because that is
what the adapters were trained on (shared/mimic_format.context_line); showing
them captions waits for a retrain on captioned data.

Lore: the caption is appended to the question text, so the session, its
persistence and compaction stay plain text.

The prompt is prompts/caption.md, kept identical to history-service's, so a
live caption reads like an archived one.
"""

import base64
import io
import logging
import dataclasses
import os
from collections import OrderedDict
from typing import Iterable, Optional

from PIL import Image

from config import VISION_CACHE_SIZE, VISION_MAX_BYTES, VISION_MAX_NEW_PER_BURST, VISION_MAX_SIDE, VISION_MAX_TOKENS
from prompt_loader import load
from proxy_client import ProxyClient, ProxyError

logger = logging.getLogger("mimic-bot.vision")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def is_image(attachment) -> bool:
    """Whether an attachment is a still image (or a GIF, read by its first frame)."""
    ctype = (getattr(attachment, "content_type", None) or "").split(";")[0]
    if ctype:
        return ctype.startswith("image/") and ctype != "image/svg+xml"
    return os.path.splitext(getattr(attachment, "filename", "") or "")[1].lower() in _IMAGE_EXTS


def encode(data: bytes, max_side: int = VISION_MAX_SIDE) -> str:
    """
    A data: URL as JPEG. llama.cpp's image loader reads no WebP, a GIF is
    described by its first frame, and alpha says nothing to a caption.
    """
    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Captioner:
    """Captions with a cache keyed by attachment id, shared by every feature."""

    def __init__(self, proxy: ProxyClient, cache_size: int = VISION_CACHE_SIZE):
        self.proxy = proxy
        self._cache: OrderedDict[int, str] = OrderedDict()
        self._size = cache_size

    def cached(self, attachment) -> Optional[str]:
        caption = self._cache.get(attachment.id)
        if caption is not None:
            self._cache.move_to_end(attachment.id)
        return caption

    async def caption(self, attachment, model: str, lora: Optional[list[dict]] = None) -> Optional[str]:
        """
        One image's description, or None when it cannot be had (not an image,
        too large, unreadable, the proxy failed). Never raises: a missing
        caption leaves the plain marker, which is what the bot showed before.
        """
        hit = self.cached(attachment)
        if hit is not None:
            return hit
        if not is_image(attachment) or (getattr(attachment, "size", 0) or 0) > VISION_MAX_BYTES:
            return None
        try:
            data_url = encode(await attachment.read())
            text = await self.proxy.chat(
                model,
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": load("caption")},
                ]}],
                lora=lora,
                temperature=0.2,
                max_tokens=VISION_MAX_TOKENS,
                # A description is mechanical work; the lore agent's preset
                # thinks by default, which here would only cost time.
                chat_template_kwargs={"enable_thinking": False},
            )
        except ProxyError as e:
            logger.warning("Caption failed for %s: %s", getattr(attachment, "filename", "?"), e)
            return None
        except Exception as e:  # unreadable image, download failure
            logger.info("Could not read %s: %s", getattr(attachment, "filename", "?"), e)
            return None
        caption = " ".join(text.split())
        self._cache[attachment.id] = caption
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        logger.info("Captioned %s: %s", getattr(attachment, "filename", "?"), caption[:80])
        return caption

    async def captions(self, attachments: Iterable, model: str, lora: Optional[list[dict]] = None) -> list[str]:
        """Captions for every image among `attachments` that could be described, in order."""
        out = []
        for att in attachments:
            if is_image(att):
                c = await self.caption(att, model, lora)
                if c:
                    out.append(c)
        return out


def with_captions(text: str, captions: Iterable[str]) -> str:
    """`text` followed by one "[image: ...]" per caption."""
    tags = " ".join(f"[image: {c}]" for c in captions)
    return f"{text} {tags}".strip() if tags else text


async def annotate(
    captioner: Optional[Captioner],
    lines: list,
    model: str,
    lora: Optional[list[dict]] = None,
    max_new: int = VISION_MAX_NEW_PER_BURST,
) -> list:
    """
    `lines` (mimic.transcript.Line) with `captions` filled from their images.
    Newest images are captioned first, and at most `max_new` that are not
    already cached, so a burst after a screenshot dump is not held up; the
    rest are described on later bursts. Cached captions are always applied.
    """
    if captioner is None:
        return lines
    budget = max_new
    out = list(lines)
    for i in range(len(out) - 1, -1, -1):
        line = out[i]
        if not line.images:
            continue
        caps = []
        for att in line.images:
            c = captioner.cached(att)
            if c is None and budget > 0:
                budget -= 1
                c = await captioner.caption(att, model, lora)
            if c:
                caps.append(c)
        if caps:
            out[i] = dataclasses.replace(line, captions=tuple(caps))
    return out
