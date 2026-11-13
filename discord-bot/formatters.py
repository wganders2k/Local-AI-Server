"""
Model output -> Discord-shaped text: disclaimer stripping, splitting to
Discord's limits, and lore embeds.

Discord caps a message at 2000 characters and an embed description at 4096, so
anything long is split — at a paragraph or line boundary where possible, so a
code block or list is not cut mid-line. One splitter serves every caller.
"""

import re
from typing import TYPE_CHECKING

from config import MAX_EMBED_DESCRIPTION_LENGTH, MAX_MESSAGE_LENGTH

# Applied to persona posts and /chat replies. Models occasionally append
# baked-in disclaimers that break character. Anchored to the end of the
# response, which is why stripping happens once on the final buffer rather
# than per streamed chunk.
DISCLAIMER_PATTERNS: list[str] = [
    r"\n+This is general.*$",
    r"\n+This is not legal.*$",
    r"\n+This is not medical.*$",
    r"\n+This is not financial.*$",
    r"\n+Note:.*?(disclaimer|advice|professional).*$",
    r"\n+Please consult.*$",
    r"\n+Please note.*$",
    r"\n+Please be aware.*$",
    r"\n+I'm an AI.*$",
    r"\n+I am an AI.*$",
    r"\n+As an AI.*$",
    r"\n+Remember, I'm.*$",
    r"\n+Remember, I am.*$",
]

if TYPE_CHECKING:
    import discord


def strip_disclaimers(text: str) -> str:
    """
    Remove baked-in trailing disclaimers.

    Applies each regex pattern from DISCLAIMER_PATTERNS above
    to strip trailing disclaimers.
    """
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def find_split_boundary(buffer: str, max_length: int) -> int:
    """
    Find the best split index in a buffer exceeding max_length.

    Prefers splitting at paragraph breaks (\n\n), then line breaks (\n),
    within the first max_length characters. Falls back to max_length
    if no suitable break is found.

    Args:
        buffer: The accumulated text buffer.
        max_length: Maximum desired chunk length.

    Returns:
        The index to split at.
    """
    if len(buffer) <= max_length:
        return len(buffer)

    # Search within the first max_length characters
    search_range = buffer[:max_length]

    # Prefer paragraph breaks (\n\n)
    idx = search_range.rfind("\n\n")
    if idx >= 0:
        return idx + 2  # Include both newlines in the split point

    # Next prefer line breaks (\n)
    idx = search_range.rfind("\n")
    if idx >= 0:
        return idx + 1

    # Fallback: hard split at max_length
    return max_length


def take_chunk(text: str, max_length: int) -> tuple[str, str]:
    """(the first chunk of `text` that fits, the rest), split at the best boundary."""
    split_at = find_split_boundary(text, max_length)
    return text[:split_at].rstrip(), text[split_at:].lstrip("\n")


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Break text into chunks of at most `max_length` at the best available boundaries."""
    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        chunk, text = take_chunk(text, max_length)
        chunks.append(chunk)
    return chunks


_EMBED_TITLE = "\U0001f4da Lore"
_EMBED_COLOUR = 0x5865F2


def build_lore_embeds(text: str) -> list["discord.Embed"]:
    """
    Build one or more discord.Embed instances for a lore response.

    If the text exceeds Discord's embed description limit (~4096 chars),
    it is split into multiple embeds with pagination footers.
    """
    import discord  # noqa: F811

    if not text:
        return [discord.Embed(title=_EMBED_TITLE, description="(No results)", colour=_EMBED_COLOUR)]

    chunks = split_message(text, MAX_EMBED_DESCRIPTION_LENGTH)

    embeds = []
    for i, chunk in enumerate(chunks):
        embed = discord.Embed(
            title=_EMBED_TITLE,
            description=chunk,
            colour=_EMBED_COLOUR,
        )
        if len(chunks) > 1:
            embed.set_footer(text=f"Part {i + 1}/{len(chunks)}")
        embeds.append(embed)

    return embeds
