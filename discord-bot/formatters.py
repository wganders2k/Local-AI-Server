"""
Response formatting, disclaimer stripping, and truncation.

Post-processing applied to mimic responses before posting to Discord.
"""

import re
from typing import TYPE_CHECKING

from config import (
    DISCLAIMER_PATTERNS,
    MAX_EMBED_DESCRIPTION_LENGTH,
    MAX_MESSAGE_LENGTH,
)

if TYPE_CHECKING:
    import discord


def strip_disclaimers(text: str) -> str:
    """
    Remove baked-in disclaimers from mimic responses.

    Applies each regex pattern from config.DISCLAIMER_PATTERNS
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


def format_mimic_response(text: str) -> str:
    """
    Apply post-processing for mimic responses.

    Strips disclaimers from the final accumulated chunk.
    Truncation is no longer needed — streaming handles message splitting.
    """
    return strip_disclaimers(text)


def build_lore_embed_discord(text: str, chunk_count: int = 0) -> "discord.Embed":
    """
    Build a lore response as an actual discord.Embed instance.

    Use this version when discord is imported in the calling scope.
    """
    import discord  # noqa: F811

    embed = discord.Embed(
        title="\U0001f4da nullposting Lore",
        description=text[:MAX_EMBED_DESCRIPTION_LENGTH],
        colour=0x5865F2,
    )

    if chunk_count > 0:
        embed.set_footer(text=f"Sources: {chunk_count} lore entries retrieved")

    return embed
