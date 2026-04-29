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


def truncate_response(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """
    Truncate a response to fit within Discord's message length limit.

    Attempts to break at the last complete sentence before the limit.
    If no sentence boundary is found, hard-truncates at the limit.
    """
    if len(text) <= max_length:
        return text

    # Try to find a sentence boundary near the limit
    truncated = text[:max_length]

    # Look for the last sentence-ending punctuation
    last_period = truncated.rfind(".")
    last_exclaim = truncated.rfind("!")
    last_question = truncated.rfind("?")

    boundary = max(last_period, last_exclaim, last_question)

    if boundary > max_length * 0.5:
        # Only use sentence boundary if it's in the latter half
        return text[: boundary + 1].rstrip()

    # Hard truncate — no good sentence boundary found
    return truncated.rstrip() + "..."


def format_mimic_response(text: str) -> str:
    """
    Apply all post-processing steps for mimic responses.

    1. Strip disclaimers
    2. Truncate to Discord message limit
    """
    text = strip_disclaimers(text)
    text = truncate_response(text, MAX_MESSAGE_LENGTH)
    return text


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
