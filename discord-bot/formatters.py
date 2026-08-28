"""
Response formatting, disclaimer stripping, and truncation.

Post-processing applied to mimic responses before posting to Discord.
"""

import re
from typing import TYPE_CHECKING

from config import MAX_EMBED_DESCRIPTION_LENGTH

# Applied to mimic responses only. The base model occasionally appends
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
    Remove baked-in disclaimers from mimic responses.

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


def format_mimic_response(text: str) -> str:
    """
    Apply post-processing for mimic responses.

    Strips disclaimers from the final accumulated chunk.
    Truncation is no longer needed — streaming handles message splitting.
    """
    return strip_disclaimers(text)


def build_lore_embeds(text: str) -> list["discord.Embed"]:
    """
    Build one or more discord.Embed instances for a lore response.
    
    If the text exceeds Discord's embed description limit (~4096 chars),
    it is split into multiple embeds with pagination footers.
    """
    import discord  # noqa: F811

    if not text:
        return [discord.Embed(title="\U0001f4da nullposting Lore", description="(No results)", colour=0x5865F2)]

    # Split text into chunks that fit within Discord's embed description limit
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= MAX_EMBED_DESCRIPTION_LENGTH:
            chunks.append(remaining)
            break
        
        split_idx = find_split_boundary(remaining, MAX_EMBED_DESCRIPTION_LENGTH)
        # Safety fallback: if no break point found, hard split
        if split_idx == 0:
            split_idx = MAX_EMBED_DESCRIPTION_LENGTH
            
        chunks.append(remaining[:split_idx].rstrip())
        remaining = remaining[split_idx:].lstrip()

    embeds = []
    for i, chunk in enumerate(chunks):
        embed = discord.Embed(
            title="\U0001f4da nullposting Lore",
            description=chunk,
            colour=0x5865F2,
        )
        if len(chunks) > 1:
            embed.set_footer(text=f"Part {i + 1}/{len(chunks)}")
        embeds.append(embed)

    return embeds
