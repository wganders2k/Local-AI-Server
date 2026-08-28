"""
Getting model output into Discord: streaming, splitting, posting.

Discord caps a message at 2000 characters, so anything long has to be broken
up. Splitting happens at paragraph or line boundaries where possible
(formatters.find_split_boundary) so a code block or list does not get cut
mid-line.
"""

import logging
from typing import Awaitable, Callable, Optional

import discord

from config import MAX_MESSAGE_LENGTH
from formatters import find_split_boundary, format_mimic_response
from proxy_client import ProxyClient

logger = logging.getLogger("mimic-bot.io")

# Anything that takes a string and posts it: interaction.followup.send,
# channel.send, thread.send.
Sender = Callable[[str], Awaitable[object]]


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Break text into Discord-sized chunks at the best available boundary."""
    chunks: list[str] = []
    buffer = text
    while buffer:
        if len(buffer) <= max_length:
            chunks.append(buffer)
            break
        split_at = find_split_boundary(buffer, max_length)
        chunks.append(buffer[:split_at].rstrip())
        buffer = buffer[split_at:].lstrip("\n")
    return chunks


async def stream_and_send(
    send: Sender,
    proxy_client: ProxyClient,
    model: str,
    messages: list[dict],
) -> str:
    """
    Stream a completion, posting each full chunk as it becomes available.

    Chunks are flushed as soon as the buffer passes the message limit, so a
    long answer starts appearing instead of landing all at once at the end.
    Disclaimer stripping is applied to the final buffer only — the patterns are
    anchored to the end of the response, so applying them mid-stream would
    match text the model has not finished writing.

    Returns:
        The complete response, for the caller to record in history.
    """
    buffer = ""
    full_response = ""

    async for token in proxy_client.chat_stream(model, messages):
        buffer += token
        full_response += token

        while len(buffer) >= MAX_MESSAGE_LENGTH:
            split_at = find_split_boundary(buffer, MAX_MESSAGE_LENGTH)
            chunk = buffer[:split_at].rstrip()
            buffer = buffer[split_at:].lstrip("\n")
            if chunk:
                await send(chunk)

    buffer = format_mimic_response(buffer).strip()
    if buffer:
        await send(buffer)

    return full_response


async def post_lore_answer(
    thread: discord.Thread,
    answer: str,
    footer: str,
    edit_first: Optional[discord.Message] = None,
) -> None:
    """
    Post an answer to a lore thread, split at paragraph boundaries.

    Args:
        edit_first: If given, this message is edited to hold the first chunk
            instead of a new one being sent — so the status message turns into
            the answer rather than leaving a stale progress line behind.
    """
    chunks = split_message(answer) or ["(empty answer)"]

    # Ride the footer along on the last chunk when there is room for it.
    tail = "\n" + footer
    footer_attached = len(chunks[-1]) + len(tail) <= MAX_MESSAGE_LENGTH
    if footer_attached:
        chunks[-1] += tail

    for i, chunk in enumerate(chunks):
        if i == 0 and edit_first is not None:
            await edit_first.edit(content=chunk)
        else:
            await thread.send(chunk)

    if not footer_attached:
        await thread.send(footer)
