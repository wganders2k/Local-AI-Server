"""
Getting model output into Discord: streaming and posting. The text shaping —
splitting to Discord's limit, stripping disclaimers — is formatters.py's.
"""

import logging
from typing import Awaitable, Callable, Optional

import discord

from config import MAX_MESSAGE_LENGTH
from formatters import split_message, strip_disclaimers, take_chunk
from proxy_client import ProxyClient

logger = logging.getLogger("mimic-bot.io")

# Anything that takes a string and posts it: interaction.followup.send,
# channel.send, thread.send.
Sender = Callable[[str], Awaitable[object]]

# Lore answers are model output as plain message text, which can ping: an
# "@everyone" in an answer would fire. Embeds never pinged, so neither does this.
NO_PINGS = discord.AllowedMentions.none()


async def post_plain_answer(
    channel: discord.abc.Messageable,
    text: str,
    reply_to: Optional[discord.Message] = None,
    fallback: Optional[Sender] = None,
) -> None:
    """
    Post an answer as plain messages, split at paragraph boundaries. Only the
    first is a reply (to ``reply_to``); the rest follow as ordinary messages.

    Args:
        fallback: Used for any chunk the channel refuses (no send permission,
            or a DM the bot cannot open) — an interaction's followup.send,
            which works wherever the command was allowed.
    """
    for i, chunk in enumerate(split_message(text) or ["(No results)"]):
        try:
            if i == 0 and reply_to is not None:
                await reply_to.reply(chunk, mention_author=False, allowed_mentions=NO_PINGS)
            else:
                await channel.send(chunk, allowed_mentions=NO_PINGS)
        except discord.HTTPException:
            if fallback is None:
                raise
            logger.warning("Channel refused an answer chunk — sending it as a followup", exc_info=True)
            await fallback(chunk)


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
            chunk, buffer = take_chunk(buffer, MAX_MESSAGE_LENGTH)
            if chunk:
                await send(chunk)

    buffer = strip_disclaimers(buffer)
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
            await edit_first.edit(content=chunk, allowed_mentions=NO_PINGS)
        else:
            await thread.send(chunk, allowed_mentions=NO_PINGS)

    if not footer_attached:
        await thread.send(footer)
