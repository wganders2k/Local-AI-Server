"""
Streaming one completion from the agent model.

Lives apart from lore.agent so lore.compaction can use it without importing the
agent loop, which itself calls into compaction.
"""

import logging
import time
from typing import Optional

from lore.progress import ProgressReporter
from proxy_client import ProxyClient

logger = logging.getLogger("mimic-bot.lore.agent")

# How often the "generating final answer" status message is refreshed while the
# synthesis streams. Discord allows roughly 5 edits per 5s on one message; the
# point here is only to show the answer is still coming, so keep it well under.
_SYNTHESIS_PROGRESS_INTERVAL = 10.0


async def stream_answer(
    proxy_client: ProxyClient,
    model: str,
    messages: list[dict],
    *,
    status: Optional[ProgressReporter] = None,
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = True,
) -> tuple[str, dict]:
    """
    Stream one completion, refreshing a status message as it arrives.

    Streamed rather than buffered everywhere it is used, for two reasons. The
    read timeout is a gap-between-reads, so on a buffered call it becomes a
    deadline on the entire generation — a ~50k-token synthesis that prefills for
    50s and then writes a long answer ran past it routinely. And dropping the
    socket on a streamed request actually cancels the work upstream, instead of
    leaving the backend to generate into nothing while holding the GPU.

    Args:
        tools: Pass the same tools as the surrounding calls even when the model
            must not use them — see ProxyClient.chat_stream for why dropping
            them cold-prefills the whole conversation.
        enable_thinking: False for mechanical work (summarising, compression).

    Returns:
        (text, usage) — the joined completion, unstripped, and the backend's
        usage dict (empty if it reported none).
    """
    usage: dict = {}
    parts: list[str] = []
    chars = 0
    last_edit = time.monotonic()

    async for token in proxy_client.chat_stream(
        model,
        messages,
        usage_sink=usage,
        tools=tools,
        enable_thinking=enable_thinking,
    ):
        parts.append(token)
        chars += len(token)
        if status is not None and time.monotonic() - last_edit >= _SYNTHESIS_PROGRESS_INTERVAL:
            last_edit = time.monotonic()
            await status.generating(chars)

    text = "".join(parts)
    logger.info("Streamed %d chars in %d chunk(s)", len(text), len(parts))
    return text, usage
