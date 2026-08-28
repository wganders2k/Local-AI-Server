"""
Turning a tool-calling conversation into plain research text.

Two jobs, both about the same problem: the model imitates whatever shape it
sees. Leaving assistant/``tool_calls`` and ``role="tool"`` messages in context
teaches it to answer with tool-call markup, so anything that has to be *read*
rather than *continued* is rendered as flat labelled text instead.

Chunks repeated across searches are dropped, keeping the first occurrence, so
what has to be prefilled stays proportional to what was actually found rather
than to how many overlapping queries the agent happened to run.
"""

import json
import logging
from typing import Optional

from lore.prompts import build_synthesis_prompt
from lore.session import chunk_key

logger = logging.getLogger("mimic-bot.lore.research")


# How the RAG service joins retrieved chunks inside a single tool result
# (rag/retrieve.py :: format_context_block). Splitting on it lets repeated
# chunks be dropped across searches. If that formatter ever changes separator
# this degrades safely: the split yields one element and dedup still catches
# whole results that are byte-identical.
_RAG_CHUNK_SEPARATOR = "\n\n---\n\n"

def _flatten_research_for_synthesis(
    messages: list[dict],
    user_question: str,
) -> list[dict]:
    """
    Collapse a tool-calling conversation into a flat two-message exchange.

    Replacing only the system prompt was not enough to stop the model ending a
    run by emitting tool-call markup as its answer. The assistant/``tool_calls``
    and ``role="tool"`` messages stayed in context, and a dozen rounds of
    "assistant emits a tool call, tool returns a result" is a stronger signal
    than a single instruction not to emit tool calls. When the model imitates
    that pattern on the synthesis call the markup reaches the user verbatim:
    the request carries no tools, so the backend has no tool parser to lift it
    out of the response the way it does during the loop itself.

    Rendering the same research as plain text under one user turn leaves no
    tool-calling pattern in context to continue.

    Chunks repeated across searches are dropped, keeping the first occurrence.
    The flattened prompt is new text every time and so cannot reuse the KV
    cache prefix the incremental conversation enjoyed; dropping redundant
    chunks keeps what has to be prefilled proportional to what was actually
    found rather than to how many overlapping queries the agent happened to run.

    Args:
        messages: The full agent-loop conversation.
        user_question: The original question from the /lore command.

    Returns:
        A two-message list — the synthesis system prompt, and one user turn
        carrying the question plus every tool result gathered during the loop.
    """
    blocks, _ = render_research_blocks(messages)

    if blocks:
        user_content = (
            f"Question: {user_question}\n\n"
            "Below is everything already retrieved from the Discord archive to "
            "answer it. No further searching is possible — write the final "
            "answer using only what appears here.\n\n"
            + "\n\n".join(blocks)
        )
    else:
        user_content = (
            f"Question: {user_question}\n\n"
            "No usable results came back from the Discord archive. Tell the "
            "user plainly that you could not find this information."
        )

    return [
        {"role": "system", "content": build_synthesis_prompt()},
        {"role": "user", "content": user_content},
    ]


def render_research_blocks(
    messages: list[dict],
    seen_keys: Optional[set[str]] = None,
    start_index: int = 0,
) -> tuple[list[str], int]:
    """
    Render the tool results in ``messages`` as labelled plain-text blocks.

    Args:
        messages: A conversation containing assistant/``tool_calls`` messages
            and their matching ``role="tool"`` results.
        seen_keys: Chunk identities already shown. Mutated in place, so a
            lore session can pass its own set and have a later turn skip
            material an earlier turn already put in the transcript. Pass None
            for a self-contained run.
        start_index: Number to start "[Search N]" labels from, so blocks
            appended across turns keep counting up instead of restarting at 1.

    Returns:
        (blocks, dropped) — the rendered blocks, and how many duplicate chunks
        were suppressed.
    """
    # tool_call_id -> description of what was searched, so each result block
    # keeps its provenance. Rendered as prose rather than echoing function
    # names, which would put tool-calling vocabulary back into context.
    search_labels: dict[str, str] = {}
    for msg in messages:
        for tc in msg.get("tool_calls") or ():
            function = tc.get("function", {}) or {}
            raw_args = function.get("arguments", "")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (json.JSONDecodeError, TypeError):
                args = {}

            query = args.get("query")
            channel = args.get("channel_name")
            start_date = args.get("start_date")
            end_date = args.get("end_date")

            parts: list[str] = []
            if query:
                parts.append(f'"{query}"')
            if channel:
                parts.append(f"in {channel}")
            if start_date or end_date:
                parts.append(f"between {start_date or 'any'} and {end_date or 'any'}")

            search_labels[tc.get("id", "")] = " ".join(parts) if parts else "(unlabelled search)"

    # The agent tends to issue near-identical queries across rounds, which return
    # heavily overlapping chunks. Keeping only the first occurrence of each chunk
    # preserves every distinct fact while cutting the prompt that has to be
    # prefilled from scratch on the synthesis call.
    keys = seen_keys if seen_keys is not None else set()
    fresh_count = 0
    dropped = 0

    blocks: list[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        result = (msg.get("content") or "").strip()
        if not result:
            continue
        label = search_labels.get(msg.get("tool_call_id", "")) or "(unlabelled search)"

        fresh: list[str] = []
        for chunk in result.split(_RAG_CHUNK_SEPARATOR):
            chunk = chunk.strip()
            if not chunk:
                continue
            key = chunk_key(chunk)
            if key in keys:
                dropped += 1
                continue
            keys.add(key)
            fresh_count += 1
            fresh.append(chunk)

        # A search whose every hit was already shown adds nothing to synthesise.
        if not fresh:
            continue

        blocks.append(
            f"[Search {start_index + len(blocks) + 1}] {label}\n"
            + _RAG_CHUNK_SEPARATOR.join(fresh)
        )

    logger.info(
        "Research payload: %d new chunk(s) across %d search block(s), "
        "%d duplicate chunk(s) dropped",
        fresh_count, len(blocks), dropped,
    )
    return blocks, dropped
