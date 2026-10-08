"""
Reclaiming context, in a long-running lore thread and inside a single run.

Compaction is the one operation allowed to break the append-only rule. It
rewrites the oldest research into a summary, which knowingly voids the prefix
cache and costs a full re-prefill on the next request — cheaper than the
alternative, which is hitting the context ceiling and refusing to answer at all.

Two callers, one summariser:

* ``maybe_compact_session`` — between turns of a thread, on its transcript.
* ``compact_tool_rounds``   — between rounds of the tool-calling loop, on the
  in-flight conversation. A 25-round opening /lore run was observed adding
  ~10k tokens a round and overflowing a 96k window at round 10, before any
  thread existed for the session-level pass to act on.
"""

import logging
from typing import Optional

from config import AGENT_CTX_COMPACT_PCT, AGENT_MODEL
from lore.context_window import limit as ctx_limit
from lore.metrics import AgentMetrics
from lore.prompts import build_compaction_messages
from lore.research import CONDENSED_KIND, render_research_blocks
from lore.session import LoreSession
from lore.streaming import stream_answer
from proxy_client import ProxyClient, ProxyError

logger = logging.getLogger("mimic-bot.lore.compaction")

# Tool results are counted in characters until the next usage report arrives.
# Measured at ~3.1 chars/token on RAG excerpts (a 34,088-char result grew the
# next request by 10,901 tokens), so 3 errs toward acting a little early.
TOOL_CHARS_PER_TOKEN = 3


async def _summarise(
    excerpts: str,
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
) -> Optional[str]:
    """
    Condense research text, or return None and leave the caller's state alone.

    No tools here on purpose: compaction already invalidates the prefix, so
    there is no cache to protect, and a summariser has no use for them.

    No reasoning either. The agent model is a hybrid reasoner and deliberates
    by default even on mechanical work — a summarising call of this shape was
    measured at 52.2s and 724 completion tokens with thinking on, versus 1.0s
    and 27 tokens with it off, for the same output. Compression is extraction,
    not judgement.
    """
    try:
        text, usage = await stream_answer(
            proxy_client, AGENT_MODEL, build_compaction_messages(excerpts),
            enable_thinking=False,
        )
    except ProxyError as e:
        logger.warning("Compaction failed, leaving research intact: %s", e)
        return None

    summary = text.strip()
    if not summary:
        logger.warning("Compaction produced nothing, leaving research intact")
        return None

    metrics.record_usage(usage)
    logger.info(
        "Compaction done: %d chars -> %d chars (%.0f%% saved)",
        len(excerpts), len(summary),
        (1 - len(summary) / max(len(excerpts), 1)) * 100,
    )
    return summary


def _condensed_content(n: int, scope: str, summary: str) -> str:
    return f"[Condensed research {n}] {scope}, summarised to save context\n{summary}"


async def _compact_session(
    session: LoreSession,
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
) -> bool:
    """
    Condense the oldest half of a session's research to reclaim context.

    Returns:
        True if the transcript was rewritten.
    """
    indices = session.research_indices()
    if len(indices) < 2:
        return False  # Nothing meaningful to merge.

    victims = indices[: max(1, len(indices) // 2)]
    original = "\n\n".join(session.transcript[i]["content"] for i in victims)

    logger.info(
        "Compacting lore session %d: %d of %d research block(s), %d chars",
        session.thread_id, len(victims), len(indices), len(original),
    )

    summary = await _summarise(original, proxy_client, metrics)
    if summary is None:
        return False

    session.compactions += 1
    condensed = {
        "role": "user",
        "content": _condensed_content(session.compactions, "earlier searches", summary),
        "kind": "research",
    }

    # Replace the first victim in place and drop the rest, so surrounding
    # questions and answers keep their order.
    keep: list[dict] = []
    for i, msg in enumerate(session.transcript):
        if i == victims[0]:
            keep.append(condensed)
        elif i in victims:
            continue
        else:
            keep.append(msg)
    session.transcript = keep
    return True


async def maybe_compact_session(
    session: LoreSession,
    proxy_client: ProxyClient,
) -> bool:
    """
    Compact a session if it has crossed the compaction threshold.

    Called after a turn has been answered and posted, so the cost lands between
    messages rather than in front of the user's answer.
    """
    pct = session.pct_of(ctx_limit())
    if pct < AGENT_CTX_COMPACT_PCT:
        return False
    logger.info(
        "Session %d at %.0f%% of context (>= %.0f%%) — compacting",
        session.thread_id, pct * 100, AGENT_CTX_COMPACT_PCT * 100,
    )
    metrics = AgentMetrics()
    return await _compact_session(session, proxy_client, metrics)


async def compact_tool_rounds(
    messages: list[dict],
    tool_messages: list[dict],
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
    n: int,
) -> int:
    """
    Condense the oldest half of a run's tool results, mid-loop.

    Both lists are rewritten in place. ``messages`` is the live conversation
    and gets an untagged user message where the first victim was;
    ``tool_messages`` gets the same text tagged CONDENSED_KIND, so the
    synthesis prompt and a seeded thread still see what was found.

    Each assistant/``tool_calls`` message is removed together with its result,
    so the conversation never holds a call without its answer.

    Args:
        n: Which compaction this is within the run, for the summary's label.

    Returns:
        Estimated tokens reclaimed — 0 if nothing was changed.
    """
    results = [i for i, m in enumerate(tool_messages) if m.get("role") == "tool"]
    if len(results) < 2:
        return 0  # Nothing meaningful to merge.

    # The loop appends each call directly before its result.
    victims: list[dict] = []
    for i in results[: len(results) // 2]:
        victims.extend(tool_messages[i - 1 : i + 1])
    victim_ids = {id(m) for m in victims}

    blocks, _ = render_research_blocks(victims)
    original = "\n\n".join(blocks)
    removed_chars = sum(
        len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        for m in victims
    )
    logger.info(
        "Compacting in-flight research: %d of %d result(s), %d chars",
        len(victims) // 2, len(results), removed_chars,
    )

    # Results that were all empty have nothing to keep: drop them outright.
    content = ""
    if blocks:
        summary = await _summarise(original, proxy_client, metrics)
        if summary is None:
            return 0
        content = _condensed_content(n, "earlier searches from this run", summary)

    for target, tag in ((messages, {}), (tool_messages, {"kind": CONDENSED_KIND})):
        first = next(i for i, m in enumerate(target) if id(m) in victim_ids)
        target[:] = [m for m in target if id(m) not in victim_ids]
        if content:
            target.insert(first, {"role": "user", "content": content, **tag})

    return max(0, (removed_chars - len(content)) // TOOL_CHARS_PER_TOKEN)
