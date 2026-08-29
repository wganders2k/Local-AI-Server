"""
Reclaiming context in a long-running lore thread.

Compaction is the one operation allowed to break the session's append-only
rule. It rewrites the oldest research into a summary, which knowingly voids the
prefix cache and costs a full re-prefill on the next turn — cheaper than the
alternative, which is the thread hitting the context ceiling and refusing to
answer at all.
"""

import logging

from config import AGENT_CTX_COMPACT_PCT, AGENT_MODEL
from lore.context_window import limit as ctx_limit
from lore.agent import stream_answer
from lore.metrics import AgentMetrics
from lore.prompts import build_compaction_messages
from lore.session import LoreSession
from proxy_client import ProxyClient, ProxyError

logger = logging.getLogger("mimic-bot.lore.compaction")


async def _compact_session(
    session: LoreSession,
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
) -> bool:
    """
    Condense the oldest half of a session's research to reclaim context.

    This is the one operation that rewrites the transcript, so it knowingly
    voids the prefix cache and costs a full re-prefill on the next turn. That
    is cheaper than the alternative, which is the thread hitting the context
    ceiling and refusing to answer at all.

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

    prompt = build_compaction_messages(original)

    # No tools here on purpose: compaction already invalidates the prefix, so
    # there is no cache to protect, and a summariser has no use for them.
    #
    # No reasoning either. The agent model is a hybrid reasoner and deliberates
    # by default even on mechanical work — a summarising call of this shape was
    # measured at 52.2s and 724 completion tokens with thinking on, versus 1.0s
    # and 27 tokens with it off, for the same output. Compression is extraction,
    # not judgement.
    try:
        text, usage = await stream_answer(
            proxy_client, AGENT_MODEL, prompt, enable_thinking=False
        )
    except ProxyError as e:
        logger.warning("Compaction failed, leaving transcript intact: %s", e)
        return False

    summary = text.strip()
    if not summary:
        logger.warning("Compaction produced nothing, leaving transcript intact")
        return False

    metrics.record_usage(usage)
    session.compactions += 1
    condensed = {
        "role": "user",
        "content": (
            f"[Condensed research {session.compactions}] earlier searches, "
            f"summarised to save context\n{summary}"
        ),
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

    logger.info(
        "Compaction done: %d chars -> %d chars (%.0f%% saved)",
        len(original), len(summary),
        (1 - len(summary) / max(len(original), 1)) * 100,
    )
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
