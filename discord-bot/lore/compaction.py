"""
Reclaiming context, in a long-running lore thread and inside a single run.

Compaction is the one operation allowed to break the append-only rule. It
rewrites the oldest research into a summary, which knowingly voids the prefix
cache and costs a full re-prefill on the next request — cheaper than the
alternative, which is hitting the context ceiling and refusing to answer at all.

Every check is made against the window of the model about to receive the
context (lore.context_window.window). Four callers, one summariser:

* ``maybe_compact_session`` — between turns of a thread, on its transcript.
* ``compact_tool_rounds``   — between rounds of the tool-calling loop, on the
  in-flight conversation. A 25-round opening /lore run was observed adding
  ~10k tokens a round and overflowing a 96k window at round 10, before any
  thread existed for the session-level pass to act on.
* ``fit_research_for`` and ``fit_session_for`` — right before a run hands its
  research to a different writer model. Research is budgeted against the
  research model's window while it is gathered, which can be larger than what
  the writer takes, so it is condensed again until it fits.

The summariser is always whichever model is already loaded — the research
model mid-run and before a handover, the writer after a thread's answer — so
compaction never costs a model swap of its own.
"""

import logging
from typing import Optional

from config import AGENT_CTX_COMPACT_PCT
from lore.context_window import limit as ctx_limit, window
from lore.models import writer_model
from lore.metrics import TOOL_CHARS_PER_TOKEN, AgentMetrics
from lore.prompts import build_compaction_messages
from lore.research import (
    CONDENSED_KIND,
    flatten_research_for_synthesis,
    render_research_blocks,
    split_chunks,
)
from lore.session import LoreSession, chunk_key
from lore.streaming import stream_answer
from proxy_client import ProxyClient, ProxyError

logger = logging.getLogger("mimic-bot.lore.compaction")

async def _summarise(
    excerpts: str,
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
    model: str,
) -> Optional[str]:
    """
    Condense research text, or return None and leave the caller's state alone.

    No tools here on purpose: compaction already invalidates the prefix, so
    there is no cache to protect, and a summariser has no use for them.

    No reasoning either. The lore models are hybrid reasoners and deliberates
    by default even on mechanical work — a summarising call of this shape was
    measured at 52.2s and 724 completion tokens with thinking on, versus 1.0s
    and 27 tokens with it off, for the same output. Compression is extraction,
    not judgement.
    """
    try:
        text, usage = await stream_answer(
            proxy_client, model, build_compaction_messages(excerpts),
            enable_thinking=False,
        )
    except ProxyError as e:
        logger.warning("Compaction failed, leaving research intact: %s", e)
        return None

    summary = text.strip()
    if not summary:
        logger.warning("Compaction produced nothing, leaving research intact")
        return None

    metrics.record_usage(usage, model=model)
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
    model: str,
) -> bool:
    """
    Condense the oldest half of a session's research to reclaim context, with
    ``model`` writing the summary.

    Returns:
        True if the transcript was rewritten.
    """
    units = session.research_units()
    if len(units) < 2:
        return False  # Nothing meaningful to merge.

    victims = units[: max(1, len(units) // 2)]
    parts: list[str] = []
    for unit in victims:
        msgs = [session.transcript[i] for i in unit]
        if msgs[0].get("kind") == "research":
            parts.append(msgs[0]["content"])
        else:
            parts.extend(render_research_blocks(msgs)[0])
    original = "\n\n".join(parts)
    gone = {i for unit in victims for i in unit}

    logger.info(
        "Compacting lore session %d: %d of %d research unit(s), %d chars",
        session.thread_id, len(victims), len(units), len(original),
    )

    summary = await _summarise(original, proxy_client, metrics, model) if original else ""
    if summary is None:
        return False

    if summary:
        session.compactions += 1
    condensed = {
        "role": "user",
        "content": _condensed_content(session.compactions, "earlier searches", summary),
        "kind": "research",
    }

    # Replace the first victim in place and drop the rest, so surrounding
    # questions and answers keep their order. A call always goes with its
    # result, so no call is left without its answer.
    first = victims[0][0]
    keep: list[dict] = []
    for i, msg in enumerate(session.transcript):
        if i == first:
            if summary:
                keep.append(condensed)
        elif i not in gone:
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

    Measured against context_window.limit(speed, writer): the research model
    and the writer both receive the whole transcript on the next turn, so it
    has to fit the smaller of the two. The writer summarises: it wrote the
    answer just before this, so it is the model already loaded.
    """
    pct = session.pct_of(ctx_limit(session.speed, session.writer))
    if pct < AGENT_CTX_COMPACT_PCT:
        return False
    logger.info(
        "Session %d at %.0f%% of context (>= %.0f%%) — compacting",
        session.thread_id, pct * 100, AGENT_CTX_COMPACT_PCT * 100,
    )
    model = writer_model(session.speed, session.writer)
    metrics = AgentMetrics(speed=session.speed, model=model)
    return await _compact_session(session, proxy_client, metrics, model)


async def compact_tool_rounds(
    messages: list[dict],
    tool_messages: list[dict],
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
    n: int,
    model: str,
    shown_keys: Optional[set[str]] = None,
    free_tokens: Optional[int] = None,
) -> int:
    """
    Condense the oldest half of a run's tool results, mid-loop — or, with
    ``free_tokens``, just enough of the oldest to free about that much.

    Both lists are rewritten in place. ``messages`` is the live conversation
    and gets an untagged user message where the first victim was;
    ``tool_messages`` gets the same text tagged CONDENSED_KIND, so the
    synthesis prompt and a seeded thread still see what was found.

    Each assistant/``tool_calls`` message is removed together with its result,
    so the conversation never holds a call without its answer.

    Args:
        n: Which compaction this is within the run, for the summary's label.
        model: Writes the summary — the research model, which is loaded.
        shown_keys: The loop's record of chunks in context. The condensed
            chunks are removed from it, so a later search that finds one again
            shows it in full rather than calling it already shown.
        free_tokens: Size the cut to the overage instead of halving. Results
            vary from a few hundred chars to tens of thousands, so halving by
            count was seen spending a whole pass on 192 chars.

    Returns:
        Estimated tokens reclaimed — 0 if nothing was changed.
    """
    results = [i for i, m in enumerate(tool_messages) if m.get("role") == "tool"]
    if len(results) < 2:
        return 0  # Nothing meaningful to merge.

    # The loop appends each call directly before its result.
    if free_tokens is None:
        take = results[: len(results) // 2]
    else:
        # Oldest first, never the newest, until the cut covers the overage
        # with a margin for the summary that replaces it.
        take, chars = [], 0
        for i in results[:-1]:
            take.append(i)
            chars += len(tool_messages[i].get("content") or "")
            if chars // TOOL_CHARS_PER_TOKEN >= free_tokens * 1.25:
                break
    victims: list[dict] = []
    for i in take:
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
        summary = await _summarise(original, proxy_client, metrics, model)
        if summary is None:
            return 0
        content = _condensed_content(n, "earlier searches from this run", summary)

    if shown_keys is not None:
        for m in victims:
            if m.get("role") == "tool":
                for chunk in split_chunks(m.get("content") or ""):
                    shown_keys.discard(chunk_key(chunk))

    for target, tag in ((messages, {}), (tool_messages, {"kind": CONDENSED_KIND})):
        first = next(i for i, m in enumerate(target) if id(m) in victim_ids)
        target[:] = [m for m in target if id(m) not in victim_ids]
        if content:
            target.insert(first, {"role": "user", "content": content, **tag})

    return max(0, (removed_chars - len(content)) // TOOL_CHARS_PER_TOKEN)


def estimate_tokens(messages: list[dict]) -> int:
    """
    Size of a request that has not been sent yet, from its characters.

    At the research ratio (TOOL_CHARS_PER_TOKEN), which overestimates ordinary
    prose slightly: the point is to stay under a window, not to hit it.
    """
    return sum(len(m.get("content") or "") for m in messages) // TOOL_CHARS_PER_TOKEN


def _budget(model: str) -> int:
    """Prompt tokens a request to ``model`` may carry: the compaction threshold
    of its window, which leaves the rest for the answer."""
    return int(AGENT_CTX_COMPACT_PCT * window(model))


async def fit_research_for(
    model: str,
    messages: list[dict],
    tool_messages: list[dict],
    user_question: str,
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
    summariser: str,
) -> list[dict]:
    """
    The flattened synthesis prompt for a run's research, condensed until it
    fits ``model``.

    Each pass condenses as many of the oldest results still held in full as
    cover the overage, so the newest research is the last to go and one pass
    is usually enough. ``tool_messages`` is rewritten in
    place, so a thread seeded from the run carries the condensed form too.

    Args:
        messages: The run's conversation. Not modified; only needed because
            compact_tool_rounds keeps the two lists in step.
        summariser: The model that condenses — the research model, still
            loaded when the handover starts.

    Returns:
        The synthesis messages, ready to send. If nothing is left to condense
        and they are still over budget, they are returned as they are, with a
        warning: the request may still fit, as the budget leaves answer room.
    """
    budget = _budget(model)
    synthesis = flatten_research_for_synthesis(tool_messages, user_question)
    scratch = list(messages)
    n = sum(1 for m in tool_messages if m.get("kind") == CONDENSED_KIND)
    while (size := estimate_tokens(synthesis)) >= budget:
        logger.info(
            "Research is ~%s tokens, over %s's budget of %s — condensing",
            f"{size:,}", model, f"{budget:,}",
        )
        n += 1
        if not await compact_tool_rounds(
            scratch, tool_messages, proxy_client, metrics, n, summariser,
            free_tokens=size - budget,
        ):
            logger.warning("Nothing left to condense — sending ~%s tokens to %s", f"{size:,}", model)
            break
        synthesis = flatten_research_for_synthesis(tool_messages, user_question)
    return synthesis


async def fit_session_for(
    model: str,
    session: LoreSession,
    extra: list[dict],
    proxy_client: ProxyClient,
    metrics: AgentMetrics,
    summariser: str,
) -> list[dict]:
    """
    A thread's request for ``model``, with the session compacted until it fits.

    The thread-level counterpart of fit_research_for: the research model may
    have searched past what the writer takes.

    Args:
        extra: Ephemeral tail messages, as for session.build_messages().
        summariser: The model that condenses — the research model, which
            just ran this turn's rounds.

    Returns:
        session.build_messages(extra), within budget unless there was nothing
        left to compact.
    """
    budget = _budget(model)
    previous = None
    while (size := estimate_tokens(request := session.build_messages(extra=extra))) >= budget:
        # Condensed blocks are research too, so a pass can re-summarise a
        # summary; stop once that no longer buys anything.
        if previous is not None and size > previous * 0.95:
            logger.warning("Compaction stopped shrinking the thread — sending ~%s tokens to %s", f"{size:,}", model)
            break
        logger.info(
            "Thread %d is ~%s tokens, over %s's budget of %s — compacting",
            session.thread_id, f"{size:,}", model, f"{budget:,}",
        )
        if not await _compact_session(session, proxy_client, metrics, summariser):
            logger.warning("Nothing left to compact — sending ~%s tokens to %s", f"{size:,}", model)
            break
        previous = size
    return request
