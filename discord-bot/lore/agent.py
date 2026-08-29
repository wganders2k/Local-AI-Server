"""
The /lore tool-calling loop.

Three entry points, one round-handling implementation:

* ``run_agent_loop``  — the opening /lore run, from a cold question.
* ``run_lore_turn``   — one follow-up turn inside an existing thread.
* ``seed_session_from_run`` — turn a finished opening run into a thread session
  without re-searching anything.

The recurring concern in here is the backend's prefix cache. llama.cpp keys it
on the longest common prefix of the previous request, so anything that changes
early tokens — a ticking timestamp in the system prompt, dropping the tools
payload between calls, rewriting the transcript — costs a full cold prefill of
~50k tokens. Where that trade-off is made deliberately, the comment says so.

This module must not import discord: progress goes through
``lore.progress.ProgressReporter``.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import (
    AGENT_CTX_HARD_PCT,
    AGENT_CTX_SOFT_PCT,
    AGENT_MODEL,
    AGENT_MAX_ROUNDS,
    AGENT_MAX_ROUNDS_HARD_CAP,
    AGENT_TEMPERATURE,
    LORE_FOLLOWUP_MAX_ROUNDS,
)
from lore.context_window import limit as ctx_limit
from lore.metrics import AgentMetrics
from lore.progress import ProgressReporter
from lore.prompts import build_lore_followup_prompt, build_system_prompt
from lore.research import _flatten_research_for_synthesis, render_research_blocks
from lore.session import LoreSession
from lore.tools import TOOLS, execute_tool
from proxy_client import ProxyClient, ProxyError
from rag_client import RAGClient

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


@dataclass
class LoreRunResult:
    """
    What one opening /lore run produced.

    ``tool_messages`` is kept so the run can seed a follow-up session without
    re-searching: it is the raw assistant/``tool_calls`` + ``role="tool"``
    material, ready for render_research_blocks().
    """
    answer: str
    metrics: AgentMetrics
    tool_messages: list[dict] = field(default_factory=list)
    ok: bool = True


# ---------------------------------------------------------------------------
# Shared tool-calling loop
# ---------------------------------------------------------------------------


async def _run_tool_rounds(
    messages: list[dict],
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    metrics: AgentMetrics,
    max_rounds: int,
    status: Optional[ProgressReporter] = None,
    label: str = "Agent",
) -> tuple[Optional[str], list[dict]]:
    """
    Drive tool-calling rounds until the model answers or the budget runs out.

    Shared by the opening /lore run and by thread follow-ups so there is one
    implementation of round handling, tool dispatch, reasoning passthrough and
    progress reporting.

    Args:
        messages: Conversation to extend. Mutated in place with the assistant
            and tool messages produced by each round.
        metrics: Updated with usage, timings and tool counts.
        max_rounds: Hard cap on rounds.
        status: Progress reporter, or None.
        label: Prefix for log lines, so the two callers are distinguishable.

    Returns:
        (direct_answer, tool_messages). ``direct_answer`` is the model's prose
        when it chose to answer rather than search again, or None if the round
        budget was exhausted — in which case the caller must synthesise from
        ``tool_messages``.
    """
    tool_messages: list[dict] = []

    for round_num in range(max_rounds):
        metrics.rounds_executed += 1
        logger.info(
            "--- %s round %d/%d (messages_in_context=%d) ---",
            label, round_num + 1, max_rounds, len(messages),
        )
        if status:
            await status.thinking(round_num + 1, max_rounds)

        llm_start = time.monotonic()
        try:
            result = await proxy_client.chat_with_tools(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=AGENT_TEMPERATURE,
            )
        finally:
            llm_elapsed = time.monotonic() - llm_start
            metrics.llm_response_times.append(llm_elapsed)
            logger.info("LLM response in %.2fs", llm_elapsed)

        metrics.record_usage(result.get("usage"))
        logger.info("Round %d %s", round_num + 1, metrics.context_line())

        content = result.get("content")
        tool_calls = result.get("tool_calls")
        reasoning = (result.get("reasoning_content") or "").strip()

        if not tool_calls:
            if content and content.strip():
                logger.info("Model returned final answer (%d chars)", len(content))
                return content.strip(), tool_messages
            # Empty response — nudge and retry rather than abandoning the run.
            logger.warning(
                "Model returned empty response (no content, no tool_calls) in round %d",
                round_num + 1,
            )
            messages.append({
                "role": "assistant",
                "content": "I didn't find enough information. Let me try a different search.",
            })
            continue

        names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        logger.info("Model wants to call %d tool(s): %s", len(tool_calls), names)
        if status:
            await status.searching(round_num + 1, max_rounds)

        for tc in tool_calls:
            function_name = tc.get("function", {}).get("name", "")
            raw_args = tc.get("function", {}).get("arguments", "{}")
            tool_call_id = tc.get("id", "")
            try:
                function_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse tool args for %s: %r", function_name, raw_args
                )
                function_args = {}

            metrics.total_tool_calls += 1
            tool_result = await execute_tool(
                function_name, function_args, rag_client, metrics=metrics
            )
            logger.info("Tool %s returned %d chars", function_name, len(tool_result))

            # The model's own reasoning rides along on the first tool call of
            # the round: dropping it left the model able to see *what* it had
            # searched but not why, so it re-derived the same plan — and at
            # temperature 0.1 that meant reissuing byte-identical queries.
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tc],
            }
            if reasoning and tc is tool_calls[0]:
                assistant_msg["reasoning_content"] = reasoning
            tool_msg = {
                "role": "tool",
                "name": function_name,
                "content": tool_result,
                "tool_call_id": tool_call_id,
            }
            messages.append(assistant_msg)
            messages.append(tool_msg)
            tool_messages.append(assistant_msg)
            tool_messages.append(tool_msg)

        if status:
            await status.analyzing(round_num + 1, max_rounds)

    logger.warning("Max rounds (%d) reached — forcing final answer", max_rounds)
    return None, tool_messages


async def run_agent_loop(
    user_question: str,
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    status: ProgressReporter,
    channel_names: Optional[list[str]] = None,
    max_rounds: Optional[int] = None,
) -> LoreRunResult:
    """
    Run the opening agentic RAG turn for /lore.

    Flow:
    1. Send user question + system prompt to the agent model with tools attached.
    2. Rounds of tool_calls -> execute -> append, via _run_tool_rounds().
    3. If the model answers without exhausting the budget, that is the answer.
    4. Otherwise flatten the research and synthesise one.

    Args:
        user_question: The question from /lore.
        proxy_client: Proxy client for the agent model.
        rag_client: RAG client for tool execution.
        status: Progress reporter. Required — the caller owns it, because
            opening one is a Discord concern and this module has no business
            knowing what an interaction is. LoreStatus no-ops when its message
            could not be sent, so there is never a None to guard against.
        channel_names: Channel names for the system prompt.
        max_rounds: Round cap (defaults to AGENT_MAX_ROUNDS, hard-capped).

    Returns:
        LoreRunResult carrying the answer, the metrics, and the raw tool
        messages so a follow-up thread can be seeded without re-searching.
    """
    if max_rounds is None:
        max_rounds = AGENT_MAX_ROUNDS
    max_rounds = max(1, min(max_rounds, AGENT_MAX_ROUNDS_HARD_CAP))

    metrics = AgentMetrics()
    logger.info("=" * 60)
    logger.info("AGENT LOOP START — question=%r", user_question[:80])
    logger.info("=" * 60)

    messages = [
        {"role": "system", "content": build_system_prompt(channel_names or [])},
        {"role": "user", "content": user_question},
    ]

    try:
        direct_answer, tool_messages = await _run_tool_rounds(
            messages, proxy_client, rag_client, metrics, max_rounds,
            status=status, label="Agent",
        )
    except ProxyError as e:
        metrics.error = str(e)
        logger.error("ProxyError in agent loop: %s", e)
        await status.failed()
        return LoreRunResult(f"⚠️ Backend error: {e}", metrics, [], ok=False)

    if direct_answer:
        await status.writing()
        logger.info(metrics.summary())
        logger.info("Final answer preview: %s...", direct_answer[:100])
        await status.complete(user_question)
        return LoreRunResult(direct_answer, metrics, tool_messages)

    # Budget exhausted — synthesise from everything gathered.
    await status.generating()
    try:
        llm_start = time.monotonic()
        # Flatten the tool-calling history into a single plain-text user turn.
        # Replacing just the system prompt left the assistant/tool_calls and
        # role="tool" messages in context and the model kept imitating them,
        # emitting tool-call markup as its answer. See the helper for why that
        # markup reaches the user instead of being parsed out.
        synthesis_messages = _flatten_research_for_synthesis(messages, user_question)
        # No tools in this request either way — the synthesis prompt carries no
        # tool-calling pattern to continue, which is the whole point of
        # flattening it. See stream_answer() for why it is streamed.
        answer, usage = await stream_answer(
            proxy_client, AGENT_MODEL, synthesis_messages, status=status
        )
        llm_elapsed = time.monotonic() - llm_start
        metrics.llm_response_times.append(llm_elapsed)
        # The synthesis prompt is the largest single request the run makes, so
        # this is normally where peak_context_tokens is set.
        metrics.record_usage(
            usage,
            fallback_prompt_chars=sum(len(m.get("content") or "") for m in synthesis_messages),
            fallback_completion_chars=len(answer),
        )
        logger.info(
            "Final answer generation took %.2fs (%d chars) — %s",
            llm_elapsed, len(answer), metrics.context_line(),
        )
        answer = (answer or "I've exhausted my search rounds. Please try rephrasing your question.").strip()
        logger.info(metrics.summary())
        logger.info("Final answer preview: %s...", answer[:100])
        await status.complete(user_question)
        return LoreRunResult(answer, metrics, tool_messages)
    except ProxyError as e:
        metrics.error = str(e)
        logger.exception("Failed to generate final answer")
        await status.failed()
        return LoreRunResult(
            f"⚠️ Failed to generate final answer: {e}", metrics, tool_messages, ok=False
        )
# ---------------------------------------------------------------------------
# Lore follow-up threads
# ---------------------------------------------------------------------------


def _budget_notice(used: int, limit: int) -> dict:
    """
    Tell the model how much room is left, as an ephemeral tail message.

    Appended, never merged into the system prompt: the system prompt is the
    head of the cached prefix, so editing it to carry a number that changes
    every turn would re-prefill the entire thread each time.
    """
    remaining = max(0, limit - used)
    pct = (used / limit * 100) if limit else 0.0
    return {
        "role": "user",
        "content": (
            f"CONTEXT BUDGET: this thread has used {used:,} of {limit:,} tokens "
            f"({pct:.0f}%); about {remaining:,} remain. Prefer answering from the "
            "material already gathered. Search again only if the question truly "
            "cannot be answered without it, and keep any new search narrow."
        ),
    }


# The answer call keeps the tools attached even though it must not use them —
# see chat_stream(tools=...) for why dropping them would cold-prefill the whole
# thread. So the instruction not to call them has to be explicit and firm.
_ANSWER_NUDGE = {
    "role": "user",
    "content": (
        "Now write the answer to the most recent question, using the material "
        "above. Do NOT call any tool, and do NOT output any tool calls, "
        "function names, or XML tags — searching is finished for this turn. "
        "Reply with the answer itself, as prose."
    ),
}

async def run_lore_turn(
    session: LoreSession,
    question: str,
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    max_rounds: Optional[int] = None,
    status: Optional[ProgressReporter] = None,
) -> tuple[str, AgentMetrics, bool]:
    """
    Run one follow-up turn against an existing lore session.

    Two paths. If the model decides the thread already holds what it needs, it
    answers on the first call and that answer is used as-is — no second request,
    and the prefix cache makes it cheap. If it calls tools, the results are
    rendered into one research block, appended, and the answer is generated by a
    separate streamed call.

    Progress is reported through the same status vocabulary the opening
    /lore run uses, so a thread shows identical indicators.

    On caching: every request here carries the same tools payload, so an
    append-only transcript reuses the prefix almost entirely — measured at 98%
    on a third turn, answering in 6.3s against 36.2s for the opening turn.
    There is exactly one unavoidable miss. A turn that searches leaves the
    backend holding the loop's raw assistant/``tool_calls`` + ``role="tool"``
    messages, while the transcript keeps the same results rendered as a flat
    research block; the two are different text, so the turn *after* a searching
    turn re-prefills from the first research block onward (~44% hit observed).
    Paying it here rather than deferring it is not worth an extra generation —
    the flattened form has to be prefilled once either way — so the direct
    answer is used when the loop produces one.

    The session is mutated (question, any research, and the answer are appended)
    but NOT persisted — the caller owns the store so it can persist once after
    posting to Discord succeeds.

    Args:
        session: The thread's session. Mutated in place.
        question: The new user message.
        proxy_client: Proxy client for the agent model.
        rag_client: RAG client for tool execution.
        max_rounds: Tool rounds allowed this turn. Defaults to
            LORE_FOLLOWUP_MAX_ROUNDS.
        status: Progress reporter, or None.

    Returns:
        (answer, metrics, searched) — the answer text, this turn's metrics, and
        whether any tool was actually run.
    """
    if max_rounds is None:
        max_rounds = LORE_FOLLOWUP_MAX_ROUNDS
    max_rounds = max(1, min(max_rounds, AGENT_MAX_ROUNDS_HARD_CAP))

    metrics = AgentMetrics()
    used = session.last_context_tokens
    ctx_total = ctx_limit()
    pct = session.pct_of(ctx_total)

    logger.info("=" * 60)
    logger.info(
        "LORE FOLLOW-UP turn %d (thread=%d) — question=%r",
        session.turns + 1, session.thread_id, question[:80],
    )
    logger.info(
        "Session context before turn: %s (%d%% of limit), %d search block(s)",
        f"{used:,}/{ctx_total:,}", pct * 100, session.searches,
    )
    logger.info("=" * 60)

    session.append_question(question)

    # Above the hard ceiling the model does not get a vote: searching again
    # would push the thread out of context entirely.
    allow_tools = pct < AGENT_CTX_HARD_PCT
    if not allow_tools:
        logger.warning(
            "Context at %.0f%% (>= %.0f%% hard limit) — skipping tools, answering directly",
            pct * 100, AGENT_CTX_HARD_PCT * 100,
        )

    extra: list[dict] = []
    if allow_tools and pct >= AGENT_CTX_SOFT_PCT:
        logger.info("Context at %.0f%% — telling the model its budget", pct * 100)
        extra.append(_budget_notice(used, ctx_total))

    direct_answer: Optional[str] = None
    tool_messages: list[dict] = []

    if allow_tools:
        work = session.build_messages(extra=extra)
        direct_answer, tool_messages = await _run_tool_rounds(
            work, proxy_client, rag_client, metrics, max_rounds,
            status=status, label="Lore follow-up",
        )
        if direct_answer:
            logger.info(
                "No further searching needed — answered from existing context (%d chars)",
                len(direct_answer),
            )

    searched = bool(tool_messages)

    # Collapse this turn's results into one block, skipping anything an earlier
    # turn already put in the transcript.
    if searched:
        blocks, dropped = render_research_blocks(
            tool_messages,
            seen_keys=session.seen_keys,
            start_index=session.searches,
        )
        if blocks:
            session.append_research("\n\n".join(blocks))
        else:
            logger.info(
                "Every chunk this turn was already in the thread (%d dropped) — "
                "nothing appended",
                dropped,
            )

    if direct_answer:
        answer = direct_answer
        if status:
            await status.writing()
    else:
        # Streamed so a long answer cannot trip the read timeout, and so the
        # thread shows progress instead of going silent.
        logger.info("Generating answer (streamed)...")
        if status:
            await status.generating()
        llm_start = time.monotonic()
        answer_messages = session.build_messages(extra=[_ANSWER_NUDGE])
        # Tools stay attached even though this turn must not call them: the
        # prompt shape has to match the surrounding calls or the whole thread
        # re-prefills. _ANSWER_NUDGE is what actually holds the model back.
        text, usage = await stream_answer(
            proxy_client, AGENT_MODEL, answer_messages, status=status, tools=TOOLS
        )
        answer = text.strip()

        if not answer:
            # The model answered with a tool call instead of prose despite the
            # nudge, so there were no content deltas to collect. Retry without
            # tools, which removes the option entirely at the cost of one cold
            # prefill.
            logger.warning(
                "Streamed answer was empty (likely a tool call) — retrying without tools"
            )
            text, usage = await stream_answer(
                proxy_client, AGENT_MODEL, answer_messages, status=status
            )
            answer = text.strip()

        metrics.llm_response_times.append(time.monotonic() - llm_start)
        metrics.record_usage(usage)
        logger.info(
            "Answer generated in %.2fs (%d chars) — %s",
            metrics.llm_response_times[-1], len(answer), metrics.context_line(),
        )

    if not answer:
        answer = "I wasn't able to produce an answer for that. Try rephrasing?"

    session.append_answer(answer)
    session.record_context(metrics.context_used)
    logger.info(metrics.summary())
    if status:
        await status.complete(question)

    return answer, metrics, searched


def seed_session_from_run(
    thread_id: int,
    channel_names: list[str],
    question: str,
    result: LoreRunResult,
    now: Optional[str] = None,
) -> LoreSession:
    """
    Build a follow-up session from a completed opening /lore run.

    The opening run already did the searching, so a thread opens knowing
    everything the embed was built from without repeating a single query. Its
    raw excerpts are carried over verbatim.

    Condensing them up front was tried and abandoned: it cost a whole extra
    generation (96-197s measured) on every /lore, and the model writes the
    summary one token at a time, so the wait scaled with how much it kept. The
    thread already has compaction for when context actually runs short, and
    that only pays the cost for threads that reach the limit — which few do.
    Storing the excerpts in the meantime is just text on disk.

    Args:
        thread_id: Thread the session backs, or 0 for a session that is still a
            pending offer.
        channel_names: Channels for the session's system prompt.
        question: The original /lore question.
        result: What run_agent_loop() returned.
        now: Timestamp to pin. Defaults to the current time.

    Returns:
        A session whose transcript is [question, research, answer].
    """
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session = LoreSession(
        thread_id=thread_id,
        system_prompt=build_lore_followup_prompt(channel_names, now=now),
        original_question=question,
        pinned_now=now,
    )
    session.append_question(question)

    blocks, _ = render_research_blocks(
        result.tool_messages, seen_keys=session.seen_keys
    )
    if blocks:
        session.append_research("\n\n".join(blocks))

    session.append_answer(result.answer)
    # From the transcript itself, not from the opening run's peak. The run's
    # peak includes its longest single request, which is not what this thread
    # will carry; the transcript is. Replaced by a real usage report as soon as
    # the first follow-up returns one.
    session.record_context(session.estimated_tokens())
    logger.info(
        "Lore session seeded for thread %d — %d research block(s), "
        "estimated context %s/%s",
        thread_id, len(blocks),
        f"{session.last_context_tokens:,}", f"{ctx_limit():,}",
    )
    return session
