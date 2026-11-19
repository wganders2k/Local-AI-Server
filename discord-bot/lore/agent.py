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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import (
    AGENT_CTX_COMPACT_PCT,
    AGENT_CTX_HARD_PCT,
    AGENT_CTX_SOFT_PCT,
    AGENT_MAX_ROUNDS,
    AGENT_MAX_ROUNDS_HARD_CAP,
    AGENT_MAX_REPEATED_SEARCHES,
    AGENT_TEMPERATURE,
    LORE_DEFAULT_SPEED,
    LORE_DEFAULT_WRITER,
    LORE_FOLLOWUP_MAX_ROUNDS,
    LORE_MAX_ROUND_TOKENS,
    LORE_MAX_SIMULATIONS,
    LORE_STRIKE_RECOVERY_ROUNDS,
    LORE_SYSTEM_NUDGE_MODELS,
    LORE_TOOL_CALL_STRIKES,
)
from lore.citations import Citations
from lore.compaction import compact_tool_rounds, fit_research_for, fit_session_for
from lore.context_window import limit as ctx_limit, window
from lore.metrics import TOOL_CHARS_PER_TOKEN, AgentMetrics, fraction
from lore.models import hands_off, research_model, writer_model
from lore.progress import ProgressReporter
from lore.prompts import build_lore_followup_prompt, build_system_prompt
from lore.research import (
    SEARCH_NOTE_PREFIX,
    flatten_research_for_synthesis,
    drop_seen_chunks,
)
from lore.session import LoreSession
from lore.streaming import stream_answer
from lore.tools import ToolContext, build_tools, execute_tool
from personas import PersonaCatalog
from proxy_client import ProxyClient, ProxyError
from rag_client import RAGClient

logger = logging.getLogger("mimic-bot.lore.agent")

# What llama-server says when the model's tool call is not valid JSON —
# typically one cut off by max_tokens after the model looped inside the
# arguments. A failed round, not a failed backend.
_BAD_TOOL_CALL = "failed to parse tool call"


@dataclass
class LoreRunResult:
    """
    What one opening /lore run produced.

    ``tool_messages`` is kept so the run can seed a follow-up session without
    re-searching: it is the raw assistant/``tool_calls`` + ``role="tool"``
    material, which seeds a thread's transcript as it is.
    """
    answer: str
    metrics: AgentMetrics
    tool_messages: list[dict] = field(default_factory=list)
    ok: bool = True
    # The [mN] refs the run's research carries; a thread seeded from the run
    # inherits them, so its answers keep linking the same messages.
    citations: Citations = field(default_factory=Citations)
    # "fast" or "slow", and the writer choice; a thread seeded from the run
    # keeps both.
    speed: str = LORE_DEFAULT_SPEED
    writer: str = LORE_DEFAULT_WRITER


# ---------------------------------------------------------------------------
# Shared tool-calling loop
# ---------------------------------------------------------------------------


async def _run_tool_rounds(
    messages: list[dict],
    proxy_client: ProxyClient,
    ctx: ToolContext,
    tools: list[dict],
    metrics: AgentMetrics,
    max_rounds: int,
    status: Optional[ProgressReporter] = None,
    label: str = "Agent",
    seen_keys: Optional[set[str]] = None,
    *,
    model: str,
) -> tuple[Optional[str], list[dict]]:
    """
    Drive tool-calling rounds until the model answers or the budget runs out.

    Shared by the opening /lore run and by thread follow-ups so there is one
    implementation of round handling, tool dispatch, reasoning passthrough,
    context guarding and progress reporting.

    Each result is stripped of chunks the conversation already holds before it
    is appended, and an exact repeat of an earlier call is refused rather than
    run — at temperature 0.1 a search that found nothing useful was reissued
    verbatim for 17 rounds. After AGENT_MAX_REPEATED_SEARCHES refusals the loop
    stops searching.

    Context is checked after every round, before the next request is sent:
    at AGENT_CTX_COMPACT_PCT the oldest half of the results is condensed, and if
    that still leaves the next request at AGENT_CTX_HARD_PCT or more, searching
    stops and the caller answers from what it has. Without this, a long run
    grows ~10k tokens a round until the backend rejects the request outright.

    Args:
        messages: Conversation to extend. Mutated in place with the assistant
            and tool messages produced by each round.
        ctx: What the tools run with (lore.tools.ToolContext).
        tools: The schemas sent with every request (lore.tools.build_tools).
        metrics: Updated with usage, timings and tool counts.
        max_rounds: Hard cap on rounds.
        status: Progress reporter, or None.
        label: Prefix for log lines, so the two callers are distinguishable.
        seen_keys: chunk_key() of chunks already in ``messages`` from earlier
            turns. Copied, not mutated.
        model: The research model — every round goes to it.

    Returns:
        (direct_answer, tool_messages). ``direct_answer`` is the model's prose
        when it chose to answer rather than search again, or None if the round
        budget was exhausted, context ran short or the model kept repeating
        itself — in which case the caller must synthesise from
        ``tool_messages``.
    """
    tool_messages: list[dict] = []
    compactions = 0
    shown = set(seen_keys or ())
    ran: dict[str, int] = {}  # call signature -> round it first ran in
    repeats = 0
    simulations = 0
    # Malformed tool calls cost a strike each; LORE_STRIKE_RECOVERY_ROUNDS
    # good rounds in a row earn one back, up to the starting budget. Searching
    # stops when none are left. Kept out of the model's context entirely: it
    # is only ever told that its last call was discarded.
    strikes = LORE_TOOL_CALL_STRIKES
    clean_rounds = 0

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
                model=model,
                messages=messages,
                tools=tools,
                temperature=AGENT_TEMPERATURE,
                max_tokens=LORE_MAX_ROUND_TOKENS,
            )
        except ProxyError as e:
            if _BAD_TOOL_CALL not in str(e).lower():
                raise
            # Seen on the Gemma MoE: it looped inside a tool call's arguments,
            # the round cap cut it off, and the backend could not parse what
            # was left. That used to end the whole run as a backend error.
            strikes -= 1
            clean_rounds = 0
            logger.warning(
                "Round %d produced a malformed tool call (%d/%d strikes left): %s",
                round_num + 1, strikes, LORE_TOOL_CALL_STRIKES, str(e)[:160],
            )
            if strikes <= 0:
                logger.warning("Out of strikes for malformed tool calls — stopping search")
                return None, tool_messages
            messages.append(_note(
                "Your last tool call was malformed — it ran past the length limit and "
                "was discarded. Make one short, well-formed call with different wording, "
                "or answer from what you already have.",
                model,
            ))
            continue
        finally:
            llm_elapsed = time.monotonic() - llm_start
            metrics.llm_response_times.append(llm_elapsed)
            logger.info("LLM response in %.2fs", llm_elapsed)

        metrics.record_usage(result.get("usage"), model=model)
        logger.info("Round %d %s", round_num + 1, metrics.context_line())
        if strikes < LORE_TOOL_CALL_STRIKES:
            clean_rounds += 1
            if clean_rounds >= LORE_STRIKE_RECOVERY_ROUNDS:
                strikes += 1
                clean_rounds = 0
                logger.info("%d good rounds — a strike back (%d/%d)",
                            LORE_STRIKE_RECOVERY_ROUNDS, strikes, LORE_TOOL_CALL_STRIKES)

        if result.get("finish_reason") == "length":
            logger.warning(
                "Round %d hit the %d-token cap (the model was looping) — its output is cut short",
                round_num + 1, LORE_MAX_ROUND_TOKENS,
            )
        content = result.get("content")
        tool_calls = result.get("tool_calls")
        reasoning = (result.get("reasoning_content") or "").strip()

        if not tool_calls:
            if content and content.strip():
                logger.info("Model returned final answer (%d chars)", len(content))
                return content.strip(), tool_messages
            # Empty response — nudge and retry rather than abandoning the run.
            # As an instruction, not as words put in the model's mouth: a canned
            # assistant line ("I didn't find enough information...") was copied
            # verbatim into the top of a later answer.
            logger.warning(
                "Model returned empty response (no content, no tool_calls) in round %d",
                round_num + 1,
            )
            messages.append(_note(
                "Your last reply was empty. Either make your next tool call, or write the answer.",
                model,
            ))
            continue

        names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        logger.info("Model wants to call %d tool(s): %s", len(tool_calls), names)
        if status:
            await status.searching(round_num + 1, max_rounds)

        result_chars = 0
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
            signature = json.dumps([function_name, function_args], sort_keys=True)
            if function_name == "simulate_conversation" and simulations >= LORE_MAX_SIMULATIONS:
                logger.warning("Simulation %d refused (cap %d)", simulations + 1, LORE_MAX_SIMULATIONS)
                tool_result = (
                    f"{SEARCH_NOTE_PREFIX} You have already run a simulation for this "
                    "question; its transcript is above. Answer from it and your research."
                )
            elif signature in ran:
                repeats += 1
                logger.warning(
                    "Repeat of round %d's %s call — refused (%d/%d)",
                    ran[signature], function_name, repeats, AGENT_MAX_REPEATED_SEARCHES,
                )
                tool_result = (
                    f"{SEARCH_NOTE_PREFIX} You already ran this exact search in "
                    f"round {ran[signature]}, and its results are above. Running it "
                    "again returns the same thing. For more matches from a semantic "
                    "search, repeat it with the next page; otherwise change the "
                    "wording, the tool or the scope (a channel or date range), or "
                    "answer from what you have."
                )
            else:
                ran[signature] = round_num + 1
                if function_name == "simulate_conversation":
                    simulations += 1
                    if status:
                        await status.simulating()
                raw = await execute_tool(function_name, function_args, ctx)
                tool_result = drop_seen_chunks(raw, shown)
                logger.info(
                    "Tool %s returned %d chars, %d new", function_name, len(raw), len(tool_result),
                )
            result_chars += len(tool_result)

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

        if repeats >= AGENT_MAX_REPEATED_SEARCHES:
            logger.warning(
                "%d repeated search(es) refused — stopping search", repeats
            )
            return None, tool_messages

        # The last usage report predates this round's results, so add them on.
        # The next request goes to the research model, so its window applies.
        limit = window(model)
        projected = metrics.context_used + result_chars // TOOL_CHARS_PER_TOKEN
        if projected < AGENT_CTX_COMPACT_PCT * limit:
            continue

        compactions += 1
        if status:
            await status.condensing(round_num + 1, max_rounds)
        logger.info(
            "Next request projected at %s/%s (%.0f%%) — compacting research",
            f"{projected:,}", f"{limit:,}", projected / limit * 100,
        )
        projected -= await compact_tool_rounds(
            messages, tool_messages, proxy_client, metrics, compactions, model,
            shown_keys=shown,
        )
        if projected >= AGENT_CTX_HARD_PCT * limit:
            logger.warning(
                "Still projected at %.0f%% after compaction (>= %.0f%% hard limit) "
                "— stopping search",
                projected / limit * 100, AGENT_CTX_HARD_PCT * 100,
            )
            return None, tool_messages

    logger.warning("Max rounds (%d) reached — forcing final answer", max_rounds)
    return None, tool_messages


_ANSWER_NUDGE_TEXT = (
        "Now write the answer to the most recent question, using the material "
        "above. Do NOT call any tool, and do NOT output any tool calls, "
        "function names, or XML tags — searching is finished for this turn. "
        "Reply with the answer itself, as prose. Cite with refs written exactly "
        "as [m12], [m12, m15] or [m12-m14] — no backticks, and no channel "
        "names or other text inside the brackets."
)


def _note(text: str, model: str) -> dict:
    """
    An instruction appended mid-conversation, in the role ``model`` takes best:
    system where its template allows one (LORE_SYSTEM_NUDGE_MODELS), user
    otherwise. See _answer_nudge for why the role matters to the cache.
    """
    role = "system" if model in LORE_SYSTEM_NUDGE_MODELS else "user"
    return {"role": role, "content": text}


def _answer_nudge(model: str) -> dict:
    """
    The instruction to stop searching and answer, in the role ``model`` takes best.

    A system message for models in LORE_SYSTEM_NUDGE_MODELS, a user message
    otherwise. Gemma 4's template keeps an assistant turn's reasoning only after
    the last user message, so a user-role nudge rewrites every earlier turn and
    re-prefills the whole run; a system turn changes nothing before it. Qwen's
    template rejects a system message anywhere but first, so it gets the user
    role — its preserve_thinking keeps the prefix intact either way.
    """
    return _note(_ANSWER_NUDGE_TEXT, model)

# Tool-call markup the model might write as text. With tools attached the
# backend's parser normally lifts a call out of the reply, leaving it empty;
# this catches one that slipped through as prose instead.
_TOOL_MARKUP = re.compile(r"<tool_call|<function[=\s>]|<tool_response|</?parameter[=\s>]")


async def _answer_in_place(
    proxy_client: ProxyClient,
    messages: list[dict],
    tools: list[dict],
    metrics: AgentMetrics,
    status: Optional[ProgressReporter],
    model: str,
) -> str:
    """
    Ask for the answer on the loop's own conversation, so the prefix is cached.
    ``model`` is the research model, which holds that conversation.

    The request is the loop's messages, unchanged, plus the answer nudge, with the
    same tools attached: everything before the nudge is what the backend
    already holds, so only the nudge is prefilled. The flattened synthesis
    prompt shares nothing with that conversation — not even the system
    prompt, which carries the tool list — and cost a full re-prefill (34.6k
    tokens, ~25s, on a typical run).

    Keeping the tools is also what keeps tool-call markup out of the answer:
    if the model calls a tool anyway, the backend parses it out and the text
    comes back empty.

    Returns:
        The answer, or "" if the model called a tool or wrote markup, in which
        case the caller falls back to the flattened synthesis.
    """
    llm_start = time.monotonic()
    text, usage = await stream_answer(
        proxy_client, model, messages + [_answer_nudge(model)], status=status, tools=tools,
    )
    elapsed = time.monotonic() - llm_start
    metrics.llm_response_times.append(elapsed)
    metrics.record_usage(usage, fallback_completion_chars=len(text), model=model)
    if not text.strip():
        logger.warning("In-place answer was empty (likely a tool call) — falling back to synthesis")
        return ""
    if _TOOL_MARKUP.search(text):
        logger.warning("In-place answer contained tool-call markup — falling back to synthesis")
        return ""
    logger.info("Answer generated in place in %.2fs (%d chars) — %s", elapsed, len(text), metrics.context_line())
    return text.strip()


async def _generate_answer(
    proxy_client: ProxyClient,
    messages: list[dict],
    metrics: AgentMetrics,
    status: Optional[ProgressReporter],
    tools: Optional[list[dict]] = None,
    *,
    model: str,
) -> str:
    """
    Stream one answer and record its timing and usage. Streamed so a long
    answer cannot trip the read timeout, and so the user sees progress — see
    stream_answer().

    With tools attached, an empty answer means the model emitted a tool call
    instead of prose despite being told not to, so there were no content deltas
    to collect. It is retried once without tools, which removes the option
    entirely at the cost of one cold prefill.
    """
    llm_start = time.monotonic()
    text, usage = await stream_answer(proxy_client, model, messages, status=status, tools=tools)
    if tools and not text.strip():
        logger.warning("Streamed answer was empty (likely a tool call) — retrying without tools")
        text, usage = await stream_answer(proxy_client, model, messages, status=status)
    elapsed = time.monotonic() - llm_start
    metrics.llm_response_times.append(elapsed)
    metrics.record_usage(
        usage,
        fallback_prompt_chars=sum(len(m.get("content") or "") for m in messages),
        fallback_completion_chars=len(text),
        model=model,
    )
    logger.info(
        "Answer generated by %s in %.2fs (%d chars) — %s",
        model, elapsed, len(text), metrics.context_line(),
    )
    return text.strip()


async def run_agent_loop(
    user_question: str,
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    status: ProgressReporter,
    channel_names: Optional[list[str]] = None,
    max_rounds: Optional[int] = None,
    catalog: Optional[PersonaCatalog] = None,
    speed: str = LORE_DEFAULT_SPEED,
    writer: str = LORE_DEFAULT_WRITER,
) -> LoreRunResult:
    """
    Run the opening agentic RAG turn for /lore.

    Flow:
    1. Send user question + system prompt to the research model (picked by
       ``speed``) with tools attached.
    2. Rounds of tool_calls -> execute -> append, via _run_tool_rounds().
    3. Writer is the research model: if it answers without exhausting the
       budget, that is the answer; otherwise it answers on its own cached
       conversation, and only if that fails is the research flattened and an
       answer synthesised.
    4. Writer is another model: the research is flattened and the writer
       writes the answer, whether or not the research model already wrote one.
       The research was budgeted against the research model's window, so it is
       condensed first until it fits the writer's. A run that never searched
       has nothing to hand over and keeps its direct answer.

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
        catalog: The personas simulate_conversation may use; None leaves the
            tool out.
        speed: "fast" or "slow" — which model researches; see
            config.LORE_RESEARCH_MODELS.
        writer: Which model writes the answer — a config.LORE_WRITERS key, or
            LORE_WRITER_SAME for the research model.

    Returns:
        LoreRunResult carrying the answer, the metrics, and the raw tool
        messages so a follow-up thread can be seeded without re-searching.
    """
    if max_rounds is None:
        max_rounds = AGENT_MAX_ROUNDS
    max_rounds = max(1, min(max_rounds, AGENT_MAX_ROUNDS_HARD_CAP))

    researcher = research_model(speed)
    author = writer_model(speed, writer)
    metrics = AgentMetrics(
        speed=speed, model=researcher, research_model=researcher,
        writer_model=author, max_rounds=max_rounds,
    )
    citations = Citations()
    logger.info("=" * 60)
    logger.info(
        "AGENT LOOP START (%s: research %s, writer %s) — question=%r",
        speed, researcher, author, user_question[:80],
    )
    logger.info("=" * 60)

    messages = [
        {"role": "system", "content": build_system_prompt(channel_names or [])},
        {"role": "user", "content": user_question},
    ]

    tools = build_tools(catalog)
    try:
        direct_answer, tool_messages = await _run_tool_rounds(
            messages, proxy_client, ToolContext(rag_client, proxy_client, catalog, metrics, citations),
            tools, metrics, max_rounds, status=status, label="Agent", model=researcher,
        )
    except ProxyError as e:
        metrics.error = str(e)
        logger.error("ProxyError in agent loop: %s", e)
        await status.failed()
        return LoreRunResult(f"⚠️ Backend error: {e}", metrics, [], ok=False, speed=speed, writer=writer)

    if hands_off(speed, writer) and tool_messages:
        if direct_answer:
            logger.info("Setting aside the research model's answer for %s's", author)
        await status.generating()
        try:
            synthesis = await fit_research_for(
                author, messages, tool_messages, user_question,
                proxy_client, metrics, researcher,
            )
            answer = await _generate_answer(
                proxy_client, synthesis, metrics, status, model=author,
            )
        except ProxyError as e:
            metrics.error = str(e)
            logger.exception("Writer %s failed to write the answer", author)
            await status.failed()
            return LoreRunResult(
                f"⚠️ Failed to generate final answer: {e}", metrics, tool_messages, ok=False,
                citations=citations, speed=speed, writer=writer,
            )
        answer = answer or direct_answer or "I couldn't write an answer from that research. Please try rephrasing."
        logger.info(metrics.summary())
        logger.info("Final answer preview: %s...", answer[:100])
        await status.complete(user_question, metrics.run_line())
        return LoreRunResult(answer, metrics, tool_messages, citations=citations, speed=speed, writer=writer)

    if direct_answer:
        await status.writing()
        logger.info(metrics.summary())
        logger.info("Final answer preview: %s...", direct_answer[:100])
        await status.complete(user_question, metrics.run_line())
        return LoreRunResult(direct_answer, metrics, tool_messages, citations=citations, speed=speed, writer=writer)

    # Searching stopped without an answer: ask for one on the conversation
    # the backend already has cached, and only if that fails, flatten.
    await status.generating()
    try:
        answer = await _answer_in_place(proxy_client, messages, tools, metrics, status, researcher)
        if not answer:
            # Flatten the tool-calling history into a single plain-text user
            # turn and answer with no tools. This re-prefills everything, but
            # leaves no tool-calling pattern in context to imitate, and with no
            # tools attached nothing could be parsed out if the model did. From
            # tool_messages rather than messages: it carries any research
            # condensed mid-loop in a form the flattener recognises.
            synthesis_messages = flatten_research_for_synthesis(tool_messages, user_question)
            answer = await _generate_answer(
                proxy_client, synthesis_messages, metrics, status, model=researcher,
            )
        answer = answer or "I've exhausted my search rounds. Please try rephrasing your question."
        logger.info(metrics.summary())
        logger.info("Final answer preview: %s...", answer[:100])
        await status.complete(user_question, metrics.run_line())
        return LoreRunResult(answer, metrics, tool_messages, citations=citations, speed=speed, writer=writer)
    except ProxyError as e:
        metrics.error = str(e)
        logger.exception("Failed to generate final answer")
        await status.failed()
        return LoreRunResult(
            f"⚠️ Failed to generate final answer: {e}", metrics, tool_messages, ok=False,
            citations=citations, speed=speed, writer=writer,
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
    return {
        "role": "user",
        "content": (
            f"CONTEXT BUDGET: this thread has used {used:,} of {limit:,} tokens "
            f"({fraction(used, limit):.0%}); about {remaining:,} remain. Prefer answering from the "
            "material already gathered. Search again only if the question truly "
            "cannot be answered without it, and keep any new search narrow."
        ),
    }


async def run_lore_turn(
    session: LoreSession,
    question: str,
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    max_rounds: Optional[int] = None,
    status: Optional[ProgressReporter] = None,
    catalog: Optional[PersonaCatalog] = None,
) -> tuple[str, AgentMetrics, bool]:
    """
    Run one follow-up turn against an existing lore session.

    The whole turn runs on one model: the thread's writer if it has one
    (``session.writer``), otherwise its research model. Searching on one model
    and writing on another meant two swaps a turn, each starting from an empty
    cache, so a writer thread re-read everything twice per reply. The opening
    run still researches on the research model; only follow-ups move over.

    If the model decides the thread already holds what it needs, it answers on
    the first call and that answer is used as-is. If it searches, its tool
    rounds are kept in the transcript exactly as they were sent, and the answer
    is asked for on that same conversation — so it extends what the backend
    already holds rather than re-reading the research in another form. Chunks
    the thread has already shown are dropped from each result before it enters
    the context (drop_seen_chunks); compaction condenses old rounds when the
    thread actually grows large.

    Caching, measured: a turn that answers from context hits ~98%. What still
    misses: the first follow-up (its system prompt differs from the opening
    run's), any model swap or idle unload, and on Gemma the previous turn's
    rounds, since its template drops earlier reasoning at a new question.

    The session is mutated (question, tool rounds, answer) but NOT persisted —
    the caller owns the store so it can persist once after posting to Discord
    succeeds.

    Returns:
        (answer, metrics, searched) — the answer text, this turn's metrics, and
        whether any tool was actually run.
    """
    if max_rounds is None:
        max_rounds = LORE_FOLLOWUP_MAX_ROUNDS
    max_rounds = max(1, min(max_rounds, AGENT_MAX_ROUNDS_HARD_CAP))

    model = writer_model(session.speed, session.writer)
    metrics = AgentMetrics(
        speed=session.speed, model=model, research_model=model,
        writer_model=model, max_rounds=max_rounds,
    )
    # Built once for the turn: every request in it must send identical tools,
    # or the next one cold-prefills the thread.
    tools = build_tools(catalog)
    ctx_total = window(model)

    logger.info("=" * 60)
    logger.info(
        "LORE FOLLOW-UP turn %d (thread=%d, %s) — question=%r",
        session.turns + 1, session.thread_id, model, question[:80],
    )

    session.append_question(question)
    # A thread opened from a run whose research outgrew this model's window
    # (a 128k research model handing a thread to a 56k writer) is condensed
    # before the first request, not rejected by the backend.
    await fit_session_for(model, session, [], proxy_client, metrics, model)

    used = session.last_context_tokens
    pct = session.pct_of(ctx_total)
    logger.info(
        "Session context before turn: %s (%d%% of limit), %d search result(s)",
        f"{used:,}/{ctx_total:,}", pct * 100, session.searches,
    )
    logger.info("=" * 60)

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

    work = session.build_messages(extra=extra)
    direct_answer: Optional[str] = None
    tool_messages: list[dict] = []
    if allow_tools:
        direct_answer, tool_messages = await _run_tool_rounds(
            work, proxy_client, ToolContext(rag_client, proxy_client, catalog, metrics, session.citations),
            tools, metrics, max_rounds,
            status=status, label="Lore follow-up", seen_keys=session.seen_keys,
            model=model,
        )
        if direct_answer and not tool_messages:
            logger.info("No searching needed — answered from existing context (%d chars)", len(direct_answer))

    if direct_answer:
        answer = direct_answer
        if status:
            await status.writing()
    else:
        # On the loop's own conversation, ephemeral budget notice included, so
        # only the nudge is new to the backend. Tools stay attached for the same
        # reason; the nudge is what holds the model back from using them.
        if status:
            await status.generating()
        answer = await _generate_answer(
            proxy_client, work + [_answer_nudge(model)], metrics, status,
            tools=tools, model=model,
        )

    if not answer:
        answer = "I wasn't able to produce an answer for that. Try rephrasing?"

    session.append_rounds(tool_messages)
    session.append_answer(answer)
    session.record_context(metrics.context_used)
    logger.info(metrics.summary())
    if status:
        await status.complete(question, metrics.run_line())

    return answer, metrics, bool(tool_messages)


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
        citations=Citations.from_dict(result.citations.to_dict()),
        speed=result.speed,
        writer=result.writer,
    )
    session.append_question(question)
    # The run's tool rounds as they were, condensed summaries included.
    session.append_rounds(result.tool_messages)
    session.append_answer(result.answer)
    # From the transcript itself, not from the opening run's peak. The run's
    # peak includes its longest single request, which is not what this thread
    # will carry; the transcript is. Replaced by a real usage report as soon as
    # the first follow-up returns one.
    session.record_context(session.estimated_tokens())
    logger.info(
        "Lore session seeded for thread %d — %d search result(s), "
        "estimated context %s/%s",
        thread_id, session.searches,
        f"{session.last_context_tokens:,}", f"{ctx_limit(session.speed, session.writer):,}",
    )
    return session
