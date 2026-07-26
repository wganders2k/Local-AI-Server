"""
Agentic RAG — Tool schemas, system prompt builder, tool executor, and agent loop.

The agent uses brain-dense (Qwen3.6-27B) as a tool-calling orchestrator that
iteratively queries the RAG service and synthesizes answers from retrieved context.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord

from config import (
    AGENT_MODEL,
    AGENT_MAX_ROUNDS,
    AGENT_MAX_ROUNDS_HARD_CAP,
    AGENT_TEMPERATURE,
    AGENT_TOP_K,
    LORE_CONTEXT_PATH,
)
from proxy_client import ProxyClient, ProxyError
from rag_client import RAGClient

logger = logging.getLogger("mimic-bot.agent")


# ---------------------------------------------------------------------------
# Metrics / timing tracker
# ---------------------------------------------------------------------------


@dataclass
class AgentMetrics:
    """Collects timing and usage metrics for a single agent run."""
    start_time: float = field(default_factory=time.monotonic)
    total_tool_calls: int = 0
    rounds_executed: int = 0
    tool_call_times: list[float] = field(default_factory=list)
    llm_response_times: list[float] = field(default_factory=list)
    rag_query_times: list[float] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_duration(self) -> float:
        return time.monotonic() - self.start_time

    def summary(self) -> str:
        return (
            f"AgentMetrics — duration={self.total_duration:.1f}s, "
            f"rounds={self.rounds_executed}, tool_calls={self.total_tool_calls}, "
            f"tools={self.tools_used!r}, "
            f"avg_llm={sum(self.llm_response_times)/max(len(self.llm_response_times),1):.2f}s, "
            f"avg_rag={sum(self.rag_query_times)/max(len(self.rag_query_times),1):.2f}s"
        )


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_discord_history",
            "description": "Search the entire Discord history across all channels for relevant context about a question. Use this as the first step when answering a lore question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in natural language. Should capture the key entities and concepts from the user's question.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional start date filter in ISO 8601 format (e.g. '2024-01-01T00:00:00Z'). Only include results after this date.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date filter in ISO 8601 format (e.g. '2024-12-31T23:59:59Z'). Only include results before this date.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve. Default 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_channel_history",
            "description": "Search a specific Discord channel's history for relevant context. Use this when the question is scoped to a particular channel or when you need more focused results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in natural language.",
                    },
                    "channel_name": {
                        "type": "string",
                        "description": "Exact channel name to search (e.g. '#general', '#lore').",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional start date filter in ISO 8601 format (e.g. '2024-01-01T00:00:00Z').",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date filter in ISO 8601 format (e.g. '2024-12-31T23:59:59Z').",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve. Default 10.",
                    },
                },
                "required": ["query", "channel_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_channel",
            "description": "Get a summary of recent conversations in a specific channel. Useful for understanding ongoing discussions, lore development, or channel activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_name": {
                        "type": "string",
                        "description": "Exact channel name to summarize (e.g. '#general', '#lore').",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional start date in ISO 8601 format.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date in ISO 8601 format.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve for summarization. Default 20.",
                    },
                },
                "required": ["channel_name"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
# This module is the single source of truth for the /lore agent's prompts.
# Server-specific background (member aliases, persona) is NOT hardcoded here —
# it is loaded from the gitignored file at config.LORE_CONTEXT_PATH.

AGENT_IDENTITY: str = (
    "You are an expert research assistant that answers questions about "
    "a Discord server's history and lore."
)

_lore_context_cache: Optional[str] = None


def load_lore_context() -> str:
    """
    Load server-specific background knowledge (member alias index, persona
    notes) from the file at config.LORE_CONTEXT_PATH.

    The file is deliberately not committed to the repo. If it is absent the
    agent still works — it just answers without the alias index.

    Returns:
        File contents stripped of surrounding whitespace, or "" if unavailable.
    """
    global _lore_context_cache
    if _lore_context_cache is not None:
        return _lore_context_cache

    path = Path(LORE_CONTEXT_PATH)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    try:
        _lore_context_cache = path.read_text(encoding="utf-8").strip()
        logger.info("Loaded lore context from %s (%d chars)", path, len(_lore_context_cache))
    except FileNotFoundError:
        logger.warning(
            "Lore context file not found at %s — running without server-specific "
            "background knowledge. Copy prompts/lore_context.example.md to create it.",
            path,
        )
        _lore_context_cache = ""
    except OSError as e:
        logger.error("Failed to read lore context file %s: %s", path, e)
        _lore_context_cache = ""

    return _lore_context_cache


def _background_block() -> str:
    """Render the server-specific background section, or "" if none is configured."""
    lore_context = load_lore_context()
    if not lore_context:
        return ""
    return f"\nGENERAL BACKGROUND KNOWLEDGE:\n{lore_context}\n"


def build_system_prompt(channel_names: list[str]) -> str:
    """
    Build the system prompt for the RAG agent.

    Args:
        channel_names: List of available Discord channel names the agent can search.

    Returns:
        Complete system prompt string.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    channels = "\n".join(f"  - {ch}" for ch in sorted(channel_names))
    background = _background_block()

    return f"""{AGENT_IDENTITY}

CURRENT DATE/TIME: {now}

AVAILABLE CHANNELS TO SEARCH:
{channels}
{background}
YOUR INSTRUCTIONS:
1. You have access to three tools: search_discord_history, search_channel_history, and summarize_channel.
2. Use these tools to find relevant information BEFORE answering.
3. When a user asks a question, first formulate a good search query and call your search tool(s).
4. If initial results are insufficient, try additional searches with different keywords or scope.
5. After gathering enough context, synthesize a comprehensive answer based on what you found.
6. If no relevant information exists in the search results, honestly state that you couldn't find relevant information.
7. Always cite which channel(s) and approximate time period your information comes from.
8. Be specific with dates, author names, and direct quotes when they appear in the context.

RESPONSE FORMAT:
- When you have enough information to answer, respond naturally (do NOT call tools again).
- Structure longer answers with clear paragraphs or bullet points.
- Mention source channels: "According to discussions in #general..."
- If conflicting information exists, present both sides.

IMPORTANT:
- Do NOT fabricate information that isn't in the retrieved context.
- If the question is unclear, ask for clarification instead of searching blindly.
- You can make multiple tool calls in a single turn if needed (e.g., search different channels)."""


def build_synthesis_prompt() -> str:
    """
    Build the system prompt used when max_rounds is exhausted.

    This replaces the tool-calling prompt so the model stops emitting tool
    calls and simply synthesizes an answer from the tool results already in
    the conversation. It keeps the same background knowledge so member names
    are still resolved correctly in the final answer.

    Returns:
        Complete synthesis system prompt string.
    """
    return f"""{AGENT_IDENTITY}
{_background_block()}
You have completed your searches and gathered information. Please now \
synthesize a final answer based on all the tool results in this conversation. \
Write a natural language response directed at the user. Do NOT output any \
tool calls, function names, or XML tags."""

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


async def execute_tool(
    tool_name: str,
    tool_args: dict,
    rag_client: RAGClient,
    metrics: Optional[AgentMetrics] = None,
) -> str:
    """
    Execute a single tool call and return the result as a string.

    Args:
        tool_name: Name of the tool to execute.
        tool_args: Arguments dict parsed from the LLM's function call.
        rag_client: Instance of the RAG service client.
        metrics: Optional metrics tracker for timing.

    Returns:
        The tool result as a formatted string for the LLM to consume.
    """
    logger.info(
        "Executing tool=%s with args={%s}",
        tool_name,
        ", ".join(f"{k}={v!r}" for k, v in tool_args.items()),
    )

    if metrics is not None:
        metrics.tools_used.append(tool_name)

    tool_start = time.monotonic()

    try:
        if tool_name == "search_discord_history":
            query = tool_args.get("query", "")
            start_date = tool_args.get("start_date")
            end_date = tool_args.get("end_date")
            top_k = tool_args.get("top_k", AGENT_TOP_K)
            context, chunk_count = await rag_client.retrieve(
                query,
                top_k=top_k,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(
                "RAG search_discord_history: query=%r, top_k=%d → %d chunks",
                query, top_k, chunk_count,
            )
            if chunk_count == 0:
                return (
                    f"No results found for query '{query}'"
                    f"{' in channel ' + channel_name if 'channel_name' in tool_args else ''}"
                    f"{' between ' + str(start_date) + ' and ' + str(end_date) if start_date or end_date else ''}."
                    " This may mean: (a) the topic was never discussed, (b) the date range is too narrow, "
                    "or (c) the query terms don't match how users phrased things. Try broadening the search "
                    "by removing date filters, using different keywords, or searching without a channel filter."
                )
            return context

        elif tool_name == "search_channel_history":
            query = tool_args.get("query", "")
            channel_name = tool_args.get("channel_name", "")
            start_date = tool_args.get("start_date")
            end_date = tool_args.get("end_date")
            top_k = tool_args.get("top_k", AGENT_TOP_K)
            context, chunk_count = await rag_client.retrieve(
                query,
                top_k=top_k,
                channel_name=channel_name,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(
                "RAG search_channel_history: query=%r, channel=%r, top_k=%d → %d chunks",
                query, channel_name, top_k, chunk_count,
            )
            if chunk_count == 0:
                return (
                    f"No results found for query '{query}' in channel '{channel_name}'"
                    f"{' between ' + str(start_date) + ' and ' + str(end_date) if start_date or end_date else ''}."
                    " This may mean the topic wasn't discussed in this channel, the date range is too narrow, "
                    "or the query terms don't match. Try a different channel, broader date range, or different keywords."
                )
            return context

        elif tool_name == "summarize_channel":
            channel_name = tool_args.get("channel_name", "")
            start_date = tool_args.get("start_date")
            end_date = tool_args.get("end_date")
            top_k = tool_args.get("top_k", 20)
            # Reuse search but with higher top_k and a generic summarization query
            query = f"Recent conversations and discussions in {channel_name}"
            context, chunk_count = await rag_client.retrieve(
                query,
                top_k=top_k,
                channel_name=channel_name,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(
                "RAG summarize_channel: channel=%r, top_k=%d → %d chunks",
                channel_name, top_k, chunk_count,
            )
            if chunk_count == 0:
                return (
                    f"No message history found for channel '{channel_name}'"
                    f"{' between ' + str(start_date) + ' and ' + str(end_date) if start_date or end_date else ''}."
                    " The channel may not exist, has no ingested messages, or the date range has no activity."
                )
            return context

        else:
            logger.warning("Unknown tool requested: %s", tool_name)
            return f"Error: Unknown tool '{tool_name}'."

    except Exception as e:
        logger.exception("Tool execution failed for %s", tool_name)
        return f"Error executing {tool_name}: {type(e).__name__}: {e}"
    finally:
        if metrics is not None:
            elapsed = time.monotonic() - tool_start
            metrics.tool_call_times.append(elapsed)
            logger.info("Tool %s completed in %.2fs", tool_name, elapsed)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def run_agent_loop(
    user_question: str,
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    interaction: discord.Interaction,
    channel_names: Optional[list[str]] = None,
    max_rounds: Optional[int] = None,
) -> str:
    """
    Run the full agentic RAG loop.

    Flow:
    1. Send user question + system prompt to brain-dense with tools attached.
    2. If the model returns tool_calls → execute them → append results → repeat.
    3. If the model returns content (no tool_calls) → that's the final answer.
    4. Cap at max_rounds to prevent infinite loops.

    Args:
        user_question: The original user question from /lore command.
        proxy_client: Proxy client for sending requests to brain-dense.
        rag_client: RAG client for querying the vector store.
        interaction: Discord interaction for sending intermediate messages.
        channel_names: Optional list of channel names for the system prompt.
        max_rounds: Max rounds before forcing final answer (defaults to AGENT_MAX_ROUNDS=5, capped at 20).

    Returns:
        The final synthesized answer string from the agent.
    """
    # Clamp user-specified rounds to valid range
    if max_rounds is None:
        max_rounds = AGENT_MAX_ROUNDS
    max_rounds = max(1, min(max_rounds, AGENT_MAX_ROUNDS_HARD_CAP))
    metrics = AgentMetrics()
    logger.info("=" * 60)
    logger.info("AGENT LOOP START — question=%r", user_question[:80])
    logger.info("=" * 60)

    system_prompt = build_system_prompt(channel_names or [])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    # Send an initial "searching" message via followup (interaction may already be deferred by bot.py)
    status_message: Optional[discord.Message] = None
    try:
        status_message = await interaction.followup.send(
            "🔍 Searching Discord history...",
        )
    except Exception:
        pass  # Followup may fail if interaction is not yet deferred

    for round_num in range(max_rounds):
        metrics.rounds_executed += 1
        logger.info("--- Agent round %d/%d (messages_in_context=%d) ---",
                     round_num + 1, max_rounds, len(messages))
        # Update status to show planning phase
        if status_message:
            try:
                await status_message.edit(
                    content=f"🔍 Thinking... (round {round_num + 1}/{max_rounds})"
                )
            except Exception:
                pass
        # --- Send request to the agent model with tools attached ---
        llm_start = time.monotonic()
        try:
            result = await proxy_client.chat_with_tools(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=AGENT_TEMPERATURE,
            )
        except ProxyError as e:
            metrics.error = str(e)
            logger.error("ProxyError in agent loop: %s", e)
            if status_message:
                try:
                    await status_message.edit(content=f"❌ Search failed")
                except Exception:
                    pass
            return f"⚠️ Backend error: {e}"
        finally:
            llm_elapsed = time.monotonic() - llm_start
            metrics.llm_response_times.append(llm_elapsed)
            logger.info("LLM response in %.2fs", llm_elapsed)

        content = result.get("content")
        tool_calls = result.get("tool_calls")

        # Log what the model returned
        if tool_calls:
            tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            logger.info("Model wants to call %d tool(s): %s", len(tool_calls), tool_names)
        elif content:
            logger.info("Model returned final answer (%d chars)", len(content))

        if tool_calls:
            # Model wants to call tools — show which tools are being called
            tool_names_list = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            if status_message:
                try:
                    await status_message.edit(
                        content=f"🔍 Searching... ({round_num + 1}/{max_rounds})"
                    )
                except Exception:
                    pass

            # Model wants to call tools — execute them and append results
            for tc in tool_calls:
                function_name = tc.get("function", {}).get("name", "")
                function_args_json = tc.get("function", {}).get("arguments", "{}")
                tool_call_id = tc.get("id", "")

                # Parse arguments
                try:
                    function_args = json.loads(function_args_json) if isinstance(function_args_json, str) else function_args_json
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to parse tool args for %s: %r", function_name, function_args_json)
                    function_args = {}

                # Execute the tool
                metrics.total_tool_calls += 1
                tool_result = await execute_tool(function_name, function_args, rag_client, metrics=metrics)

                # Log tool result length for debugging
                logger.info("Tool %s returned %d chars", function_name, len(tool_result))

                # Append assistant's tool call and the tool result to conversation
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                })
                messages.append({
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result,
                    "tool_call_id": tool_call_id,
                })

            # Update status after tool execution — show analysis phase
            if status_message:
                try:
                    await status_message.edit(
                        content=f"📝 Analyzing results... (round {round_num + 1}/{max_rounds})"
                    )
                except Exception:
                    pass

        elif content:
            # Model has a final answer — show it's writing before returning
            if status_message:
                try:
                    await status_message.edit(content="✍️ Writing answer...")
                except Exception:
                    pass
            logger.info(metrics.summary())
            logger.info("Final answer preview: %s...", content[:100])

            # Mark search as complete before returning
            if status_message:
                try:
                    question_preview = user_question[:100]
                    if len(user_question) > 100:
                        question_preview += "..."
                    await status_message.edit(
                        content=f"✅ Search complete — \"{question_preview}\""
                    )
                except Exception:
                    pass
            return content.strip()

        else:
            # Empty response — safety fallback
            logger.warning("Model returned empty response (no content, no tool_calls) in round %d", round_num + 1)
            messages.append({
                "role": "assistant",
                "content": "I didn't find enough information. Let me try a different search.",
            })

 # Max rounds reached — force a final answer from whatever context we have
    logger.warning("Max rounds (%d) reached — forcing final answer", max_rounds)
    if status_message:
        try:
            await status_message.edit(
                content=f"📝 Generating final answer..."
            )
        except Exception:
            pass

    try:
        llm_start = time.monotonic()
        # Strip the original tool-calling system prompt (first message) and replace
        # it with a pure synthesis instruction. This prevents the model from
        # continuing its tool-calling pattern and emitting XML markup as text.
        messages[0]["content"] = build_synthesis_prompt()
        # Use chat() instead of chat_with_tools() — no tools in the request at all.
        answer = await proxy_client.chat(
            model=AGENT_MODEL,
            messages=messages,
        )
        llm_elapsed = time.monotonic() - llm_start
        metrics.llm_response_times.append(llm_elapsed)
        logger.info("Final answer generation took %.2fs", llm_elapsed)
        answer = (answer or "I've exhausted my search rounds. Please try rephrasing your question.").strip()
        logger.info(metrics.summary())
        logger.info("Final answer preview: %s...", answer[:100])

        # Mark search as complete before returning
        if status_message:
            try:
                question_preview = user_question[:100]
                if len(user_question) > 100:
                    question_preview += "..."
                await status_message.edit(
                    content=f"✅ Search complete — \"{question_preview}\""
                )
            except Exception:
                pass
        return answer
    except ProxyError as e:
        metrics.error = str(e)
        logger.exception("Failed to generate final answer")

        # Mark search as failed before returning
        if status_message:
            try:
                await status_message.edit(content="❌ Search failed")
            except Exception:
                pass
        return f"⚠️ Failed to generate final answer: {e}"
