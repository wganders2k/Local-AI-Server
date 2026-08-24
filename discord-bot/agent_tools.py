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
            "description": "SEMANTIC search across all channels. Finds messages that are *about* a topic even when they never use your exact words \u2014 good for themes, vibes, opinions and paraphrases (\"what do people think of X\"). It returns only the top_k most similar chunks out of the whole archive, ranked by meaning, so it CANNOT tell you what came first, count anything, or guarantee it has found every mention. For an exact word or name, for \"first/earliest\" questions, or for anything needing complete coverage, use search_exact_chronological instead.",
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
            "description": "SEMANTIC search restricted to one channel. Same meaning-based matching and same limits as search_discord_history \u2014 top_k results by similarity, no ordering, no counting. Use when the question is clearly scoped to a channel.",
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
    {
        "type": "function",
        "function": {
            "name": "search_exact_chronological",
            "description": (
                "EXACT-TEXT search over the ENTIRE archive, returned in time order. "
                "Matches the literal characters you give (case-insensitive), not the meaning, "
                "and scans every message rather than a top-k sample \u2014 so it is the only tool "
                "that can answer 'who said X first', 'when did X start', 'what was the earliest/"
                "latest X', or give a complete list of mentions. Use it for exact words, names, "
                "in-jokes and coined terms (which semantic search often misses entirely), and "
                "whenever the question contains first/earliest/last/latest/original. "
                "Set author to get only chunks where that person spoke; set both term and author "
                "to return only messages that person actually wrote containing that term - the "
                "matching message is quoted at the top of each result. Because matching is substring-based, the oldest hit can be incidental (the term inside a larger word, or only in a link URL) - request several results and pick the one that genuinely answers the question. It reports the TOTAL number of "
                "matches even when it returns fewer, so you can tell how complete your picture is. "
                "It cannot find paraphrases \u2014 if the exact wording is unknown, use "
                "search_discord_history instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Exact text to find, matched case-insensitively as a substring. Keep it short - one word or a short phrase. Optional if author is given.",
                    },
                    "author": {
                        "type": "string",
                        "description": "Exact username (not a nickname or display name) whose messages to match. Optional if term is given.",
                    },
                    "order": {
                        "type": "string",
                        "enum": ["earliest", "latest"],
                        "description": "'earliest' returns the oldest matches first - use this for 'first/original' questions. 'latest' returns newest first. Default 'earliest'.",
                    },
                    "channel_name": {
                        "type": "string",
                        "description": "Optional - restrict to one channel.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional ISO 8601 lower bound, e.g. '2024-01-01T00:00:00Z'.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional ISO 8601 upper bound.",
                    },
                    "whole_word": {
                        "type": "boolean",
                        "description": "Default false: the term matches anywhere, so 'oink' also matches 'oinking', 'oinked' AND 'yoink'. Set true to require the match to start at a word boundary - still matches 'oinking'/'oinks' but not 'yoink'/'sploinky'. Use it when a short term is colliding with unrelated words.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum chunks to return after ordering. Default 20.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_messages",
            "description": (
                "COUNT matching messages and group them by author, channel or month, without "
                "returning the messages themselves. Use for 'how many times', 'who says X most', "
                "'which channel is X discussed in', 'when was X most active', or to check how "
                "common something is before deciding whether to search for it. Counting is done "
                "per message and attributed to the person who actually wrote it. Because it "
                "returns a small summary instead of raw conversation, prefer it over repeated "
                "searching when the question is about volume, ranking or distribution rather "
                "than about what was actually said. Use exclude_channels to drop bulk-posted "
                "catalogue channels that would otherwise skew an author ranking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Exact text to count, matched case-insensitively. Omit to count all messages in scope.",
                    },
                    "author": {
                        "type": "string",
                        "description": "Optional - count only this person's messages.",
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["author", "channel", "month"],
                        "description": "How to group the counts. 'author' answers who-says-it-most, 'channel' where, 'month' when. Default 'author'.",
                    },
                    "channel_name": {
                        "type": "string",
                        "description": "Optional - restrict to one channel.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional ISO 8601 lower bound.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional ISO 8601 upper bound.",
                    },
                    "whole_word": {
                        "type": "boolean",
                        "description": "Default false: the term matches anywhere, so 'oink' also matches 'oinking', 'oinked' AND 'yoink'. Set true to require the match to start at a word boundary - still matches 'oinking'/'oinks' but not 'yoink'/'sploinky'. Use it when a short term is colliding with unrelated words.",
                    },
                    "exclude_channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional - channel names to leave out of the count. Use this for "
                            "bulk-posted catalogue or log channels, where one person posts a long "
                            "reference document rather than talking: their entries are counted as "
                            "messages and can dominate an author ranking without reflecting how the "
                            "term is actually used in conversation. '#grant-chronicle' is the main "
                            "one on this server - a solo album-review catalogue. Pass "
                            "['grant-chronicle'] when the question is about what people say to each "
                            "other, and leave it out when the question is about the archive as a whole."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many groups to report. Default 25.",
                    },
                },
                "required": [],
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
CHOOSING A TOOL — this matters more than the query you write:

  Two different kinds of search are available, and they fail in opposite ways.

  SEMANTIC (search_discord_history, search_channel_history, summarize_channel)
    Matches meaning, so it finds paraphrases and related discussion. But it only
    ever returns the top_k closest chunks out of the whole archive, ranked by
    similarity. It therefore cannot tell you what came FIRST, cannot count, and
    can silently miss an exact word — rare words, names and in-jokes are poorly
    represented by embeddings.
    -> Use for: themes, opinions, "what do people think about X", "what is the
       story behind X", when you do not know the exact wording.

  EXACT (search_exact_chronological, count_messages)
    Matches literal text case-insensitively across EVERY message, and orders by
    time. It reports the total number of matches, so you know your coverage.
    It cannot find paraphrases.
    -> Use for: exact words, names, in-jokes and coined terms; any question
       containing first / earliest / original / last / latest / when did X
       start; anything needing a complete list; and per-person questions via
       the `author` parameter.
    -> Use count_messages for how many / who most / where / when-most-active,
       instead of searching repeatedly and counting by hand.

CHANNELS THAT SKEW COUNTS

  #grant-chronicle is not a conversation. It is a solo catalogue trystero49
  bulk-posted, one album review per message, so its lines count as messages and
  can bury everyone else in an author ranking — for 'oink' it supplies 113 of
  trystero49's 173 hits while nobody else has a single line in there.
  When a counting question is about what people say to EACH OTHER — who says X
  most, who is the biggest X-poster — pass exclude_channels=['grant-chronicle']
  and say that you excluded it. When the question is about the archive itself,
  or about trystero49's catalogue, leave it in.

COMBINING TOOLS — most good answers use more than one call:

  "Everyone's first X" / "each person's X" / "who all did X"
    1. count_messages(term='X', group_by='author')  -> the complete roster of
       who actually said it, and how often. This is the only way to know you
       have everyone. Add exclude_channels=['grant-chronicle'] if the question
       is about conversation rather than about the archive as a whole.
    2. For each name returned: search_exact_chronological(term='X',
       author=<name>, order='earliest', limit=5) -> their earliest few.
       Ask for several, never limit=1. Matching is substring-based, so the
       single oldest hit is frequently incidental: the term sitting inside a
       larger word ('oink' inside 'yoink' or 'zoinked'), or appearing only in
       a GIF/link URL rather than something the person actually said. Read the
       '>>> MATCH' lines and pick the earliest one where the term is genuinely
       used the way the question means it — not merely the first row returned.
       Say so briefly if you skipped earlier incidental matches. If all five
       look incidental, retry that person with whole_word=true, which ignores
       matches inside larger words while still matching 'oinking'/'oinks'.
    Do NOT try to answer this from a single search: one search returns only the
    globally oldest matches, which are usually all from the same one or two
    people, and you will silently miss everyone else.

  "When did X start / who said it first"
    search_exact_chronological(term='X', order='earliest', limit=5), then
    optionally search_discord_history around that date for the context and
    reaction that the exact match alone does not explain.

  "Where / when is X discussed most"
    count_messages(term='X', group_by='channel' or 'month') first, then search
    only the channel or period that actually matters.

  Always check the totals a tool reports. If a search says more matches exist
  than it showed you, your view is partial — narrow it or enumerate per author
  before you answer.

YOUR INSTRUCTIONS:
1. Pick the tool that matches the KIND of question, using the guide above.
2. Use these tools to find relevant information BEFORE answering.
3. If a tool reports zero matches over the whole archive, that answer is
   conclusive for that exact wording — change your approach or your tool, and
   never reissue a query you have already run. Re-running an identical search
   wastes a round and returns identical results.
4. If initial results are insufficient, change ONE thing at a time: the tool,
   the exact term, or the scope. Broaden before narrowing.
5. After gathering enough context, synthesize a comprehensive answer based on
   what you found.
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

        elif tool_name == "search_exact_chronological":
            term = tool_args.get("term") or None
            author = tool_args.get("author") or None
            order = tool_args.get("order") or "earliest"
            channel_name = tool_args.get("channel_name")
            start_date = tool_args.get("start_date")
            end_date = tool_args.get("end_date")
            limit = tool_args.get("limit", 20)
            if not term and not author:
                return (
                    "Error: search_exact_chronological needs at least one of 'term' or 'author'. "
                    "If you do not know the exact wording, use search_discord_history instead."
                )
            context, total = await rag_client.search_literal(
                term=term, author=author, channel_name=channel_name,
                start_date=start_date, end_date=end_date, order=order, limit=limit,
                whole_word=bool(tool_args.get("whole_word", False)),
            )
            logger.info(
                "RAG search_exact_chronological: term=%r author=%r order=%s -> %d total",
                term, author, order, total,
            )
            if total == 0:
                scope = f"'{term}'" if term else ""
                if author:
                    scope = f"{scope} by {author}".strip()
                return (
                    f"No messages anywhere in the archive literally contain {scope}. "
                    "This is an exact-text search over everything, so this result is conclusive "
                    "for that exact wording — do not repeat it. Either the phrasing differs "
                    "(try search_discord_history for the concept, or a shorter/simpler term), "
                    "or it genuinely was never said."
                )
            shown = context.count("\n\n---\n\n") + 1
            header = (
                f"Found {total} matching chunk(s); showing the {shown} "
                f"{'oldest' if order == 'earliest' else 'newest'} in time order.\n"
            )
            if total > shown:
                # The agent cannot see what it was not sent, so say plainly that
                # the view is partial and how to complete it. A per-person
                # question answered from a truncated window looks complete and
                # is wrong.
                header += (
                    f"WARNING: {total - shown} further match(es) exist and are NOT shown, "
                    f"so this is a partial view. Do NOT treat it as the full picture. "
                    f"For a per-person or 'everyone' question, call count_messages with "
                    f"group_by='author' to get the complete roster of who said it, then "
                    f"call this tool once per person with author=<name> and limit=5, "
                    f"picking the earliest result that genuinely matches the intent rather "
                    f"than the first row. "
                    f"To page through time instead, repeat with start_date set just after "
                    f"the last result shown.\n"
                )
            return header + "\n" + context

        elif tool_name == "count_messages":
            term = tool_args.get("term") or None
            author = tool_args.get("author") or None
            group_by = tool_args.get("group_by") or "author"
            # The model tends to write channels the way Discord renders them.
            # Metadata stores the bare name, so a leading '#' would silently
            # match nothing and quietly exclude nothing.
            raw_exclude = tool_args.get("exclude_channels") or []
            if isinstance(raw_exclude, str):
                raw_exclude = [raw_exclude]
            exclude_channels = [c.lstrip("#").strip() for c in raw_exclude if c and c.strip()]
            report, total = await rag_client.aggregate(
                term=term, author=author,
                channel_name=tool_args.get("channel_name"),
                start_date=tool_args.get("start_date"),
                end_date=tool_args.get("end_date"),
                group_by=group_by,
                top_n=tool_args.get("top_n", 25),
                whole_word=bool(tool_args.get("whole_word", False)),
                exclude_channels=exclude_channels or None,
            )
            logger.info(
                "RAG count_messages: term=%r author=%r group_by=%s exclude=%r -> %d",
                term, author, group_by, exclude_channels, total,
            )
            if total == 0:
                return (
                    f"No messages matched, so there is nothing to count. This counted over the "
                    f"whole archive, so the answer is zero for that exact wording — do not repeat "
                    f"this search."
                )
            return report

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
    seen_chunks: set[str] = set()
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
            if chunk in seen_chunks:
                dropped += 1
                continue
            seen_chunks.add(chunk)
            fresh.append(chunk)

        # A search whose every hit was already shown adds nothing to synthesise.
        if not fresh:
            continue

        blocks.append(
            f"[Search {len(blocks) + 1}] {label}\n"
            + _RAG_CHUNK_SEPARATOR.join(fresh)
        )

    logger.info(
        "Synthesis payload: %d unique chunk(s) across %d search block(s), "
        "%d duplicate chunk(s) dropped",
        len(seen_chunks), len(blocks), dropped,
    )

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
        reasoning = (result.get("reasoning_content") or "").strip()

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

                # Append assistant's tool call and the tool result to conversation.
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
                messages.append(assistant_msg)
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
        # Flatten the tool-calling history into a single plain-text user turn.
        # Replacing just the system prompt left the assistant/tool_calls and
        # role="tool" messages in context and the model kept imitating them,
        # emitting tool-call markup as its answer. See the helper for why that
        # markup reaches the user instead of being parsed out.
        synthesis_messages = _flatten_research_for_synthesis(messages, user_question)
        # Use chat() instead of chat_with_tools() — no tools in the request at all.
        answer = await proxy_client.chat(
            model=AGENT_MODEL,
            messages=synthesis_messages,
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
