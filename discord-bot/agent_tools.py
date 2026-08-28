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
    AGENT_CTX_COMPACT_PCT,
    AGENT_CTX_HARD_PCT,
    AGENT_CTX_LIMIT,
    AGENT_CTX_SOFT_PCT,
    AGENT_MODEL,
    AGENT_MAX_ROUNDS,
    AGENT_MAX_ROUNDS_HARD_CAP,
    AGENT_TEMPERATURE,
    AGENT_TOP_K,
    LORE_CONTEXT_PATH,
    LORE_FOLLOWUP_MAX_ROUNDS,
)
from lore_session import LoreSession, chunk_key
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

    # Token accounting. prompt/completion/cached describe the most recent call;
    # peak_context_tokens is the high-water mark across the whole run, which is
    # the number that matters against AGENT_CTX_LIMIT.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    peak_context_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def total_duration(self) -> float:
        return time.monotonic() - self.start_time

    def record_usage(
        self,
        usage: Optional[dict],
        fallback_prompt_chars: int = 0,
        fallback_completion_chars: int = 0,
    ) -> None:
        """
        Absorb one backend usage dict.

        The backend reports real counts on both the streamed and buffered
        paths, so the character estimate is only a guard against a build that
        drops the field — it uses the same ~4 chars/token conversion as
        proxy/main.py's Prometheus fallback.
        """
        if usage:
            self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            self.completion_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            self.cached_tokens = int(details.get("cached_tokens") or 0)
        else:
            self.prompt_tokens = fallback_prompt_chars // 4
            self.completion_tokens = fallback_completion_chars // 4
            self.cached_tokens = 0
        self.total_completion_tokens += self.completion_tokens
        self.peak_context_tokens = max(
            self.peak_context_tokens, self.prompt_tokens + self.completion_tokens
        )

    @property
    def context_used(self) -> int:
        """Tokens the most recent call actually occupied in the slot."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def context_pct(self) -> float:
        """Peak context occupancy as a fraction of the model's window."""
        return self.peak_context_tokens / AGENT_CTX_LIMIT if AGENT_CTX_LIMIT else 0.0

    def context_line(self) -> str:
        """One-line context summary for the round log."""
        used = self.context_used
        pct = (used / AGENT_CTX_LIMIT * 100) if AGENT_CTX_LIMIT else 0.0
        return (
            f"context={used:,}/{AGENT_CTX_LIMIT:,} ({pct:.0f}%) "
            f"cached={self.cached_tokens:,}"
        )

    def summary(self) -> str:
        return (
            f"AgentMetrics — duration={self.total_duration:.1f}s, "
            f"rounds={self.rounds_executed}, tool_calls={self.total_tool_calls}, "
            f"tools={self.tools_used!r}, "
            f"avg_llm={sum(self.llm_response_times)/max(len(self.llm_response_times),1):.2f}s, "
            f"avg_rag={sum(self.rag_query_times)/max(len(self.rag_query_times),1):.2f}s, "
            f"peak_context={self.peak_context_tokens:,}/{AGENT_CTX_LIMIT:,} "
            f"({self.context_pct * 100:.0f}%), "
            f"generated={self.total_completion_tokens:,} tok"
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

# Rules that govern how an answer is WRITTEN, as opposed to how the archive is
# searched. Both prompts below need them: the synthesis turn is the only one
# that produces user-visible prose, and it had drifted into carrying none of
# these — no anti-fabrication rule, no citation rule, no date. Keeping one copy
# means the two prompts cannot silently diverge again.
ANSWER_RULES: str = """WRITING THE ANSWER:
- Do NOT fabricate information that isn't in the retrieved context. Where the
  archive does not cover something, say so plainly instead of filling the gap.
- Always cite which channel(s) and approximate time period your information
  comes from: "According to discussions in #general around mid-2023...".
- Be specific with dates, author names, and direct quotes wherever the context
  supports it.
- If conflicting information exists, present both sides.
- Structure longer answers with clear paragraphs or bullet points."""

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


def build_system_prompt(channel_names: list[str], now: Optional[str] = None) -> str:
    """
    Build the system prompt for the RAG agent.

    Args:
        channel_names: List of available Discord channel names the agent can search.
        now: Pre-rendered timestamp. Pass one to pin it for the life of a
            lore thread session — this string sits at the front of the cached
            prefix, so letting it tick would re-prefill the whole conversation
            on every turn. Defaults to the current time.

    Returns:
        Complete system prompt string.
    """
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
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
6. If no relevant information exists in the search results, honestly state that
   you couldn't find relevant information.
7. When you have enough information to answer, respond naturally (do NOT call
   tools again).
8. If the question is unclear, ask for clarification instead of searching blindly.
9. You can make multiple tool calls in a single turn if needed (e.g., search
   different channels).

{ANSWER_RULES}"""


# The clause that makes a lore thread a conversation rather than a series of cold
# /lore runs. Without it the model treats every follow-up as a fresh research
# task and re-searches material already sitting in the thread — which is both
# slow and the fastest way to exhaust the context window.
LORE_FOLLOWUP_CLAUSE: str = """ONGOING CONVERSATION:
You are answering inside a continuing thread. Everything retrieved so far — for
the first question and every follow-up since — already appears above as
[Search N] blocks, and you can read all of it.

Search again ONLY if the new message asks for something that material genuinely
does not cover. If it can be answered from what is already present — including
rephrasing it, expanding on it, comparing parts of it, or drawing conclusions
from it — then answer directly and call no tools at all. Re-running a search you
have already run spends the thread's remaining context to return text you can
already see."""


def build_lore_followup_prompt(
    channel_names: list[str],
    now: Optional[str] = None,
) -> str:
    """
    Build the system prompt for follow-up turns in a lore thread.

    Identical to the /lore agent prompt plus the follow-up clause. Build this
    ONCE per session and store the result: it is the head of the cached prefix,
    so rebuilding it per turn — and letting its timestamp tick — would change
    the very first tokens and force a full re-prefill of the whole thread.

    Args:
        channel_names: Channels the agent may search.
        now: Pre-rendered timestamp to pin. Defaults to the current time.

    Returns:
        Complete follow-up system prompt string.
    """
    base = build_system_prompt(channel_names, now=now)
    return f"{base}\n\n{LORE_FOLLOWUP_CLAUSE}"


def build_synthesis_prompt(now: Optional[str] = None) -> str:
    """
    Build the system prompt used when max_rounds is exhausted.

    This replaces the tool-calling prompt so the model stops emitting tool
    calls and simply synthesizes an answer from the tool results already in
    the conversation. It keeps the same background knowledge so member names
    are still resolved correctly in the final answer, and the same ANSWER_RULES
    the tool-calling prompt carries — this is the turn that actually writes the
    user-facing prose, so it is the turn that most needs them.

    Args:
        now: Pre-rendered timestamp; defaults to the current time. See
            build_system_prompt() for why a caller would pin it.

    Returns:
        Complete synthesis system prompt string.
    """
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""{AGENT_IDENTITY}

CURRENT DATE/TIME: {now}
{_background_block()}
You have completed your searches and gathered information. Please now \
synthesize a final answer based on all the tool results in this conversation. \
Write a natural language response directed at the user. Do NOT output any \
tool calls, function names, or XML tags.

{ANSWER_RULES}"""

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

# How often the "generating final answer" status message is refreshed while the
# synthesis streams. Discord allows roughly 5 edits per 5s on one message; the
# point here is only to show the answer is still coming, so keep it well under.
_SYNTHESIS_PROGRESS_INTERVAL = 10.0


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


# ---------------------------------------------------------------------------
# Shared progress reporting
# ---------------------------------------------------------------------------

_STATUS_START = "\U0001f50d Searching Discord history..."


class LoreStatus:
    """
    The single source of /lore's progress vocabulary.

    Both the opening run and thread follow-ups report through this, so the
    indicators a user sees in a thread are literally the same strings the
    slash command shows — there is no second copy to drift.

    Every edit is best-effort: a status update must never be the thing that
    fails a run, so all Discord errors are swallowed.
    """

    def __init__(self, message: Optional[discord.Message] = None):
        self._message = message

    @classmethod
    async def from_interaction(cls, interaction: discord.Interaction) -> "LoreStatus":
        """Open a status message as a followup on a (deferred) interaction."""
        try:
            return cls(await interaction.followup.send(_STATUS_START))
        except Exception:
            # Followup fails if the interaction was never deferred; the run
            # should still proceed, just silently.
            return cls(None)

    @classmethod
    async def from_thread(cls, thread: discord.Thread) -> "LoreStatus":
        """Open a status message inside a thread."""
        try:
            return cls(await thread.send(_STATUS_START))
        except Exception:
            return cls(None)

    @property
    def message(self) -> Optional[discord.Message]:
        """The underlying message, so a caller can edit it into the answer."""
        return self._message

    async def _set(self, content: str) -> None:
        if self._message is None:
            return
        try:
            await self._message.edit(content=content)
        except Exception:
            pass

    async def waiting(self) -> None:
        """Queued behind another turn in the same thread."""
        await self._set("\u23f3 Waiting for the previous question to finish...")

    async def thinking(self, round_num: int, max_rounds: int) -> None:
        await self._set(f"\U0001f50d Thinking... (round {round_num}/{max_rounds})")

    async def searching(self, round_num: int, max_rounds: int) -> None:
        await self._set(f"\U0001f50d Searching... ({round_num}/{max_rounds})")

    async def analyzing(self, round_num: int, max_rounds: int) -> None:
        await self._set(f"\U0001f4dd Analyzing results... (round {round_num}/{max_rounds})")

    async def writing(self) -> None:
        await self._set("\u270d\ufe0f Writing answer...")

    async def generating(self, chars: Optional[int] = None) -> None:
        if chars is None:
            await self._set("\U0001f4dd Generating final answer...")
        else:
            await self._set(f"\U0001f4dd Generating final answer\u2026 ({chars:,} chars)")

    async def complete(self, question: str) -> None:
        preview = question[:100] + ("..." if len(question) > 100 else "")
        await self._set(f"\u2705 Search complete \u2014 \"{preview}\"")

    async def failed(self) -> None:
        await self._set("\u274c Search failed")


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
    status: Optional[LoreStatus] = None,
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
    interaction: discord.Interaction,
    channel_names: Optional[list[str]] = None,
    max_rounds: Optional[int] = None,
    status: Optional[LoreStatus] = None,
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
        interaction: Interaction used to open a status message when none is given.
        channel_names: Channel names for the system prompt.
        max_rounds: Round cap (defaults to AGENT_MAX_ROUNDS, hard-capped).
        status: Progress reporter. One is opened on the interaction if omitted.

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

    if status is None:
        status = await LoreStatus.from_interaction(interaction)

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
        # Stream instead of chat() — no tools in the request either way. The
        # synthesis prefills the whole research payload cold (~50k tokens) and
        # then writes a long answer, which regularly ran past 200s against a
        # 120s read timeout. Buffered, that timeout is a deadline on the entire
        # generation; streamed, it only fires if the model stalls mid-answer,
        # and dropping the socket now actually cancels the work upstream
        # instead of leaving it to finish into nothing while holding the GPU.
        parts: list[str] = []
        chars = 0
        last_edit = time.monotonic()
        usage: dict = {}
        async for token in proxy_client.chat_stream(
            AGENT_MODEL,
            synthesis_messages,
            usage_sink=usage,
        ):
            parts.append(token)
            chars += len(token)
            if time.monotonic() - last_edit >= _SYNTHESIS_PROGRESS_INTERVAL:
                last_edit = time.monotonic()
                await status.generating(chars)
        answer = "".join(parts)
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
            "Final answer generation took %.2fs (%d chars in %d chunk(s)) — %s",
            llm_elapsed, len(answer), len(parts), metrics.context_line(),
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

    prompt = [
        {
            "role": "system",
            "content": (
                "You compress research notes without inventing anything. Keep "
                "every distinct fact, name, channel reference, date and direct "
                "quote that appears. Drop only repetition and filler. Write "
                "dense prose or bullets, no preamble."
            ),
        },
        {
            "role": "user",
            "content": (
                "Condense the following Discord archive excerpts. Preserve "
                "channel names, dates, authors and quotes verbatim where they "
                "carry meaning.\n\n" + original
            ),
        },
    ]

    # No tools here on purpose: compaction already invalidates the prefix, so
    # there is no cache to protect, and a summariser has no use for them.
    #
    # No reasoning either. The agent model is a hybrid reasoner and deliberates
    # by default even on mechanical work — a summarising call of this shape was
    # measured at 52.2s and 724 completion tokens with thinking on, versus 1.0s
    # and 27 tokens with it off, for the same output. Compression is extraction,
    # not judgement.
    usage: dict = {}
    parts: list[str] = []
    try:
        async for token in proxy_client.chat_stream(
            AGENT_MODEL, prompt, usage_sink=usage, enable_thinking=False
        ):
            parts.append(token)
    except ProxyError as e:
        logger.warning("Compaction failed, leaving transcript intact: %s", e)
        return False

    summary = "".join(parts).strip()
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


async def run_lore_turn(
    session: LoreSession,
    question: str,
    proxy_client: ProxyClient,
    rag_client: RAGClient,
    max_rounds: Optional[int] = None,
    status: Optional[LoreStatus] = None,
) -> tuple[str, AgentMetrics, bool]:
    """
    Run one follow-up turn against an existing lore session.

    Two paths. If the model decides the thread already holds what it needs, it
    answers on the first call and that answer is used as-is — no second request,
    and the prefix cache makes it cheap. If it calls tools, the results are
    rendered into one research block, appended, and the answer is generated by a
    separate streamed call.

    Progress is reported through the same LoreStatus vocabulary the opening
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
    pct = session.pct_of(AGENT_CTX_LIMIT)

    logger.info("=" * 60)
    logger.info(
        "LORE FOLLOW-UP turn %d (thread=%d) — question=%r",
        session.turns + 1, session.thread_id, question[:80],
    )
    logger.info(
        "Session context before turn: %s (%d%% of limit), %d search block(s)",
        f"{used:,}/{AGENT_CTX_LIMIT:,}", pct * 100, session.searches,
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
        extra.append(_budget_notice(used, AGENT_CTX_LIMIT))

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
        usage: dict = {}
        parts: list[str] = []
        chars = 0
        last_edit = time.monotonic()
        async for token in proxy_client.chat_stream(
            AGENT_MODEL,
            answer_messages,
            usage_sink=usage,
            tools=TOOLS,
        ):
            parts.append(token)
            chars += len(token)
            if status and time.monotonic() - last_edit >= _SYNTHESIS_PROGRESS_INTERVAL:
                last_edit = time.monotonic()
                await status.generating(chars)
        answer = "".join(parts).strip()

        if not answer:
            # The model answered with a tool call instead of prose despite the
            # nudge, so there were no content deltas to collect. Retry without
            # tools, which removes the option entirely at the cost of one cold
            # prefill.
            logger.warning(
                "Streamed answer was empty (likely a tool call) — retrying without tools"
            )
            usage = {}
            parts = []
            async for token in proxy_client.chat_stream(
                AGENT_MODEL, answer_messages, usage_sink=usage
            ):
                parts.append(token)
            answer = "".join(parts).strip()

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
        f"{session.last_context_tokens:,}", f"{AGENT_CTX_LIMIT:,}",
    )
    return session


async def maybe_compact_session(
    session: LoreSession,
    proxy_client: ProxyClient,
) -> bool:
    """
    Compact a session if it has crossed the compaction threshold.

    Called after a turn has been answered and posted, so the cost lands between
    messages rather than in front of the user's answer.
    """
    pct = session.pct_of(AGENT_CTX_LIMIT)
    if pct < AGENT_CTX_COMPACT_PCT:
        return False
    logger.info(
        "Session %d at %.0f%% of context (>= %.0f%%) — compacting",
        session.thread_id, pct * 100, AGENT_CTX_COMPACT_PCT * 100,
    )
    metrics = AgentMetrics()
    return await _compact_session(session, proxy_client, metrics)
