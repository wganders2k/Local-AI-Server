"""
The /lore agent's tools: their schemas, and the code that runs them.

Schemas and executor live together deliberately. A tool's ``description`` is
prompt text, but it is also an API contract sent verbatim to the model and
coupled line-for-line to the branch that implements it — splitting the two
across files is how they drift apart.

Every handler degrades to a sentence of prose rather than an exception: the
model reads whatever comes back, so a failure has to be legible to it. A
zero-result answer in particular says whether the result is *conclusive*
(exact search over the whole archive) or merely *empty* (top-k semantic
search), because the agent's next move differs.
"""

import logging
import time
from typing import Awaitable, Callable, Optional

from config import AGENT_TOP_K
from lore.metrics import AgentMetrics
from rag_client import RAGClient

logger = logging.getLogger("mimic-bot.lore.tools")


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
# Handlers
# ---------------------------------------------------------------------------
#
# One coroutine per tool, each returning the string the model will read. They
# are dispatched from the table at the bottom of this module; execute_tool()
# owns only logging, timing and error containment.


def _date_scope(start_date: Optional[str], end_date: Optional[str]) -> str:
    """The " between X and Y" clause of a no-results message, or ""."""
    if not (start_date or end_date):
        return ""
    return f" between {start_date} and {end_date}"


def _semantic_miss(
    query: str,
    channel_name: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> str:
    """
    What to tell the model when a semantic search returns nothing.

    Semantic search only ever sees the top-k nearest chunks, so an empty result
    is weak evidence — it says the query did not match, not that the topic is
    absent. The advice therefore pushes towards broadening, never towards
    concluding.
    """
    scope = f" in channel '{channel_name}'" if channel_name else ""
    return (
        f"No results found for query '{query}'{scope}"
        f"{_date_scope(start_date, end_date)}."
        " This may mean: (a) the topic was never discussed, (b) the date range is too narrow, "
        "or (c) the query terms don't match how users phrased things. Try broadening the search "
        "by removing date filters, using different keywords, or searching without a channel filter."
    )


async def _search_discord_history(args: dict, rag: RAGClient) -> str:
    query = args.get("query", "")
    # The model sometimes passes a channel to the un-scoped search. Honour it
    # rather than dropping it silently — and read it from args, never from a
    # variable that only exists on a sibling branch.
    channel_name = args.get("channel_name") or None
    start_date, end_date = args.get("start_date"), args.get("end_date")
    top_k = args.get("top_k", AGENT_TOP_K)

    context, chunk_count = await rag.retrieve(
        query, top_k=top_k, channel_name=channel_name,
        start_date=start_date, end_date=end_date,
    )
    logger.info(
        "RAG search_discord_history: query=%r, channel=%r, top_k=%d → %d chunks",
        query, channel_name, top_k, chunk_count,
    )
    if chunk_count == 0:
        return _semantic_miss(query, channel_name, start_date, end_date)
    return context


async def _search_channel_history(args: dict, rag: RAGClient) -> str:
    query = args.get("query", "")
    channel_name = args.get("channel_name", "")
    start_date, end_date = args.get("start_date"), args.get("end_date")
    top_k = args.get("top_k", AGENT_TOP_K)

    context, chunk_count = await rag.retrieve(
        query, top_k=top_k, channel_name=channel_name,
        start_date=start_date, end_date=end_date,
    )
    logger.info(
        "RAG search_channel_history: query=%r, channel=%r, top_k=%d → %d chunks",
        query, channel_name, top_k, chunk_count,
    )
    if chunk_count == 0:
        return (
            f"No results found for query '{query}' in channel '{channel_name}'"
            f"{_date_scope(start_date, end_date)}."
            " This may mean the topic wasn't discussed in this channel, the date range is too narrow, "
            "or the query terms don't match. Try a different channel, broader date range, or different keywords."
        )
    return context


async def _summarize_channel(args: dict, rag: RAGClient) -> str:
    channel_name = args.get("channel_name", "")
    start_date, end_date = args.get("start_date"), args.get("end_date")
    top_k = args.get("top_k", 20)
    # Reuse search but with higher top_k and a generic summarization query
    query = f"Recent conversations and discussions in {channel_name}"

    context, chunk_count = await rag.retrieve(
        query, top_k=top_k, channel_name=channel_name,
        start_date=start_date, end_date=end_date,
    )
    logger.info(
        "RAG summarize_channel: channel=%r, top_k=%d → %d chunks",
        channel_name, top_k, chunk_count,
    )
    if chunk_count == 0:
        return (
            f"No message history found for channel '{channel_name}'"
            f"{_date_scope(start_date, end_date)}."
            " The channel may not exist, has no ingested messages, or the date range has no activity."
        )
    return context


async def _search_exact_chronological(args: dict, rag: RAGClient) -> str:
    term = args.get("term") or None
    author = args.get("author") or None
    order = args.get("order") or "earliest"
    if not term and not author:
        return (
            "Error: search_exact_chronological needs at least one of 'term' or 'author'. "
            "If you do not know the exact wording, use search_discord_history instead."
        )

    context, total = await rag.search_literal(
        term=term, author=author,
        channel_name=args.get("channel_name"),
        start_date=args.get("start_date"), end_date=args.get("end_date"),
        order=order, limit=args.get("limit", 20),
        whole_word=bool(args.get("whole_word", False)),
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

    from lore.research import _RAG_CHUNK_SEPARATOR

    shown = context.count(_RAG_CHUNK_SEPARATOR) + 1
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


async def _count_messages(args: dict, rag: RAGClient) -> str:
    term = args.get("term") or None
    author = args.get("author") or None
    group_by = args.get("group_by") or "author"
    # The model tends to write channels the way Discord renders them.
    # Metadata stores the bare name, so a leading '#' would silently
    # match nothing and quietly exclude nothing.
    raw_exclude = args.get("exclude_channels") or []
    if isinstance(raw_exclude, str):
        raw_exclude = [raw_exclude]
    exclude_channels = [c.lstrip("#").strip() for c in raw_exclude if c and c.strip()]

    report, total = await rag.aggregate(
        term=term, author=author,
        channel_name=args.get("channel_name"),
        start_date=args.get("start_date"), end_date=args.get("end_date"),
        group_by=group_by, top_n=args.get("top_n", 25),
        whole_word=bool(args.get("whole_word", False)),
        exclude_channels=exclude_channels or None,
    )
    logger.info(
        "RAG count_messages: term=%r author=%r group_by=%s exclude=%r -> %d",
        term, author, group_by, exclude_channels, total,
    )
    if total == 0:
        return (
            "No messages matched, so there is nothing to count. This counted over the "
            "whole archive, so the answer is zero for that exact wording — do not repeat "
            "this search."
        )
    return report


Handler = Callable[[dict, RAGClient], Awaitable[str]]

HANDLERS: dict[str, Handler] = {
    "search_discord_history": _search_discord_history,
    "search_channel_history": _search_channel_history,
    "summarize_channel": _summarize_channel,
    "search_exact_chronological": _search_exact_chronological,
    "count_messages": _count_messages,
}

# Every advertised tool must be runnable, and nothing unadvertised should be.
assert {t["function"]["name"] for t in TOOLS} == set(HANDLERS), (
    "TOOLS and HANDLERS disagree: "
    f"{ {t['function']['name'] for t in TOOLS} ^ set(HANDLERS) }"
)


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
        The tool result as a formatted string for the LLM to consume. Errors are
        returned, not raised — the model has to be able to read and react to them.
    """
    logger.info(
        "Executing tool=%s with args={%s}",
        tool_name,
        ", ".join(f"{k}={v!r}" for k, v in tool_args.items()),
    )

    if metrics is not None:
        metrics.tools_used.append(tool_name)

    handler = HANDLERS.get(tool_name)
    if handler is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return f"Error: Unknown tool '{tool_name}'."

    tool_start = time.monotonic()
    try:
        return await handler(tool_args, rag_client)
    except Exception as e:
        logger.exception("Tool execution failed for %s", tool_name)
        return f"Error executing {tool_name}: {type(e).__name__}: {e}"
    finally:
        if metrics is not None:
            elapsed = time.monotonic() - tool_start
            metrics.tool_call_times.append(elapsed)
            logger.info("Tool %s completed in %.2fs", tool_name, elapsed)
