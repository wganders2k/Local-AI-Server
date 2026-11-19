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
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from config import (
    AGENT_TOP_K,
    MIMIC_CHAT_DEFAULT_SIM_LENGTH,
    MIMIC_CHAT_MAX_LENGTH,
    MIMIC_CHAT_MAX_PERSONAS,
)
from lore.citations import Citations
from lore.metrics import AgentMetrics
from lore.research import CHUNK_SEPARATOR, SEARCH_NOTE_PREFIX
from mimic.engine import render_transcript, simulate
from personas import PersonaCatalog
from proxy_client import ProxyClient
from rag_client import RAGClient

logger = logging.getLogger("mimic-bot.lore.tools")


_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_discord_history",
            "description": "SEMANTIC search across all channels. Finds messages that are *about* a topic even when they never use your exact words \u2014 good for themes, vibes, opinions and paraphrases (\"what do people think of X\"). It returns the top_k most similar chunks out of the whole archive, ranked by meaning (use `page` to see the next-best ones), so it CANNOT tell you what came first, count anything, or guarantee it has found every mention. For an exact word or name, for \"first/earliest\" questions, or for anything needing complete coverage, use search_exact_chronological instead.",
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
                    "page": {
                        "type": "integer",
                        "description": "Which page of the ranking to show: 1 (default) is the top_k best matches, 2 the next top_k, and so on. When a search's matches are on target and you want more of them, repeat it with the next page rather than rewording it.",
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
                    "page": {
                        "type": "integer",
                        "description": "Which page of the ranking to show: 1 (default) is the top_k best matches, 2 the next top_k, and so on. When a search's matches are on target and you want more of them, repeat it with the next page rather than rewording it.",
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
            "name": "search_images",
            "description": (
                "SEMANTIC search over descriptions of the IMAGES people posted: memes, screenshots, "
                "photos, art. Each image was described by a vision model, including any text visible "
                "in it. Use it when the question is about a picture - 'that meme about X', 'the "
                "screenshot of Y', 'who posted the picture of Z', 'what images did people share "
                "about W'. Ordinary searches do not look inside images, so this is the only way to "
                "find one by what it shows. Each result is the conversation the image was posted "
                "in, led by the matching image. Same limits as any semantic search: top_k by "
                "similarity, no ordering, no counting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What the image shows or says, in natural language (e.g. 'cat wearing sunglasses', 'screenshot of a Steam sale').",
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
                    "top_k": {
                        "type": "integer",
                        "description": "Number of conversations to return. Default 5.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Which page of the ranking to show: 1 (default) is the top_k best matches, 2 the next top_k, and so on. When a search's matches are on target and you want more of them, repeat it with the next page rather than rewording it.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_surrounding_messages",
            "description": (
                "Read the messages just BEFORE or AFTER a message you have already seen, in the "
                "same channel, regardless of where excerpts were cut. Excerpts are slices of a "
                "conversation and can stop mid-exchange; when the part that matters to the "
                "question (an answer, a reaction, how an argument ended, what a reply was "
                "replying to) lies just outside an excerpt, read it with this tool instead of "
                "guessing. Name the message by its ref, e.g. 'm42'. The anchor is marked "
                ">>> ANCHOR, and long silences are noted, so you can tell a cut conversation "
                "from one that ended. To keep reading, call again anchored on the last ref returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "The ref of a message shown in the research, e.g. 'm42'.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["after", "before", "both"],
                        "description": "'after' to read on, 'before' for what led up to it, 'both' for each side. Default 'after'.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Messages to read on each requested side, 1-60. Default 20.",
                    },
                },
                "required": ["ref"],
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
                        # Which channels those are on this server is host-local
                        # knowledge, so it lives in the lore context file
                        # (prompts/lore_context.example.md), not here.
                        "description": (
                            "Optional - channel names to leave out of the count. Use this for "
                            "bulk-posted catalogue or log channels, where one person posts a long "
                            "reference document rather than talking: their entries are counted as "
                            "messages and can dominate an author ranking without reflecting how the "
                            "term is actually used in conversation. Pass them when the question is "
                            "about what people say to each other, and leave them out when the "
                            "question is about the archive as a whole."
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


def _simulation_tool(names: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "simulate_conversation",
            "description": "Runs a SHORT SIMULATED chat between mimic personas: models trained on members' messages, imitating how they talk. Use it only when the question asks for a hypothetical — what members WOULD say, how they would react to or argue about something — never to find out what anyone actually said. Its output is invented, is never evidence, and must be presented as a simulation. Put the facts you found by searching into the premise, because the personas know nothing about the archive. At most one call per question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "personas": {
                        "type": "array",
                        "items": {"type": "string", "enum": names},
                        "minItems": 1,
                        "maxItems": MIMIC_CHAT_MAX_PERSONAS,
                        "description": "Who takes part, by persona name.",
                    },
                    "premise": {
                        "type": "string",
                        "description": "The opening message of the chat, as if someone had just said it in the channel: the topic or situation, plus any facts from your research the personas should react to.",
                    },
                    "length": {
                        "type": "integer",
                        "description": f"Persona turns to generate (a persona following up on itself counts once), 2-{MIMIC_CHAT_MAX_LENGTH}. Default {MIMIC_CHAT_DEFAULT_SIM_LENGTH}.",
                    },
                },
                "required": ["personas", "premise"],
            },
        },
    }


def build_tools(catalog: Optional[PersonaCatalog] = None) -> list[dict]:
    """
    The tool schemas for one run. simulate_conversation is offered only when
    personas are available, and its persona enum is their display names.

    Deterministic for a given persona set, so a thread's every request carries
    byte-identical tools and the prefix cache holds: rebuilding per run costs
    nothing, and a roster change shows up without a restart.
    """
    names = sorted(p.display_name for p in catalog.all()) if catalog is not None else []
    return _SEARCH_TOOLS + ([_simulation_tool(names)] if names else [])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
#
# One coroutine per tool, each taking the model's arguments and a ToolContext
# and returning the string the model will read. They are dispatched from the
# table at the bottom of this module; execute_tool() owns only logging, timing
# and error containment.


@dataclass
class ToolContext:
    """What a tool may use. Any of it can be absent; a tool that needs it says so in prose."""
    rag: Optional[RAGClient] = None
    proxy: Optional[ProxyClient] = None
    catalog: Optional[PersonaCatalog] = None
    metrics: Optional[AgentMetrics] = None
    # Swaps the RAG service's raw message tags for this conversation's [mN]
    # refs. None leaves results untouched.
    citations: Optional[Citations] = None


# Every account handle in the archive, read once from the RAG service. The model
# reaches for display names ("Alice") where `author` needs the handle
# ("alice_e"); that matched nothing, and it went on to build answers from
# messages that merely mention the name. So a name that is not a handle is
# resolved through the server background's alias index where that is
# unambiguous, and otherwise answered with the list of real handles.
_handles: Optional[list[str]] = None
_HANDLE_LINE = re.compile(r"^\s+(\S.*?):\s+\d+\s*$", re.M)


async def _archive_handles(ctx: ToolContext) -> list[str]:
    global _handles
    if _handles is None:
        report, total = await ctx.rag.aggregate(group_by="author", top_n=100)
        if not total:
            return []
        _handles = _HANDLE_LINE.findall(report)
    return _handles


def _background_people(background: str) -> list[tuple[str, list[str], list[str], list[str]]]:
    """
    (name, every name and alias, current aliases, old accounts) for each
    person in the background's alias index.

    An entry is a "Name:" line followed by lines such as "Real Name: ..." and
    "Aliases: a, b (formerly c)" (see prompts/lore_context.example.md).
    "(formerly ...)" aliases are old accounts.
    """
    people: list[tuple[str, list[str], list[str], list[str]]] = []
    for line in background.splitlines():
        head, sep, rest = line.partition(":")
        if not sep:
            continue
        key, rest = head.strip(), rest.strip()
        if not rest and key and not key.isupper():
            people.append((key, re.split(r"[()]", key), [], []))
        elif people and key.lower() == "real name":
            people[-1][1].extend([rest, *rest.replace(",", " ").split()])
        elif people and key.lower() == "aliases":
            former = re.findall(r"\(\s*formerly\s+([^)]*)\)", rest, flags=re.I)
            current = re.sub(r"\(\s*formerly[^)]*\)", "", rest, flags=re.I)
            aliases = [a.strip() for a in current.split(",") if a.strip()]
            people[-1][2].extend(aliases)
            people[-1][1].extend(aliases)
            people[-1][3].extend(a.strip() for f in former for a in f.split(",") if a.strip())
    return people


def _accounts(background: str) -> dict[str, tuple[str, bool]]:
    """
    {account handle -> (person's name, is an old account)}.

    An old account counts only when one person claims it: "Deleted User" is
    Discord's label for any departed account, so if two people list it, it
    could be either and belongs to neither.
    """
    people = _background_people(background)
    claims: dict[str, int] = {}
    for _, _, _, former in people:
        for a in set(former):
            claims[a] = claims.get(a, 0) + 1
    accounts: dict[str, tuple[str, bool]] = {}
    for name, _, current, former in people:
        for a in current:
            accounts.setdefault(a, (name, False))
        for a in former:
            if claims[a] == 1:
                accounts.setdefault(a, (name, True))
    return accounts


def _person_accounts(background: str, handles: list[str]) -> dict[str, list[str]]:
    """
    {person's name -> their accounts that exist in the archive}, current ones
    first, in the archive's own spelling.
    """
    lower = {h.lower(): h for h in handles}
    out: dict[str, list[str]] = {}
    for account, (name, _) in _accounts(background).items():
        if account.lower() in lower:
            out.setdefault(name, []).append(lower[account.lower()])
    return out


def _alias_index(background: str, handles: list[str]) -> dict[str, set[str]]:
    """
    {lowercased name or account -> author filters of the people it may mean}.

    A filter is all of one person's accounts in the archive, comma-separated
    (the RAG service matches any of them), so asking for a person finds their
    old accounts' messages too. A name is ambiguous only when it fits two
    different people.
    """
    accounts = _person_accounts(background, handles)
    index: dict[str, set[str]] = {}
    for name, names, _, _ in _background_people(background):
        if name not in accounts:
            continue
        target = ",".join(accounts[name])
        for n in [*names, *accounts[name]]:
            if n.strip().strip(","):
                index.setdefault(n.strip().strip(",").lower(), set()).add(target)
    return index


# Where a speaker's handle sits in a tool result: after a message tag
# ("[@1/2] alice_e: hi", "[m12] alice_e: hi") or at the head of a count line
# ("  alice_e: 120"). Text inside messages is never touched.
_SPEAKER = r"(?m)(\] |^  )({handles})(?=: )"


def _name_speakers(result: str, background: str) -> str:
    """
    Show each speaker as "Name (handle)", or "Name (handle, old account)",
    using the background's names.

    Results otherwise carry bare handles, and answers repeated them ("Tech-
    Priest alice_e") where the background asks for names — or treated a
    person's old account as somebody else.
    """
    accounts = _accounts(background)
    if not accounts:
        return result
    pattern = _SPEAKER.format(handles="|".join(map(re.escape, sorted(accounts, key=len, reverse=True))))

    def label(m: re.Match) -> str:
        name, old = accounts[m.group(2)]
        return f"{m.group(1)}{name} ({m.group(2)}{', old account' if old else ''})"

    return re.sub(pattern, label, result)


_COUNT_LINE = re.compile(r"^  (.+?): (\d[\d,]*)$")


def _merge_people(report: str, background: str) -> str:
    """
    A count grouped by author, with each person's accounts added together.

    The backend counts per account, so someone who changed accounts appears
    as two rows that the model would have to recognise and add up from the
    background — and did not. Merged rows show their breakdown; single-account
    rows are left for _name_speakers to label. Re-sorted by total.
    """
    accounts = _accounts(background)
    lines = report.splitlines()
    rows = [(i, m.group(1), int(m.group(2).replace(",", ""))) for i, line in enumerate(lines)
            if (m := _COUNT_LINE.match(line))]
    by_person: dict[str, list[tuple[str, int]]] = {}
    for _, account, n in rows:
        if account in accounts:
            by_person.setdefault(accounts[account][0], []).append((account, n))
    if not any(len(v) > 1 for v in by_person.values()):
        return report

    merged: list[tuple[int, str]] = []
    done: set[str] = set()
    for _, account, n in rows:
        person = accounts[account][0] if account in accounts else None
        if person is None or len(by_person[person]) == 1:
            merged.append((n, f"  {account}: {n:,}"))
        elif person not in done:
            done.add(person)
            parts = by_person[person]
            total = sum(c for _, c in parts)
            breakdown = " + ".join(
                f"{a} {c:,}{' (old account)' if accounts[a][1] else ''}" for a, c in parts
            )
            merged.append((total, f"  {person}: {total:,} — {breakdown}"))
    merged.sort(key=lambda r: -r[0])
    first, last = rows[0][0], rows[-1][0]
    return "\n".join(lines[:first] + [line for _, line in merged] + lines[last + 1:])


async def _resolve_author(author: str, ctx: ToolContext) -> tuple[Optional[str], str]:
    """
    (author filter to search, note for the result) for what the model passed.

    A name, or any one of a person's accounts, becomes all of that person's
    accounts in the archive, with a note when that is not exactly what was
    passed — someone's old account holds messages a search by name should
    find. A handle no background entry claims comes back unchanged. Anything
    else comes back as (None, hint): searching it would only return zero.
    """
    from lore.prompts import load_lore_context

    handles = await _archive_handles(ctx)
    if not handles:
        return author, ""
    key = author.strip().lower()
    targets = _alias_index(load_lore_context(), handles).get(key, set())
    if len(targets) == 1:
        target = next(iter(targets))
        if target.lower() == key:
            return target, ""
        accounts = target.split(",")
        searched = (
            f"was searched as {target}, the handle the background gives for that name"
            if len(accounts) == 1 else
            f"was searched across all of that person's accounts: {', '.join(accounts)}"
        )
        return target, f"(`author` takes account handles: '{author}' {searched}.)\n"
    if key in (h.lower() for h in handles):
        return author, ""
    could_be = (
        f" It could be any of: {', '.join(sorted(t.split(',')[0] for t in targets))}." if targets else ""
    )
    return None, (
        f"'{author}' is not an account handle, so it matches nothing.{could_be} `author` "
        "takes the handle exactly as it appears before the colon in messages — the "
        "background maps people's names to their handles. Handles in the archive, most "
        f"active first: {', '.join(handles)}. Repeat the call with the right handle."
    )


def _date_scope(start_date: Optional[str], end_date: Optional[str]) -> str:
    """The " between X and Y" clause of a no-results message, or ""."""
    if not (start_date or end_date):
        return ""
    return f" between {start_date} and {end_date}"


def _page(args: dict, top_k: int) -> tuple[int, int]:
    """(page, offset) for a semantic search: page 1 is the best top_k."""
    try:
        page = max(1, int(args.get("page") or 1))
    except (TypeError, ValueError):
        page = 1
    return page, (page - 1) * top_k


def _with_page_note(result: str, page: int, offset: int, count: int, top_k: int) -> str:
    """
    The results with a closing note on where they sit in the ranking, so the
    model knows it can ask for more of a good search instead of rewording it.

    A search note, not a chunk: it is never deduplicated as an excerpt and
    never handed to the writer as research.
    """
    note = (
        f"{SEARCH_NOTE_PREFIX} These are matches {offset + 1}-{offset + count} of this "
        f"search, best first. Lower-ranked matches continue past them: repeat the same "
        f"search with page={page + 1} to see the next {top_k}."
    )
    return result + CHUNK_SEPARATOR + note


def _past_the_end(query: str, page: int) -> str:
    """What a page beyond the last match says."""
    return (
        f"{SEARCH_NOTE_PREFIX} No more matches for '{query}' past page {page - 1} with "
        "these filters — every match has been shown. Try different wording, another "
        "tool or a wider scope."
    )


async def _semantic_search(args: dict, ctx: ToolContext, *, scoped: bool) -> str:
    """
    search_discord_history (scoped=False) and search_channel_history.

    The model sometimes passes a channel to the un-scoped search. That is
    honoured rather than dropped — read from args, so both tools filter the same
    way.

    Semantic search only ever sees the top-k nearest chunks, so an empty result
    is weak evidence — it says the query did not match, not that the topic is
    absent. The advice therefore pushes towards broadening, never towards
    concluding.
    """
    query = args.get("query", "")
    channel_name = args.get("channel_name") or None
    start_date, end_date = args.get("start_date"), args.get("end_date")
    top_k = args.get("top_k", AGENT_TOP_K)
    page, offset = _page(args, top_k)

    context, chunk_count = await ctx.rag.retrieve(
        query, top_k=top_k, channel_name=channel_name,
        start_date=start_date, end_date=end_date, offset=offset,
    )
    logger.info(
        "RAG %s: query=%r, channel=%r, top_k=%d, page=%d → %d chunks",
        "search_channel_history" if scoped else "search_discord_history",
        query, channel_name, top_k, page, chunk_count,
    )
    if chunk_count:
        return _with_page_note(context, page, offset, chunk_count, top_k)
    if page > 1:
        return _past_the_end(query, page)
    scope = f" in channel '{channel_name}'" if channel_name else ""
    miss = f"No results found for query '{query}'{scope}{_date_scope(start_date, end_date)}."
    if scoped:
        return miss + (
            " This may mean the topic wasn't discussed in this channel, the date range is too narrow, "
            "or the query terms don't match. Try a different channel, broader date range, or different keywords."
        )
    return miss + (
        " This may mean: (a) the topic was never discussed, (b) the date range is too narrow, "
        "or (c) the query terms don't match how users phrased things. Try broadening the search "
        "by removing date filters, using different keywords, or searching without a channel filter."
    )


async def _search_discord_history(args: dict, ctx: ToolContext) -> str:
    return await _semantic_search(args, ctx, scoped=False)


async def _search_channel_history(args: dict, ctx: ToolContext) -> str:
    return await _semantic_search(args, ctx, scoped=True)


async def _search_images(args: dict, ctx: ToolContext) -> str:
    query = args.get("query", "")
    channel_name = args.get("channel_name") or None
    start_date, end_date = args.get("start_date"), args.get("end_date")
    top_k = args.get("top_k", 5)
    page, offset = _page(args, top_k)
    context, count = await ctx.rag.search_images(
        query, top_k=top_k, channel_name=channel_name,
        start_date=start_date, end_date=end_date, offset=offset,
    )
    logger.info(
        "RAG search_images: query=%r channel=%r page=%d -> %d conversation(s)",
        query, channel_name, page, count,
    )
    if count:
        return _with_page_note(context, page, offset, count, top_k)
    if page > 1:
        return _past_the_end(query, page)
    scope = f" in channel '{channel_name}'" if channel_name else ""
    return (
        f"No images matched '{query}'{scope}{_date_scope(start_date, end_date)}. Image search "
        "only sees the vision model's description of each picture, so describe what it would "
        "visibly show, or drop the filters."
    )


async def _read_surrounding_messages(args: dict, ctx: ToolContext) -> str:
    # The model may write the ref the way it appears in research: "[m42]".
    ref = str(args.get("ref") or "").strip().strip("[]").strip().lower()
    target = ctx.citations.refs.get(ref) if ctx.citations is not None else None
    if target is None:
        return (
            f"Error: '{ref}' is not a ref from this conversation's research. Use the ref "
            "printed at the start of a message you were shown, e.g. 'm12'."
        )
    direction = args.get("direction") or "after"
    if direction not in ("after", "before", "both"):
        direction = "after"
    try:
        count = int(args.get("count") or 20)
    except (TypeError, ValueError):
        count = 20

    context, returned = await ctx.rag.surrounding(target[0], target[1], direction=direction, count=count)
    logger.info("RAG read_surrounding_messages: %s %s x%d -> %d", ref, direction, count, returned)
    if context:
        return context
    return (
        f"Could not read around {ref}: the message is not in the searchable archive, or "
        "the archive search is unavailable. Answer from what you have."
    )


async def _summarize_channel(args: dict, ctx: ToolContext) -> str:
    channel_name = args.get("channel_name", "")
    start_date, end_date = args.get("start_date"), args.get("end_date")
    top_k = args.get("top_k", 20)
    # Reuse search but with higher top_k and a generic summarization query
    query = f"Recent conversations and discussions in {channel_name}"

    context, chunk_count = await ctx.rag.retrieve(
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


async def _search_exact_chronological(args: dict, ctx: ToolContext) -> str:
    term = args.get("term") or None
    author = args.get("author") or None
    order = args.get("order") or "earliest"
    if not term and not author:
        return (
            "Error: search_exact_chronological needs at least one of 'term' or 'author'. "
            "If you do not know the exact wording, use search_discord_history instead."
        )

    note = ""
    if author:
        author, note = await _resolve_author(author, ctx)
        if author is None:
            return note
    context, total = await ctx.rag.search_literal(
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
        return note + (
            f"No messages anywhere in the archive literally contain {scope}. "
            "This is an exact-text search over everything, so this result is conclusive "
            "for that exact wording — do not repeat it. Either the phrasing differs "
            "(try search_discord_history for the concept, or a shorter/simpler term), "
            "or it genuinely was never said."
        )

    shown = context.count(CHUNK_SEPARATOR) + 1
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
    return note + header + "\n" + context


async def _count_messages(args: dict, ctx: ToolContext) -> str:
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

    note = ""
    if author:
        author, note = await _resolve_author(author, ctx)
        if author is None:
            return note
    report, total = await ctx.rag.aggregate(
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
        return note + (
            "No messages matched, so there is nothing to count. This counted over the "
            "whole archive, so the answer is zero for that exact wording — do not repeat "
            "this search."
        )
    if group_by == "author":
        from lore.prompts import load_lore_context
        report = _merge_people(report, load_lore_context())
    return note + report


SIMULATION_HEADER = (
    "SIMULATED CONVERSATION — generated by persona models imitating these members, "
    "not real messages. Never cite it as something anyone said; present it as a simulation."
)


async def _simulate_conversation(args: dict, ctx: ToolContext) -> str:
    catalog = ctx.catalog
    available = sorted(p.display_name for p in catalog.all())
    raw_names = args.get("personas") or []
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    by_name = catalog.by_name()
    unknown = [n for n in raw_names if str(n).strip().lower() not in by_name]
    if unknown or not raw_names:
        return (
            f"Error: unknown persona(s) {unknown or raw_names}. "
            f"Choose from: {', '.join(available)}."
        )
    personas = list(dict.fromkeys(by_name[str(n).strip().lower()] for n in raw_names))[:MIMIC_CHAT_MAX_PERSONAS]
    premise = (args.get("premise") or "").strip()
    if not premise:
        return "Error: simulate_conversation needs a premise — the chat's opening message."
    try:
        length = int(args.get("length") or MIMIC_CHAT_DEFAULT_SIM_LENGTH)
    except (TypeError, ValueError):
        length = MIMIC_CHAT_DEFAULT_SIM_LENGTH
    length = max(2, min(length, MIMIC_CHAT_MAX_LENGTH))

    lines = await simulate(ctx.proxy, catalog, personas, premise, length)
    logger.info("Simulated %d line(s) between %s", len(lines) - 1, personas)
    cast = ", ".join(catalog.get(p).display_name for p in personas)
    return f"{SIMULATION_HEADER}\nPersonas: {cast}\n\n{render_transcript(lines)}"


Handler = Callable[[dict, ToolContext], Awaitable[str]]


@dataclass(frozen=True)
class _Tool:
    run: Handler
    needs: str  # the ToolContext field it cannot run without
    missing: str  # what the model is told when that field is absent


_NO_RAG = "Error: archive search is disabled on this bot (RAG_ENABLED=false). Answer without searching."
_NO_PERSONAS = "Error: simulate_conversation is not available right now — no personas are loaded."

HANDLERS: dict[str, _Tool] = {
    "search_discord_history": _Tool(_search_discord_history, "rag", _NO_RAG),
    "search_channel_history": _Tool(_search_channel_history, "rag", _NO_RAG),
    "summarize_channel": _Tool(_summarize_channel, "rag", _NO_RAG),
    "search_images": _Tool(_search_images, "rag", _NO_RAG),
    "read_surrounding_messages": _Tool(_read_surrounding_messages, "rag", _NO_RAG),
    "search_exact_chronological": _Tool(_search_exact_chronological, "rag", _NO_RAG),
    "count_messages": _Tool(_count_messages, "rag", _NO_RAG),
    "simulate_conversation": _Tool(_simulate_conversation, "catalog", _NO_PERSONAS),
}

# Every advertised tool must be runnable, and nothing unadvertised should be.
assert {t["function"]["name"] for t in _SEARCH_TOOLS} | {"simulate_conversation"} == set(HANDLERS), (
    "tool schemas and HANDLERS disagree"
)


async def execute_tool(tool_name: str, tool_args: dict, ctx: ToolContext) -> str:
    """
    Execute a single tool call and return the result as a string.

    Errors are returned, not raised — the model has to be able to read and
    react to them.
    """
    logger.info(
        "Executing tool=%s with args={%s}",
        tool_name,
        ", ".join(f"{k}={v!r}" for k, v in tool_args.items()),
    )
    if ctx.metrics is not None:
        ctx.metrics.tools_used.append(tool_name)

    tool = HANDLERS.get(tool_name)
    if tool is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return f"Error: Unknown tool '{tool_name}'."
    if getattr(ctx, tool.needs) is None or (tool.needs == "catalog" and ctx.proxy is None):
        logger.warning("Tool %s refused: no %s", tool_name, tool.needs)
        return tool.missing

    tool_start = time.monotonic()
    try:
        from lore.prompts import load_lore_context

        result = _name_speakers(await tool.run(tool_args, ctx), load_lore_context())
        return ctx.citations.tag(result) if ctx.citations is not None else result
    except Exception as e:
        logger.exception("Tool execution failed for %s", tool_name)
        return f"Error executing {tool_name}: {type(e).__name__}: {e}"
    finally:
        if ctx.metrics is not None:
            elapsed = time.monotonic() - tool_start
            ctx.metrics.tool_call_times.append(elapsed)
            logger.info("Tool %s completed in %.2fs", tool_name, elapsed)
