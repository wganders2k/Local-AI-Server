"""
Tool dispatch, and the prose each tool returns when it finds nothing.

The empty-result text matters as much as the happy path: the model reads it and
decides what to do next, so a semantic miss has to read as "try something else"
while an exact miss has to read as "this question is settled".
"""


import pytest

from lore.tools import HANDLERS, SIMULATION_HEADER, ToolContext, build_tools, execute_tool
from lore.metrics import AgentMetrics
import asyncio
from lore.citations import Citations
from mimic import controller


def speaker_enum(response_format: dict) -> list[str]:
    """The names a speak decision may give."""
    schema = response_format["json_schema"]["schema"]
    speak = next(b for b in schema.get("anyOf", [schema]) if "speak" in b["properties"]["action"]["enum"])
    return speak["properties"]["speaker"]["enum"]


def test_every_advertised_tool_has_a_handler(catalog):
    assert {t["function"]["name"] for t in build_tools(catalog)} == set(HANDLERS)


def test_simulation_is_offered_only_with_personas_and_names_them(catalog):
    assert "simulate_conversation" not in {t["function"]["name"] for t in build_tools(None)}
    sim = next(t for t in build_tools(catalog) if t["function"]["name"] == "simulate_conversation")
    enum = sim["function"]["parameters"]["properties"]["personas"]["items"]["enum"]
    assert enum == [f"User{i}" for i in range(1, 8)]
    # Rebuilt per run, so it must come out byte-identical for the prefix cache.
    assert build_tools(catalog) == build_tools(catalog)


async def test_search_tools_say_so_when_retrieval_is_disabled():
    out = await execute_tool("search_discord_history", {"query": "x"}, ToolContext(rag=None))
    assert "disabled" in out and "Traceback" not in out


class ScriptedPersonas:
    """
    A proxy whose persona lines are numbered, and whose controller takes the
    personas in turn and declines a follow-up once the turns are used up.
    """

    def __init__(self):
        self.calls: list[dict] = []

    async def chat(self, model, messages, lora=None, response_format=None, temperature=None, **kw):
        self.calls.append({"lora": lora, "response_format": response_format})
        if response_format is not None:
            speakers = speaker_enum(response_format)
            if len(speakers) == 1 and "end" in controller.schema_actions(response_format):
                return '{"reason": "r", "action": "end", "speaker": "none"}'
            n = sum(1 for c in self.calls if c["response_format"] is not None)
            return f'{{"reason": "r", "action": "speak", "speaker": "{speakers[n % len(speakers)]}"}}'
        return f"line {sum(1 for c in self.calls if c['response_format'] is None)}"


async def test_simulation_returns_a_labelled_transcript_of_the_requested_length(fake_rag, catalog):
    proxy = ScriptedPersonas()

    out = await execute_tool(
        "simulate_conversation",
        {"personas": ["User1", "user2"], "premise": "pineapple on pizza?", "length": 3},
        ToolContext(fake_rag, proxy, catalog),
    )

    assert out.startswith(SIMULATION_HEADER)
    assert "[someone]: pineapple on pizza?" in out
    assert "line 3" in out and "line 4" not in out


async def test_simulation_rejects_unknown_personas_and_needs_a_proxy(fake_rag, catalog):
    out = await execute_tool(
        "simulate_conversation", {"personas": ["Nobody"], "premise": "x"},
        ToolContext(fake_rag, ScriptedPersonas(), catalog),
    )
    assert out.startswith("Error: unknown persona")
    assert "User1" in out

    out = await execute_tool(
        "simulate_conversation", {"personas": ["User1"], "premise": "x"}, ToolContext(fake_rag, catalog=catalog)
    )
    assert out.startswith("Error:")


async def test_unknown_tool_returns_an_error_string_not_an_exception(fake_rag):
    out = await execute_tool("no_such_tool", {}, ToolContext(fake_rag))
    assert out.startswith("Error: Unknown tool")


async def test_handler_exception_is_returned_as_readable_text(fake_rag):
    class Boom:
        async def retrieve(self, *a, **kw):
            raise RuntimeError("rag exploded")

    out = await execute_tool("search_discord_history", {"query": "x"}, ToolContext(Boom()))
    assert "Error executing search_discord_history" in out
    assert "rag exploded" in out


async def test_channel_name_on_the_unscoped_search_does_not_crash(fake_rag):
    """
    Regression: the zero-result message used to interpolate a `channel_name`
    that was never bound on this branch, so a search that returned nothing while
    the model had passed a channel raised NameError — swallowed into an error
    string the agent then read as a tool failure rather than "no results".
    """
    fake_rag.retrieve_result = ("", 0)
    out = await execute_tool(
        "search_discord_history", {"query": "oink", "channel_name": "general"}, ToolContext(fake_rag)
    )
    assert "NameError" not in out
    assert "in channel 'general'" in out
    # And the channel is actually honoured rather than silently dropped.
    assert fake_rag.calls[0][1]["channel_name"] == "general"


async def test_semantic_miss_advises_broadening(fake_rag):
    fake_rag.retrieve_result = ("", 0)
    out = await execute_tool("search_discord_history", {"query": "oink"}, ToolContext(fake_rag))
    assert "broadening" in out
    # Never tell the model a semantic miss is conclusive — it only saw top-k.
    assert "conclusive" not in out


async def test_exact_miss_is_reported_as_conclusive(fake_rag):
    fake_rag.literal_result = ("", 0)
    out = await execute_tool("search_exact_chronological", {"term": "oink"}, ToolContext(fake_rag))
    assert "conclusive" in out
    assert "do not repeat" in out


async def test_exact_search_needs_a_term_or_an_author(fake_rag):
    out = await execute_tool("search_exact_chronological", {}, ToolContext(fake_rag))
    assert "needs at least one of" in out
    assert not fake_rag.calls  # never reached the service


async def test_partial_results_are_flagged_as_partial(fake_rag):
    fake_rag.literal_result = ("chunk a\n\n---\n\nchunk b", 57)
    out = await execute_tool("search_exact_chronological", {"term": "oink"}, ToolContext(fake_rag))
    assert "Found 57 matching chunk(s); showing the 2 oldest" in out
    assert "55 further match(es) exist and are NOT shown" in out


async def test_complete_results_carry_no_warning(fake_rag):
    fake_rag.literal_result = ("chunk a\n\n---\n\nchunk b", 2)
    out = await execute_tool("search_exact_chronological", {"term": "oink"}, ToolContext(fake_rag))
    assert "WARNING" not in out


async def test_excluded_channels_are_stripped_of_their_hash(fake_rag):
    # The model writes channels the way Discord renders them; metadata stores
    # the bare name, so a leading '#' would quietly exclude nothing.
    fake_rag.aggregate_result = ("report", 5)
    await execute_tool(
        "count_messages",
        {"term": "oink", "exclude_channels": ["#grant-chronicle", "  ", ""]},
        ToolContext(fake_rag),
    )
    assert fake_rag.calls[0][1]["exclude_channels"] == ["grant-chronicle"]


async def test_a_single_excluded_channel_string_is_accepted(fake_rag):
    fake_rag.aggregate_result = ("report", 5)
    await execute_tool(
        "count_messages", {"exclude_channels": "#grant-chronicle"}, ToolContext(fake_rag)
    )
    assert fake_rag.calls[0][1]["exclude_channels"] == ["grant-chronicle"]


async def test_metrics_record_the_call(fake_rag):
    metrics = AgentMetrics()
    fake_rag.retrieve_result = ("ctx", 1)
    await execute_tool("search_discord_history", {"query": "x"}, ToolContext(fake_rag, metrics=metrics))
    assert metrics.tools_used == ["search_discord_history"]
    assert len(metrics.tool_call_times) == 1


class RosterRAG:
    """An archive of four handles; any other author matches nothing."""

    HANDLES = {"alice_e": 120, "bob1234": 80, "Deleted User": 40, "al_b": 10}

    def __init__(self):
        self.roster_reads = 0
        self.searched: list[str] = []

    async def aggregate(self, author=None, term=None, group_by="author", **kw):
        if author is None and term is None:
            self.roster_reads += 1
            lines = "\n".join(f"  {h}: {n}" for h, n in self.HANDLES.items())
            return f"Counts of messages — grouped by author, 210 total:\n{lines}", 210
        self.searched.append(author)
        n = self.HANDLES.get(author, 0)
        return (f"Counts of messages by {author}, {n} total:\n  general: {n}", n) if n else ("", 0)

    async def search_literal(self, author=None, **kw):
        self.searched.append(author)
        return ("", 0)


# The shape of prompts/lore_context.example.md. "Al" names two people; Bob
# also posted from an account the archive keeps as "Deleted User".
BACKGROUND = """CORE ACTIVE SERVER MEMBER ALIAS INDEX:
Alice:
Real Name: Alice Example
Aliases: alice_e, aliceee

Bob(Al):
Real Name: Robert Example
Aliases: bob1234, bobby (formerly Deleted User)

Carol:
Aliases: al, al_b
"""


@pytest.fixture
def archive(monkeypatch):
    from lore import prompts, tools
    monkeypatch.setattr(tools, "_handles", None)
    monkeypatch.setattr(prompts, "_lore_context_cache", BACKGROUND)
    return RosterRAG()


async def test_a_name_from_the_background_is_searched_as_its_handle(archive):
    out = await execute_tool("count_messages", {"author": "Alice", "group_by": "channel"},
                             ToolContext(rag=archive))
    assert archive.searched == ["alice_e"]
    assert "'Alice' was searched as alice_e" in out and "120 total" in out


async def test_real_names_and_people_with_old_accounts_resolve(archive):
    # Bob's "(formerly Deleted User)" is his old account: a search for him by
    # any name, or by either account, covers both.
    for name in ("Robert", "Bob", "bobby", "bob1234", "Deleted User"):
        await execute_tool("search_exact_chronological", {"author": name}, ToolContext(rag=archive))
    assert archive.searched == ["bob1234,Deleted User"] * 5


async def test_a_search_across_accounts_says_so(archive):
    out = await execute_tool("count_messages", {"author": "Bob", "group_by": "channel"},
                             ToolContext(rag=archive))
    assert "searched across all of that person's accounts: bob1234, Deleted User" in out


async def test_an_ambiguous_or_unknown_name_lists_the_real_handles(archive):
    ctx = ToolContext(rag=archive)
    for name, args in (("count_messages", {"author": "Al", "group_by": "channel"}),
                       ("search_exact_chronological", {"author": "Zed"})):
        out = await execute_tool(name, args, ctx)
        assert f"'{args['author']}' is not an account handle" in out
        assert "alice_e, bob1234, Deleted User, al_b" in out
    assert archive.searched == []  # nothing was searched that could only return zero
    assert archive.roster_reads == 1  # read once, then cached


async def test_an_ambiguous_name_says_who_it_could_be(archive):
    out = await execute_tool("search_exact_chronological", {"author": "al"}, ToolContext(rag=archive))
    assert "It could be any of: al_b, bob1234." in out


async def test_a_real_handle_with_no_matches_is_a_genuine_zero(archive):
    out = await execute_tool("search_exact_chronological", {"author": "alice_e", "term": "zzz"},
                             ToolContext(rag=archive))
    assert archive.searched == ["alice_e"]
    assert "not an account handle" not in out and "No messages" in out


def test_speakers_are_shown_with_their_names():
    from lore.tools import _name_speakers

    raw = (
        ">>> MATCH — [@1/2] alice_e: tell bob1234: hi\n"
        "[#general | 2021-01-01 00:00]\n"
        "[@1/3] bob1234: hello alice_e\n"
        "[m4] stranger: who are you\n"
        "Counts of messages — grouped by author, 3 total:\n"
        "  alice_e: 2\n"
        "  Deleted User: 1"
    )
    assert _name_speakers(raw, BACKGROUND) == (
        ">>> MATCH — [@1/2] Alice (alice_e): tell bob1234: hi\n"
        "[#general | 2021-01-01 00:00]\n"
        "[@1/3] Bob(Al) (bob1234): hello alice_e\n"
        "[m4] stranger: who are you\n"
        "Counts of messages — grouped by author, 3 total:\n"
        "  Alice (alice_e): 2\n"
        "  Bob(Al) (Deleted User, old account): 1"  # claimed by one person only
    )


SHARED = BACKGROUND + """
Dana:
Aliases: dana_d (formerly Deleted User)
"""


def test_an_old_account_two_people_claim_belongs_to_neither():
    from lore.tools import _merge_people, _name_speakers

    assert _name_speakers("[m1] Deleted User: hi", SHARED) == "[m1] Deleted User: hi"
    report = "Counts — grouped by author, 5 total:\n  bob1234: 3\n  Deleted User: 2"
    assert _merge_people(report, SHARED) == report


def test_counts_merge_a_persons_accounts():
    from lore.tools import _merge_people, _name_speakers

    report = (
        "Counts of messages containing 'oink' — grouped by author, 170 total:\n"
        "  alice_e: 60\n"
        "  bob1234: 50\n"
        "  Deleted User: 40\n"
        "  stranger: 20"
    )
    merged = _name_speakers(_merge_people(report, BACKGROUND), BACKGROUND)
    assert merged.splitlines() == [
        "Counts of messages containing 'oink' — grouped by author, 170 total:",
        "  Bob(Al): 90 — bob1234 50 + Deleted User 40 (old account)",
        "  Alice (alice_e): 60",
        "  stranger: 20",
    ]


def test_without_a_background_results_are_untouched():
    from lore.tools import _name_speakers

    assert _name_speakers("[@1/2] alice_e: hi", "") == "[@1/2] alice_e: hi"


# ---------------------------------------------------------------------------
# Paging through a semantic search
# ---------------------------------------------------------------------------


class PagingRAG:
    """A ranking of 25 chunks, served top_k at a time from ``offset``."""

    def __init__(self):
        self.calls: list[dict] = []

    async def retrieve(self, query, top_k=10, offset=0, **kw):
        self.calls.append({"top_k": top_k, "offset": offset})
        hits = [f"[@1/{i}] a: match {i}" for i in range(offset + 1, min(offset + top_k, 25) + 1)]
        return "\n\n---\n\n".join(hits), len(hits)


def _search(rag, **args):
    from lore.tools import _search_discord_history
    ctx = ToolContext(rag, None, None, AgentMetrics(), Citations())
    return asyncio.run(_search_discord_history({"query": "pigs", **args}, ctx))


def test_page_two_asks_for_the_next_top_k():
    rag = PagingRAG()
    out = _search(rag, page=2)
    assert rag.calls == [{"top_k": 10, "offset": 10}]
    assert "match 11" in out and "match 10\n" not in out
    assert "matches 11-20" in out and "page=3" in out


def test_the_first_page_says_more_are_available():
    out = _search(PagingRAG())
    assert "matches 1-10" in out and "page=2" in out


def test_a_page_past_the_end_says_so():
    out = _search(PagingRAG(), page=4)
    assert out.startswith("[Search note]") and "past page 3" in out


def test_a_bad_page_is_page_one():
    rag = PagingRAG()
    _search(rag, page="soon")
    assert rag.calls[0]["offset"] == 0
