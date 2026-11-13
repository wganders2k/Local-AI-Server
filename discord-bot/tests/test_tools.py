"""
Tool dispatch, and the prose each tool returns when it finds nothing.

The empty-result text matters as much as the happy path: the model reads it and
decides what to do next, so a semantic miss has to read as "try something else"
while an exact miss has to read as "this question is settled".
"""


from lore.tools import HANDLERS, SIMULATION_HEADER, ToolContext, build_tools, execute_tool
from lore.metrics import AgentMetrics
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
