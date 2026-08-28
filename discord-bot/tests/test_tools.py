"""
Tool dispatch, and the prose each tool returns when it finds nothing.

The empty-result text matters as much as the happy path: the model reads it and
decides what to do next, so a semantic miss has to read as "try something else"
while an exact miss has to read as "this question is settled".
"""

import pytest

from lore.tools import HANDLERS, TOOLS, execute_tool
from lore.metrics import AgentMetrics


def test_every_advertised_tool_has_a_handler():
    assert {t["function"]["name"] for t in TOOLS} == set(HANDLERS)


async def test_unknown_tool_returns_an_error_string_not_an_exception(fake_rag):
    out = await execute_tool("no_such_tool", {}, fake_rag)
    assert out.startswith("Error: Unknown tool")


async def test_handler_exception_is_returned_as_readable_text(fake_rag):
    class Boom:
        async def retrieve(self, *a, **kw):
            raise RuntimeError("rag exploded")

    out = await execute_tool("search_discord_history", {"query": "x"}, Boom())
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
        "search_discord_history", {"query": "oink", "channel_name": "general"}, fake_rag
    )
    assert "NameError" not in out
    assert "in channel 'general'" in out
    # And the channel is actually honoured rather than silently dropped.
    assert fake_rag.calls[0][1]["channel_name"] == "general"


async def test_semantic_miss_advises_broadening(fake_rag):
    fake_rag.retrieve_result = ("", 0)
    out = await execute_tool("search_discord_history", {"query": "oink"}, fake_rag)
    assert "broadening" in out
    # Never tell the model a semantic miss is conclusive — it only saw top-k.
    assert "conclusive" not in out


async def test_exact_miss_is_reported_as_conclusive(fake_rag):
    fake_rag.literal_result = ("", 0)
    out = await execute_tool("search_exact_chronological", {"term": "oink"}, fake_rag)
    assert "conclusive" in out
    assert "do not repeat" in out


async def test_exact_search_needs_a_term_or_an_author(fake_rag):
    out = await execute_tool("search_exact_chronological", {}, fake_rag)
    assert "needs at least one of" in out
    assert not fake_rag.calls  # never reached the service


async def test_partial_results_are_flagged_as_partial(fake_rag):
    fake_rag.literal_result = ("chunk a\n\n---\n\nchunk b", 57)
    out = await execute_tool("search_exact_chronological", {"term": "oink"}, fake_rag)
    assert "Found 57 matching chunk(s); showing the 2 oldest" in out
    assert "55 further match(es) exist and are NOT shown" in out


async def test_complete_results_carry_no_warning(fake_rag):
    fake_rag.literal_result = ("chunk a\n\n---\n\nchunk b", 2)
    out = await execute_tool("search_exact_chronological", {"term": "oink"}, fake_rag)
    assert "WARNING" not in out


async def test_excluded_channels_are_stripped_of_their_hash(fake_rag):
    # The model writes channels the way Discord renders them; metadata stores
    # the bare name, so a leading '#' would quietly exclude nothing.
    fake_rag.aggregate_result = ("report", 5)
    await execute_tool(
        "count_messages",
        {"term": "oink", "exclude_channels": ["#grant-chronicle", "  ", ""]},
        fake_rag,
    )
    assert fake_rag.calls[0][1]["exclude_channels"] == ["grant-chronicle"]


async def test_a_single_excluded_channel_string_is_accepted(fake_rag):
    fake_rag.aggregate_result = ("report", 5)
    await execute_tool(
        "count_messages", {"exclude_channels": "#grant-chronicle"}, fake_rag
    )
    assert fake_rag.calls[0][1]["exclude_channels"] == ["grant-chronicle"]


async def test_metrics_record_the_call(fake_rag):
    metrics = AgentMetrics()
    fake_rag.retrieve_result = ("ctx", 1)
    await execute_tool("search_discord_history", {"query": "x"}, fake_rag, metrics)
    assert metrics.tools_used == ["search_discord_history"]
    assert len(metrics.tool_call_times) == 1
