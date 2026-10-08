"""
The tool-calling loop, driven end to end against a fake backend.

This is the test the restructure was for: lore/ has no discord dependency, so
the loop can be exercised without a gateway, a token, or a network.
"""

import json

import pytest

from lore.agent import LoreRunResult, run_agent_loop, stream_answer
from lore.progress import ProgressReporter


class FakeProxy:
    """Replays a scripted sequence of backend responses."""

    def __init__(self, turns, stream_text="synthesised answer"):
        self._turns = list(turns)
        self._stream_text = stream_text
        self.tool_payloads: list[object] = []

    async def chat_with_tools(self, model, messages, tools, temperature):
        self.tool_payloads.append(tools)
        return self._turns.pop(0)

    async def chat_stream(self, model, messages, usage_sink=None, tools=None,
                          enable_thinking=True):
        self.tool_payloads.append(tools)
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 10, "completion_tokens": 4})
        for token in self._stream_text.split(" "):
            yield token + " "


class RecordingStatus:
    """A ProgressReporter that just remembers what it was told."""

    def __init__(self):
        self.events: list[str] = []

    async def waiting(self): self.events.append("waiting")
    async def thinking(self, r, m): self.events.append("thinking")
    async def searching(self, r, m): self.events.append("searching")
    async def analyzing(self, r, m): self.events.append("analyzing")
    async def condensing(self, r, m): self.events.append("condensing")
    async def writing(self): self.events.append("writing")
    async def generating(self, chars=None): self.events.append("generating")
    async def complete(self, q): self.events.append("complete")
    async def failed(self): self.events.append("failed")


def tool_call(name="search_discord_history", args='{"query": "pigs"}'):
    return {
        "content": None,
        "tool_calls": [{"id": "tc1", "function": {"name": name, "arguments": args}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "reasoning_content": "because pigs",
    }


def final(content="here is the answer"):
    return {"content": content, "tool_calls": None,
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


def test_recording_status_satisfies_the_protocol():
    assert isinstance(RecordingStatus(), ProgressReporter)


async def test_a_direct_answer_ends_the_run_without_synthesis(fake_rag):
    proxy = FakeProxy([final("direct")])
    status = RecordingStatus()

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=status, channel_names=["general"],
    )

    assert isinstance(result, LoreRunResult)
    assert result.answer == "direct"
    assert result.ok
    assert result.metrics.rounds_executed == 1
    assert "generating" not in status.events  # no synthesis pass
    assert status.events[-1] == "complete"


async def test_a_tool_round_then_an_answer(fake_rag):
    fake_rag.retrieve_result = ("retrieved chunk", 1)
    proxy = FakeProxy([tool_call(), final("informed answer")])
    status = RecordingStatus()

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=status, channel_names=["general"],
    )

    assert result.answer == "informed answer"
    assert result.metrics.total_tool_calls == 1
    assert result.metrics.tools_used == ["search_discord_history"]
    # The raw tool messages are kept so a thread can be seeded without re-searching.
    assert any(m["role"] == "tool" for m in result.tool_messages)
    assert "searching" in status.events


async def test_reasoning_rides_along_on_the_first_tool_call(fake_rag):
    # Without it the model cannot see *why* it searched and re-derives the same
    # plan, reissuing identical queries at temperature 0.1.
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = FakeProxy([tool_call(), final()])
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[],
    )
    assistant = next(m for m in result.tool_messages if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "because pigs"


async def test_exhausting_the_round_budget_falls_back_to_synthesis(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = FakeProxy([tool_call(), tool_call()], stream_text="synthesised answer")
    status = RecordingStatus()

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=status, channel_names=[], max_rounds=2,
    )

    assert result.answer.strip() == "synthesised answer"
    assert result.metrics.rounds_executed == 2
    assert "generating" in status.events
    # The synthesis request must carry no tools — there is no tool-calling
    # pattern left in context for the model to imitate.
    assert proxy.tool_payloads[-1] is None


async def test_max_rounds_is_hard_capped(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = FakeProxy([tool_call()] * 30)
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=9999,
    )
    assert result.metrics.rounds_executed == 25  # AGENT_MAX_ROUNDS_HARD_CAP


async def test_an_empty_model_response_is_nudged_rather_than_abandoned(fake_rag):
    empty = {"content": "", "tool_calls": None, "usage": {}}
    proxy = FakeProxy([empty, final("recovered")])
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[],
    )
    assert result.answer == "recovered"


async def test_a_backend_failure_reports_rather_than_raises(fake_rag):
    from proxy_client import ProxyError

    class Broken:
        async def chat_with_tools(self, **kw):
            raise ProxyError("backend down")

    status = RecordingStatus()
    result = await run_agent_loop(
        user_question="q", proxy_client=Broken(), rag_client=fake_rag,
        status=status, channel_names=[],
    )
    assert not result.ok
    assert "backend down" in result.answer
    assert status.events[-1] == "failed"


async def test_stream_answer_returns_text_and_usage():
    proxy = FakeProxy([], stream_text="one two three")
    text, usage = await stream_answer(proxy, "model", [])
    assert text.strip() == "one two three"
    assert usage["prompt_tokens"] == 10


async def test_stream_answer_forwards_tools_to_keep_the_prompt_shape():
    # Dropping the tools payload between calls cold-prefills the conversation.
    proxy = FakeProxy([])
    await stream_answer(proxy, "model", [], tools=[{"a": 1}])
    assert proxy.tool_payloads == [[{"a": 1}]]


# ---------------------------------------------------------------------------
# Context guard
# ---------------------------------------------------------------------------


class RecordingProxy(FakeProxy):
    """A FakeProxy that keeps every request and tells summaries apart."""

    def __init__(self, turns):
        super().__init__(turns, stream_text="synthesised answer")
        self.tool_requests: list[list[dict]] = []
        self.stream_requests: list[tuple[list[dict], bool]] = []

    async def chat_with_tools(self, model, messages, tools, temperature):
        self.tool_requests.append([dict(m) for m in messages])
        return await super().chat_with_tools(model, messages, tools, temperature)

    async def chat_stream(self, model, messages, usage_sink=None, tools=None,
                          enable_thinking=True):
        self.stream_requests.append((messages, enable_thinking))
        # Compaction is the only call made with thinking off.
        text = "the gist" if not enable_thinking else self._stream_text
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 10, "completion_tokens": 4})
        yield text


def tool_call_at(prompt_tokens, call_id):
    turn = tool_call(args=json.dumps({"query": call_id}))
    turn["tool_calls"][0]["id"] = call_id
    turn["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 20}
    return turn


async def test_nearing_the_limit_condenses_the_oldest_results(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    # The window is the 96k fallback: 75k lands past the 75% compaction mark.
    proxy = RecordingProxy([
        tool_call_at(100, "a"), tool_call_at(200, "b"), tool_call_at(75_000, "c"),
        final("done"),
    ])
    status = RecordingStatus()

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=status, channel_names=[],
    )

    assert result.answer == "done"
    assert status.events.count("condensing") == 1
    compactions = [m for m, thinking in proxy.stream_requests if not thinking]
    assert len(compactions) == 1

    sent = proxy.tool_requests[-1]
    assert any(
        m["role"] == "user" and m["content"].startswith("[Condensed research 1]")
        for m in sent
    )
    # The oldest result went; the call that produced it went with it.
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == ["b", "c"]
    assert [m["tool_calls"][0]["id"] for m in sent if m.get("tool_calls")] == ["b", "c"]
    # The tag that lets synthesis find the summary never reaches the backend.
    assert all("kind" not in m for m in sent)
    assert any(m.get("kind") == "condensed" for m in result.tool_messages)


async def test_condensed_research_reaches_the_synthesis_prompt(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = RecordingProxy([
        tool_call_at(100, "a"), tool_call_at(200, "b"), tool_call_at(75_000, "c"),
    ])

    await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=3,
    )

    synthesis, thinking = proxy.stream_requests[-1]
    assert thinking
    assert "the gist" in synthesis[-1]["content"]


async def test_still_too_full_after_compacting_stops_searching(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    # One result cannot be condensed, so 90k (94%) is still over the 85% line.
    proxy = RecordingProxy([tool_call_at(90_000, "a")] + [tool_call()] * 5)

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[],
    )

    assert result.ok
    assert result.metrics.rounds_executed == 1
    assert result.answer.strip() == "synthesised answer"
