"""
The tool-calling loop, driven end to end against a fake backend.

This is the test the restructure was for: lore/ has no discord dependency, so
the loop can be exercised without a gateway, a token, or a network.
"""

import json

from config import AGENT_CTX_LIMIT_FALLBACK, AGENT_MAX_ROUNDS_HARD_CAP
from lore.agent import LoreRunResult, run_agent_loop, stream_answer
from lore.progress import ProgressReporter
from config import LORE_SYSTEM_NUDGE_MODELS
from lore.research import split_chunks
from lore.models import research_model

# The role a mid-conversation instruction takes on the default research model.
NUDGE_ROLE = "system" if research_model("fast") in LORE_SYSTEM_NUDGE_MODELS else "user"


class FakeProxy:
    """Replays a scripted sequence of backend responses."""

    def __init__(self, turns, stream_text="synthesised answer"):
        self._turns = list(turns)
        self._stream_text = stream_text
        self.tool_payloads: list[object] = []

    async def chat_with_tools(self, model, messages, tools, temperature, max_tokens=None):
        self.tool_payloads.append(tools)
        return self._turns.pop(0)

    async def chat_stream(self, model, messages, usage_sink=None, tools=None,
                          enable_thinking=True, max_tokens=None):
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
    async def simulating(self): self.events.append("simulating")
    async def writing(self): self.events.append("writing")
    async def generating(self, chars=None): self.events.append("generating")
    async def complete(self, q, details=None):
        self.events.append("complete")
        self.details = details
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
    # Answered in place: the same tools as the loop, so the prompt keeps the
    # shape the backend has cached.
    assert proxy.tool_payloads[-1] == proxy.tool_payloads[0]
    assert len([p for p in proxy.tool_payloads if p is None]) == 0


class ScriptedStreamProxy(FakeProxy):
    """Streams a scripted reply per chat_stream call and keeps each request."""

    def __init__(self, turns, replies):
        super().__init__(turns)
        self._replies = list(replies)
        self.loop_messages: list[list[dict]] = []
        self.stream_requests: list[tuple[list[dict], object]] = []

    async def chat_with_tools(self, model, messages, tools, temperature, max_tokens=None):
        self.loop_messages.append(list(messages))
        return await super().chat_with_tools(model, messages, tools, temperature, max_tokens)

    async def chat_stream(self, model, messages, usage_sink=None, tools=None,
                          enable_thinking=True, max_tokens=None):
        self.stream_requests.append((list(messages), tools))
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 10, "completion_tokens": 4})
        reply = self._replies.pop(0)
        if reply:
            yield reply


async def test_the_in_place_answer_extends_the_cached_conversation(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = ScriptedStreamProxy([tool_call(), tool_call(args='{"query": "cows"}')], ["the answer"])

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=2,
    )

    assert result.answer == "the answer"
    [(sent, tools)] = proxy.stream_requests
    assert tools
    # Everything the last loop request held, then its round's call and
    # result, then the nudge — nothing earlier rewritten.
    last = proxy.loop_messages[-1]
    assert sent[: len(last)] == last
    # In the role the research model's template keeps cached: system for
    # Gemma 4, which drops earlier reasoning after a user turn; user for Qwen,
    # which rejects a late system message.
    assert [m["role"] for m in sent[len(last):]] == ["assistant", "tool", NUDGE_ROLE]
    assert "Do NOT call any tool" in sent[-1]["content"]


async def test_an_empty_in_place_answer_falls_back_to_flattened_synthesis(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = ScriptedStreamProxy([tool_call()], ["", "synthesised"])

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=1,
    )

    assert result.answer == "synthesised"
    (_, first_tools), (flat, flat_tools) = proxy.stream_requests
    assert first_tools and flat_tools is None
    assert [m["role"] for m in flat] == ["system", "user"]


async def test_tool_markup_in_the_in_place_answer_falls_back(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    leaked = '<tool_call>\n<function=search_discord_history>\n<parameter=query>x'
    proxy = ScriptedStreamProxy([tool_call()], [leaked, "synthesised"])

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=1,
    )

    assert result.answer == "synthesised"
    assert len(proxy.stream_requests) == 2


async def test_max_rounds_is_hard_capped(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    # Distinct queries, or the repeat guard would end the run first.
    proxy = FakeProxy([
        tool_call(args=json.dumps({"query": f"q{i}"})) for i in range(AGENT_MAX_ROUNDS_HARD_CAP + 5)
    ])
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=9999,
    )
    assert result.metrics.rounds_executed == AGENT_MAX_ROUNDS_HARD_CAP


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

    async def chat_with_tools(self, model, messages, tools, temperature, max_tokens=None):
        self.tool_requests.append([dict(m) for m in messages])
        return await super().chat_with_tools(model, messages, tools, temperature, max_tokens)

    async def chat_stream(self, model, messages, usage_sink=None, tools=None,
                          enable_thinking=True, max_tokens=None):
        self.stream_requests.append((messages, enable_thinking))
        # Compaction is the only call made with thinking off.
        text = "the gist" if not enable_thinking else self._stream_text
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 10, "completion_tokens": 4})
        yield text


PAST_COMPACT_MARK = int(0.80 * AGENT_CTX_LIMIT_FALLBACK)


def tool_call_at(prompt_tokens, call_id):
    turn = tool_call(args=json.dumps({"query": call_id}))
    turn["tool_calls"][0]["id"] = call_id
    turn["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 20}
    return turn


async def test_nearing_the_limit_condenses_the_oldest_results(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    # The window is the fallback: 80% is past the 75% compaction mark and
    # short of the 85% hard limit.
    proxy = RecordingProxy([
        tool_call_at(100, "a"), tool_call_at(200, "b"), tool_call_at(PAST_COMPACT_MARK, "c"),
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
        tool_call_at(100, "a"), tool_call_at(200, "b"), tool_call_at(PAST_COMPACT_MARK, "c"),
    ])

    await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=3,
    )

    answer_request, thinking = proxy.stream_requests[-1]
    assert thinking
    assert any("the gist" in (m.get("content") or "") for m in answer_request)


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


# ---------------------------------------------------------------------------
# Repeats
# ---------------------------------------------------------------------------


async def test_a_repeated_search_is_refused_not_rerun(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    proxy = RecordingProxy([tool_call_at(100, "a"), tool_call_at(100, "a"), final("ok")])

    await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[],
    )

    assert len(fake_rag.calls) == 1
    refusal = [m for m in proxy.tool_requests[-1] if m["role"] == "tool"][-1]
    assert "already ran this exact search in round 1" in refusal["content"]


async def test_repeating_itself_ends_the_search(fake_rag):
    fake_rag.retrieve_result = ("chunk", 1)
    # The loop that motivated this: one query, reissued until the budget ran out.
    proxy = RecordingProxy([tool_call_at(100, "a")] * 25)

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=25,
    )

    assert result.metrics.rounds_executed == 3  # AGENT_MAX_REPEATED_SEARCHES = 2
    assert result.answer.strip() == "synthesised answer"


async def test_chunks_already_in_context_are_not_appended_again(fake_rag):
    proxy = RecordingProxy([tool_call_at(100, "a"), tool_call_at(100, "b"), final("ok")])
    first = True

    async def retrieve(query, **kw):
        nonlocal first
        result = ("old", 1) if first else ("old\n\n---\n\nnew", 2)
        first = False
        return result

    fake_rag.retrieve = retrieve
    await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[],
    )

    results = [m["content"] for m in proxy.tool_requests[-1] if m["role"] == "tool"]
    assert split_chunks(results[0])[0] == "old"
    assert results[1].startswith("new")
    assert "1 of 2 excerpt(s) omitted" in results[1]
    # The page note is not an excerpt: identical on both results, never
    # deduplicated, and still on the second.
    assert "page=2" in results[1]
    # The earlier result is untouched, so the cached prefix still matches.
    assert proxy.tool_requests[-1][:4] == proxy.tool_requests[-2][:4]


async def test_condensed_chunks_can_be_shown_again():
    from lore.compaction import compact_tool_rounds
    from lore.metrics import AgentMetrics
    from lore.session import chunk_key

    def pair(i, text):
        return [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": i, "function": {"name": "search_discord_history",
                                       "arguments": json.dumps({"query": i})}}]},
            {"role": "tool", "tool_call_id": i, "content": text},
        ]

    messages = pair("a", "old") + pair("b", "kept")
    tool_messages = list(messages)
    shown = {chunk_key("old"), chunk_key("kept")}

    await compact_tool_rounds(
        messages, tool_messages, RecordingProxy([]), AgentMetrics(), 1, "lore-model",
        shown_keys=shown,
    )

    assert shown == {chunk_key("kept")}


async def test_only_one_simulation_runs_per_question(fake_rag, catalog):
    class SimProxy(FakeProxy):
        def __init__(self, turns):
            super().__init__(turns)
            self.persona_lines = 0

        async def chat(self, model, messages, lora=None, response_format=None, temperature=None, **kw):
            if response_format is not None:
                return '{"reason": "r", "action": "speak", "speaker": "mimic_user1"}'
            self.persona_lines += 1
            return "simulated line"

    first = '{"personas": ["User1"], "premise": "pizza?", "length": 2}'
    second = '{"personas": ["User1"], "premise": "tacos?", "length": 2}'
    proxy = SimProxy([
        tool_call("simulate_conversation", first),
        tool_call("simulate_conversation", second),
        final("answer"),
    ])
    status = RecordingStatus()

    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=status, channel_names=[], catalog=catalog,
    )

    assert result.answer == "answer"
    assert proxy.persona_lines == 2  # the first simulation only
    tool_results = [m["content"] for m in result.tool_messages if m["role"] == "tool"]
    assert tool_results[0].startswith("SIMULATED CONVERSATION")
    assert "already run a simulation" in tool_results[1]
    assert status.events.count("simulating") == 1


class MalformedOnce(FakeProxy):
    """The backend rejects the model's tool call as unparseable N times."""

    def __init__(self, turns, failures=1):
        super().__init__(turns)
        self.failures = failures
        self.requests: list[list[dict]] = []

    async def chat_with_tools(self, model, messages, tools, temperature, max_tokens=None):
        from proxy_client import ProxyError
        self.requests.append(list(messages))
        if self.failures:
            self.failures -= 1
            raise ProxyError(
                'Proxy returned error 500: {"error":{"code":500,"message":"Failed to parse '
                'tool call arguments as JSON: parse error"}}', status_code=500,
            )
        return await super().chat_with_tools(model, messages, tools, temperature, max_tokens)


async def test_a_malformed_tool_call_is_a_failed_round_not_a_failed_run(fake_rag):
    proxy = MalformedOnce([final("recovered")])
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=3,
    )
    assert result.ok and result.answer == "recovered"
    # The retry is told what happened, appended so the cached prefix holds.
    note = proxy.requests[1][-1]
    assert note["role"] == NUDGE_ROLE and "malformed" in note["content"]
    assert proxy.requests[1][:-1] == proxy.requests[0]


async def test_three_malformed_tool_calls_stop_the_search(fake_rag):
    proxy = MalformedOnce([], failures=3)
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=10,
    )
    assert result.ok  # answered from what it had, not a backend error
    assert len(proxy.requests) == 3


class Scripted(FakeProxy):
    """Plays a script where "bad" is a malformed tool call and anything else a turn."""

    def __init__(self, script):
        super().__init__([t for t in script if t != "bad"])
        self.script = list(script)
        self.requests: list[list[dict]] = []

    async def chat_with_tools(self, model, messages, tools, temperature, max_tokens=None):
        from proxy_client import ProxyError
        self.requests.append(list(messages))
        if self.script.pop(0) == "bad":
            raise ProxyError("Failed to parse tool call arguments as JSON", status_code=500)
        return await super().chat_with_tools(model, messages, tools, temperature, max_tokens)


def good_rounds(n, tag):
    return [tool_call(args=json.dumps({"query": f"{tag}{i}"})) for i in range(n)]


async def run_script(script, fake_rag):
    fake_rag.retrieve_result = ("[@5/10] a: pigs", 1)
    proxy = Scripted(script)
    result = await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=len(script),
    )
    return proxy, result


async def test_five_good_rounds_earn_a_strike_back(fake_rag):
    # 3 -> 1, five good rounds -> 2, then one more failure leaves 1: still going.
    script = ["bad", "bad", *good_rounds(5, "a"), "bad", final("made it")]
    proxy, result = await run_script(script, fake_rag)
    assert result.answer == "made it"
    assert len(proxy.requests) == len(script)


async def test_good_rounds_at_a_full_budget_are_not_banked(fake_rag):
    script = [*good_rounds(5, "a"), "bad", "bad", "bad", final("never reached")]
    proxy, result = await run_script(script, fake_rag)
    assert len(proxy.requests) == 8  # the third failure stopped the search
    assert result.answer != "never reached"


async def test_the_model_never_hears_about_strikes(fake_rag):
    script = ["bad", "bad", *good_rounds(5, "a"), final("done")]
    proxy, _ = await run_script(script, fake_rag)
    seen = " ".join(str(m.get("content")) for m in proxy.requests[-1]).lower()
    assert "malformed" in seen
    assert "strike" not in seen
