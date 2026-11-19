"""
/lore models: ``speed`` picks the research model, ``writer`` the model that
writes the answer (by default the research model). Which model writes the
answer, what it is sent, and what a thread remembers are the things that must
not drift.
"""

import pytest

from config import AGENT_CTX_LIMIT_FALLBACK, LORE_WRITERS
from lore import context_window
from lore.agent import run_agent_loop, run_lore_turn, seed_session_from_run
from lore.models import research_model
from lore.session import LoreSession
from tests.test_agent import RecordingStatus, final, tool_call

FAST = research_model("fast")
SLOW = research_model("slow")
GEMMA = LORE_WRITERS["gemma-31b"]


@pytest.fixture(autouse=True)
def _clean_cache():
    context_window.reset()
    yield
    context_window.reset()


class ModelProxy:
    """Scripted tool rounds; records which model each streamed answer went to."""

    def __init__(self, turns, reply="written answer"):
        self._turns = list(turns)
        self._reply = reply
        self.streams: list[dict] = []
        self.rounds: list[str] = []  # the model each tool round went to

    async def chat_with_tools(self, model, messages, tools, temperature, max_tokens=None):
        self.rounds.append(model)
        return self._turns.pop(0)

    async def chat_stream(self, model, messages, usage_sink=None, tools=None,
                          enable_thinking=True, max_tokens=None):
        self.streams.append({"model": model, "messages": list(messages), "tools": tools})
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 10, "completion_tokens": 4})
        yield self._reply


async def run(proxy, fake_rag, speed, max_rounds=3, writer="same"):
    return await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=RecordingStatus(), channel_names=[], max_rounds=max_rounds, speed=speed,
        writer=writer,
    )


@pytest.mark.parametrize("speed, model", [("fast", FAST), ("slow", SLOW)])
async def test_speed_picks_the_research_model(fake_rag, speed, model):
    fake_rag.retrieve_result = ("[@5/10] a: pigs", 1)
    proxy = ModelProxy([tool_call(), final("own answer")])

    result = await run(proxy, fake_rag, speed)

    assert set(proxy.rounds) == {model}
    assert result.answer == "own answer"  # the default writer is the research model
    assert proxy.streams == []


async def test_a_writer_gets_the_research_handed_over(fake_rag):
    fake_rag.retrieve_result = ("[@5/10] a: pigs everywhere", 1)
    # The research model answers by itself; a separate writer sets that aside.
    proxy = ModelProxy([tool_call(), final("the researcher's own answer")], reply="gemma answer")

    result = await run(proxy, fake_rag, "slow", writer="gemma-31b")

    assert result.answer == "gemma answer"
    assert (result.speed, result.writer) == ("slow", "gemma-31b")
    assert set(proxy.rounds) == {SLOW}
    [stream] = proxy.streams
    assert stream["model"] == GEMMA
    assert stream["tools"] is None
    # The flattened synthesis prompt: one system and one user message, with the
    # research in it.
    assert [m["role"] for m in stream["messages"]] == ["system", "user"]
    assert "pigs everywhere" in stream["messages"][1]["content"]


async def test_a_writer_without_research_keeps_the_direct_answer(fake_rag):
    proxy = ModelProxy([final("nothing to look up")])

    result = await run(proxy, fake_rag, "slow", writer="gemma-31b")

    assert result.answer == "nothing to look up"
    assert proxy.streams == []  # no swap for an answer with nothing behind it


async def test_a_writer_naming_the_research_model_is_no_handover(fake_rag):
    fake_rag.retrieve_result = ("[@5/10] a: pigs", 1)
    proxy = ModelProxy([tool_call(), final("own answer")])

    result = await run(proxy, fake_rag, "fast", writer="qwen-35b-moe")

    assert LORE_WRITERS["qwen-35b-moe"] == FAST
    assert result.answer == "own answer"
    assert proxy.streams == []


async def test_the_research_model_answers_in_place(fake_rag):
    fake_rag.retrieve_result = ("[@5/10] a: pigs", 1)
    proxy = ModelProxy([tool_call()], reply="fast answer")

    result = await run(proxy, fake_rag, "fast", max_rounds=1)

    assert result.answer == "fast answer"
    [stream] = proxy.streams
    assert stream["model"] == FAST
    # Qwen's template rejects a system message after the first, so the nudge
    # is a user message.
    assert stream["messages"][-1]["role"] == "user"


def session(speed, writer="same"):
    s = LoreSession(
        thread_id=1, system_prompt="p", original_question="q", pinned_now="n",
        speed=speed, writer=writer,
    )
    s.append_question("q")
    s.append_answer("a")
    return s


async def test_a_thread_with_a_writer_runs_every_follow_up_on_the_writer(fake_rag):
    # Searching on one model and writing on another was two cold swaps a turn.
    fake_rag.retrieve_result = ("[@5/10] a: pigs", 1)
    proxy = ModelProxy([tool_call()], reply="gemma follow-up")
    s = session("slow", "gemma-31b")

    answer, metrics, searched = await run_lore_turn(s, "and then?", proxy, fake_rag, max_rounds=1)

    assert answer == "gemma follow-up" and searched
    assert proxy.rounds == [GEMMA]
    [stream] = proxy.streams
    assert stream["model"] == GEMMA
    # Gemma's template takes the cache-safe system nudge.
    assert stream["messages"][-1]["role"] == "system"
    assert stream["tools"]  # kept, so a tool call is parsed out, not left as text
    assert metrics.speed == "slow"


async def test_a_slow_thread_researches_and_answers_on_the_slow_model(fake_rag):
    proxy = ModelProxy([final("from the 27b")])

    answer, _, _ = await run_lore_turn(session("slow"), "and then?", proxy, fake_rag)

    assert answer == "from the 27b"
    assert proxy.rounds == [SLOW]
    assert proxy.streams == []


async def test_a_fast_thread_keeps_a_direct_answer(fake_rag):
    proxy = ModelProxy([final("straight from context")])

    answer, _, _ = await run_lore_turn(session("fast"), "and then?", proxy, fake_rag)

    assert answer == "straight from context"
    assert proxy.streams == []


async def test_a_seeded_thread_keeps_the_run_speed_and_writer(fake_rag):
    proxy = ModelProxy([final("x")])
    result = await run(proxy, fake_rag, "slow", writer="gemma-26b-moe")
    s = seed_session_from_run(0, [], "q", result)
    assert (s.speed, s.writer) == ("slow", "gemma-26b-moe")
    restored = LoreSession.from_dict(s.to_dict())
    assert (restored.speed, restored.writer) == ("slow", "gemma-26b-moe")


def test_sessions_from_before_speeds_load_at_the_default():
    d = session("slow").to_dict()
    del d["speed"]
    assert LoreSession.from_dict(d).speed == "fast"


@pytest.mark.parametrize("stored", [None, "a-removed-model"])
def test_sessions_without_a_known_writer_answer_on_the_research_model(stored):
    d = session("slow").to_dict()
    if stored is None:
        del d["writer"]
    else:
        d["writer"] = stored
    assert LoreSession.from_dict(d).writer == "same"


class SizesProxy:
    def __init__(self, sizes):
        self._sizes = sizes

    async def model_context_sizes(self):
        return self._sizes


async def test_a_thread_is_held_to_the_smaller_of_research_and_writer():
    await context_window.discover(SizesProxy({FAST: 131072, SLOW: 98304, GEMMA: 57344}))
    assert context_window.limit("fast") == 131072
    assert context_window.limit("slow") == 98304
    assert context_window.limit("fast", "gemma-31b") == 57344
    assert context_window.limit("fast", "qwen-27b") == 98304


async def test_a_missing_writer_is_flagged_but_research_still_works(caplog):
    with caplog.at_level("WARNING"):
        assert await context_window.discover(SizesProxy({FAST: 131072, SLOW: 98304}))
    assert context_window.limit("fast") == 131072
    assert GEMMA in caplog.text


def test_every_combination_uses_the_fallback_before_discovery():
    assert context_window.limit("fast") == context_window.limit("slow", "gemma-31b") == AGENT_CTX_LIMIT_FALLBACK


# ---------------------------------------------------------------------------
# Each request fits the model it goes to
# ---------------------------------------------------------------------------

CHUNK_CHARS = 3000  # ~1,000 tokens at the research ratio


def distinct_results(fake_rag):
    """Each search finds a different ~1k-token chunk, so none is deduplicated."""
    async def retrieve(query, **kw):
        return (f"[@5/10] a: {query} " + "x" * CHUNK_CHARS, 1)
    fake_rag.retrieve = retrieve


def window_for(prompt_tokens):
    """A window whose compaction budget is exactly ``prompt_tokens``."""
    from config import AGENT_CTX_COMPACT_PCT
    return int(prompt_tokens / AGENT_CTX_COMPACT_PCT) + 1


def searches(n):
    return [tool_call(args=f'{{"query": "q{i}"}}') for i in range(n)]


async def test_handed_over_research_is_condensed_to_fit_the_writer(fake_rag):
    from lore.compaction import estimate_tokens
    from lore.research import flatten_research_for_synthesis

    distinct_results(fake_rag)
    base = estimate_tokens(flatten_research_for_synthesis([], "q"))
    dense = window_for(base + 1500)  # room for one and a half chunks
    await context_window.discover(SizesProxy({FAST: 196608, SLOW: 196608, GEMMA: dense}))
    proxy = ModelProxy([*searches(3), final("researcher answer")], reply="short")

    result = await run(proxy, fake_rag, "fast", max_rounds=4, writer="gemma-31b")

    # Research never came near the research model's window, so the loop kept
    # it all; the writer's window forced one pass, sized to the overage, which
    # the research model (still loaded) summarised.
    summary, answer = proxy.streams
    assert summary["model"] == FAST
    assert "q2" not in summary["messages"][-1]["content"]  # the newest result stays whole
    assert answer["model"] == GEMMA
    assert estimate_tokens(answer["messages"]) < base + 1500
    assert result.answer == "short"
    # A thread seeded from the run carries the condensed research.
    assert sum(1 for m in result.tool_messages if m.get("kind") == "condensed") == 1


async def test_research_is_held_only_to_the_models_the_run_uses(fake_rag):
    distinct_results(fake_rag)
    await context_window.discover(SizesProxy({FAST: 196608, SLOW: 2000, GEMMA: 2000}))
    proxy = ModelProxy(searches(3), reply="fast")

    await run(proxy, fake_rag, "fast", max_rounds=3)

    # Small windows elsewhere are irrelevant to a run that never uses them.
    assert [s["model"] for s in proxy.streams] == [FAST]


async def test_a_thread_is_compacted_to_fit_its_writer(fake_rag):
    from lore.compaction import estimate_tokens

    s = session("slow", "gemma-31b")
    s.append_research("[@1/1] a: " + "y" * CHUNK_CHARS)
    s.append_question("more?")
    s.append_research("[@2/2] b: " + "z" * CHUNK_CHARS)
    s.append_answer("ok")
    base = estimate_tokens(s.build_messages()) - 2 * CHUNK_CHARS // 3
    dense = window_for(base + 1500)
    await context_window.discover(SizesProxy({FAST: 196608, SLOW: 196608, GEMMA: dense}))
    proxy = ModelProxy([final("writer answer")])

    answer, _, _ = await run_lore_turn(s, "and then?", proxy, fake_rag)

    # Condensed before the first request, by the model the turn runs on.
    [compaction] = proxy.streams
    assert compaction["model"] == GEMMA
    assert proxy.rounds == [GEMMA]
    assert answer == "writer answer"
    assert s.compactions == 1


def test_a_calls_context_is_measured_against_its_own_model():
    from lore.metrics import AgentMetrics

    context_window._sizes = {SLOW: 10_000, GEMMA: 1_000}
    m = AgentMetrics(speed="slow", model=SLOW)
    m.record_usage({"prompt_tokens": 2_000, "completion_tokens": 0}, model=SLOW)
    assert m.ctx_limit == 10_000 and m.context_pct == 0.2
    m.record_usage({"prompt_tokens": 800, "completion_tokens": 0}, model=GEMMA)
    assert m.ctx_limit == 1_000 and m.context_pct == 0.8


# ---------------------------------------------------------------------------
# The run summary on the finished status message
# ---------------------------------------------------------------------------


def test_the_run_line_names_both_models_on_a_handover():
    from lore.metrics import AgentMetrics

    m = AgentMetrics(research_model=FAST, writer_model=GEMMA, max_rounds=20)
    m.rounds_executed, m.total_tool_calls = 5, 12
    m.record_usage({"prompt_tokens": 1, "completion_tokens": 100,
                    "timings": {"predicted_n": 100, "predicted_ms": 1000.0}})
    m.record_usage({"prompt_tokens": 1, "completion_tokens": 300,
                    "timings": {"predicted_n": 300, "predicted_ms": 1000.0}})
    line = m.run_line()
    assert f"{FAST} → ✍️ {GEMMA}" in line
    assert "5/20 rounds" in line and "12 tool calls" in line
    assert "200 tok/s" in line  # 400 tokens over 2s of decoding, not a mean of means


def test_the_run_line_names_one_model_when_it_also_writes():
    from lore.metrics import AgentMetrics

    m = AgentMetrics(research_model=SLOW, writer_model=SLOW, max_rounds=5)
    line = m.run_line()
    assert SLOW in line and "✍️" not in line
    assert "tok/s" not in line  # nothing measured, nothing claimed


async def test_a_finished_run_reports_its_summary(fake_rag):
    from tests.test_agent import RecordingStatus

    status = RecordingStatus()
    proxy = ModelProxy([final("done")])
    await run_agent_loop(
        user_question="q", proxy_client=proxy, rag_client=fake_rag,
        status=status, channel_names=[], max_rounds=7, speed="slow", writer="gemma-31b",
    )
    assert status.details and "1/7 rounds" in status.details and SLOW in status.details


# ---------------------------------------------------------------------------
# Thread turns keep their tool rounds raw, so the answer extends the cache
# ---------------------------------------------------------------------------


async def test_the_answer_extends_the_turns_own_conversation(fake_rag):
    fake_rag.retrieve_result = ("[@5/10] a: pigs everywhere", 1)
    proxy = ModelProxy([tool_call()], reply="answer")
    s = session("fast")

    await run_lore_turn(s, "and then?", proxy, fake_rag, max_rounds=1)

    [stream] = proxy.streams
    sent = stream["messages"]
    # The search round's call and result, as sent, then only the nudge: no
    # re-rendered research block for the backend to read again.
    assert [m["role"] for m in sent[-3:]] == ["assistant", "tool", "user"]
    assert "pigs everywhere" in sent[-2]["content"]
    # The transcript now holds exactly those messages, so the next turn's
    # request starts with what the backend already has.
    nxt = s.build_messages()
    assert nxt[: len(sent) - 1] == sent[:-1]
    assert nxt[-1] == {"role": "assistant", "content": "answer"}


async def test_a_seeded_thread_holds_the_runs_rounds_raw(fake_rag):
    fake_rag.retrieve_result = ("[@5/10] a: pigs", 1)
    proxy = ModelProxy([tool_call(), final("done")])
    result = await run(proxy, fake_rag, "fast")

    s = seed_session_from_run(0, [], "q", result)

    assert [m.get("kind") for m in s.transcript] == ["question", "call", "result", "answer"]
    assert s.searches == 1 and s.seen_keys


async def test_compaction_condenses_rounds_and_never_orphans_a_call():
    from lore.compaction import _compact_session
    from lore.metrics import AgentMetrics

    s = session("fast")
    for i in range(4):
        s.append_rounds([
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "function": {"name": "search_discord_history", "arguments": f'{{"query": "q{i}"}}'}}]},
            {"role": "tool", "name": "search_discord_history", "content": f"[@{i}/1] a: result {i}", "tool_call_id": f"c{i}"},
        ])
    proxy = ModelProxy([], reply="summary of q0 and q1")

    assert await _compact_session(s, proxy, AgentMetrics(), FAST)

    kinds = [m.get("kind") for m in s.transcript]
    assert kinds == ["question", "answer", "research", "call", "result", "call", "result"]
    assert "summary of q0 and q1" in s.transcript[2]["content"]
    assert "result 0" in proxy.streams[0]["messages"][-1]["content"]  # what was summarised
    ids = [m["tool_calls"][0]["id"] for m in s.transcript if m.get("kind") == "call"]
    assert ids == ["c2", "c3"]
