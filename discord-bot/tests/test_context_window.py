"""
Discovering the agent model's context window from the proxy.

The value drives every budget decision a lore thread makes, so the cases that
matter most here are the failure ones: every path that cannot produce a real
number must leave the documented fallback in place rather than a zero or a guess.
"""

import pytest

from config import AGENT_CTX_LIMIT_FALLBACK, AGENT_MODEL
from lore import context_window


@pytest.fixture(autouse=True)
def _clean_cache():
    """The cache is a module global; no test may leak into the next."""
    context_window.reset()
    yield
    context_window.reset()


class FakeProxy:
    """Returns a scripted {model: ctx_size} mapping, and counts the calls."""

    def __init__(self, sizes):
        self._sizes = sizes
        self.calls = 0

    async def model_context_sizes(self):
        self.calls += 1
        return self._sizes


DISCOVERED = 65536  # deliberately not the fallback


def test_limit_is_the_fallback_before_discovery():
    assert context_window.limit() == AGENT_CTX_LIMIT_FALLBACK
    assert not context_window.discovered()
    assert "fallback" in context_window.describe()


async def test_discovery_caches_the_backend_value():
    proxy = FakeProxy({AGENT_MODEL: DISCOVERED, "other": 4096})

    assert await context_window.discover(proxy) is True
    assert context_window.limit() == DISCOVERED
    assert context_window.discovered()
    assert "/v1/models" in context_window.describe()


async def test_a_value_matching_the_fallback_is_still_discovered():
    proxy = FakeProxy({AGENT_MODEL: AGENT_CTX_LIMIT_FALLBACK})
    assert await context_window.discover(proxy) is True
    assert context_window.discovered()


async def test_unreachable_proxy_keeps_the_fallback():
    # model_context_sizes() degrades to {} rather than raising.
    proxy = FakeProxy({})

    assert await context_window.discover(proxy) is False
    assert context_window.limit() == AGENT_CTX_LIMIT_FALLBACK
    assert not context_window.discovered()


async def test_agent_model_missing_from_the_backend_keeps_the_fallback(caplog):
    proxy = FakeProxy({"some-other-model": 4096})

    with caplog.at_level("WARNING"):
        assert await context_window.discover(proxy) is False

    assert context_window.limit() == AGENT_CTX_LIMIT_FALLBACK
    assert AGENT_MODEL in caplog.text  # names the model that is missing


async def test_a_stale_fallback_constant_is_flagged(caplog):
    proxy = FakeProxy({AGENT_MODEL: DISCOVERED})

    with caplog.at_level("WARNING"):
        await context_window.discover(proxy)

    # The discovered value wins; the warning is what replaces the old
    # "keep these two in step by hand" comment.
    assert context_window.limit() == DISCOVERED
    assert "AGENT_CTX_LIMIT_FALLBACK" in caplog.text


async def test_ensure_discovered_is_a_no_op_once_cached():
    proxy = FakeProxy({AGENT_MODEL: DISCOVERED})

    await context_window.ensure_discovered(proxy)
    await context_window.ensure_discovered(proxy)
    await context_window.ensure_discovered(proxy)

    assert proxy.calls == 1  # never touches the network again


async def test_ensure_discovered_retries_after_a_failure():
    failing = FakeProxy({})
    await context_window.ensure_discovered(failing)
    assert not context_window.discovered()

    recovered = FakeProxy({AGENT_MODEL: DISCOVERED})
    await context_window.ensure_discovered(recovered)
    assert context_window.limit() == DISCOVERED


async def test_an_explicit_model_can_be_queried():
    proxy = FakeProxy({"lore": 32768, AGENT_MODEL: DISCOVERED})
    assert await context_window.discover(proxy, model="lore") is True
    assert context_window.limit() == 32768


async def test_metrics_track_the_discovered_limit():
    """The accessor exercised through its real consumer."""
    from lore.metrics import AgentMetrics

    await context_window.discover(FakeProxy({AGENT_MODEL: 1000}))
    metrics = AgentMetrics()
    metrics.record_usage({"prompt_tokens": 400, "completion_tokens": 100})

    assert metrics.context_pct == 0.5
    assert "500/1,000 (50%)" in metrics.context_line()


# ---------------------------------------------------------------------------
# ProxyClient.model_context_sizes — parsing llama-server's argv
# ---------------------------------------------------------------------------


def make_entry(model_id, args, status_value="unloaded"):
    return {"id": model_id, "status": {"value": status_value, "args": args}}


async def sizes_from(entries):
    """Run model_context_sizes() against a canned /v1/models payload."""
    from proxy_client import ProxyClient

    client = ProxyClient("http://proxy")

    async def payload():
        return entries

    client._models_payload = payload
    return await client.model_context_sizes()


async def test_ctx_size_is_read_from_the_argv():
    # The real shape: llama-server reports the full command line it would run.
    entries = [make_entry("brain-dense-heretic",
                          ["--alias", "brain-dense-heretic", "--ctx-size", "96000",
                           "--flash-attn", "on", "--n-gpu-layers", "-1"])]
    assert await sizes_from(entries) == {"brain-dense-heretic": 96000}


async def test_short_form_flag_is_accepted():
    assert await sizes_from([make_entry("m", ["-c", "8192"])]) == {"m": 8192}


async def test_models_without_a_size_are_omitted_not_guessed():
    entries = [
        make_entry("sized", ["--ctx-size", "4096"]),
        make_entry("unsized", ["--alias", "unsized"]),
    ]
    assert await sizes_from(entries) == {"sized": 4096}


async def test_a_malformed_size_is_skipped():
    entries = [
        make_entry("bad", ["--ctx-size", "not-a-number"]),
        make_entry("truncated", ["--ctx-size"]),  # flag with no value
        make_entry("good", ["--ctx-size", "2048"]),
    ]
    assert await sizes_from(entries) == {"good": 2048}


async def test_entries_missing_id_or_args_are_skipped():
    entries = [
        {"id": "no-status"},
        {"status": {"args": ["--ctx-size", "1"]}},   # no id
        make_entry("fine", ["--ctx-size", "512"]),
    ]
    assert await sizes_from(entries) == {"fine": 512}


async def test_empty_payload_yields_empty_mapping():
    assert await sizes_from([]) == {}
