"""
Tests for idle eviction.

llama-server's router keeps a model resident once loaded, so without eviction a
background job would be starved on any day with LLM traffic — it would be told it
may run while 18 GB of model sat on the card. Eviction must *verify* the unload
rather than trusting the response, because telling the arbiter there is room is
what lets it start a job.
"""

import asyncio

import pytest
import httpx

import main


def _models_payload(*resident):
    return {
        "data": [
            {"id": m, "status": {"value": "loaded" if m in resident else "unloaded"}}
            for m in ("brain", "chat", "lore")
        ]
    }


class FakeRouter:
    """
    Stands in for llama-swappable's router endpoints.

    Models the real router's contract faithfully, including its rejections. An
    earlier version of this stub accepted any body, which let a real bug ship:
    the evictor posted `{}` and the router answered 400 "model is not found",
    so nothing was ever unloaded in production while the tests stayed green.
    """

    def __init__(self, resident=(), unload_effective=True):
        self.resident = set(resident)
        self.unload_effective = unload_effective
        self.unload_calls = []

    async def get(self, url, **kw):
        assert url.endswith("/v1/models")
        # request= is required for raise_for_status() to work on a synthetic response
        return httpx.Response(
            200,
            json=_models_payload(*self.resident),
            request=httpx.Request("GET", url),
        )

    async def post(self, url, **kw):
        assert url.endswith("/models/unload")
        body = kw.get("json") or {}
        model = body.get("model")
        self.unload_calls.append(model)

        # The real router requires a model name and 400s without one.
        if not model:
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": "model is not found",
                                "type": "invalid_request_error"}},
                request=httpx.Request("POST", url),
            )
        if self.unload_effective:
            self.resident.discard(model)
        return httpx.Response(200, json={"success": True},
                              request=httpx.Request("POST", url))


class FakeArbiter:
    """The evictor's only job besides unloading: say when there is really room."""

    def __init__(self):
        self.calls = []

    async def release(self):
        self.calls.append("release")

    async def acquire(self):
        self.calls.append("acquire")
        return True, "cleared"

    async def aclose(self):
        pass


@pytest.fixture
def arbiter(monkeypatch):
    a = FakeArbiter()
    monkeypatch.setattr(main, "arbiter", a)
    return a


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    main.state.current_model = None
    main.state.last_request_at = None
    while main.state.llm_inflight:
        main.state.llm_request_finished()
    monkeypatch.setattr(main, "IDLE_EVICT_UNLOAD_TIMEOUT", 1.0)
    yield


@pytest.mark.asyncio
async def test_reads_resident_models(monkeypatch):
    monkeypatch.setattr(main, "_http_client", FakeRouter(resident=["brain"]))
    assert await main._resident_models() == ["brain"]


@pytest.mark.asyncio
async def test_unload_names_the_model(monkeypatch):
    """
    The regression: posting an empty body gets a 400 and unloads nothing, so
    the GPU never frees and external jobs are starved forever.
    """
    router = FakeRouter(resident=["brain"])
    monkeypatch.setattr(main, "_http_client", router)
    main.state.current_model = "brain"

    assert await main._unload_resident_models() is True
    assert router.unload_calls == ["brain"], \
        f"unload must name the model; sent {router.unload_calls}"
    assert main.state.current_model is None


@pytest.mark.asyncio
async def test_unloads_every_resident_model(monkeypatch):
    router = FakeRouter(resident=["brain", "chat"])
    monkeypatch.setattr(main, "_http_client", router)

    assert await main._unload_resident_models() is True
    assert sorted(router.unload_calls) == ["brain", "chat"]


@pytest.mark.asyncio
async def test_rejected_unload_is_reported_as_failure(monkeypatch):
    """A 400 must not be mistaken for success — that would free no VRAM."""
    router = FakeRouter(resident=["brain"])
    monkeypatch.setattr(main, "_http_client", router)

    # Simulate the old behaviour: call with a model list containing an empty name.
    assert await main._unload_resident_models([""]) is False


@pytest.mark.asyncio
async def test_unload_that_does_not_take_effect_is_reported_as_failure(monkeypatch):
    """Trusting the 200 here would hand a job a GPU that is still occupied."""
    router = FakeRouter(resident=["brain"], unload_effective=False)
    monkeypatch.setattr(main, "_http_client", router)

    assert await main._unload_resident_models() is False


@pytest.mark.asyncio
async def test_unreachable_router_is_not_treated_as_success(monkeypatch):
    class Dead:
        async def post(self, *a, **kw):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(main, "_http_client", Dead())
    assert await main._unload_resident_models() is False


# -- telling the arbiter there is room --

@pytest.mark.asyncio
async def test_arbiter_is_told_only_after_the_unload_lands(monkeypatch, arbiter):
    """
    Ordering that matters: releasing before the model is actually gone invites
    the arbiter to start a 16 GB job under an 18 GB model, which is the OOM this
    whole arrangement exists to prevent.
    """
    router = FakeRouter(resident=["brain"], unload_effective=False)
    monkeypatch.setattr(main, "_http_client", router)
    monkeypatch.setattr(main, "IDLE_EVICT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(main, "IDLE_EVICT_SECONDS", 0.0)
    main.state.record_request()

    task = asyncio.create_task(main._idle_evictor())
    await asyncio.sleep(0.3)
    task.cancel()

    assert router.unload_calls, "should have attempted the unload"
    assert arbiter.calls == [], "must not report room while the model is still resident"


@pytest.mark.asyncio
async def test_arbiter_is_told_once_the_slot_is_free(monkeypatch, arbiter):
    router = FakeRouter(resident=["brain"])
    monkeypatch.setattr(main, "_http_client", router)
    monkeypatch.setattr(main, "IDLE_EVICT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(main, "IDLE_EVICT_SECONDS", 0.0)
    main.state.record_request()

    task = asyncio.create_task(main._idle_evictor())
    await asyncio.sleep(0.3)
    task.cancel()

    assert "release" in arbiter.calls


@pytest.mark.asyncio
async def test_nothing_is_evicted_while_a_request_is_in_flight(monkeypatch, arbiter):
    router = FakeRouter(resident=["brain"])
    monkeypatch.setattr(main, "_http_client", router)
    monkeypatch.setattr(main, "IDLE_EVICT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(main, "IDLE_EVICT_SECONDS", 0.0)
    main.state.record_request()
    main.state.llm_request_started()

    task = asyncio.create_task(main._idle_evictor())
    await asyncio.sleep(0.2)
    task.cancel()

    assert router.unload_calls == []
    assert arbiter.calls == []


# -- the in-flight refcount that gates all of the above --

def test_llm_refcount():
    main.state.llm_request_started()
    main.state.llm_request_started()
    assert main.state.llm_inflight == 2
    main.state.llm_request_finished()
    assert main.state.llm_inflight == 1
    main.state.llm_request_finished()
    main.state.llm_request_finished()  # over-release must not go negative
    assert main.state.llm_inflight == 0
