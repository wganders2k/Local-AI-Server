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
from dispatcher import Ticket
from tests.fakes import install_dispatcher


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


@pytest.fixture
def arbiter(monkeypatch):
    """The evictor's only job besides unloading: say when there is really room."""
    service, d = install_dispatcher(monkeypatch)
    # As after a chat: the lease is still held and the model resident.
    service.holder = "llm"
    d.lease.held = "interactive"
    return service


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    main.state.current_model = None
    main.state.last_request_at = None
    monkeypatch.setattr(main, "IDLE_EVICT_UNLOAD_TIMEOUT", 1.0)
    for server in main.image_servers.values():
        monkeypatch.setattr(server, "running", False)
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
    assert arbiter.names("release") == [], "must not report room while the model is still resident"


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

    assert arbiter.names("release") == ["llm"]


@pytest.mark.asyncio
async def test_nothing_is_evicted_while_a_request_is_in_flight(monkeypatch, arbiter):
    router = FakeRouter(resident=["brain"])
    monkeypatch.setattr(main, "_http_client", router)
    monkeypatch.setattr(main, "IDLE_EVICT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(main, "IDLE_EVICT_SECONDS", 0.0)
    main.state.record_request()
    main.dispatcher.running = Ticket("interactive")

    task = asyncio.create_task(main._idle_evictor())
    await asyncio.sleep(0.2)
    task.cancel()

    assert router.unload_calls == []
    assert arbiter.names("release") == []


@pytest.mark.asyncio
async def test_a_request_arriving_during_eviction_waits_for_it(monkeypatch, arbiter):
    """
    The race this closes: the evictor used to unload and poll for up to a
    minute without holding anything, so a request that started meanwhile could
    have the image server stopped and the lease released under it.
    """
    router = FakeRouter(resident=["brain"])
    monkeypatch.setattr(main, "_http_client", router)
    slow = asyncio.Event()
    real_unload = main._unload_resident_models

    async def unload(resident=None):
        await slow.wait()
        return await real_unload(resident)
    monkeypatch.setattr(main, "_unload_resident_models", unload)

    async with main.dispatcher.maintenance() as ok:
        assert ok
        turn = asyncio.create_task(main.dispatcher.turn("interactive", label="brain"))
        await asyncio.sleep(0.05)
        assert not turn.done(), "a request started while the evictor held the backend"
    ticket = await turn
    main.dispatcher.finish(ticket)


@pytest.mark.asyncio
async def test_maintenance_is_refused_while_anything_waits(arbiter):
    main.dispatcher.running = Ticket("interactive")
    async with main.dispatcher.maintenance() as ok:
        assert not ok


@pytest.mark.asyncio
async def test_a_stale_lease_from_before_a_restart_is_cleared(arbiter):
    """
    The proxy restarted mid-request: the arbiter still holds `llm` for it, which
    is kind: none and never reaped. Holding nothing, the proxy releases both
    names so the stale lease cannot outrank every job forever.
    """
    main.dispatcher.lease.held = None
    arbiter.holder = "llm"
    await main.dispatcher.release_lease_if_idle()
    assert arbiter.holder is None
