"""
Endpoint-level tests for the VRAM handover, driven through the FastAPI app.

The property under test is the one the whole design rests on: the proxy must not
issue a single byte to the model backend until every registered external job has
confirmed its VRAM is released. Here that is asserted by counting calls to a
stubbed backend.
"""

import asyncio

import pytest
import httpx
from httpx import ASGITransport

import main
from state import ExternalJobState


@pytest.fixture(autouse=True)
def clean_state():
    """Fresh coordination state per test — these are module-level singletons."""
    main.external_jobs = ExternalJobState()
    main.state.current_model = None
    main.state.last_request_at = None
    yield


class Backend:
    """
    Stub forwarder that records every attempt to reach the model backend.

    ``gate`` lets a test hold a request open so it is genuinely in flight,
    rather than completing before the assertion runs.
    """

    def __init__(self):
        self.calls = []
        self.gate = asyncio.Event()
        self.gate.set()

    def block(self):
        self.gate.clear()

    def unblock(self):
        self.gate.set()

    async def forward(self, target_base_url, request, model=None, lock=None, on_complete=None):
        self.calls.append(model)
        try:
            await self.gate.wait()
        finally:
            if lock:
                lock.release()
            if on_complete:
                on_complete()
        return httpx.Response(200, json={"ok": True, "model": model})


@pytest.fixture
def backend(monkeypatch):
    b = Backend()
    monkeypatch.setattr(main, "_forward", b.forward)
    return b


@pytest.fixture
def backend_calls(backend):
    return backend.calls


@pytest.fixture
async def client():
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        yield c


async def _register(client, job_id):
    r = await client.post("/external-job/register", json={"job_id": job_id})
    assert r.status_code == 200
    return r.json()


async def _confirm(client, job_id):
    r = await client.post("/external-job/yield", json={"job_id": job_id, "action": "yield"})
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_no_external_job_passes_straight_through(client, backend_calls):
    r = await client.post("/v1/chat/completions", json={"model": "brain"})
    assert r.status_code == 200
    assert backend_calls == ["brain"]


@pytest.mark.asyncio
async def test_backend_untouched_until_job_confirms(client, backend_calls):
    await _register(client, "job1")

    task = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "brain"})
    )
    await asyncio.sleep(0.2)

    assert backend_calls == [], "proxy reached the backend while VRAM was still held"

    status = (await client.get("/external-job/status", params={"job_id": "job1"})).json()
    assert status["yield_requested"] is True

    await _confirm(client, "job1")
    r = await task

    assert r.status_code == 200
    assert backend_calls == ["brain"]


@pytest.mark.asyncio
async def test_refuses_rather_than_loading_over_a_held_gpu(client, backend_calls):
    """A job that never confirms must cost us a 503, never an OOM."""
    await _register(client, "job1")

    r = await client.post("/v1/chat/completions", json={"model": "brain"})

    assert r.status_code == 503
    assert r.json()["error"] == "vram_unavailable"
    assert r.headers["Retry-After"] == "30"
    assert backend_calls == [], "proxy must not touch the backend on timeout"


@pytest.mark.asyncio
async def test_yield_released_after_request_completes(client, backend_calls):
    await _register(client, "job1")

    task = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "brain"})
    )
    await asyncio.sleep(0.1)
    await _confirm(client, "job1")
    await task

    status = (await client.get("/external-job/status", params={"job_id": "job1"})).json()
    assert status["may_run"] is True, "job must be allowed to resume once the LLM is done"


@pytest.mark.asyncio
async def test_yield_held_while_a_second_request_is_in_flight(client, backend):
    """Releasing on the first completion would let the job restart mid-conversation."""
    await _register(client, "job1")
    backend.block()

    t1 = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "brain"}))
    t2 = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "brain"}))
    await asyncio.sleep(0.1)
    await _confirm(client, "job1")
    await asyncio.sleep(0.1)

    assert main.external_jobs.llm_inflight == 2

    status = (await client.get("/external-job/status", params={"job_id": "job1"})).json()
    assert status["may_run"] is False, "job must stay yielded while LLM work is in flight"

    backend.unblock()
    await asyncio.gather(t1, t2)

    status = (await client.get("/external-job/status", params={"job_id": "job1"})).json()
    assert status["may_run"] is True


@pytest.mark.asyncio
async def test_autocomplete_bypasses_the_handover(client, backend_calls):
    """The permanent slot is already resident — it must not stall on a job."""
    await _register(client, "job1")

    r = await client.post("/v1/completions", json={"model": "autocomplete"})

    assert r.status_code == 200
    assert backend_calls == ["autocomplete"]
    status = (await client.get("/external-job/status", params={"job_id": "job1"})).json()
    assert status["yield_requested"] is False


@pytest.mark.asyncio
async def test_unknown_model_still_triggers_handover(client, backend_calls):
    """Unlisted models still route to the swappable slot, so they still need VRAM."""
    await _register(client, "job1")

    r = await client.post("/v1/chat/completions", json={"model": "some-new-model"})

    assert r.status_code == 503
    assert backend_calls == []


@pytest.mark.asyncio
async def test_unregistering_unblocks_a_waiting_request(client, backend_calls):
    """A job that exits outright has released its VRAM."""
    await _register(client, "job1")

    task = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "brain"})
    )
    await asyncio.sleep(0.1)
    await client.post("/external-job/unregister", json={"job_id": "job1"})
    r = await task

    assert r.status_code == 200
    assert backend_calls == ["brain"]
