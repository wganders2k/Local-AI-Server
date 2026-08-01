"""
The proxy must not touch the model backend until the arbiter has cleared the GPU.

This is the one property the proxy still has to get right about the GPU. It used
to be enforced here with a whole handover state machine; now it is one call, and
what remains to prove is that the call is actually respected — that a refusal
becomes a 503 rather than a request that loads a model onto an occupied card.

Asserted by counting calls to a stubbed backend, so "did not touch it" is a fact
rather than an inference.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import asyncio

import httpx
import pytest
from httpx import ASGITransport

import main


class FakeArbiter:
    """Records the sequence, which is what the ordering guarantee lives in."""

    def __init__(self):
        self.calls = []
        self.granted = True
        self.reason = "refused"
        self.gate = asyncio.Event()
        self.gate.set()

    def block(self):
        """Hold acquire open so a test can observe the window before it returns."""
        self.gate.clear()

    def unblock(self):
        self.gate.set()

    async def acquire(self):
        self.calls.append("acquire")
        await self.gate.wait()
        return (True, "cleared") if self.granted else (False, self.reason)

    async def release(self):
        self.calls.append("release")

    async def aclose(self):
        pass


class Backend:
    """Stub forwarder recording every attempt to reach the model backend."""

    def __init__(self):
        self.calls = []

    async def forward(self, target_base_url, request, model=None, lock=None, on_complete=None):
        self.calls.append(model)
        if lock:
            lock.release()
        if on_complete:
            on_complete()
        return httpx.Response(200, json={"ok": True, "model": model})


@pytest.fixture
def arbiter(monkeypatch):
    a = FakeArbiter()
    monkeypatch.setattr(main, "arbiter", a)
    return a


@pytest.fixture
def backend(monkeypatch):
    b = Backend()
    monkeypatch.setattr(main, "_forward", b.forward)
    return b


@pytest.fixture(autouse=True)
def clean_state():
    main.state.current_model = None
    main.state.last_request_at = None
    while main.state.llm_inflight:
        main.state.llm_request_finished()
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        yield c


# -- the gate --

async def test_the_gpu_is_acquired_before_the_backend_is_touched(client, backend, arbiter):
    r = await client.post("/v1/chat/completions", json={"model": "brain"})

    assert r.status_code == 200
    assert backend.calls == ["brain"]
    assert arbiter.calls[0] == "acquire"


async def test_backend_untouched_while_the_arbiter_is_still_clearing(client, backend, arbiter):
    """The window that matters: a slow stop must not race the model load."""
    arbiter.block()
    task = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "brain"}))
    await asyncio.sleep(0.1)

    assert backend.calls == [], "proxy reached the backend before the GPU was clear"

    arbiter.unblock()
    assert (await task).status_code == 200
    assert backend.calls == ["brain"]


async def test_a_refusal_becomes_503_and_never_reaches_the_backend(client, backend, arbiter):
    """
    Refusing is the safe failure. Loading a model over a job that may still hold
    several GB OOMs both workloads, so a refused acquire must not be routed
    around.
    """
    arbiter.granted = False
    arbiter.reason = "could not stop lora-trainer"

    r = await client.post("/v1/chat/completions", json={"model": "brain"})

    assert r.status_code == 503
    assert r.headers["Retry-After"] == "30"
    assert "could not stop lora-trainer" in r.json()["detail"]
    assert backend.calls == []


async def test_a_refusal_does_not_leak_queue_depth(client, backend, arbiter):
    """A stuck queue depth would make the idle evictor believe work is pending."""
    arbiter.granted = False

    await client.post("/v1/chat/completions", json={"model": "brain"})

    assert main.state.queue_depth == 0
    assert main.state.llm_inflight == 0


# -- handing it back --

async def test_the_arbiter_is_told_when_the_last_request_finishes(client, backend, arbiter):
    await client.post("/v1/chat/completions", json={"model": "brain"})
    await asyncio.sleep(0)  # release is fired as a task

    assert "release" in arbiter.calls


async def test_autocomplete_never_touches_the_arbiter(client, backend, arbiter):
    """
    The permanent slot is always resident and never competes for the swappable
    model's VRAM. Acquiring for it would stop a training run for nothing.
    """
    r = await client.post("/v1/chat/completions", json={"model": "autocomplete"})

    assert r.status_code == 200
    assert backend.calls == ["autocomplete"]
    assert arbiter.calls == []


# -- what goes on the wire --

async def test_the_proxy_sends_its_name_and_nothing_else():
    """
    Identity only. The proxy must not send a priority, a VRAM figure, or anything
    else that amounts to assessing its own importance — the arbiter looks all of
    that up from the name, and a caller that could assert it would be grading its
    own homework. The LLM is not exempt just because it ranks highest.
    """
    from arbiter import ArbiterClient

    seen = []

    def handler(request):
        seen.append((request.url.path, __import__("json").loads(request.content)))
        return httpx.Response(200, json={"granted": True, "detail": "cleared"})

    c = ArbiterClient("http://arbiter:11438", "llm")
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await c.acquire()
    await c.release()
    await c.aclose()

    assert seen == [("/gpu/acquire", {"name": "llm"}), ("/gpu/release", {"name": "llm"})]
