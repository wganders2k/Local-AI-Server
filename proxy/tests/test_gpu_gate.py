"""
The proxy must not touch the model backend until the arbiter has granted it
the GPU, and it must schedule its own requests so a person never queues behind
background work.

Asserted by counting calls to a stubbed backend and to a fake arbiter that
applies the real admission rules (tests/fakes.py), so "did not touch it" and
"asked under this name" are facts rather than inferences.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import asyncio

import httpx
import pytest
from httpx import ASGITransport

import main
from tests.fakes import Backend, install_dispatcher, settle


@pytest.fixture
def arb(monkeypatch):
    service, _ = install_dispatcher(monkeypatch)
    return service


@pytest.fixture
def backend(monkeypatch):
    b = Backend()
    monkeypatch.setattr(main, "_forward", b.forward)
    return b


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    main.state.current_model = None
    main.state.last_request_at = None
    for server in main.image_servers.values():
        monkeypatch.setattr(server, "running", False)
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        yield c


def chat(client, model):
    return asyncio.create_task(client.post("/v1/chat/completions", json={"model": model}))


# -- the gate --

async def test_the_gpu_is_acquired_before_the_backend_is_touched(client, backend, arb):
    r = await client.post("/v1/chat/completions", json={"model": "brain"})

    assert r.status_code == 200
    assert backend.calls == ["brain"]
    assert arb.calls[0] == ("acquire", "llm")


async def test_backend_untouched_while_the_arbiter_is_still_clearing(client, backend, arb):
    """The window that matters: a slow stop must not race the model load."""
    arb.block()
    task = chat(client, "brain")
    await asyncio.sleep(0.1)

    assert backend.calls == [], "proxy reached the backend before the GPU was clear"

    arb.unblock()
    assert (await task).status_code == 200
    assert backend.calls == ["brain"]


async def test_a_refusal_becomes_503_and_never_reaches_the_backend(client, backend, arb):
    """
    Refusing is the safe failure. Loading a model over a job that may still hold
    several GB OOMs both workloads, so a refused acquire must not be routed
    around.
    """
    arb.refusals["llm"] = "could not stop lora-trainer"

    r = await client.post("/v1/chat/completions", json={"model": "brain"})

    assert r.status_code == 503
    assert r.headers["Retry-After"] == "30"
    assert "could not stop lora-trainer" in r.json()["detail"]
    assert backend.calls == []


async def test_a_refusal_does_not_leak_a_turn(client, backend, arb):
    """A stuck turn would block every later request and the idle evictor."""
    arb.refusals["llm"] = "no"

    await client.post("/v1/chat/completions", json={"model": "brain"})

    assert main.dispatcher.queue_depth == 0
    assert main.dispatcher.running is None


async def test_requests_run_one_at_a_time_in_arrival_order(client, backend, arb):
    hold = backend.hold_model("a")
    first = chat(client, "a")
    await asyncio.sleep(0.05)
    second, third = chat(client, "b"), chat(client, "c")
    await asyncio.sleep(0.05)

    assert backend.calls == ["a"]
    assert main.dispatcher.queue_depth == 2

    hold.set()
    await asyncio.gather(first, second, third)
    assert backend.calls == ["a", "b", "c"]


# -- handing it back --

async def test_the_lease_is_released_when_the_last_request_finishes(client, backend, arb):
    await client.post("/v1/chat/completions", json={"model": "brain"})
    await settle()

    assert ("release", "llm") in arb.calls
    assert arb.holder is None


async def test_the_lease_is_kept_while_more_work_is_queued(client, backend, arb):
    """Releasing between back-to-back requests would invite a job onto the card mid-conversation."""
    hold = backend.hold_model("a")
    first = chat(client, "a")
    await asyncio.sleep(0.05)
    second = chat(client, "b")
    await asyncio.sleep(0.05)
    hold.set()
    await asyncio.gather(first, second)
    await settle()

    assert arb.names("release") == ["llm"], "released once, after the queue drained"


# -- batch traffic --

async def test_a_batch_model_asks_under_its_own_lower_ranked_name(client, backend, arb):
    r = await client.post("/v1/chat/completions", json={"model": "judge"})

    assert r.status_code == 200 and backend.calls == ["judge"]
    assert arb.names("acquire") == ["llm-batch"]


async def test_a_batch_request_waits_out_interactive_traffic(client, backend, arb):
    """A person waits for at most the batch call already running, never for a queue of them."""
    main.dispatcher.batch_yield_seconds = 0.3
    hold = backend.hold_model("mimic-multi")
    person = chat(client, "mimic-multi")
    await asyncio.sleep(0.05)
    judge = chat(client, "judge")
    await asyncio.sleep(0.1)
    assert backend.calls == ["mimic-multi"]

    hold.set()
    await person
    await asyncio.sleep(0.1)
    assert backend.calls == ["mimic-multi"], "the batch request did not wait out the quiet period"

    await judge
    assert backend.calls == ["mimic-multi", "judge"]


async def test_a_person_goes_ahead_of_every_queued_batch_request(client, backend, arb):
    """
    The bug this replaced: when the quiet window opened, every waiting batch
    request passed the gate at once and queued on a FIFO lock, so the next
    person waited behind all of them.
    """
    hold = backend.hold_model("judge")
    batch = [chat(client, "judge") for _ in range(4)]
    await asyncio.sleep(0.1)
    assert backend.calls == ["judge"], "only one batch request may be on the backend"
    assert main.dispatcher.batch_waiting == 3

    person = chat(client, "mimic-multi")
    await asyncio.sleep(0.05)
    hold.set()
    await person
    assert backend.calls == ["judge", "mimic-multi"], "the person queued behind batch work"
    await asyncio.gather(*batch)


async def test_a_refused_batch_request_waits_and_retries_rather_than_failing(client, backend, arb):
    """Its client is a background job: a refusal is the proxy's to wait out, not a 503 to count as a failure."""
    arb.holder = "profile-segment"
    judge = chat(client, "judge")
    await asyncio.sleep(0.15)
    assert not judge.done() and backend.calls == []
    assert arb.names("acquire").count("llm-batch") >= 2, "should have asked again"

    await arb.release("profile-segment")
    r = await judge
    assert r.status_code == 200 and backend.calls == ["judge"]


async def test_moving_up_from_batch_to_llm_hands_over_without_unloading(client, backend, arb):
    """The proxy reclaiming from itself: the batch lease goes straight back, nothing is freed."""
    freed = []

    async def free_gpu():
        freed.append(True)
        return True
    main.dispatcher.free_gpu = free_gpu

    hold = backend.hold_model("judge")
    judge = chat(client, "judge")
    await asyncio.sleep(0.05)
    assert arb.holder == "llm-batch"
    person = chat(client, "mimic-multi")
    await asyncio.sleep(0.05)
    hold.set()
    await asyncio.gather(judge, person)

    assert backend.calls == ["judge", "mimic-multi"]
    assert ("release", "llm-batch") in arb.calls
    assert freed == [], "a self-handover must not unload the model"


async def test_a_cooperative_reclaim_finishes_the_batch_call_frees_the_gpu_and_releases(client, backend, arb):
    """llm-batch is cooperative: something ranked above it gets a GPU that is really free."""
    freed = []

    async def free_gpu():
        freed.append(list(backend.calls))
        return True
    main.dispatcher.free_gpu = free_gpu

    hold = backend.hold_model("judge")
    first = chat(client, "judge")
    await asyncio.sleep(0.05)
    queued = chat(client, "judge")
    await asyncio.sleep(0.05)

    profiler = asyncio.create_task(arb.acquire("profile-segment"))
    await asyncio.sleep(0.05)
    assert not profiler.done(), "granted while a batch call was still running"

    hold.set()
    granted, _ = await profiler
    assert granted and arb.holder == "profile-segment"
    assert freed == [["judge"]], "the GPU is freed after the running call, before the release"
    await first
    await asyncio.sleep(0.1)
    assert backend.calls == ["judge"], "batch work resumed while the profiler held the card"

    await arb.release("profile-segment")
    await queued
    assert backend.calls == ["judge", "judge"]


# -- clients that went away --

class GoneRequest:
    async def is_disconnected(self):
        return True


async def test_a_request_whose_client_left_is_dropped_and_never_runs(monkeypatch, backend, arb):
    hold = backend.hold_model("a")
    running = asyncio.create_task(main.dispatcher.turn("interactive", label="a"))
    ticket = await running
    with pytest.raises(main.Abandoned):
        await main.dispatcher.turn("batch", request=GoneRequest(), label="judge")
    assert main.dispatcher.batch_waiting == 0
    main.dispatcher.finish(ticket)
    hold.set()


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
        if request.method == "GET":
            seen.append((request.url.path, dict(request.url.params)))
            return httpx.Response(200, json={"reclaim": False})
        seen.append((request.url.path, __import__("json").loads(request.content)))
        return httpx.Response(200, json={"granted": True, "detail": "cleared"})

    c = ArbiterClient("http://arbiter:11438", "llm-batch")
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await c.acquire()
    assert await c.reclaim_notice() is False
    await c.release()
    await c.aclose()

    assert seen == [
        ("/gpu/acquire", {"name": "llm-batch"}),
        ("/gpu/reclaim-notice", {"name": "llm-batch"}),
        ("/gpu/release", {"name": "llm-batch"}),
    ]


async def test_an_unreachable_arbiter_is_a_none_notice_not_a_reclaim():
    from arbiter import ArbiterClient

    def down(request):
        raise httpx.ConnectError("refused")

    c = ArbiterClient("http://arbiter:11438", "llm-batch")
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(down))
    assert await c.reclaim_notice() is None
    await c.aclose()
