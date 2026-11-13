"""
Image generation behind the proxy: text prompts only, the same GPU gate as
chat, and the image server never sharing the card with a model too big for it.
"""

import asyncio

import httpx
import pytest
from httpx import ASGITransport

import imagegen
import main
from tests.fakes import Backend, install_dispatcher, settle

SIZES = {"1024x1024", "768x768"}


# ---- clean_request -------------------------------------------------------

def test_only_prompt_and_size_survive():
    body = {"prompt": " a goose ", "size": "768x768", "image": "data:...", "init_image": "x",
            "strength": 0.5, "model": "other", "response_format": "url"}
    assert imagegen.clean_request(body, SIZES, 100, "1024x1024") == {
        "prompt": "a goose", "size": "768x768", "n": 1, "response_format": "b64_json"}


@pytest.mark.parametrize("body, msg", [
    ({}, "prompt"),
    ({"prompt": "   "}, "prompt"),
    ({"prompt": "x" * 101}, "longer"),
    ({"prompt": "x", "size": "4096x4096"}, "size"),
    ({"prompt": "x", "n": 4}, "n=1"),
    ([], "object"),
])
def test_bad_requests_are_refused(body, msg):
    with pytest.raises(imagegen.RequestError, match=msg):
        imagegen.clean_request(body, SIZES, 100, "1024x1024")


# ---- the route -----------------------------------------------------------

class FakeImageServer:
    def __init__(self, running=False, stop_ok=True):
        self.running = running
        self.stop_ok = stop_ok
        self.bodies = []
        self.events = []

    async def generate(self, client, body):
        self.events.append("generate")
        self.running = True
        self.bodies.append(body)
        return httpx.Response(200, json={"data": [{"b64_json": "AAAA"}]})

    async def stop(self, client):
        self.events.append("stop")
        if self.stop_ok:
            self.running = False
        return self.stop_ok


class Router:
    def __init__(self, resident=()):
        self.resident = list(resident)
        self.unloaded = []

    async def resident_models(self):
        return list(self.resident)

    async def unload(self, resident=None):
        self.unloaded.append(list(resident or self.resident))
        self.resident = []
        return True


@pytest.fixture
def setup(monkeypatch):
    arb, _ = install_dispatcher(monkeypatch)
    server = FakeImageServer()
    monkeypatch.setattr(main, "image_server", server)
    monkeypatch.setattr(main, "IMAGEGEN_ENABLED", True)
    monkeypatch.setattr(main, "IMAGEGEN_SIZES", SIZES)
    router = Router()
    monkeypatch.setattr(main, "_resident_models", router.resident_models)
    monkeypatch.setattr(main, "_unload_resident_models", router.unload)
    backend = Backend()
    monkeypatch.setattr(main, "_forward", backend.forward)
    main.state.current_model = None
    return arb, server, router, backend


@pytest.fixture
async def client():
    async with httpx.AsyncClient(transport=ASGITransport(app=main.app), base_url="http://proxy") as c:
        yield c


async def test_disabled_by_default(client, setup, monkeypatch):
    monkeypatch.setattr(main, "IMAGEGEN_ENABLED", False)
    r = await client.post("/v1/images/generations", json={"prompt": "x"})
    assert r.status_code == 503 and setup[1].bodies == []


async def test_generation_is_gated_and_sends_only_the_clean_body(client, setup):
    arb, server, router, _ = setup
    r = await client.post("/v1/images/generations", json={"prompt": "goose", "image": "data:x"})
    assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == "AAAA"
    assert server.bodies == [{"prompt": "goose", "size": "1024x1024", "n": 1, "response_format": "b64_json"}]
    assert arb.calls[0] == ("acquire", "llm")
    await settle()
    assert ("release", "llm") in arb.calls
    assert main.dispatcher.running is None


async def test_refused_gpu_never_reaches_the_image_server(client, setup):
    arb, server, _, _ = setup
    arb.refusals["llm"] = "no"
    r = await client.post("/v1/images/generations", json={"prompt": "goose"})
    assert r.status_code == 503 and server.bodies == []
    assert main.dispatcher.queue_depth == 0 and main.dispatcher.running is None


@pytest.mark.parametrize("path", ["/v1/images/edits", "/v1/images/variations"])
async def test_image_inputs_are_refused(client, setup, path):
    r = await client.post(path, json={"prompt": "x", "image": "data:x"})
    assert r.status_code == 403 and setup[1].bodies == []


async def test_a_small_model_stays_a_big_one_is_unloaded(client, setup):
    _, _, router, _ = setup
    router.resident = ["mimic-multi"]
    await client.post("/v1/images/generations", json={"prompt": "goose"})
    assert router.unloaded == []
    router.resident = ["brain-dense-heretic"]
    await client.post("/v1/images/generations", json={"prompt": "goose"})
    assert router.unloaded == [["brain-dense-heretic"]]


async def test_a_big_chat_model_stops_the_image_server_first(client, setup):
    _, server, _, backend = setup
    server.running = True
    await client.post("/v1/chat/completions", json={"model": "mimic-multi"})
    assert server.events == [] and server.running
    await client.post("/v1/chat/completions", json={"model": "brain-dense-heretic"})
    assert server.events == ["stop"] and backend.calls == ["mimic-multi", "brain-dense-heretic"]


async def test_a_failed_stop_never_loads_the_big_model(client, setup):
    _, server, _, backend = setup
    server.running, server.stop_ok = True, False
    with pytest.raises(RuntimeError):
        await client.post("/v1/chat/completions", json={"model": "brain-dense-heretic"})
    assert backend.calls == [] and main.dispatcher.running is None


# ---- ImageServer against a fake Docker API ---------------------------------

async def test_image_server_start_stop_and_generate():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path, dict(req.url.params)))
        if req.url.host == "docker":
            return httpx.Response(204)
        if req.url.path == "/v1/images/generations":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, text="ok")

    srv = imagegen.ImageServer("http://img:8080", "http://docker:2375", "imagegen", 5, 5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await srv.generate(c, {"prompt": "x"})
        assert srv.running
        assert await srv.stop(c) and not srv.running
    assert calls[0] == ("POST", "/containers/imagegen/start", {})
    assert ("POST", "/containers/imagegen/stop", {"t": "0"}) in calls


async def test_a_crashed_server_is_started_again_and_the_request_retried():
    """
    sd-server is the container's only process: nothing listening means it has
    exited and freed its memory. `running` used to stay set, so every later
    request went to a dead server until the idle evictor happened to stop it.
    """
    calls = []
    alive = {"up": False}

    def handler(req):
        calls.append((req.method, req.url.path))
        if req.url.host == "docker":
            if req.url.path.endswith("/start"):
                alive["up"] = True
            return httpx.Response(204)
        if not alive["up"]:
            raise httpx.ConnectError("refused")
        if req.url.path == "/v1/images/generations":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, text="ok")

    srv = imagegen.ImageServer("http://img:8080", "http://docker:2375", "imagegen", 5, 5)
    srv.running = True  # believed up; actually crashed
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        resp = await srv.generate(c, {"prompt": "x"})
    assert resp.status_code == 200 and srv.running
    assert ("POST", "/containers/imagegen/start") in calls


async def test_a_slow_server_is_not_assumed_dead():
    """A read timeout may mean it is still generating, holding its memory."""
    def handler(req):
        raise httpx.ReadTimeout("slow")

    srv = imagegen.ImageServer("http://img:8080", "http://docker:2375", "imagegen", 5, 5)
    srv.running = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(httpx.ReadTimeout):
            await srv.generate(c, {"prompt": "x"})
    assert srv.running


async def test_failed_stop_keeps_running_set():
    srv = imagegen.ImageServer("http://img:8080", "http://docker:2375", "imagegen", 5, 5)
    srv.running = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as c:
        assert not await srv.stop(c)
    assert srv.running


# ---- idle eviction ---------------------------------------------------------

@pytest.mark.parametrize("stop_ok", [True, False])
async def test_idle_evictor_stops_the_image_server_before_reporting_room(setup, monkeypatch, stop_ok):
    arb, server, router, _ = setup
    server.running, server.stop_ok = True, stop_ok
    monkeypatch.setattr(main, "IDLE_EVICT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(main, "IDLE_EVICT_SECONDS", 0.0)
    arb.holder, main.dispatcher.lease.held = "llm", "interactive"
    main.state.record_request()
    task = asyncio.create_task(main._idle_evictor())
    await asyncio.sleep(0.2)
    task.cancel()
    assert "stop" in server.events
    # No room is reported while the image server may still hold its 11 GB.
    assert (("release", "llm") in arb.calls) is stop_ok
