"""
Image generation behind the proxy: allowlisted requests only, the same GPU gate as
chat, and the image server never sharing the card with a model too big for it.
"""

import asyncio
import base64
import io

import httpx
import pytest
from httpx import ASGITransport
from PIL import Image

import imagegen
import main
from tests.fakes import Backend, install_dispatcher, settle

SIZES = {"1024x1024", "768x768"}


def steps_tag(n):
    return f'<sd_cpp_extra_args>{{"sample_params":{{"sample_steps":{n}}}}}</sd_cpp_extra_args>'


# ---- clean_request -------------------------------------------------------

def test_only_prompt_and_size_survive():
    body = {"prompt": " a goose ", "size": "768x768", "init_image": "x", "mask": "x",
            "strength": 0.5, "model": "other", "response_format": "url"}
    assert imagegen.clean_request(body, SIZES, 100, "1024x1024") == {
        "prompt": "a goose" + steps_tag(8), "size": "768x768", "n": 1, "response_format": "b64_json"}


@pytest.mark.parametrize("body, msg", [
    ({}, "prompt"),
    ({"prompt": "   "}, "prompt"),
    ({"prompt": "x" * 101}, "longer"),
    ({"prompt": "x", "size": "4096x4096"}, "size"),
    ({"prompt": "x", "size": ["768x768"]}, "size"),  # unhashable: a 400, not a TypeError
    ({"prompt": "x", "size": 768}, "size"),
    ({"prompt": "x", "n": 4}, "'n'"),
    ({"prompt": "x", "n": 0}, "'n'"),
    ({"prompt": "x", "n": "2"}, "'n'"),
    ({"prompt": "x", "n": True}, "'n'"),
    ({"prompt": 'a goose <sd_cpp_extra_args>{"steps": 999}</sd_cpp_extra_args>'}, "extra_args"),
    ({"prompt": "a goose <SD_CPP_EXTRA_ARGS>{}"}, "extra_args"),
    ([], "object"),
])
def test_bad_requests_are_refused(body, msg):
    with pytest.raises(imagegen.RequestError, match=msg):
        imagegen.clean_request(body, SIZES, 100, "1024x1024", max_images=3)


def test_up_to_the_cap_of_images_per_request():
    assert imagegen.clean_request({"prompt": "x", "n": 3}, SIZES, 100, "1024x1024", max_images=3)["n"] == 3
    with pytest.raises(imagegen.RequestError, match="'n'"):
        imagegen.clean_request({"prompt": "x", "n": 2}, SIZES, 100, "1024x1024")


def test_cfg_is_sent_only_when_set_and_only_inside_its_range():
    tag = imagegen.clean_request({"prompt": "x", "cfg": 2.5}, SIZES, 100, "1024x1024", cfg_range=(1.0, 7.0))["prompt"]
    assert tag == 'x<sd_cpp_extra_args>{"sample_params":{"sample_steps":8,"guidance":{"txt_cfg":2.5}}}</sd_cpp_extra_args>'
    assert "guidance" not in imagegen.clean_request({"prompt": "x"}, SIZES, 100, "1024x1024")["prompt"]
    native = img2img({"image": png_b64(64, 64), "cfg": 3}, cfg_range=(1.0, 7.0))
    assert native["sample_params"] == {"sample_steps": 8, "guidance": {"txt_cfg": 3.0}}
    for bad in (0.5, 7.5, "2", True):
        with pytest.raises(imagegen.RequestError, match="'cfg'"):
            imagegen.clean_request({"prompt": "x", "cfg": bad}, SIZES, 100, "1024x1024", cfg_range=(1.0, 7.0))


def test_steps_come_only_from_the_allowlist():
    body = imagegen.clean_request({"prompt": "x", "steps": 16}, SIZES, 100, "1024x1024", steps=(8, 16))
    assert body["prompt"] == "x" + steps_tag(16)
    for bad in (12, "16", True, None):
        with pytest.raises(imagegen.RequestError, match="'steps'"):
            imagegen.clean_request({"prompt": "x", "steps": bad}, SIZES, 100, "1024x1024", steps=(8, 16))


# ---- image-to-image inputs -------------------------------------------------

def png_b64(width, height, fmt="PNG", color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def img2img(body, **kw):
    return imagegen.clean_request({"prompt": "watercolor", **body}, SIZES, 100, "1024x1024", max_images=3,
                                  steps=(8, 16), max_input_bytes=1024 * 1024, max_input_pixels=4_000_000, **kw)


def test_an_image_becomes_a_native_body_with_a_fresh_png_of_the_requested_size():
    # A wide JPEG sent as a data: URL, into a square: covered, then centre-cropped.
    body = img2img({"image": "data:image/jpeg;base64," + png_b64(1600, 900, "JPEG"), "size": "768x768",
                    "strength": 0.4, "steps": 16, "n": 2})
    with Image.open(io.BytesIO(base64.b64decode(body.pop("init_image")))) as im:
        assert (im.format, im.size, im.mode) == ("PNG", (768, 768), "RGB")
    assert body == {"prompt": "watercolor", "strength": 0.4, "width": 768, "height": 768, "batch_count": 2,
                    "sample_params": {"sample_steps": 16}, "output_format": "png"}


def test_the_crop_keeps_the_middle_of_the_image():
    im = Image.new("RGB", (300, 100), (0, 0, 255))
    im.paste((255, 0, 0), (90, 0, 210, 100))  # red middle, a little wider than the crop; blue sides
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    out = img2img({"image": base64.b64encode(buf.getvalue()).decode(), "size": "768x768"})["init_image"]
    with Image.open(io.BytesIO(base64.b64decode(out))) as im:
        assert im.getpixel((0, 384)) == im.getpixel((767, 384)) == (255, 0, 0)


def test_rgba_and_gif_inputs_are_flattened_to_rgb():
    for mode, fmt in (("RGBA", "PNG"), ("P", "GIF")):
        buf = io.BytesIO()
        Image.new(mode, (64, 64)).save(buf, format=fmt)
        out = img2img({"image": base64.b64encode(buf.getvalue()).decode(), "size": "768x768"})["init_image"]
        with Image.open(io.BytesIO(base64.b64decode(out))) as im:
            assert im.mode == "RGB"


@pytest.mark.parametrize("image, msg", [
    ("", "base64 string"),
    (123, "base64 string"),
    ("not base64!", "valid base64"),
    (base64.b64encode(b"just some text").decode(), "readable"),
    (png_b64(2001, 2000), "pixels"),
    ("A" * (2 * 1024 * 1024), "larger than"),
])
def test_bad_images_are_refused(image, msg):
    with pytest.raises(imagegen.RequestError, match=msg):
        img2img({"image": image})


@pytest.mark.parametrize("strength", [0.05, 0.99, 1, "0.5", True, None])
def test_strength_stays_in_range(strength):
    with pytest.raises(imagegen.RequestError, match="'strength'"):
        img2img({"image": png_b64(64, 64), "strength": strength})


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
    monkeypatch.setattr(main, "image_servers", {"default": server})
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
    r = await client.post("/v1/images/generations", json={"prompt": "goose", "init_image": "data:x"})
    assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == "AAAA"
    assert server.bodies == [{"prompt": "goose" + steps_tag(8), "size": "1024x1024", "n": 1,
                              "response_format": "b64_json"}]
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


async def test_an_image_request_reaches_the_server_as_a_native_body(client, setup):
    _, server, _, _ = setup
    r = await client.post("/v1/images/generations", json={"prompt": "goose", "image": png_b64(64, 64),
                                                          "size": "768x768", "strength": 0.5})
    assert r.status_code == 200
    assert server.bodies[0]["init_image"] and server.bodies[0]["strength"] == 0.5
    r = await client.post("/v1/images/generations", json={"prompt": "goose", "image": "nope"})
    assert r.status_code == 400 and len(server.bodies) == 1


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


async def test_a_request_picks_its_server_and_stops_the_other(client, setup, monkeypatch):
    _, default, _, _ = setup
    alt = FakeImageServer()
    monkeypatch.setattr(main, "image_servers", {"default": default, "alt": alt})
    await client.post("/v1/images/generations", json={"prompt": "goose", "model": "alt"})
    assert alt.bodies and not default.bodies and alt.running
    # An unknown name is an OpenAI client's own model field: the default serves it.
    await client.post("/v1/images/generations", json={"prompt": "goose", "model": "dall-e-3"})
    assert alt.events == ["generate", "stop"] and default.events == ["generate"]
    assert "model" not in default.bodies[0]


async def test_a_failed_stop_of_the_other_server_never_starts_this_one(client, setup, monkeypatch):
    _, default, _, _ = setup
    alt = FakeImageServer(running=True, stop_ok=False)
    monkeypatch.setattr(main, "image_servers", {"default": default, "alt": alt})
    r = await client.post("/v1/images/generations", json={"prompt": "goose"})
    assert r.status_code == 503 and default.bodies == []


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


def job_server(statuses, calls):
    """An image server whose native job reports each of `statuses` in turn."""
    def handler(req):
        calls.append((req.method, req.url.path))
        if req.url.host == "docker":
            return httpx.Response(204)
        if req.url.path == "/sdcpp/v1/img_gen":
            return httpx.Response(202, json={"id": "job_1", "status": "queued"})
        if req.url.path == "/sdcpp/v1/jobs/job_1":
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            result = {"images": [{"index": 0, "b64_json": "AAAA"}]} if status == "completed" else None
            return httpx.Response(200, json={"id": "job_1", "status": status, "result": result,
                                             "error": "out of memory" if status == "failed" else None})
        if req.url.path == "/sdcpp/v1/jobs/job_1/cancel":
            return httpx.Response(200, json={"cancel_requested": True})
        return httpx.Response(200, text="ok")

    srv = imagegen.ImageServer("http://img:8080", "http://docker:2375", "imagegen", 5, 0.3)
    srv.poll_interval = 0.01
    return srv, httpx.MockTransport(handler)


async def test_an_image_request_runs_as_a_native_job_and_answers_like_the_openai_route():
    calls = []
    srv, transport = job_server(["queued", "generating", "completed"], calls)
    async with httpx.AsyncClient(transport=transport) as c:
        resp = await srv.generate(c, {"prompt": "x", "init_image": "AAAA"})
    assert resp.status_code == 200 and resp.json() == {"data": [{"b64_json": "AAAA"}]}
    assert ("POST", "/sdcpp/v1/img_gen") in calls and ("POST", "/v1/images/generations") not in calls


async def test_a_failed_job_is_an_error_response():
    srv, transport = job_server(["failed"], [])
    async with httpx.AsyncClient(transport=transport) as c:
        resp = await srv.generate(c, {"prompt": "x", "init_image": "AAAA"})
    assert resp.status_code == 500 and "out of memory" in resp.text


async def test_a_job_that_outlives_the_timeout_is_cancelled():
    calls = []
    srv, transport = job_server(["generating"], calls)
    async with httpx.AsyncClient(transport=transport) as c:
        with pytest.raises(httpx.ReadTimeout):
            await srv.generate(c, {"prompt": "x", "init_image": "AAAA"})
    assert calls[-1] == ("POST", "/sdcpp/v1/jobs/job_1/cancel")


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
