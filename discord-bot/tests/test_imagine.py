"""/imagine: text only, rate capped, and failures reported in the channel."""

import base64
from types import SimpleNamespace

import httpx
import pytest

import cogs.imagine as imagine
from proxy_client import ProxyClient, ProxyError


class FakeResponse:
    def __init__(self):
        self.messages, self.deferred = [], False

    async def send_message(self, text, ephemeral=False):
        self.messages.append((text, ephemeral))

    async def defer(self, thinking=False):
        self.deferred = True


class FakeFollowup:
    def __init__(self):
        self.sent = []

    async def send(self, content=None, **kw):
        self.sent.append((content, kw))


def interaction(uid=1):
    return SimpleNamespace(
        user=SimpleNamespace(id=uid, name="erin", display_name="Erin"),
        response=FakeResponse(), followup=FakeFollowup(),
    )


class FakeProxy:
    """`rewrite` is what the rewriter answers; None makes its request fail, so the original is drawn."""

    def __init__(self, fail=False, rewrite=None):
        self.calls, self.fail = [], fail
        self.rewrite, self.chats = rewrite, []

    async def chat(self, model, messages, **kw):
        self.chats.append((model, messages, kw))
        if self.rewrite is None:
            raise ProxyError("mimic-multi returned error 503: busy")
        return self.rewrite

    async def generate_image(self, prompt, size):
        self.calls.append((prompt, size))
        if self.fail:
            raise ProxyError("Image generation returned error 503: busy")
        return b"\x89PNG..."

    async def get_queue_depth(self):
        return 0


BASE = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]


def undiscovered(key):
    raise LookupError("not discovered")


def cog(proxy, per_hour=10, lora=lambda key: BASE):
    services = SimpleNamespace(
        proxy=proxy, limiter=SimpleNamespace(is_allowed=lambda uid: True), personas=SimpleNamespace(lora=lora),
    )
    c = imagine.ImagineCog(SimpleNamespace(services=services))
    c.hourly.max_requests = per_hour
    return c


async def test_the_command_takes_text_only():
    params = {p.name for p in imagine.ImagineCog.imagine.parameters}
    assert params == {"prompt", "style", "mood", "framing", "shape", "rewrite"}


async def test_generates_and_attaches_the_image():
    proxy = FakeProxy()
    i = interaction()
    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose")
    assert proxy.calls == [("a goose", "1024x1024")]
    (_, kw), = i.followup.sent
    assert kw["file"].filename == "imagine.png"
    assert kw["embed"].description == "a goose"


async def test_shape_choice_sets_the_size():
    proxy = FakeProxy()
    shape = SimpleNamespace(value="832x1216")
    await imagine.ImagineCog.imagine.callback(cog(proxy), interaction(), "a goose", shape=shape)
    assert proxy.calls == [("a goose", "832x1216")]


async def test_hourly_cap_refuses_without_generating():
    proxy = FakeProxy()
    c = cog(proxy, per_hour=1)
    await imagine.ImagineCog.imagine.callback(c, interaction(), "one")
    i = interaction()
    await imagine.ImagineCog.imagine.callback(c, i, "two")
    assert len(proxy.calls) == 1
    assert i.response.messages and i.response.messages[0][1] is True  # ephemeral refusal


async def test_failure_is_reported():
    i = interaction()
    await imagine.ImagineCog.imagine.callback(cog(FakeProxy(fail=True)), i, "a goose")
    (content, _), = i.followup.sent
    assert content.startswith("⚠️")


async def test_client_decodes_the_image(monkeypatch):
    png = b"\x89PNG\r\n"

    def handler(req):
        assert req.url.path == "/v1/images/generations"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    client = ProxyClient("http://proxy")
    client._client = httpx.AsyncClient(base_url="http://proxy", transport=httpx.MockTransport(handler))
    assert await client.generate_image("x", "1024x1024") == png
    client._client = httpx.AsyncClient(base_url="http://proxy",
                                       transport=httpx.MockTransport(lambda r: httpx.Response(403, text="no")))
    with pytest.raises(ProxyError):
        await client.generate_image("x", "1024x1024")


async def test_the_rewrite_is_drawn_and_shown_under_the_original():
    proxy = FakeProxy(rewrite='"A goose in a top hat, oil painting, warm light."')
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose")

    assert proxy.calls == [("A goose in a top hat, oil painting, warm light.", "1024x1024")]
    model, messages, kw = proxy.chats[0]
    assert model == "mimic-multi" and kw["lora"] == BASE
    assert "Z-Image" in messages[0]["content"] and messages[1] == {"role": "user", "content": "a goose"}
    (_, sent), = i.followup.sent
    assert sent["embed"].description == "a goose"
    assert sent["embed"].fields[0].value == "A goose in a top hat, oil painting, warm light."


async def test_rewrite_off_sends_the_prompt_as_typed():
    proxy = FakeProxy(rewrite="anything")
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=False)

    assert proxy.chats == [] and proxy.calls == [("a goose", "1024x1024")]
    assert not i.followup.sent[0][1]["embed"].fields


@pytest.mark.parametrize("reply", [None, "", "I can't help with that.", "Sorry, no.", "x" * 1501])
async def test_an_unusable_rewrite_falls_back_to_the_original(reply):
    proxy = FakeProxy(rewrite=reply)
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose")

    assert proxy.calls == [("a goose", "1024x1024")]
    assert not i.followup.sent[0][1]["embed"].fields


async def test_before_discovery_the_rewrite_sends_no_adapter_scales():
    proxy = FakeProxy(rewrite="a goose, detailed")

    await imagine.ImagineCog.imagine.callback(cog(proxy, lora=undiscovered), interaction(), "a goose")

    assert proxy.chats[0][2]["lora"] is None
    assert proxy.calls == [("a goose, detailed", "1024x1024")]


async def test_details_follow_the_prompt_as_labelled_sentences():
    proxy = FakeProxy(rewrite="a goose, painted")
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", style="oil painting.", framing="close-up")

    sent = "a goose. Art style: oil painting. Framing: close-up."
    assert proxy.chats[0][1][1]["content"] == sent
    assert i.followup.sent[0][1]["embed"].description == sent


async def test_details_reach_the_image_unrewritten_when_rewrite_is_off():
    proxy = FakeProxy()

    await imagine.ImagineCog.imagine.callback(cog(proxy), interaction(), "a goose...", mood="eerie fog", rewrite=False)

    assert proxy.calls == [("a goose. Mood and lighting: eerie fog.", "1024x1024")]


def test_without_details_the_prompt_is_untouched():
    assert imagine.with_details("a goose....  ") == "a goose....  "


def test_every_description_fits_discords_limit_and_gives_examples():
    for p in imagine.ImagineCog.imagine.parameters:
        assert len(p.description) <= 100 and "e.g." in p.description, p.name
