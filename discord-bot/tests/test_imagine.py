"""/imagine: text or image-to-image, rate capped, and failures reported in the channel."""

import base64
import json
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


def interaction(uid=1, guild_id=None):
    return SimpleNamespace(
        user=SimpleNamespace(id=uid, name="erin", display_name="Erin"), guild_id=guild_id,
        response=FakeResponse(), followup=FakeFollowup(),
    )


class FakeProxy:
    """`rewrite` is what the rewriter answers; None makes its request fail, so the original is drawn."""

    def __init__(self, fail=False, rewrite=None):
        self.calls, self.counts, self.models, self.steps, self.fail = [], [], [], [], fail
        self.images, self.cfgs = [], []
        self.rewrite, self.chats = rewrite, []

    async def chat(self, model, messages, **kw):
        self.chats.append((model, messages, kw))
        if self.rewrite is None:
            raise ProxyError("mimic-multi returned error 503: busy")
        return self.rewrite

    async def generate_image(self, prompt, size, n=1, model="default", steps=None, image=None, strength=None,
                             cfg=None):
        self.cfgs.append(cfg)
        self.calls.append((prompt, size))
        self.images.append((image, strength))
        self.models.append(model)
        self.steps.append(steps)
        self.counts.append(n)
        if self.fail:
            raise ProxyError("Image generation returned error 503: busy")
        return [b"\x89PNG..."] * n

    async def get_queue_depth(self):
        return 0


BASE = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]


def undiscovered(key):
    raise LookupError("not discovered")


def cog(proxy, lora=lambda key: BASE):
    services = SimpleNamespace(
        proxy=proxy, limiter=SimpleNamespace(is_allowed=lambda uid: True), personas=SimpleNamespace(lora=lora),
    )
    return imagine.ImagineCog(SimpleNamespace(services=services))


async def test_the_command_options():
    params = {p.name for p in imagine.ImagineCog.imagine.parameters}
    assert params == {"prompt", "rewrite", "count", "style", "mood", "framing", "shape", "effort", "image", "strength",
                      "cfg"}


def attachment(content_type="image/png", size=1000, width=1600, height=900, data=b"\x89PNG..."):
    async def read():
        return data
    return SimpleNamespace(content_type=content_type, filename="pic.png", size=size, width=width, height=height,
                           url="https://cdn.example/pic.png", read=read)


def test_an_image_gets_the_shape_nearest_its_own():
    assert imagine.nearest_size(1600, 900) == "1216x832"
    assert imagine.nearest_size(900, 1600) == "832x1216"
    assert imagine.nearest_size(1000, 1050) == "1024x1024"
    assert imagine.nearest_size(None, None) == "1024x1024"


async def test_an_image_is_sent_with_the_default_strength_and_its_own_shape():
    proxy = FakeProxy()
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "as a watercolor", rewrite=False, image=attachment())

    assert proxy.images == [(b"\x89PNG...", imagine.IMAGINE_STRENGTHS["Medium"])]
    assert proxy.calls == [("as a watercolor", "1216x832")]
    embed = i.followup.sent[0][1]["embed"]
    assert embed.thumbnail.url == "https://cdn.example/pic.png"
    assert embed.footer.text.split(" · ")[3] == "from image, medium"


async def test_a_chosen_shape_and_strength_win():
    proxy = FakeProxy()
    heavy = next(c for c in imagine._STRENGTH_CHOICES if c.name == "Heavy")
    portrait = next(c for c in imagine._SHAPE_CHOICES if c.name == "Portrait")

    await imagine.ImagineCog.imagine.callback(cog(proxy), interaction(), "x", rewrite=False, image=attachment(),
                                              strength=heavy, shape=portrait)

    assert proxy.images == [(b"\x89PNG...", 0.85)] and proxy.calls[0][1] == "832x1216"


async def test_cfg_is_passed_on_and_shown_only_when_set():
    proxy = FakeProxy()
    plain, guided = interaction(), interaction()
    await imagine.ImagineCog.imagine.callback(cog(proxy), plain, "x", rewrite=False)
    await imagine.ImagineCog.imagine.callback(cog(proxy), guided, "x", rewrite=False, cfg=2.5)
    assert proxy.cfgs == [None, 2.5]
    assert "cfg" not in plain.followup.sent[0][1]["embed"].footer.text
    assert " · cfg 2.5 · " in guided.followup.sent[0][1]["embed"].footer.text


def test_the_cfg_option_offers_the_proxys_range():
    cfg = next(p for p in imagine.ImagineCog.imagine.parameters if p.name == "cfg")
    assert (cfg.min_value, cfg.max_value) == (1.0, 7.0) and not cfg.required


async def test_without_an_image_strength_is_ignored():
    proxy = FakeProxy()
    heavy = next(c for c in imagine._STRENGTH_CHOICES if c.name == "Heavy")
    i = interaction()
    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "x", rewrite=False, strength=heavy)
    assert proxy.images == [(None, None)] and "from image" not in i.followup.sent[0][1]["embed"].footer.text


@pytest.mark.parametrize("bad", [attachment(content_type="text/plain"), attachment(size=16 * 1024 * 1024)])
async def test_a_bad_attachment_is_refused_before_anything_runs(bad):
    proxy = FakeProxy()
    i = interaction()
    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "x", rewrite=False, image=bad)
    assert proxy.calls == [] and not i.response.deferred and i.response.messages[0][1] is True


def test_prompt_and_rewrite_are_the_required_options_and_come_first():
    params = imagine.ImagineCog.imagine.parameters
    assert [(p.name, p.required) for p in params[:2]] == [("prompt", True), ("rewrite", True)]
    assert not any(p.required for p in params[2:])
    assert params[2].name == "count" and (params[2].min_value, params[2].max_value) == (1, 3)


async def test_count_asks_for_that_many_and_attaches_them_all():
    proxy = FakeProxy()
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=False, count=3)

    assert proxy.calls == [("a goose", "1024x1024")] and proxy.counts == [3]
    (_, kw), = i.followup.sent
    assert [f.filename for f in kw["files"]] == ["imagine_1.png", "imagine_2.png", "imagine_3.png"]
    assert kw["embed"].footer.text.split(" · ")[-1].startswith("3 images ")


async def test_client_sends_n_and_returns_every_image():
    pngs = [b"one", b"two"]

    def handler(req):
        assert json.loads(req.content)["n"] == 2
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(p).decode()} for p in pngs]})

    client = ProxyClient("http://proxy")
    client._client = httpx.AsyncClient(base_url="http://proxy", transport=httpx.MockTransport(handler))
    assert await client.generate_image("x", "1024x1024", 2) == pngs


async def test_client_sends_the_image_as_base64_with_its_strength():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"out").decode()}]})

    client = ProxyClient("http://proxy")
    client._client = httpx.AsyncClient(base_url="http://proxy", transport=httpx.MockTransport(handler))
    await client.generate_image("x", "1024x1024", image=b"pic", strength=0.45)
    await client.generate_image("x", "1024x1024")
    assert sent[0]["image"] == base64.b64encode(b"pic").decode() and sent[0]["strength"] == 0.45
    assert "image" not in sent[1] and "strength" not in sent[1]


async def test_generates_and_attaches_the_image():
    proxy = FakeProxy()
    i = interaction()
    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=False)
    assert proxy.calls == [("a goose", "1024x1024")]
    (_, kw), = i.followup.sent
    assert [f.filename for f in kw["files"]] == ["imagine.png"]
    assert kw["embed"].image.url == "attachment://imagine.png"
    assert kw["embed"].description == "a goose"


async def test_shape_choice_sets_the_size():
    proxy = FakeProxy()
    shape = SimpleNamespace(value="832x1216")
    await imagine.ImagineCog.imagine.callback(cog(proxy), interaction(), "a goose", shape=shape, rewrite=False)
    assert proxy.calls == [("a goose", "832x1216")]


async def test_there_is_no_hourly_image_cap():
    proxy = FakeProxy()
    c = cog(proxy)
    for n in range(12):
        await imagine.ImagineCog.imagine.callback(c, interaction(), f"goose {n}", rewrite=False)
    assert len(proxy.calls) == 12


async def test_failure_is_reported():
    i = interaction()
    await imagine.ImagineCog.imagine.callback(cog(FakeProxy(fail=True)), i, "a goose", rewrite=False)
    (content, _), = i.followup.sent
    assert content.startswith("⚠️")


async def test_client_decodes_the_image(monkeypatch):
    png = b"\x89PNG\r\n"

    def handler(req):
        assert req.url.path == "/v1/images/generations"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    client = ProxyClient("http://proxy")
    client._client = httpx.AsyncClient(base_url="http://proxy", transport=httpx.MockTransport(handler))
    assert await client.generate_image("x", "1024x1024") == [png]
    client._client = httpx.AsyncClient(base_url="http://proxy",
                                       transport=httpx.MockTransport(lambda r: httpx.Response(403, text="no")))
    with pytest.raises(ProxyError):
        await client.generate_image("x", "1024x1024")


async def test_the_rewrite_is_drawn_and_shown_under_the_original():
    proxy = FakeProxy(rewrite='"A goose in a top hat, oil painting, warm light."')
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=True)

    assert proxy.calls == [("A goose in a top hat, oil painting, warm light.", "1024x1024")]
    model, messages, kw = proxy.chats[0]
    assert model == "mimic-multi" and kw["lora"] == BASE
    assert "text-to-image model" in messages[0]["content"] and messages[1] == {"role": "user", "content": "a goose"}
    (_, sent), = i.followup.sent
    assert sent["embed"].description == "a goose"
    assert sent["embed"].fields[0].value == "A goose in a top hat, oil painting, warm light."


async def test_rewrite_false_sends_the_prompt_as_typed():
    proxy = FakeProxy(rewrite="anything")
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=False)

    assert proxy.chats == [] and proxy.calls == [("a goose", "1024x1024")]
    assert not i.followup.sent[0][1]["embed"].fields


@pytest.mark.parametrize("reply", [None, "", "I can't help with that.", "Sorry, no.", "x" * 1501])
async def test_an_unusable_rewrite_falls_back_to_the_original(reply):
    proxy = FakeProxy(rewrite=reply)
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=True)

    assert proxy.calls == [("a goose", "1024x1024")]
    assert not i.followup.sent[0][1]["embed"].fields


async def test_before_discovery_the_rewrite_sends_no_adapter_scales():
    proxy = FakeProxy(rewrite="a goose, detailed")

    await imagine.ImagineCog.imagine.callback(cog(proxy, lora=undiscovered), interaction(), "a goose", rewrite=True)

    assert proxy.chats[0][2]["lora"] is None
    assert proxy.calls == [("a goose, detailed", "1024x1024")]


async def test_details_follow_the_prompt_as_labelled_sentences():
    proxy = FakeProxy(rewrite="a goose, painted")
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", style="oil painting.", framing="close-up", rewrite=True)

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


def test_the_footer_times_each_step_and_omits_a_rewrite_not_asked_for():
    assert imagine.footer_text("ann", "Model A", "1024x1024", None, 6.44) == (
        "Requested by ann · Model A · 1024x1024 · image 6.4 s")
    assert imagine.footer_text("ann", "Model A", "832x1216", 1.25, 16.0).endswith(
        "832x1216 · rewrite 1.2 s · image 16.0 s")


async def test_effort_sets_the_steps_and_only_high_shows_in_the_footer():
    proxy = FakeProxy()
    low, high = interaction(), interaction()
    high_effort = next(c for c in imagine._EFFORT_CHOICES if c.name == "High")

    await imagine.ImagineCog.imagine.callback(cog(proxy), low, "a goose", rewrite=False)
    await imagine.ImagineCog.imagine.callback(cog(proxy), high, "a goose", rewrite=False, effort=high_effort)

    assert proxy.steps == [8, 16]
    assert " steps" not in low.followup.sent[0][1]["embed"].footer.text
    assert " · 16 steps · " in high.followup.sent[0][1]["embed"].footer.text


async def test_a_listed_server_draws_with_its_own_model(monkeypatch):
    monkeypatch.setattr(imagine, "IMAGINE_GUILD_MODELS", {42: "alt"})
    monkeypatch.setattr(imagine, "IMAGINE_MODEL_LABELS", {"default": "Model A", "alt": "Model B"})
    proxy = FakeProxy()
    listed, other = interaction(guild_id=42), interaction(guild_id=7)

    await imagine.ImagineCog.imagine.callback(cog(proxy), listed, "a goose", rewrite=False)
    await imagine.ImagineCog.imagine.callback(cog(proxy), other, "a goose", rewrite=False)

    assert proxy.models == ["alt", "default"]
    assert " · Model B · " in listed.followup.sent[0][1]["embed"].footer.text
    assert " · Model A · " in other.followup.sent[0][1]["embed"].footer.text


async def test_the_sent_footer_carries_the_timings():
    proxy = FakeProxy(rewrite="a goose, detailed")
    i = interaction()

    await imagine.ImagineCog.imagine.callback(cog(proxy), i, "a goose", rewrite=True)

    footer = i.followup.sent[0][1]["embed"].footer.text
    assert " · rewrite " in footer and footer.endswith(" s") and " · image " in footer
