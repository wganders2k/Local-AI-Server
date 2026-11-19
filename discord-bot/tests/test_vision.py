"""
Live image captions: the Captioner's cache and limits, and the rule that
captions reach the mimic controller but never a persona prompt.
"""

import io
from datetime import datetime, timezone
from types import SimpleNamespace

from PIL import Image

import vision
from mimic import controller
from mimic.controller import SPEAK, WAIT
from mimic.engine import run_burst
from mimic.transcript import Cast, Line
from mimic_format import Msg
from proxy_client import ProxyError
from tests.test_mimic_chat import ScriptedProxy

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
P1, P2 = "mimic_user1", "mimic_user2"


def png(size=(20, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, (0, 255, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


class FakeAttachment:
    def __init__(self, id, filename="a.png", content_type="image/png", data=None, size=100):
        self.id, self.filename, self.content_type, self.size = id, filename, content_type, size
        self.data = png() if data is None else data
        self.reads = 0

    async def read(self):
        self.reads += 1
        return self.data


class CaptionProxy:
    def __init__(self, answers=None, fail=False):
        self.calls = []
        self.answers = list(answers or [])
        self.fail = fail

    async def chat(self, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages, **kw})
        if self.fail:
            raise ProxyError("down")
        return self.answers.pop(0) if self.answers else f"caption {len(self.calls)}"


def test_is_image():
    assert vision.is_image(FakeAttachment(1))
    assert vision.is_image(FakeAttachment(1, "x.JPG", content_type=None))
    assert not vision.is_image(FakeAttachment(1, "clip.mp4", content_type="video/mp4"))
    assert not vision.is_image(FakeAttachment(1, "x.svg", content_type="image/svg+xml"))
    assert not vision.is_image(FakeAttachment(1, "notes.txt", content_type=None))


def test_encode_downscales_to_rgb_jpeg():
    url = vision.encode(png((4000, 1000)), max_side=1536)
    assert url.startswith("data:image/jpeg;base64,")


async def test_caption_is_cached_and_sent_without_thinking():
    proxy = CaptionProxy(["  a green\n square "])
    cap = vision.Captioner(proxy)
    att = FakeAttachment(7)
    assert await cap.caption(att, "gemma4-12b-mimic-u", lora=[{"id": 0, "scale": 0.0}]) == "a green square"
    assert await cap.caption(att, "gemma4-12b-mimic-u") == "a green square"
    assert len(proxy.calls) == 1 and att.reads == 1
    call = proxy.calls[0]
    assert call["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["lora"] == [{"id": 0, "scale": 0.0}]
    assert [p["type"] for p in call["messages"][0]["content"]] == ["image_url", "text"]


async def test_failures_and_oversize_give_none_and_are_not_cached():
    proxy = CaptionProxy(fail=True)
    cap = vision.Captioner(proxy)
    assert await cap.caption(FakeAttachment(1), "m") is None
    assert await cap.caption(FakeAttachment(2, size=10**9), "m") is None
    assert await cap.caption(FakeAttachment(3, data=b"junk"), "m") is None
    assert cap.cached(FakeAttachment(1)) is None
    assert len(proxy.calls) == 1  # oversize and unreadable never reach the proxy


async def test_cache_is_bounded():
    cap = vision.Captioner(CaptionProxy(), cache_size=2)
    for i in range(3):
        await cap.caption(FakeAttachment(i), "m")
    assert cap.cached(FakeAttachment(0)) is None
    assert cap.cached(FakeAttachment(2)) is not None


def test_with_captions():
    assert vision.with_captions("lol", ["a cat"]) == "lol [image: a cat]"
    assert vision.with_captions("", ["a", "b"]) == "[image: a] [image: b]"
    assert vision.with_captions("lol", []) == "lol"


def img_line(text, *atts, persona=None):
    return Line(Msg("22", "erin", T0, text, has_attachments=True), persona, images=tuple(atts))


async def test_annotate_newest_first_within_budget():
    proxy = CaptionProxy()
    cap = vision.Captioner(proxy)
    old, new = FakeAttachment(1), FakeAttachment(2)
    lines = [img_line("", old), Line(Msg("22", "erin", T0, "hm")), img_line("", new)]
    out = await vision.annotate(cap, lines, "gemma4-12b-mimic-u", max_new=1)
    assert out[2].captions == ("caption 1",) and out[0].captions == ()
    out = await vision.annotate(cap, out, "gemma4-12b-mimic-u", max_new=1)
    assert out[0].captions == ("caption 2",) and out[2].captions == ("caption 1",)
    assert await vision.annotate(None, lines, "m") is lines


def test_controller_sees_captions():
    line = Line(Msg("22", "erin", T0, "look", True), captions=("a cat on a keyboard",))
    msgs = controller.build_messages([P1], {P1: "x"}, "pizza?", [line], True, [SPEAK, WAIT], "", None)
    assert "[erin]: look [image: a cat on a keyboard]" in msgs[1]["content"]


async def test_persona_prompt_keeps_the_trained_marker(catalog):
    """The adapters were trained on "[attachment]"; captions must not reach them."""
    cast = Cast.build(catalog)
    proxy = ScriptedProxy([{"action": SPEAK, "speaker": cast.label(P1)}, {"action": WAIT}])
    cap = vision.Captioner(SimpleNamespace(chat=None))
    att = FakeAttachment(9)
    cap._cache[att.id] = "a cat on a keyboard"
    lines = [Line(Msg("22", "erin", T0, "", has_attachments=True), images=(att,))]

    await run_burst(proxy, cast, [P1, P2], "pizza?", lines, post=_ignore, captioner=cap)

    controller_text = proxy.controller_calls[0]["messages"][1]["content"]
    assert "[image: a cat on a keyboard]" in controller_text
    persona_text = " ".join(m["content"] for m in proxy.persona_calls[0]["messages"])
    assert "[attachment]" in persona_text
    assert "keyboard" not in persona_text and "image:" not in persona_text


async def _ignore(*a):
    pass
