"""Unit tests for image_captioner: encoding, proxy call, write-back."""
import base64
import io
import json
import os
import tempfile

import httpx
import pytest
from PIL import Image

import image_captioner
from jsonl_store import _user_id_cache, append_messages, load_all_records


@pytest.fixture
def archive(monkeypatch):
    _user_id_cache.clear()
    with tempfile.TemporaryDirectory() as root:
        monkeypatch.setattr("jsonl_store.ARCHIVE_DIR", os.path.join(root, "archive"))
        monkeypatch.setattr("image_captioner.ARCHIVE_MEDIA_DIR", os.path.join(root, "media"))
        os.makedirs(os.path.join(root, "media", "c"))
        yield root
    _user_id_cache.clear()


def write_image(root, rel, fmt="PNG", size=(40, 20), mode="RGBA"):
    path = os.path.join(root, rel)
    Image.new(mode, size, (255, 0, 0, 128) if mode == "RGBA" else 0).save(path, format=fmt)
    return rel


def proxy(answer, status=200, seen=None):
    def handler(req):
        if seen is not None:
            seen.append(json.loads(req.content))
        if status != 200:
            return httpx.Response(status, text="down")
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})
    return httpx.MockTransport(handler)


@pytest.fixture
def use_proxy(monkeypatch):
    real = httpx.Client

    def install(transport):
        monkeypatch.setattr("image_captioner.httpx.Client", lambda *a, **kw: real(*a, transport=transport, **kw))
    return install


def stored(aid, rel, **kw):
    a = {"attachment_id": aid, "caption_status": "pending", "media_status": "stored", "media_path": rel}
    a.update(kw)
    return a


@pytest.mark.parametrize("fmt", ["PNG", "WEBP", "GIF"])
def test_encode_image_is_rgb_jpeg_and_downscaled(archive, fmt):
    rel = write_image(archive, f"media/c/x.{fmt.lower()}", fmt=fmt, size=(3000, 1000),
                      mode="P" if fmt == "GIF" else "RGBA")
    url = image_captioner.encode_image(os.path.join(archive, rel), max_side=1536)
    assert url.startswith("data:image/jpeg;base64,")
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert im.format == "JPEG" and im.mode == "RGB" and max(im.size) == 1536


def test_batch_writes_caption_with_provenance(archive, use_proxy):
    rel = write_image(archive, "media/c/1.png")
    append_messages("u", [{"message_id": "1", "attachments": [stored("1", rel), stored("2", rel, media_status="gone")]}])
    seen = []
    use_proxy(proxy("  A red square.  ", seen=seen))
    assert image_captioner.process_pending_batch(10) == 1
    a1, a2 = load_all_records("u")[0]["attachments"]
    assert a1["caption"] == "A red square." and a1["caption_status"] == "done"
    assert a1["caption_model"] == image_captioner.IMAGE_CAPTION_MODEL
    assert a1["caption_prompt_version"] == image_captioner.PROMPT_VERSION
    assert a2["caption_status"] == "pending"
    [body] = seen
    assert body["model"] == image_captioner.IMAGE_CAPTION_MODEL
    kinds = [part["type"] for part in body["messages"][0]["content"]]
    assert kinds == ["image_url", "text"]


def test_failures_retry_then_give_up(archive, use_proxy, monkeypatch):
    monkeypatch.setattr("image_captioner.IMAGE_CAPTION_MAX_ATTEMPTS", 2)
    rel = write_image(archive, "media/c/1.png")
    append_messages("u", [{"message_id": "1", "attachments": [stored("1", rel)]}])
    use_proxy(proxy("", status=503))
    image_captioner.process_pending_batch(10)
    [a] = load_all_records("u")[0]["attachments"]
    assert a["caption_status"] == "pending" and a["caption_attempts"] == 1
    image_captioner.process_pending_batch(10)
    [a] = load_all_records("u")[0]["attachments"]
    assert a["caption_status"] == "error"
    assert image_captioner.process_pending_batch(10) == 0


def test_unreadable_file_is_marked(archive, use_proxy):
    with open(os.path.join(archive, "media/c/bad.png"), "wb") as f:
        f.write(b"not an image")
    append_messages("u", [{"message_id": "1", "attachments": [stored("1", "media/c/bad.png")]}])
    use_proxy(proxy("never called"))
    image_captioner.process_pending_batch(10)
    assert load_all_records("u")[0]["attachments"][0]["caption_status"] == "unreadable"
