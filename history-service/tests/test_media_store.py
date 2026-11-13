"""Unit tests for media_store: link refresh, download outcomes, write-back."""
import json
import os
import tempfile

import httpx
import pytest

import media_store
from jsonl_store import _user_id_cache, append_messages, load_all_records


@pytest.fixture
def archive(monkeypatch):
    _user_id_cache.clear()
    with tempfile.TemporaryDirectory() as root:
        monkeypatch.setattr("jsonl_store.ARCHIVE_DIR", os.path.join(root, "archive"))
        monkeypatch.setattr("media_store.ARCHIVE_MEDIA_DIR", os.path.join(root, "media"))
        yield root
    _user_id_cache.clear()


def att(aid, url, name="p.png", **kw):
    a = {"attachment_id": aid, "url": url, "filename": name, "content_type": "image/png", "caption_status": "pending"}
    a.update(kw)
    return a


def fake_discord(refreshed: dict, files: dict, refresh_status=200):
    """refreshed: original -> fresh url (absent = deleted). files: fresh url -> (status, bytes)."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/attachments/refresh-urls"):
            if refresh_status != 200:
                return httpx.Response(refresh_status, text="nope")
            urls = json.loads(req.content)["attachment_urls"]
            return httpx.Response(200, json={"refreshed_urls": [
                {"original": u, "refreshed": refreshed[u]} for u in urls if u in refreshed]})
        status, body = files.get(str(req.url), (404, b""))
        return httpx.Response(status, content=body)
    return httpx.MockTransport(handler)


@pytest.fixture
def transport(monkeypatch):
    holder = {}
    real = httpx.Client

    def client(*a, **kw):
        kw["transport"] = holder["t"]
        return real(*a, **kw)

    monkeypatch.setattr("media_store.httpx.Client", client)
    return holder


def test_needs_fetch():
    assert media_store.needs_fetch({"content_type": "image/png"})
    assert not media_store.needs_fetch({"content_type": ""})
    assert not media_store.needs_fetch({"content_type": "image/png", "media_status": "stored"})
    assert not media_store.needs_fetch({"content_type": "image/png", "media_status": "gone"})
    assert not media_store.needs_fetch({"content_type": "image/png", "media_attempts": 3})


def test_media_relpath():
    assert media_store.media_relpath("c1", "a9", "Shot.PNG") == "media/c1/a9.png"


def test_fetch_batch_outcomes(archive, transport):
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [
        att("1", "https://cdn/ok"), att("2", "https://cdn/deleted"), att("3", "https://cdn/dead"),
        att("4", "https://cdn/x.mp4", name="x.mp4", content_type="", caption_status="skipped"),
    ]}])
    transport["t"] = fake_discord(
        {"https://cdn/ok": "https://cdn/ok?fresh", "https://cdn/dead": "https://cdn/dead?fresh"},
        {"https://cdn/ok?fresh": (200, b"PNGDATA"), "https://cdn/dead?fresh": (404, b"")},
    )
    assert media_store.fetch_batch(10) == 3
    a1, a2, a3, a4 = load_all_records("u")[0]["attachments"]
    assert a1["media_status"] == "stored" and a1["media_path"] == "media/c/1.png"
    with open(os.path.join(archive, a1["media_path"]), "rb") as f:
        assert f.read() == b"PNGDATA"
    assert a2["media_status"] == "gone"
    assert a3["media_status"] == "gone"
    assert "media_status" not in a4
    assert media_store.fetch_batch(10) == 0  # nothing left to try


def test_failed_refresh_is_retried_not_gone(archive, transport):
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att("1", "https://cdn/ok")]}])
    transport["t"] = fake_discord({}, {}, refresh_status=500)
    media_store.fetch_batch(10)
    [a] = load_all_records("u")[0]["attachments"]
    assert a["media_status"] == "error" and a["media_attempts"] == 1
    assert media_store.needs_fetch(a)


def test_oversized_download_is_discarded(archive, transport, monkeypatch):
    monkeypatch.setattr("media_store.MEDIA_FETCH_MAX_FILE_SIZE_MB", 0)
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att("1", "https://cdn/big")]}])
    transport["t"] = fake_discord({"https://cdn/big": "https://cdn/big?f"}, {"https://cdn/big?f": (200, b"x" * 10)})
    media_store.fetch_batch(10)
    [a] = load_all_records("u")[0]["attachments"]
    assert a["media_status"] == "too_large"
    assert not os.path.exists(os.path.join(archive, "media", "c", "1.png"))
