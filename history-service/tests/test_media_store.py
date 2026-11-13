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


@pytest.fixture(autouse=True)
def fresh_backoff(monkeypatch):
    monkeypatch.setattr(media_store, "_backoff", media_store._Backoff())


def fake_discord(refreshed: dict, files: dict, refresh_status=200, poison=(), calls=None):
    """
    refreshed: original -> fresh url (absent = deleted). files: fresh url -> (status, bytes).
    poison: links that make Discord reject (400) any call containing them.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/attachments/refresh-urls"):
            if calls is not None:
                calls.append(req)
            if refresh_status != 200:
                return httpx.Response(refresh_status, text="nope")
            urls = json.loads(req.content)["attachment_urls"]
            if any(u in poison for u in urls):
                return httpx.Response(400, text="invalid url")
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


@pytest.mark.parametrize("status", [401, 500, 503, 429])
def test_a_failing_refresh_service_records_nothing_and_backs_off(archive, transport, monkeypatch, status):
    """A bad token or a Discord outage is nobody's image's fault: no attempt is spent."""
    monkeypatch.setattr("media_store.time.sleep", lambda s: None)
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att("1", "https://cdn/ok")]}])
    calls = []
    transport["t"] = fake_discord({}, {}, refresh_status=status, calls=calls)
    assert media_store.fetch_batch(10) == 0
    [a] = load_all_records("u")[0]["attachments"]
    assert "media_status" not in a and "media_attempts" not in a

    # Backing off: the next run makes no call at all.
    n = len(calls)
    assert media_store.fetch_batch(10) == 0
    assert len(calls) == n


def test_a_network_error_is_a_service_failure_not_a_crash(archive, transport):
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att("1", "https://cdn/ok")]}])

    def down(req):
        raise httpx.ConnectError("no route")
    transport["t"] = httpx.MockTransport(down)
    assert media_store.fetch_batch(10) == 0
    assert "media_attempts" not in load_all_records("u")[0]["attachments"][0]


def test_backoff_doubles_to_a_cap_and_resets_on_success(monkeypatch):
    b = media_store._Backoff()
    delays = [b.failed() for _ in range(8)]
    assert delays[:3] == [120.0, 240.0, 480.0] and delays[-1] == media_store.BACKOFF_MAX_S
    b.succeeded()
    assert not b.active() and b.failed() == 120.0


def test_one_rejected_link_costs_only_itself(archive, transport):
    """A 400 caused by one link is split down to that link; the others are fetched."""
    urls = [f"https://cdn/{i}" for i in range(5)]
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att(str(i), u) for i, u in enumerate(urls)]}])
    transport["t"] = fake_discord(
        {u: u + "?fresh" for u in urls}, {u + "?fresh": (200, b"IMG") for u in urls}, poison={urls[3]},
    )
    media_store.fetch_batch(10)
    atts = load_all_records("u")[0]["attachments"]
    assert [a["media_status"] for a in atts] == ["stored", "stored", "stored", "error", "stored"]
    assert atts[3]["media_attempts"] == 1


def test_a_link_rejected_every_time_is_given_up_on(archive, transport, monkeypatch):
    """Bounded: the rejected link stops being tried after MEDIA_FETCH_MAX_ATTEMPTS."""
    monkeypatch.setattr("media_store.MEDIA_FETCH_MAX_ATTEMPTS", 2)
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att("1", "https://cdn/bad")]}])
    transport["t"] = fake_discord({}, {}, poison={"https://cdn/bad"})
    media_store.fetch_batch(10)
    media_store.fetch_batch(10)
    [a] = load_all_records("u")[0]["attachments"]
    assert a["media_status"] == "error" and a["media_attempts"] == 2
    assert not media_store.needs_fetch(a)
    assert media_store.fetch_batch(10) == 0


def test_oversized_download_is_discarded(archive, transport, monkeypatch):
    monkeypatch.setattr("media_store.MEDIA_FETCH_MAX_FILE_SIZE_MB", 0)
    append_messages("u", [{"message_id": "1", "channel_id": "c", "attachments": [att("1", "https://cdn/big")]}])
    transport["t"] = fake_discord({"https://cdn/big": "https://cdn/big?f"}, {"https://cdn/big?f": (200, b"x" * 10)})
    media_store.fetch_batch(10)
    [a] = load_all_records("u")[0]["attachments"]
    assert a["media_status"] == "too_large"
    assert not os.path.exists(os.path.join(archive, "media", "c", "1.png"))
