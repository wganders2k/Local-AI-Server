"""
The quarterly refresh: a full re-export replaces the archive, deleted
messages drop out, and captions/media carry over by attachment id.
"""
import json
import os

import pytest

import channel_state
import dce_orchestrator
import jsonl_store
from dce_parser import ATTACHMENT_PARSER_FIELDS


def _rec(mid, uid="u1", atts=None, content="hi"):
    return {"message_id": mid, "user_id": uid, "timestamp": f"2026-01-01T00:00:{mid[-2:]}",
            "content": content, "attachments": atts or []}


def _att(aid, **extra):
    return {"attachment_id": aid, "url": f"https://cdn/{aid}?new", "filename": "a.png",
            "content_type": "image/png", "file_size_bytes": 1, "caption_status": "pending", **extra}


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    archive, raw, state = tmp_path / "archive", tmp_path / "raw", tmp_path / "state"
    for d in (archive, raw, state):
        d.mkdir()
    monkeypatch.setattr(jsonl_store, "ARCHIVE_DIR", str(archive))
    monkeypatch.setattr(dce_orchestrator, "HOST_RAW_DIR", str(raw))
    monkeypatch.setattr(dce_orchestrator, "ARCHIVE_PREV_DIR", str(tmp_path / "archive_prev"))
    monkeypatch.setattr(dce_orchestrator, "RAW_PREV_DIR", str(tmp_path / "raw_prev"))
    monkeypatch.setattr(dce_orchestrator, "EXCLUDED_CHANNELS", ["skipme"])
    monkeypatch.setattr(channel_state, "ARCHIVE_STATE_DIR", str(state))
    monkeypatch.setattr(channel_state, "CHANNEL_STATE_FILE", str(state / "channel_state.json"))
    jsonl_store._user_id_cache.clear()
    return tmp_path


class TestCarryAttachmentState:
    def test_caption_and_media_carry_parser_fields_stay_fresh(self):
        old = {"a1": _att("a1", url="https://cdn/a1?old", caption_status="done", caption="a cat",
                          media_status="stored", media_path="media/c/a1.png")}
        recs = [_rec("m01", atts=[_att("a1")])]
        assert jsonl_store.carry_attachment_state(recs, old, ATTACHMENT_PARSER_FIELDS) == 1
        att = recs[0]["attachments"][0]
        assert att["caption"] == "a cat" and att["caption_status"] == "done"
        assert att["media_path"] == "media/c/a1.png"
        assert att["url"] == "https://cdn/a1?new"

    def test_parser_decides_skipped_and_pending(self):
        old = {"a1": _att("a1", caption_status="skipped")}
        recs = [_rec("m01", atts=[_att("a1")])]
        jsonl_store.carry_attachment_state(recs, old, ATTACHMENT_PARSER_FIELDS)
        assert recs[0]["attachments"][0]["caption_status"] == "pending"

    def test_new_attachment_untouched(self):
        recs = [_rec("m01", atts=[_att("a9")])]
        assert jsonl_store.carry_attachment_state(recs, {}, ATTACHMENT_PARSER_FIELDS) == 0
        assert "caption" not in recs[0]["attachments"][0]


class TestRefreshArchive:
    def _setup_old(self, dirs):
        jsonl_store.rewrite_archive("u1", [
            _rec("m01", atts=[_att("a1", caption_status="done", caption="a cat", media_status="stored")]),
            _rec("m02", content="deleted later"),
        ])
        jsonl_store.rewrite_archive("gone", [_rec("m03", uid="gone")])
        (dirs / "raw" / "20260829T000000Z").mkdir()

    def _refresh(self, monkeypatch, dirs, parsed, fail=()):
        exported = []

        def fake_export(channel_id, after=None, before=None):
            exported.append((channel_id, after))
            if channel_id in fail:
                return None
            d = dirs / "raw" / f"new-{channel_id}"
            d.mkdir(exist_ok=True)
            return f"/out/new-{channel_id}"

        monkeypatch.setattr(dce_orchestrator, "export_channel", fake_export)
        monkeypatch.setattr(dce_orchestrator, "parse_dce_export_directory",
                            lambda d: parsed if d.endswith("new-c1") else {})
        channels = [{"id": "c1", "name": "general", "last_message_at": "2026-11-01T00:00:00+00:00"},
                    {"id": "c2", "name": "skipme"}]
        return exported, dce_orchestrator.refresh_archive(channels)

    def test_replaces_archive_and_keeps_one_copy(self, monkeypatch, dirs):
        self._setup_old(dirs)
        parsed = {"u1": [_rec("m01", atts=[_att("a1")], content="edited"), _rec("m04")]}
        exported, result = self._refresh(monkeypatch, dirs, parsed)

        assert [c for c, _ in exported] == ["c1"]          # excluded channel never exported
        assert all(after is None for _, after in exported)  # full export, no date range
        recs = {r["message_id"]: r for r in _read(dirs / "archive" / "u1.jsonl")}
        assert set(recs) == {"m01", "m04"}                  # m02 deleted on Discord
        assert recs["m01"]["content"] == "edited"
        assert recs["m01"]["attachments"][0]["caption"] == "a cat"
        assert not (dirs / "archive" / "gone.jsonl").exists()
        assert (dirs / "archive_prev" / "gone.jsonl").exists()
        assert len(_read(dirs / "archive_prev" / "u1.jsonl")) == 2
        assert os.listdir(dirs / "raw") == ["new-c1"]
        assert os.listdir(dirs / "raw_prev") == ["20260829T000000Z"]
        state = channel_state.load_state()
        assert set(state) == {"c1"}
        assert result["attachments_carried"] == 1

    def test_failed_export_changes_nothing(self, monkeypatch, dirs):
        self._setup_old(dirs)
        monkeypatch.setattr(dce_orchestrator, "EXCLUDED_CHANNELS", [])
        with pytest.raises(RuntimeError):
            self._refresh(monkeypatch, dirs, {}, fail=("c2",))
        assert len(_read(dirs / "archive" / "u1.jsonl")) == 2
        assert os.listdir(dirs / "raw") == ["20260829T000000Z"]   # partial new export removed
        assert not (dirs / "archive_prev").exists()
