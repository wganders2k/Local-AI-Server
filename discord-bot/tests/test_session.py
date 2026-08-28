"""LoreSession's append-only transcript, and the store behind it."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from lore.session import LoreSession, LoreSessionStore, chunk_key


def make_session(**kw) -> LoreSession:
    return LoreSession(
        thread_id=kw.pop("thread_id", 1),
        system_prompt=kw.pop("system_prompt", "SYS"),
        original_question=kw.pop("original_question", "q"),
        pinned_now=kw.pop("pinned_now", "PINNED"),
        **kw,
    )


def test_chunk_key_is_stable_and_whitespace_insensitive():
    assert chunk_key("  text  ") == chunk_key("text")
    assert chunk_key("a") != chunk_key("b")


def test_build_messages_projects_away_the_kind_tag():
    # "kind" is bookkeeping for compaction; sending it to the backend is a
    # protocol error waiting to happen.
    s = make_session()
    s.append_question("q")
    s.append_research("r")
    s.append_answer("a")
    msgs = s.build_messages()
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert all(set(m) == {"role", "content"} for m in msgs)


def test_extra_messages_are_appended_but_not_stored():
    s = make_session()
    s.append_question("q")
    msgs = s.build_messages(extra=[{"role": "user", "content": "nudge"}])
    assert msgs[-1]["content"] == "nudge"
    assert len(s.transcript) == 1  # the nudge did not join the transcript


def test_counters_track_turns_and_searches():
    s = make_session()
    s.append_question("q")
    s.append_research("r")
    s.append_answer("a")
    assert (s.turns, s.searches) == (1, 1)


def test_research_indices_finds_only_research():
    s = make_session()
    s.append_question("q")
    s.append_research("r")
    s.append_answer("a")
    s.append_question("q2")
    s.append_research("r2")
    assert s.research_indices() == [1, 4]


def test_record_context_tracks_the_peak():
    s = make_session()
    s.record_context(100)
    s.record_context(50)
    assert s.last_context_tokens == 50
    assert s.peak_context_tokens == 100


def test_idle_seconds_on_a_corrupt_timestamp_is_infinite():
    # A corrupt entry must be sweepable, not pinned in the store forever.
    s = make_session(last_active_at="not-a-date")
    assert s.idle_seconds() == float("inf")


def test_round_trip_through_dict_preserves_everything():
    s = make_session()
    s.append_question("q")
    s.append_research("r")
    s.append_answer("a")
    s.seen_keys.add("abc")
    s.record_context(1234)
    back = LoreSession.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.transcript == s.transcript
    assert back.seen_keys == s.seen_keys
    assert back.last_context_tokens == s.last_context_tokens


def test_from_dict_ages_legacy_entries_from_their_creation_time():
    # Sessions written before expiry existed have no last_active_at; they must
    # age from when they were made rather than never.
    created = datetime.now(timezone.utc).isoformat()
    s = LoreSession.from_dict({
        "thread_id": 1, "system_prompt": "", "original_question": "",
        "pinned_now": "", "created_at": created,
    })
    assert s.last_active_at == created


def test_store_reads_the_legacy_flat_file_shape(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"7": make_session(thread_id=7).to_dict()}))
    store = LoreSessionStore(str(path))
    assert store.get(7) is not None


def test_store_skips_malformed_entries_rather_than_failing_to_start(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"threads": {"7": {"nonsense": True}}}))
    store = LoreSessionStore(str(path))
    assert store.thread_ids() == []


def test_store_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text("{ not json")
    assert LoreSessionStore(str(path)).thread_ids() == []


def test_save_is_atomic_and_reloadable(tmp_path):
    path = tmp_path / "nested" / "sessions.json"
    store = LoreSessionStore(str(path))
    store.put(make_session(thread_id=5))
    assert not path.with_suffix(".tmp").exists()  # temp file renamed away
    assert LoreSessionStore(str(path)).get(5) is not None


def test_an_offer_is_claimed_exactly_once(tmp_path):
    store = LoreSessionStore(str(tmp_path / "s.json"))
    store.offer(100, make_session(thread_id=0))
    assert store.claim(100, 42).thread_id == 42
    assert store.claim(100, 43) is None  # second reaction gets nothing


def test_expired_offers_are_swept(tmp_path):
    store = LoreSessionStore(str(tmp_path / "s.json"))
    old = make_session(thread_id=0)
    old.created_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.offer(100, old)
    assert store.sweep_offers(ttl_seconds=3600) == 1
    assert not store.has_offer(100)


def test_offers_with_unparseable_timestamps_are_swept(tmp_path):
    store = LoreSessionStore(str(tmp_path / "s.json"))
    s = make_session(thread_id=0)
    s.created_at = "not-a-date"
    store.offer(100, s)
    assert store.sweep_offers(ttl_seconds=3600) == 1


def test_expired_threads_lists_only_idle_ones(tmp_path):
    store = LoreSessionStore(str(tmp_path / "s.json"))
    fresh = make_session(thread_id=1)
    stale = make_session(thread_id=2)
    stale.last_active_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.put(fresh)
    store.put(stale)
    assert store.expired_threads(ttl_seconds=3600) == [2]
