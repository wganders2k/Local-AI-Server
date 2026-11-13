"""
The thread directory: the registry file and the routing built from it must
never disagree, and an existing threads.json must keep loading.
"""

import json

import discord

from threads import CHAT, LORE, MIMIC_CHAT, ThreadDirectory


def live_thread(archived=False):
    """A discord.Thread as far as isinstance and .archived go."""
    thread = discord.Thread.__new__(discord.Thread)
    thread.archived = archived
    return thread


MIMIC_ENTRY = {
    "model": "mimic-multi", "name": "mimic-chat: A", "created_at": "2026-01-01T00:00:00+00:00",
    "status": "active", "kind": "mimic_chat", "personas": ["mimic_user1"], "premise": "pizza?",
    "starter_id": "22", "starter_label": "erin", "started_at": "2026-01-01T00:00:00+00:00", "length": None,
}


def write_registry(tmp_path, data):
    path = tmp_path / "threads.json"
    path.write_text(json.dumps(data))
    return str(path)


async def test_an_existing_registry_restores_every_kind(tmp_path):
    path = write_registry(tmp_path, {
        # Written before `kind` existed: a chat thread.
        "1": {"model": "brain", "name": "chat-brain", "created_at": "x", "status": "active"},
        "2": {"model": "brain-dense-heretic", "name": "lore-x", "created_at": "x", "status": "archived",
              "kind": "lore"},
        "3": {"model": "brain-dense-heretic", "name": "lore-y", "created_at": "x", "status": "active",
              "kind": "lore"},
        "4": MIMIC_ENTRY,
        "5": {"model": "brain", "name": "gone", "created_at": "x", "status": "active", "kind": "chat"},
    })
    threads = ThreadDirectory(path)

    async def fetch(thread_id):
        if thread_id == 5:
            raise discord.NotFound(type("R", (), {"status": 404, "reason": "x"})(), "gone")
        return live_thread(archived=thread_id == 2)

    await threads.restore(fetch, has_session=lambda tid: tid == 2)

    assert threads.kind_of(1) == CHAT and threads.model_of(1) == "brain"
    assert threads.kind_of(2) == LORE
    assert threads.kind_of(3) == CHAT  # lore without a session falls back to plain chat
    assert threads.kind_of(4) == MIMIC_CHAT and threads.mimic_chats[4].premise == "pizza?"
    assert threads.kind_of(5) is None
    assert "5" not in json.loads((tmp_path / "threads.json").read_text())


def test_open_and_close_keep_file_and_routing_in_step(tmp_path):
    path = str(tmp_path / "ctx" / "threads.json")
    threads = ThreadDirectory(path)

    threads.open(10, LORE, "brain-dense-heretic", "lore-q")
    assert threads.kind_of(10) == LORE
    assert json.loads(open(path).read())["10"]["kind"] == "lore"
    assert ThreadDirectory(path)._data["10"]["model"] == "brain-dense-heretic"

    threads.lock_for(10)
    assert threads.close(10)
    assert threads.kind_of(10) is None and not threads.is_busy(10)
    assert json.loads(open(path).read()) == {}
    assert not threads.close(10)  # closing twice is harmless


def test_a_corrupt_registry_starts_empty(tmp_path):
    (tmp_path / "threads.json").write_text("{ nope")
    assert ThreadDirectory(str(tmp_path / "threads.json")).kind_of(1) is None


async def test_a_mimic_chat_without_a_premise_restores(tmp_path):
    entry = {k: v for k, v in MIMIC_ENTRY.items() if k != "premise"}
    threads = ThreadDirectory(write_registry(tmp_path, {"4": entry}))

    await threads.restore(lambda tid: _live(), has_session=lambda tid: False)

    assert threads.mimic_chats[4].premise is None
    assert "premise" in threads.mimic_chats[4].to_registry()


async def _live():
    return live_thread()
