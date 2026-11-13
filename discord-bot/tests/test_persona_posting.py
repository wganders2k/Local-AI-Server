"""
Live Discord history must reach an adapter exactly as its training data did
(lora-training/build_dataset.py): the same labels, mentions, merged turns and
context cut. Any drift is a prompt the adapter never saw.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import mimic_format
import pytest

import persona_posting as pp
from tests.conftest import write_roster
from mimic import transcript
from mimic.transcript import build_persona_prompt, clean_persona_reply
from personas import PersonaCatalog

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
MEMBER = "11"  # the member mimic_user1 mimics


@pytest.fixture
def catalog(tmp_path):
    d = write_roster(tmp_path / "training", {
        "mimic_user1": {"user_ids": MEMBER, "display_name": "Alice"},
        "mimic_user2": {"user_ids": "44", "display_name": "Dana"},
    })
    (d / "names.json").write_text(json.dumps({
        "_readme": "notes",
        MEMBER: {"alias": "alice"},
        "22": {"alias": "erin"},
    }))
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "mimic_user1-lora.json").write_text(json.dumps({"source": "/runs/x", "dataset": {
        "user_id": MEMBER, "display_name": "Alice", "context_n": 3, "gap_minutes": 30.0, "merge_seconds": 120.0,
    }}))
    return PersonaCatalog(training_dir=str(d), lora_dir=str(lora), adapters=["mimic_user1", "mimic_user2"])


def user(uid, name, bot=False):
    return SimpleNamespace(id=int(uid), name=name, bot=bot)


def message(author, content, minute=0, second=0, webhook_id=None, type_value=0, attachments=(), mentions=()):
    return SimpleNamespace(
        author=author, content=content, webhook_id=webhook_id,
        type=SimpleNamespace(value=type_value), attachments=list(attachments), mentions=list(mentions),
        created_at=T0 + timedelta(minutes=minute, seconds=second),
    )


ERIN = user("22", "erin_h")
BOB = user("33", "bobby")


def reader(catalog, guild=None, known=None):
    return pp.HistoryReader(catalog, guild, lambda uid: None, known or {})


def prompt(catalog, msgs, persona="mimic_user1"):
    return build_persona_prompt(catalog, catalog.get(persona), msgs)


def test_people_are_labelled_by_alias_then_handle_and_mentions_resolve(catalog):
    line = reader(catalog).to_line(message(ERIN, "<@33> look https://x.com/a/status/1234567890123456789 <:kekw:1>",
                                           mentions=[BOB]))

    assert line.persona is None
    assert line.msg.username == "erin"
    assert line.msg.text == "@bobby look [link: x.com] :kekw:"
    assert reader(catalog).to_line(message(BOB, "hi")).msg.username == "bobby"


def test_bots_commands_and_system_messages_stay_out(catalog):
    r = reader(catalog)
    assert r.to_line(message(user("99", "musicbot", bot=True), "now playing")) is None
    assert r.to_line(message(ERIN, "!play something")) is None
    assert r.to_line(message(ERIN, "Pinned a message.", type_value=6)) is None
    assert r.to_line(message(ERIN, "", attachments=[])) is None
    assert r.to_line(message(ERIN, "replying", type_value=19)) is not None


def test_a_persona_webhook_post_reads_back_as_the_member_it_mimics(catalog):
    r = reader(catalog)
    name = pp.webhook_username(catalog.get("mimic_user1"))
    hook = SimpleNamespace(id=1, name=name, bot=True)

    line = r.to_line(message(hook, "lol", webhook_id=1))

    assert name == "Alice (mimic)"
    assert line.persona == "mimic_user1"
    assert (line.msg.user_id, line.msg.username) == (MEMBER, "alice")
    assert r.to_line(message(SimpleNamespace(id=2, name="Some Other Hook", bot=True), "x", webhook_id=2)) is None


def test_prompt_merges_turns_and_keeps_the_trained_window(catalog):
    r = reader(catalog)
    msgs = [r.to_line(m).msg for m in [
        message(BOB, "too old", minute=0),
        message(ERIN, "one", minute=1),
        message(BOB, "two", minute=2),
        message(BOB, "two again", minute=3),  # within 120s of "two": same turn
        message(ERIN, "three", minute=4),
        message(BOB, "four", minute=5),
    ]]

    out = prompt(catalog, msgs)

    assert out[0]["role"] == "system" and "mimics Alice's" in out[0]["content"]
    # context_n=3 from the build: the last three turns, oldest first.
    assert out[1] == {"role": "user", "content": "[bobby]: two\ntwo again\n[erin]: three\n[bobby]: four"}


def test_prompt_is_never_wider_than_training(catalog):
    r = reader(catalog)
    # Eight turns in the last eight minutes: a busy chat still shows only the trained three.
    msgs = [r.to_line(message(WILL if i % 2 else BOB, str(i), minute=i)).msg for i in range(8)]

    assert prompt(catalog, msgs)[1]["content"] == "[erin]: 5\n[bobby]: 6\n[erin]: 7"


def test_prompt_is_cut_at_a_silence_longer_than_training_gap(catalog, monkeypatch):
    monkeypatch.setattr(transcript, "MIMIC_MIN_CONTEXT_TURNS", 0)
    r = reader(catalog)
    msgs = [r.to_line(m).msg for m in [
        message(BOB, "yesterday's topic", minute=0),
        message(ERIN, "an hour later", minute=60),  # gap_minutes=30 from the build
    ]]

    # As in training: a turn after a long silence starts a new conversation.
    assert prompt(catalog, msgs)[1]["content"] == "[erin]: an hour later"
    # A silence before the newest turn is fine; only gaps between shown turns cut.
    assert prompt(catalog, msgs[:1])[1]["content"] == "[bobby]: yesterday's topic"
    assert prompt(catalog, []) is None


def test_a_silence_never_leaves_fewer_than_the_minimum(catalog, monkeypatch):
    monkeypatch.setattr(transcript, "MIMIC_MIN_CONTEXT_TURNS", 2)
    r = reader(catalog)
    msgs = [r.to_line(m).msg for m in [
        message(ERIN, "older", minute=0),
        message(BOB, "yesterday's topic", minute=1),
        message(ERIN, "an hour later", minute=60),
        message(BOB, "right after", minute=61),
    ]]

    # Two turns after the silence already meet the minimum: nothing from before it.
    assert prompt(catalog, msgs)[1]["content"] == "[erin]: an hour later\n[bobby]: right after"
    # One turn after it: topped up across the silence to two.
    assert prompt(catalog, msgs[:3])[1]["content"] == "[bobby]: yesterday's topic\n[erin]: an hour later"
    # And never past the trained context_n (3).
    monkeypatch.setattr(transcript, "MIMIC_MIN_CONTEXT_TURNS", 10)
    assert len(prompt(catalog, msgs)[1]["content"].splitlines()) == 3


def test_persona_without_a_build_gets_build_defaults(catalog):
    p = catalog.get("mimic_user2")  # no manifest
    assert (p.context_n, p.gap_minutes) == (mimic_format.CONTEXT_N, mimic_format.GAP_MINUTES)
    assert catalog.get("mimic_user1").gap_minutes == 30.0


def test_attachment_only_turns_leave_a_marker(catalog):
    msgs = [reader(catalog).to_line(message(ERIN, "", attachments=["a.png"])).msg]

    assert prompt(catalog, msgs)[1]["content"] == "[erin]: [attachment]"


def test_reply_is_one_speakers_line():
    assert clean_persona_reply("[alice]: lol no") == "lol no"
    assert clean_persona_reply("lol no\nfr\n[erin]: yes") == "lol no\nfr"
    assert clean_persona_reply("see [link: x.com] lol") == "see [link: x.com] lol"


def test_persona_posts_as_the_members_current_nickname_and_avatar(catalog):
    alice = catalog.get("mimic_user1")
    member = SimpleNamespace(display_name="ally", display_avatar=SimpleNamespace(url="https://cdn/a.png"))
    guild = SimpleNamespace(get_member=lambda uid: member if uid == int(MEMBER) else None)

    assert pp.webhook_username(alice, guild) == "ally (mimic)"
    assert pp.webhook_avatar(alice, guild) == "https://cdn/a.png"
    # Not in the server (or no guild): the display name and the webhook's own avatar.
    gone = SimpleNamespace(get_member=lambda uid: None)
    assert pp.webhook_username(alice, gone) == "Alice (mimic)"
    assert pp.webhook_avatar(alice, gone) is None


def test_posts_under_the_nickname_or_the_old_name_both_read_back(catalog):
    member = SimpleNamespace(display_name="ally", display_avatar=SimpleNamespace(url="u"))
    guild = SimpleNamespace(get_member=lambda uid: member if uid == int(MEMBER) else None,
                            get_channel_or_thread=lambda cid: None)
    r = reader(catalog, guild)

    for name in ("ally (mimic)", "Alice (mimic)"):
        line = r.to_line(message(SimpleNamespace(id=1, name=name, bot=True), "lol", webhook_id=1))
        assert line.persona == "mimic_user1"
    # Both names address the persona, and Discord currently shows the nickname.
    assert {"ally", "Alice", "alice"} <= r.cast.names["mimic_user1"]
    assert r.cast.shown_as["mimic_user1"] == "ally"


def test_a_name_from_before_a_nickname_change_still_reads_back(catalog, tmp_path):
    names = pp.PersonaNameRegistry(str(tmp_path / "names" / "persona_names.json"))
    names.record("old nick (mimic)", "mimic_user1")
    names.record("old nick (mimic)", "mimic_user1")  # no news, no write

    reloaded = pp.PersonaNameRegistry(str(tmp_path / "names" / "persona_names.json"))
    r = reader(catalog, known=reloaded.names)
    line = r.to_line(message(SimpleNamespace(id=1, name="old nick (mimic)", bot=True), "lol", webhook_id=1))

    assert line.persona == "mimic_user1"
    assert "old nick" in r.cast.names["mimic_user1"]
    # A name recorded for a persona that no longer exists is ignored.
    assert reader(catalog, known={"x (mimic)": "mimic_gone"}).by_webhook_name.get("x (mimic)") is None


def test_a_persona_sees_its_own_member_as_someone_else(catalog):
    r = reader(catalog)
    member = SimpleNamespace(id=int(MEMBER), name="chris_handle", bot=False)
    lines = [
        r.to_line(message(member, "i think pineapple is fine", minute=0)),
        r.cast.persona_line("mimic_user1", "nah it's disgusting", T0 + timedelta(minutes=1)),
        r.to_line(message(member, "you're me, you can't disagree", minute=2)),
    ]

    out = prompt(catalog, r.cast.seen_by("mimic_user1", lines))

    # Three turns, not one: the member is "member", the persona keeps the member's name.
    assert out[1]["content"] == (
        "[member]: i think pineapple is fine\n[alice]: nah it's disgusting\n[member]: you're me, you can't disagree"
    )
    # Another persona reading the same lines sees the member by name.
    assert r.cast.seen_by("mimic_user2", lines)[0].username == "alice"


def test_an_adapter_trained_without_the_system_prompt_is_served_without_one(catalog):
    manifest = os.path.join(catalog.lora_dir, "mimic_user1-lora.json")
    with open(manifest) as fh:
        record = json.load(fh)
    record["dataset"]["no_system"] = True
    with open(manifest, "w") as fh:
        json.dump(record, fh)
    r = reader(catalog)
    msgs = [r.to_line(message(ERIN, "hi", minute=0)).msg]

    assert prompt(catalog, msgs) == [{"role": "user", "content": "[erin]: hi"}]


# ---- reset prefixes ---------------------------------------------------------

CAROL = user("55", "carol")


def chat(r, *said):
    """(author, text) pairs a minute apart, as Lines."""
    return [r.to_line(message(author, text, minute=i)) for i, (author, text) in enumerate(said)]


def shown(r, lines, persona="mimic_user2"):
    """What mimic_user2 (build defaults: 24 turns, 600s merge) is shown."""
    return r.cast.prompt(persona, lines)[-1]["content"]


def test_a_reset_prefix_is_stripped_and_recorded(catalog):
    r = reader(catalog)

    two = r.to_line(message(ERIN, ">> new topic"))
    three = r.to_line(message(ERIN, ">>>new topic"))

    assert (two.msg.text, two.reset) == ("new topic", 4)
    assert (three.msg.text, three.reset) == ("new topic", 1)
    assert r.to_line(message(ERIN, "a > b >> c")).reset is None
    # Nothing left to say once the prefix is gone: dropped like any empty message.
    assert r.to_line(message(ERIN, ">>>")) is None


def test_a_personas_prefix_is_just_text(catalog):
    r = reader(catalog)
    hook = SimpleNamespace(id=1, name=pp.webhook_username(catalog.get("mimic_user1")), bot=True)

    line = r.to_line(message(hook, ">> lol", webhook_id=1))

    assert (line.msg.text, line.reset) == (">> lol", None)


def test_double_prefix_keeps_the_last_four_turns_then_everything_after(catalog):
    r = reader(catalog)
    lines = chat(r, (BOB, "tech support"), (CAROL, "grades"), (BOB, "a"), (CAROL, "b"), (BOB, "c"),
                 (ERIN, ">> pizza?"))
    lines.append(r.cast.persona_line("mimic_user1", "pineapple", T0 + timedelta(minutes=9)))

    assert shown(r, lines) == "[bobby]: a\n[carol]: b\n[bobby]: c\n[erin]: pizza?\n[alice]: pineapple"
    assert shown(r, lines[:-1]) == "[bobby]: a\n[carol]: b\n[bobby]: c\n[erin]: pizza?"


def test_triple_prefix_keeps_only_that_message_then_everything_after(catalog):
    r = reader(catalog)
    lines = chat(r, (BOB, "tech support"), (ERIN, "still on this"), (ERIN, ">>> pizza?"), (BOB, "yes"))

    # "still on this" would merge into the same turn; a full reset leaves it out.
    assert shown(r, lines) == "[erin]: pizza?\n[bobby]: yes"


def test_the_latest_reset_wins_and_fewer_turns_than_asked_keeps_them_all(catalog):
    r = reader(catalog)
    lines = chat(r, (BOB, "old"), (ERIN, ">>> first"), (BOB, "x"), (ERIN, ">> second"))

    assert shown(r, lines) == "[bobby]: old\n[erin]: first\n[bobby]: x\n[erin]: second"
    assert shown(r, lines[:3]) == "[erin]: first\n[bobby]: x"


def test_a_reset_is_not_undone_by_the_minimum_turns(catalog):
    r = reader(catalog)
    lines = chat(r, (BOB, "a"), (CAROL, "b"), (BOB, "c"), (ERIN, ">>> d"))

    assert shown(r, lines) == "[erin]: d"
