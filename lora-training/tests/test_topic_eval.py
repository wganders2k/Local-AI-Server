"""Which messages become probes, and how a probe is shaped. A wrong pick judges adapters on the wrong thing."""

from datetime import datetime, timedelta, timezone

import topic_eval as te
from mimic_format import Turn

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
GAP = timedelta(hours=2)


def turn(uid, minute, text, attachment_only=False):
    ts = T0 + timedelta(minutes=minute)
    return Turn(uid, f"user{uid}", ts, ts, [text], attachment_only)


def test_an_opener_follows_a_silence_and_gets_an_answer():
    turns = [
        turn("1", 0, "first message in the channel ever, hello"),
        turn("2", 1, "hey there, welcome to the server"),
        turn("1", 300, "anyone want to play something tonight?"),  # 5 h later, answered
        turn("3", 302, "yeah im down"),
        turn("2", 700, "nobody is going to answer this one sadly"),  # 6.5 h later, never answered
    ]
    assert [t.text for t in te.openers(turns, GAP, set(), set())] == [
        "first message in the channel ever, hello", "anyone want to play something tonight?"]
    # The persona's own openers are never probes.
    assert len(te.openers(turns, GAP, {"1"}, set())) == 0


def test_mid_conversation_messages_skip_short_link_and_persona_lines():
    turns = [
        turn("1", 0, "so what did everyone think of the finale"),
        turn("2", 1, "honestly the ending made no sense to me at all"),
        turn("3", 2, "lol"),
        turn("4", 3, "https://example.com/clip"),
        turn("9", 4, "the persona said this and it is long enough"),
    ]
    picked = te.mid_conversation(turns, {"9"}, set())
    assert [t.text for t in picked] == ["honestly the ending made no sense to me at all"]


def test_a_probe_is_the_newest_turn_and_never_exceeds_serving():
    base = [f"[a]: line {i}" for i in range(30)]
    ctx = te.probe_context(base, "[z]: totally unrelated")
    assert len(ctx) == te.MAX_TURNS and ctx[-1] == "[z]: totally unrelated" and ctx[0] == "[a]: line 7"


def test_replies_are_cleaned_as_the_bot_posts_them():
    assert te.clean_reply("[alice]: nah\n[erin]: speaking for erin") == "nah"


def test_the_judge_sees_the_probe_apart_from_the_conversation():
    msgs = te.judge_messages("chaos", ["[a]: one", "[b]: two", "[z]: probe"], "reply text")
    text = msgs[0]["content"]
    assert "EARLIER CONVERSATION:\n[a]: one\n[b]: two\n\nNEWEST MESSAGE:\n[z]: probe" in text
    assert '"wtf"' in text and te.judge_schema("chaos")["json_schema"]["schema"]["properties"]["verdict"]["enum"] == \
        ["ignores", "wtf", "engages", "neither"]


def test_summary_counts_wanted_verdicts():
    rows = [{"model": "m", "kind": "chaos", "verdict": v} for v in ("ignores", "wtf", "engages", "engages")]
    assert "50%" in te.summarise(rows, ["m"]).splitlines()[-1]
