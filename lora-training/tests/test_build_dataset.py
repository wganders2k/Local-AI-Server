"""
The archive -> training-set builder.

Every test here pins a property whose failure is silent in training: the run
completes, the loss falls, and the adapter has learned the wrong thing — a
caption as someone's voice, a phone number, a reply to a conversation that had
ended an hour earlier.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import json
import re

import pytest

import build_dataset as bd
import mimic_format

TARGET = "100"
OTHER = "200"
THIRD = "300"


def msg(mid, user, minute, content="", channel="c1", second=0, **extra):
    names = {TARGET: "alice", OTHER: "bob", THIRD: "carol"}
    rec = {
        "message_id": str(mid),
        "user_id": user,
        "username": names.get(user, user),
        "channel_id": channel,
        "channel_name": {"c1": "general", "c2": "bot-zone"}.get(channel, channel),
        "timestamp": f"2026-01-01T10:{minute:02d}:{second:02d}Z",
        "content": content,
        "is_bot": False,
    }
    rec.update(extra)
    return rec


def build(records, system="SYS", **opts):
    opts.setdefault("user_ids", [TARGET])
    return bd.build_examples(records, system, bd.Options(**opts))


# -- shape --

def test_reply_becomes_prompt_completion_with_labelled_context():
    examples, _ = build([
        msg(1, OTHER, 0, "anyone up"),
        msg(2, THIRD, 1, "yeah"),
        msg(3, TARGET, 2, "unfortunately yes"),
    ])

    assert len(examples) == 1
    ex = examples[0]
    assert ex["prompt"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "[bob]: anyone up\n[carol]: yeah"},
    ]
    assert ex["completion"] == [{"role": "assistant", "content": "unfortunately yes"}]
    assert ex["context_turns"] == ["[bob]: anyone up", "[carol]: yeah"]


def test_no_system_omits_the_system_turn():
    examples, _ = build([msg(1, OTHER, 0, "anyone up"), msg(2, TARGET, 1, "unfortunately yes")], system=None)

    assert [m["role"] for m in examples[0]["prompt"]] == ["user"]


def test_only_the_target_members_turns_are_completions():
    examples, _ = build([
        msg(1, TARGET, 0, "starting this off"),
        msg(2, OTHER, 1, "bob replies here"),
        msg(3, TARGET, 2, "alice answers bob"),
    ])

    assert [e["completion"][0]["content"] for e in examples] == ["alice answers bob"]


# -- context window --

def test_context_stops_at_a_silence_longer_than_the_gap():
    examples, _ = build([
        msg(1, THIRD, 0, "old conversation"),
        msg(2, OTHER, 45, "new conversation"),
        msg(3, TARGET, 46, "replying to the new one"),
    ], gap_minutes=30)

    assert examples[0]["prompt"][-1]["content"] == "[bob]: new conversation"


def test_context_is_capped_at_n_turns():
    records = [msg(i, OTHER if i % 2 else THIRD, i, f"line {i}") for i in range(1, 8)]
    records.append(msg(99, TARGET, 9, "finally replying"))

    examples, _ = build(records, context_n=3)

    assert examples[0]["prompt"][-1]["content"].count("\n") == 2
    assert "line 7" in examples[0]["prompt"][-1]["content"]


def test_opening_a_conversation_is_not_an_example():
    _, stats = build([msg(1, TARGET, 0, "first message of the day")])

    assert stats["dropped_no_context"] == 1


def test_channels_are_never_mixed():
    examples, _ = build([
        msg(1, OTHER, 0, "in another channel", channel="c9"),
        msg(2, THIRD, 1, "in this channel"),
        msg(3, TARGET, 2, "replying here"),
    ])

    assert "another channel" not in examples[0]["prompt"][-1]["content"]


def test_archive_order_does_not_matter():
    """Records arrive grouped by author file, not by time."""
    records = [msg(3, TARGET, 2, "unfortunately yes"), msg(1, OTHER, 0, "anyone up"), msg(2, THIRD, 1, "yeah")]

    examples, _ = build(records)

    assert examples[0]["prompt"][-1]["content"] == "[bob]: anyone up\n[carol]: yeah"


# -- merging --

def test_rapid_messages_from_one_author_merge_into_one_reply():
    examples, _ = build([
        msg(1, OTHER, 0, "thoughts?"),
        msg(2, TARGET, 1, "honestly", second=0),
        msg(3, TARGET, 1, "terrible idea", second=20),
    ], merge_seconds=60, min_chars=1)

    assert len(examples) == 1
    assert examples[0]["completion"][0]["content"] == "honestly\nterrible idea"


def test_messages_further_apart_than_the_merge_window_stay_separate():
    examples, _ = build([
        msg(1, OTHER, 0, "thoughts?"),
        msg(2, TARGET, 1, "honestly no"),
        msg(3, TARGET, 5, "still no actually"),
    ], merge_seconds=60)

    assert len(examples) == 2


# -- filters --

def test_attachment_only_is_not_a_target_but_marks_context():
    examples, stats = build([
        msg(1, OTHER, 0, "", attachments=[{"filename": "a.png"}]),
        msg(2, TARGET, 1, "", attachments=[{"filename": "b.png"}]),
        msg(3, OTHER, 2, "lol"),
        msg(4, TARGET, 3, "glad you liked it"),
    ])

    assert stats["dropped_attachment_only"] == 1
    assert examples[0]["prompt"][-1]["content"].startswith("[bob]: [attachment]")


def test_captions_never_appear_anywhere():
    att = [{
        "filename": "meme.png",
        "caption": "A CAPTION THE MODEL MUST NOT SEE",
        "caption_status": "done",
        "caption_excluded_from_training": True,
    }]
    examples, _ = build([
        msg(1, OTHER, 0, "", attachments=att),
        msg(2, TARGET, 1, "that is so bad", attachments=att),
    ])

    assert "CAPTION" not in json.dumps(examples)


def test_link_only_replies_are_dropped():
    _, stats = build([msg(1, OTHER, 0, "send it"), msg(2, TARGET, 1, "https://example.com/x")])

    assert stats["dropped_link_only"] == 1


def test_short_replies_are_dropped_as_targets_but_kept_as_context():
    examples, stats = build([
        msg(1, OTHER, 0, "hello"),
        msg(2, TARGET, 1, "k"),
        msg(3, OTHER, 5, "you there?"),
        msg(4, TARGET, 6, "yes I am here"),
    ], min_chars=4)

    assert stats["dropped_too_short"] == 1
    assert "[alice]: k" in examples[0]["prompt"][-1]["content"]


def test_commands_and_bots_are_removed_from_the_timeline():
    examples, stats = build([
        msg(1, OTHER, 0, "real message"),
        msg(2, OTHER, 1, "/mimic alice hi"),
        msg(3, "999", 2, "I am a bot", is_bot=True),
        msg(4, TARGET, 3, "replying to bob"),
    ])

    assert stats["dropped_command"] == 1
    assert stats["dropped_bot_message"] == 1
    assert examples[0]["prompt"][-1]["content"] == "[bob]: real message"


@pytest.mark.parametrize("text", ["!p song", ";;play song", "m!skip", "t!dice d3", "p!catch", "-p song", "--skip"])
def test_every_bot_prefix_on_this_server_is_a_command(text):
    _, stats = build([msg(1, OTHER, 0, text)])

    assert stats["dropped_command"] == 1


@pytest.mark.parametrize("text", [">greentext is a voice", "- a dash list item", "-pretty sure not", "...anyway"])
def test_punctuation_that_starts_real_messages_is_not_a_command(text):
    _, stats = build([msg(1, OTHER, 0, text)])

    assert stats["dropped_command"] == 0


def test_pin_events_are_removed_from_the_timeline():
    examples, stats = build([
        msg(1, OTHER, 0, "look at this"),
        msg(2, OTHER, 1, "Pinned a message.", type="ChannelPinnedMessage"),
        msg(3, TARGET, 2, "Pinned a message.", type="ChannelPinnedMessage"),
        msg(4, TARGET, 3, "nice one bob"),
    ])

    assert stats["dropped_system_message"] == 2
    assert len(examples) == 1
    assert examples[0]["prompt"][-1]["content"] == "[bob]: look at this"


@pytest.mark.parametrize("excluded", ["c2", "bot-zone"])
def test_excluded_channels_match_by_id_or_name(excluded):
    _, stats = build([
        msg(1, OTHER, 0, "hello there", channel="c2"),
        msg(2, TARGET, 1, "hello back", channel="c2"),
    ], exclude_channels=[excluded])

    assert stats["examples"] == 0
    assert stats["dropped_excluded_channel"] == 2


# -- normalisation --

def test_pii_is_scrubbed_from_targets_and_context():
    examples, _ = build([
        msg(1, OTHER, 0, "mail me at bob@example.com or 555-123-4567"),
        msg(2, TARGET, 1, "I live at 42 Wallaby Way, come over"),
    ])

    text = json.dumps(examples)
    assert "bob@example.com" not in text and "555-123-4567" not in text
    assert "42 Wallaby Way" not in text
    assert "[email]" in text and "[phone]" in text and "[address]" in text


def test_bare_digit_runs_are_not_phone_numbers():
    assert mimic_format.scrub_pii("scored 1234567890 points") == "scored 1234567890 points"


def test_links_become_markers_and_no_snowflake_survives():
    examples, _ = build([
        msg(1, OTHER, 0, "look https://cdn.discordapp.com/attachments/337040332107808768/434521920123456789/a.png"),
        msg(2, THIRD, 1, "see https://discord.com/channels/419718473233465355/498175798123456789/498175798987654321"),
        msg(3, TARGET, 2, "lol https://www.Twitter.com/peta/status/1100820317256208384?s=09 classic"),
    ])

    assert examples[0]["prompt"][-1]["content"] == "[bob]: look [attachment]\n[carol]: see [message link]"
    assert examples[0]["completion"][0]["content"] == "lol [link: twitter.com] classic"
    assert not re.search(r"\d{15,}", json.dumps(examples))


def test_a_reply_that_is_only_a_link_marker_is_dropped():
    _, stats = build([
        msg(1, OTHER, 0, "send it"),
        msg(2, TARGET, 1, "https://cdn.discordapp.com/attachments/1/2/clip.mp4 https://youtu.be/abc"),
    ])

    assert stats["dropped_link_only"] == 1


def test_long_digit_runs_are_masked_in_context_and_dropped_as_targets():
    examples, stats = build([
        msg(1, OTHER, 0, "10 turned into 10-9482390847234987235982375394"),
        msg(2, TARGET, 1, "e-676277797906854407421807398614769274640"),
        msg(3, TARGET, 5, "that is not how maths works"),
    ], merge_seconds=0)

    assert stats["dropped_long_number"] == 1
    assert examples[0]["prompt"][-1]["content"].startswith("[bob]: 10 turned into 10-[long number]\n")


def test_mentions_and_custom_emoji_are_rendered_readably():
    examples, _ = build([
        msg(1, OTHER, 0, "<@100> look <:kekw:12345>"),
        msg(2, TARGET, 1, "<@!200> no thanks"),
    ])

    assert examples[0]["prompt"][-1]["content"] == "[bob]: @alice look :kekw:"
    assert examples[0]["completion"][0]["content"] == "@bob no thanks"


# -- split and prompt --

def test_split_holds_out_the_newest_examples():
    examples = [{"timestamp": f"2026-01-{d:02d}"} for d in range(1, 21)]

    train, evaluation = bd.split_by_time(examples, 0.1)

    assert len(evaluation) == 2
    assert max(e["timestamp"] for e in train) < min(e["timestamp"] for e in evaluation)


def test_the_shared_persona_template_renders():
    """The template both sides render — a new placeholder in it would break this."""
    rendered = mimic_format.render_persona_prompt("mimic_user1")
    assert "mimic_user1" in rendered
    assert not rendered.endswith("\n")


def test_identified_mentions_render_as_the_username_speakers_are_labelled_with():
    examples, _ = build([
        msg(1, OTHER, 0, "who is bringing the speaker"),
        msg(2, TARGET, 1, "@Bobby Tables (Big B) is, he said so",
            mentions=[{"user_id": OTHER, "username": "bob", "display": "Bobby Tables (Big B)"}]),
    ])

    assert examples[0]["completion"][0]["content"] == "@bob is, he said so"
    assert examples[0]["prompt"][-1]["content"] == "[bob]: who is bringing the speaker"


@pytest.mark.parametrize("msg_type", [None, "Default", "Reply"])
def test_conversation_types_are_kept(msg_type):
    extra = {} if msg_type is None else {"type": msg_type}
    _, stats = build([msg(1, OTHER, 0, "anyone around", **extra), msg(2, TARGET, 1, "yeah what's up", **extra)])

    assert stats["examples"] == 1


@pytest.mark.parametrize("msg_type", ["ThreadCreated", "46", "GuildMemberJoin"])
def test_other_types_are_dropped(msg_type):
    _, stats = build([msg(1, OTHER, 0, "something happened", type=msg_type)])

    assert stats["dropped_system_message"] == 1


def test_aliases_rename_speakers_and_mentions_alike():
    examples, _ = build([
        msg(1, OTHER, 0, "who is bringing the speaker"),
        msg(2, TARGET, 1, "@Bobby Tables (Big B) is, he said so",
            mentions=[{"user_id": OTHER, "username": "bob", "display": "Bobby Tables (Big B)"}]),
    ], aliases={OTHER: "robert"})

    assert examples[0]["prompt"][-1]["content"] == "[robert]: who is bringing the speaker"
    assert examples[0]["completion"][0]["content"] == "@robert is, he said so"


def test_every_account_of_the_member_supplies_targets():
    old_account = "999"
    examples, _ = build([
        msg(1, OTHER, 0, "anyone on tonight"),
        msg(2, old_account, 1, "maybe later, at work"),
        msg(3, OTHER, 5, "still on?"),
        msg(4, TARGET, 6, "yeah getting on now"),
    ], user_ids=[TARGET, old_account], aliases={TARGET: "dana", old_account: "dana"})

    assert [e["completion"][0]["content"] for e in examples] == ["maybe later, at work", "yeah getting on now"]
    assert "[dana]: maybe later, at work" in examples[1]["prompt"][-1]["content"]


# -- junk --

WALL = " ".join([":jlulke:"] * 30)
ART = "⠀⠀⠀⡯⡯⡾⠝⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢊⠘⡮⣣⠪⠢⡑⡌\n⠀⠀⠀⠟⠝⠈⠀⠀⠀⠡⠀⠠⢈⠠⢐⢠⢂⢔⣐⢄⡂⢔⠀⡁⢉⠸⢨⢑⠕⡌"


@pytest.mark.parametrize("text,reason", [
    (WALL, "repetitive"),
    ("poggers " * 10, "repetitive"),
    (ART, "art"),
    ("𝔖𝔴𝔞𝔤 𝓢𝔀𝓪𝓰 𝕊𝕨𝕒𝕘 Ｓｗａｇ ꜱᴡᴀɢ 🆂🆆🅰️🅶 Ⓢⓦⓐⓖ 𝙎𝙬𝙖𝙜 𝔖𝔴𝔞𝔤 𝓢𝔀𝓪𝓰", "art"),
    ("BUSS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", None),
    ("ok so the thing is i really do not think that is how it works at all", None),
    ("lol :kekw: :kekw:", None),
])
def test_junk_reason(text, reason):
    assert mimic_format.junk_reason(text) == reason


def test_context_cuts_junk_to_a_stub_and_long_messages_to_the_cap():
    assert mimic_format.context_text(WALL) == WALL[:60].rstrip() + " […]"
    long = " ".join(f"word{i}" for i in range(300))
    assert mimic_format.context_text(long) == long[:1000].rstrip() + " […]"
    assert mimic_format.context_text("short and normal") == "short and normal"


def test_junk_is_cut_from_a_reply_and_stubbed_in_context():
    examples, stats = build([
        msg(1, OTHER, 0, WALL),
        msg(2, TARGET, 1, "what the hell is wrong with you man"),
        msg(3, TARGET, 2, ART),
    ])

    assert examples[0]["completion"][0]["content"] == "what the hell is wrong with you man"
    assert examples[0]["prompt"][-1]["content"] == "[bob]: " + WALL[:60].rstrip() + " […]"
    assert stats["trimmed_junk_messages"] == 1


def test_a_reply_that_is_only_junk_is_dropped_for_it():
    _, stats = build([msg(1, OTHER, 0, "say something"), msg(2, TARGET, 1, WALL)])

    assert stats["dropped_repetitive"] == 1 and stats["examples"] == 0


def test_copypasta_is_a_long_message_posted_three_times():
    pasta = "Can I get a " + "very long copypasta sentence that goes on and on " * 5
    records = [msg(1, OTHER, 0, "go on then")]
    records += [msg(10 + i, THIRD, 20 + i * 11, pasta, channel=f"c{5 + i}") for i in range(2)]
    records += [msg(2, TARGET, 1, pasta)]

    _, stats = build(records)

    assert stats["dropped_copypasta"] == 1


def test_excluded_dates_leave_the_whole_day_out():
    _, stats = build([
        msg(1, OTHER, 0, "hello there friend"),
        msg(2, TARGET, 1, "hello back to you mate"),
    ], exclude_dates=["2026-01-01"])

    assert stats["dropped_excluded_date"] == 2 and stats["examples"] == 0
