"""
The multi-persona loop against a scripted backend: who speaks, when it stops,
and that a bad controller answer never stalls the chat.
"""

import json
from datetime import datetime, timezone

import pytest

from config import MIMIC_ADAPTER_SCALE, MIMIC_CHAT_MAX_BURST
from tests.conftest import write_roster
from mimic import controller
from mimic.controller import END, SPEAK, WAIT
from mimic.engine import run_burst, seed_line, simulate
from mimic.transcript import Cast, Line
from mimic_format import Msg
from personas import PersonaCatalog

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
P1, P2 = "mimic_user1", "mimic_user2"
CAST: Cast


@pytest.fixture(autouse=True)
def _cast(catalog):
    """The fixture roster's personas, as a guild-less conversation sees them."""
    global CAST
    CAST = Cast.build(catalog)


def member_cast(training, member="22"):
    """A cast where P1 imitates account `member` — the human in most tests."""
    write_roster(training, {
        P1: {"user_ids": member, "display_name": "User1"},
        P2: {"user_ids": "12", "display_name": "User2"},
    })
    return Cast.build(PersonaCatalog(training_dir=str(training), adapters=[P1, P2]))


class ScriptedProxy:
    """Controller calls (they carry a response_format) replay `decisions`; persona calls are numbered."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.persona_calls: list[dict] = []
        self.controller_calls: list[dict] = []

    async def chat(self, model, messages, lora=None, response_format=None, temperature=None, **kw):
        if response_format is not None:
            self.controller_calls.append({"messages": messages, "schema": response_format})
            d = self.decisions.pop(0) if self.decisions else {"action": WAIT}
            return d if isinstance(d, str) else json.dumps({"reason": "r", "speaker": "none", **d})
        self.persona_calls.append({"messages": messages, "lora": lora})
        return f"line {len(self.persona_calls)}"


def speak(p):
    """The controller names a speaker as the transcript does, not by persona key."""
    return {"action": SPEAK, "speaker": CAST.label(p)}


def human(text="hey"):
    return Line(Msg("22", "erin", T0, text))


def bot(persona):
    return Line(Msg("11", "x", T0, "..."), persona)


# ---- controller ----------------------------------------------------------

def test_parse_accepts_only_allowed_actions_and_known_speakers():
    acts = [SPEAK, WAIT]
    choices = {"alice": P1}
    assert controller.parse('{"action": "speak", "speaker": "alice"}', choices, acts).speaker == P1
    assert controller.parse('{"action": "speak", "speaker": "mimic_user1"}', choices, acts) is None
    assert controller.parse('{"action": "wait", "speaker": "none"}', choices, acts).action == WAIT
    assert controller.parse('{"action": "end", "speaker": "none"}', choices, acts) is None
    assert controller.parse('{"action": "speak", "speaker": "none"}', choices, acts) is None
    assert controller.parse("not json", choices, acts) is None


def test_schema_cannot_say_speak_without_naming_a_speaker():
    branches = controller._schema(["alice", "dana"], [SPEAK, WAIT])["json_schema"]["schema"]["anyOf"]
    speak_branch, other = (b["properties"] for b in branches)
    assert speak_branch["action"]["enum"] == [SPEAK] and speak_branch["speaker"]["enum"] == ["alice", "dana"]
    assert other["action"]["enum"] == [WAIT] and other["speaker"]["enum"] == ["none"]


def test_speakers_are_offered_by_name_unless_names_collide():
    assert controller.speaker_choices([P1, P2], {P1: "alice", P2: "dana"}) == {"alice": P1, "dana": P2}
    assert controller.speaker_choices([P1, P2], {P1: "alice", P2: "alice"}) == {P1: P1, P2: P2}


def test_fallback_rotates_speakers_and_leaves_room_for_humans():
    lines = [human(), bot(P1)]
    assert controller.fallback([P1, P2], lines, [SPEAK, WAIT]) == controller.Decision(SPEAK, P2, "fallback")
    assert controller.fallback([P1, P2], lines + [bot(P2)], [SPEAK, WAIT]).action == WAIT
    # Without wait on offer (a set length), it keeps talking.
    assert controller.fallback([P1, P2], lines + [bot(P2)], [SPEAK]).action == SPEAK


# ---- run_burst -----------------------------------------------------------

async def _burst(proxy, lines=None, cast=None, **kw):
    posted = []

    async def post(persona, text):
        posted.append((persona, text))

    # Seeded now: persona lines are stamped with the real clock, and an old
    # seed would fall outside the trained gap.
    lines = lines if lines is not None else [seed_line("pizza?", "22", "erin", datetime.now(timezone.utc))]
    new = await run_burst(proxy, cast or CAST, [P1, P2], "pizza?", lines, post=post, **kw)
    return new, posted, lines


async def test_burst_speaks_until_the_controller_waits():
    proxy = ScriptedProxy([speak(P1), speak(P2), {"action": WAIT}])

    new, posted, lines = await _burst(proxy)

    assert posted == [(P1, "line 1"), (P2, "line 2")]
    assert [line.persona for line in new] == [P1, P2]
    assert lines[-1] is new[-1]
    # Each persona is served its own adapter on the shared base.
    assert [c["lora"] for c in proxy.persona_calls][0][0]["scale"] == MIMIC_ADAPTER_SCALE


async def test_each_persona_reads_the_lines_before_it():
    proxy = ScriptedProxy([speak(P1), speak(P2), {"action": WAIT}])

    await _burst(proxy)

    second = proxy.persona_calls[1]["messages"][1]["content"]
    assert second.startswith("[erin]: pizza?")
    assert "line 1" in second


async def test_burst_is_capped_however_chatty_the_controller_is():
    proxy = ScriptedProxy([speak(P1), speak(P2)] * MIMIC_CHAT_MAX_BURST)

    new, _, _ = await _burst(proxy)

    assert len(new) == MIMIC_CHAT_MAX_BURST


async def test_a_human_message_interrupts_the_burst():
    proxy = ScriptedProxy([speak(P1)] * 5)
    flag = {"n": 0}

    def interrupted():
        flag["n"] += 1
        # Checked before each line and again once it is written: a human posts
        # while the second line is being written.
        return flag["n"] > 3

    new, posted, _ = await _burst(proxy, interrupted=interrupted)

    # The second line is discarded, so the human's message stays the latest.
    assert len(new) == 1 and len(posted) == 1
    assert len(proxy.persona_calls) == 2


async def test_a_set_length_runs_without_waiting_and_cannot_end_early():
    proxy = ScriptedProxy([speak(P1), speak(P2), speak(P1), speak(P2), {"action": END}])

    new, _, _ = await _burst(proxy, length=6)

    assert len(new) == 4
    schemas = [controller.schema_actions(c["schema"]) for c in proxy.controller_calls]
    assert schemas[0] == [SPEAK]  # no ending in the first half, and never "wait"
    assert schemas[-1] == [SPEAK, END]


async def test_unusable_controller_output_falls_back_instead_of_stalling():
    proxy = ScriptedProxy(["garbage", "more garbage", "still garbage"])

    new, _, _ = await _burst(proxy, lines=[human()])

    # Fallback rotates P1 then P2, then waits after two bot lines.
    assert [line.persona for line in new] == [P1, P2]


# ---- simulate ------------------------------------------------------------

async def test_simulation_is_bot_only_and_stops_at_length():
    proxy = ScriptedProxy([speak(P1), speak(P2), speak(P1)])

    lines = await simulate(proxy, CAST.catalog, [P1, P2], "pizza?", length=3)

    assert lines[0].msg.text == "pizza?" and lines[0].persona is None
    assert [line.persona for line in lines[1:]] == [P1, P2, P1]
    for call in proxy.controller_calls:
        assert WAIT not in controller.schema_actions(call["schema"])
        assert "No real person is taking part" in call["messages"][0]["content"]


async def test_a_single_persona_simulation_needs_no_controller_until_it_may_end():
    proxy = ScriptedProxy([{"action": END}])

    lines = await simulate(proxy, CAST.catalog, [P1], "pizza?", length=4)

    assert len(lines) == 3  # seed, two forced lines, then the controller ends it
    assert len(proxy.controller_calls) == 1


async def test_a_human_line_is_always_answered_before_the_chat_can_wait():
    proxy = ScriptedProxy([speak(P1), {"action": WAIT}])

    new, _, _ = await _burst(proxy, lines=[human("fap until you die")])

    assert len(new) == 1
    enums = [controller.schema_actions(c["schema"]) for c in proxy.controller_calls]
    assert enums == [[SPEAK], [SPEAK, WAIT]]


def test_controller_tells_a_member_apart_from_the_persona_imitating_them():
    # The real member (id 11) speaks right after their own persona, whose lines carry id 11 too.
    lines = [bot(P1), Line(Msg("11", "x", T0, "for real"))]

    msgs = controller.build_messages([P1], {P1: "x"}, "pizza?", lines, True, [SPEAK, WAIT], "", None)

    assert "[x (bot)]: ...\n[x]: for real" in msgs[1]["content"]
    assert "- x (bot)" in msgs[0]["content"]


async def test_following_up_on_yourself_does_not_use_up_the_burst():
    from config import MIMIC_CHAT_FREE_FOLLOWUPS
    # P1 double-texts, then they alternate until the cap.
    proxy = ScriptedProxy([speak(P1), speak(P1)] + [speak(P2), speak(P1)] * MIMIC_CHAT_MAX_BURST)

    new, _, _ = await _burst(proxy)

    assert len(new) == MIMIC_CHAT_MAX_BURST + 1
    assert MIMIC_CHAT_FREE_FOLLOWUPS >= 1


async def test_free_follow_ups_run_out():
    from config import MIMIC_CHAT_FREE_FOLLOWUPS
    proxy = ScriptedProxy([speak(P1)] * 100)

    new, _, _ = await _burst(proxy)

    # Every (free + 1)th line from the same persona starts a new turn.
    assert len(new) == MIMIC_CHAT_MAX_BURST * (MIMIC_CHAT_FREE_FOLLOWUPS + 1)


def test_wait_counts_turns_not_lines():
    assert controller.bot_turns_since_human([human(), bot(P1), bot(P1), bot(P1)]) == 1
    assert controller.fallback([P1, P2], [human(), bot(P1), bot(P1)], [SPEAK, WAIT]).action == SPEAK


# ---- addressing ----------------------------------------------------------

def _label(p):
    return CAST.label(p)


async def test_a_persona_named_first_answers_without_the_controller():
    proxy = ScriptedProxy([{"action": WAIT}])

    new, _, _ = await _burst(proxy, lines=[human(f"{_label(P2).lower()}, pizza or not")])

    assert [line.persona for line in new] == [P2]
    # The first decision was made in code; the controller is asked only after P2 has answered.
    assert [controller.schema_actions(c["schema"]) for c in proxy.controller_calls] == [[SPEAK, WAIT]]


async def test_a_name_later_in_the_message_does_not_route():
    proxy = ScriptedProxy([speak(P1), {"action": WAIT}])

    new, _, _ = await _burst(proxy, lines=[human(f"what does {_label(P2)} think")])

    assert [line.persona for line in new] == [P1]


async def test_a_reply_to_a_persona_routes_to_it():
    proxy = ScriptedProxy([{"action": WAIT}])
    lines = [bot(P1), bot(P2), Line(Msg("22", "erin", T0, "no way"), reply_to=P1)]

    new, _, _ = await _burst(proxy, lines=lines)

    assert [line.persona for line in new] == [P1]


async def test_a_persona_does_not_answer_its_own_member_unless_called_in(training):
    cast = member_cast(training)

    proxy = ScriptedProxy([{"action": WAIT}])
    new, _, _ = await _burst(proxy, lines=[human("anyone?")], cast=cast)
    assert [line.persona for line in new] == [P2]  # P1 left out, so P2 needs no controller

    # Named without @: still left out, so the controller picks among the rest.
    proxy = ScriptedProxy([{"action": WAIT}])
    new, _, _ = await _burst(proxy, lines=[human(f"{_label(P1)} anyone?")], cast=cast)
    assert [line.persona for line in new] == [P2]

    # @mentioned or replied to by the member themself: it answers.
    proxy = ScriptedProxy([{"action": WAIT}])
    new, _, _ = await _burst(proxy, lines=[human(f"@{_label(P1)} you there")], cast=cast)
    assert [line.persona for line in new] == [P1]


async def test_a_lone_persona_answers_its_own_member(training):
    cast = member_cast(training)
    posted = []

    async def post(persona, text):
        posted.append(persona)

    await run_burst(ScriptedProxy([{"action": WAIT}]), cast, [P1], "pizza?", [human("hm")], post=post)

    assert posted == [P1]


def test_controller_sees_replies_and_discord_names():
    lines = [bot(P1), Line(Msg("22", "erin", T0, "no way"), reply_to=P1)]

    msgs = controller.build_messages(
        [P1], {P1: "x"}, "pizza?", lines, True, [SPEAK, WAIT], "", None, shown_as={P1: "Sir Lags"},
    )

    assert "[erin]: (replying to x (bot)) no way" in msgs[1]["content"]
    assert '- x (bot), shown in Discord as "Sir Lags"' in msgs[0]["content"]
    assert "- end:" not in msgs[0]["content"]  # only actions on offer are listed


async def test_only_a_one_on_one_chat_relabels_the_member(training):
    cast = member_cast(training)
    now = datetime.now(timezone.utc)
    lines = [Line(Msg("22", "erin", now, "talk to me"))]

    proxy = ScriptedProxy([{"action": WAIT}])
    await run_burst(proxy, cast, [P1], "pizza?", list(lines), post=_ignore)
    assert proxy.persona_calls[0]["messages"][1]["content"] == "[member]: talk to me"

    proxy = ScriptedProxy([speak(P1), {"action": WAIT}])
    await run_burst(proxy, cast, [P1, P2], "pizza?", list(lines), post=_ignore)
    assert proxy.persona_calls[0]["messages"][1]["content"] == "[erin]: talk to me"


async def _ignore(persona, text):
    pass


def test_controller_mentions_a_premise_only_when_there_is_one():
    lines = [Line(Msg("22", "erin", T0, "pizza?"))]

    without = controller.build_messages([P1], {P1: "x"}, None, lines, True, [SPEAK, WAIT], "", None)
    with_one = controller.build_messages([P1], {P1: "x"}, "pizza?", lines, True, [SPEAK, WAIT], "", None)

    assert "Premise" not in without[0]["content"]
    assert "{premise}" not in without[0]["content"]
    assert "Premise of the chat: pizza?\n\nActions:" in with_one[0]["content"]
