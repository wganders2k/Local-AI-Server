"""Who a human message is aimed at: a reply, or the first word only."""

from datetime import datetime, timezone

from mimic.addressing import Address, addressed, who_may_answer
from mimic.transcript import Line
from mimic_format import Msg

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
P1, P2 = "mimic_user1", "mimic_user2"
NAMES = {P1: {"alice", "Sir Lags-a-Lot"}, P2: {"dana"}}


def human(text, reply_to=None, uid="22"):
    return Line(Msg(uid, "erin", T0, text), reply_to=reply_to)


def test_first_word_names_with_punctuation_and_any_case():
    for text in ("dana what do you think", "Dana, thoughts?", "DANA: well", "dana?"):
        assert addressed(human(text), [P1, P2], NAMES) == Address((P2,), direct=False)


def test_an_at_mention_is_direct():
    assert addressed(human("@dana you there"), [P1, P2], NAMES) == Address((P2,), direct=True)


def test_only_the_first_word_counts():
    assert addressed(human("ask dana about it"), [P1, P2], NAMES) is None
    assert addressed(human("kainite is a mineral"), [P1, P2], NAMES) is None


def test_a_multi_word_nickname_is_not_matched_by_its_first_word():
    assert addressed(human("sir what"), [P1, P2], NAMES) is None


def test_a_reply_outranks_the_first_word():
    assert addressed(human("dana is wrong", reply_to=P1), [P1, P2], NAMES) == Address((P1,), direct=True)


def test_a_reply_to_a_persona_outside_the_chat_is_ignored():
    assert addressed(human("lol", reply_to="mimic_user5"), [P1, P2], NAMES) is None


def test_the_newest_address_in_a_run_of_human_lines_wins():
    lines = [Line(Msg("11", "x", T0, "..."), P1), human("dana"), human("what do you think")]
    assert who_may_answer([P1, P2], lines, NAMES) == ([P2], P2)


def test_someone_elses_mention_does_not_override_the_member_rule():
    lines = [human("@alice you there", uid="33"), human("lol he's asleep")]  # 22 is alice's member
    assert who_may_answer([P1, P2], lines, NAMES, {P1: ("22",)}) == ([P2], None)
