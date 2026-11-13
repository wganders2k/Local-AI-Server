"""The cut itself: which turns survive, and that the rest of an example is untouched."""

import cut_context as cc

EXAMPLE = {
    "prompt": [{"role": "user", "content": "[a]: one\n[b]: two\nstill two\n[c]: three"}],
    "completion": [{"role": "assistant", "content": "reply"}],
    "context_turns": ["[a]: one", "[b]: two\nstill two", "[c]: three"],
    "timestamp": "2025-01-01T00:00:00+00:00",
}


def test_a_cut_keeps_the_newest_turns_whole():
    out = cc.cut(EXAMPLE, 2)
    assert out["prompt"] == [{"role": "user", "content": "[b]: two\nstill two\n[c]: three"}]
    assert out["context_turns"] == ["[b]: two\nstill two", "[c]: three"]
    assert out["completion"] == EXAMPLE["completion"] and out["cut_from"] == 3
    assert EXAMPLE["context_turns"][0] == "[a]: one"  # the source example is not modified


def test_a_system_message_survives_a_cut():
    ex = {**EXAMPLE, "prompt": [{"role": "system", "content": "S"}] + EXAMPLE["prompt"]}
    assert [m["role"] for m in cc.cut(ex, 1)["prompt"]] == ["system", "user"]


def test_turns_are_numbered_back_from_the_newest():
    assert cc.numbered(["[a]: one", "[b]: two", "[c]: three"]) == "(3) [a]: one\n(2) [b]: two\n(1) [c]: three"


def test_the_schema_bounds_the_answer_to_the_turns_there_are():
    s = cc.schema(7, explain=False)["json_schema"]["schema"]
    assert s["properties"]["turns"] == {"type": "integer", "minimum": 1, "maximum": 7} and s["required"] == ["turns"]
    assert "why" in cc.schema(7, explain=True)["json_schema"]["schema"]["required"]


def test_the_cache_key_changes_with_anything_that_changes_the_label(monkeypatch):
    k = cc.cache_key(["[a]: x"], "r")
    assert k == cc.cache_key(["[a]: x"], "r")
    assert k != cc.cache_key(["[a]: y"], "r") != cc.cache_key(["[a]: x"], "s")
    monkeypatch.setattr(cc, "PROMPT_VERSION", cc.PROMPT_VERSION + 1)
    assert k != cc.cache_key(["[a]: x"], "r")


def test_the_uncut_share_is_stable_and_roughly_right():
    keys = [cc.cache_key([f"[a]: {i}"], "r") for i in range(2000)]
    picked = [k for k in keys if cc.keep_whole(k, 0.2)]
    assert 0.15 < len(picked) / len(keys) < 0.25
    assert picked == [k for k in keys if cc.keep_whole(k, 0.2)]
    assert not any(cc.keep_whole(k, 0.0) for k in keys)


def test_no_cut_goes_below_the_floor_or_past_the_example():
    assert cc.turns_to_keep(1, 20) == 4          # keyboard mash, a non-sequitur: the floor
    assert cc.turns_to_keep(9, 20) == 9          # the judge's label, when it is above the floor
    assert cc.turns_to_keep(1, 3) == 3           # an example shorter than the floor keeps everything
    assert cc.turns_to_keep(2, 20, min_turns=1) == 2


def test_the_prompt_asks_for_a_generous_cut():
    text = cc.prompt_for(["[a]: x", "[b]: y"], "reply", "alice")[0]["content"]
    assert "tangentially" in text and "When unsure whether a turn belongs, include it" in text


def test_a_label_cut_short_after_a_whole_number_stands():
    assert cc.parse_label('{"turns": 7}') == {"turns": 7}
    assert cc.parse_label('{\n  "turns": 11\n') == {"turns": 11, "cut_short": True}
    for broken in ('{\n  "turns": 1', '{\n  "tu'):
        try:
            cc.parse_label(broken)
        except ValueError:
            continue
        raise AssertionError(f"{broken!r} should not parse: the number may be unfinished")
