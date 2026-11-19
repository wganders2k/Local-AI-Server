"""
The roster is the one place a persona's Discord accounts are written down, so a
malformed entry must fail loudly rather than build a dataset for nobody.

    python -m pytest tests/test_persona_roster.py -q
"""

import json

import pytest

import persona_roster as personas


def _roster(tmp_path, data):
    path = tmp_path / "personas.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_readme_keys_are_not_personas(tmp_path):
    path = _roster(tmp_path, {
        "_readme": ["…"],
        "mimic_user1": {"user_ids": "1", "display_name": "a"},
    })
    assert list(personas.load_roster(path)) == ["mimic_user1"]


def test_an_entry_without_ids_is_rejected(tmp_path):
    path = _roster(tmp_path, {"mimic_user1": {"user_ids": "", "display_name": "a"}})
    with pytest.raises(personas.RosterError, match="mimic_user1"):
        personas.load_roster(path)


def test_a_missing_roster_says_where_it_looked(tmp_path):
    with pytest.raises(personas.RosterError, match="nope.json"):
        personas.load_roster(str(tmp_path / "nope.json"))


def test_select_defaults_to_everyone_and_keeps_a_given_order():
    roster = {"mimic_user2": {}, "mimic_user1": {}}
    assert personas.select(roster, "") == ["mimic_user1", "mimic_user2"]
    assert personas.select(roster, "mimic_user2 mimic_user1") == ["mimic_user2", "mimic_user1"]


def test_a_typo_lists_the_real_names():
    with pytest.raises(personas.RosterError, match="mimic_user1"):
        personas.select({"mimic_user1": {}}, "mimic_usr1")


# The shape the proxy's /v1/models really returns for [gemma4-12b-mimic-u], captured
# from the live router: one --lora flag, comma-joined paths, then more flags.
_MULTI = {
    "id": "gemma4-12b-mimic-u",
    "status": {"args": [
        "--model", "/srv/models/base.gguf",
        "--lora", "/srv/models/lora/mimic_user2-lora.gguf,/srv/models/lora/mimic_user1-lora.gguf",
        "--lora-init-without-apply", "--min-p", "0.0",
    ]},
}


def test_adapter_order_is_the_lora_list_order():
    payload = [{"id": "lore", "status": {"args": []}}, _MULTI]
    assert personas.adapter_order(payload) == ["mimic_user2", "mimic_user1"]


def test_adapter_order_is_empty_when_undiscoverable():
    assert personas.adapter_order([]) == []
    assert personas.adapter_order([{"id": "gemma4-12b-mimic-u"}]) == []
    assert personas.adapter_order([{"id": "gemma4-12b-mimic-u", "status": {"args": ["--model", "x"]}}]) == []


def test_repeated_lora_flags_append():
    entry = {"id": "gemma4-12b-mimic-u", "status": {"args": ["--lora", "/a/x-lora.gguf", "--lora", "/a/y-lora.gguf"]}}
    assert personas.adapter_order([entry]) == ["x", "y"]


def test_user_ids_split_and_trim():
    assert personas.user_ids({"user_ids": " 1, 2 ,"}) == ("1", "2")


def test_load_aliases_skips_notes_and_blanks(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"_note": "x", "1": {"alias": " bob "}, "2": {"alias": ""}}))
    assert personas.load_aliases(str(path)) == {"1": "bob"}
    assert personas.load_aliases(str(tmp_path / "missing.json")) == {}


def test_lora_scales_apply_exactly_one_adapter():
    scales = personas.lora_scales(["a", "b", "c"], "b", 0.85)
    assert scales == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.85}, {"id": 2, "scale": 0.0}]
    assert all(s["scale"] == 0.0 for s in personas.lora_scales(["a", "b"], None, 0.85))
    with pytest.raises(KeyError):
        personas.lora_scales(["a"], "z", 0.85)
