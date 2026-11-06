"""
The roster is the one place a persona's Discord accounts are written down, so a
malformed entry must fail loudly rather than build a dataset for nobody.

    python -m pytest tests/test_personas.py -q
"""

import json

import pytest

import personas


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
