"""
/mimic's names must match the persona adapters' training data exactly
(lora-training/build_dataset.py): a label or mention in any other form is a
prompt the adapter never saw.
"""

import json

import config
import member_names as mn


def _training_dir(tmp_path, names=None, stats=None):
    if names is not None:
        (tmp_path / "names.json").write_text(json.dumps(names))
    if stats is not None:
        (tmp_path / "mimic_user1").mkdir()
        (tmp_path / "mimic_user1" / "stats.json").write_text(json.dumps(stats))
    return str(tmp_path)


def test_aliases_follow_the_training_rules(tmp_path):
    d = _training_dir(tmp_path, names={
        "_readme": ["notes"],
        "1": {"alias": " will ", "handle": "pcow"},
        "2": {"alias": "", "handle": "bearkouda"},
    })

    assert mn.load_aliases(d) == {"1": "will"}


def test_missing_files_mean_no_aliases_and_no_display_name(tmp_path):
    assert mn.load_aliases(str(tmp_path)) == {}
    assert mn.persona_display_name("mimic_user1", str(tmp_path)) is None


def test_member_name_prefers_alias_then_handle():
    assert mn.member_name(1, "pcow", {"1": "will"}) == "will"
    assert mn.member_name(2, "bearkouda", {"1": "will"}) == "bearkouda"


def test_markup_is_normalised_like_the_dataset():
    names = {"1": "will", "2": "kain"}
    text = "<@1> and <@!2> in <#9> ping <@&5> <:kekw:123> <a:dance:456>"

    out = mn.normalise_message(text, lambda uid: names.get(uid, "someone"), lambda cid: "general")

    assert out == "@will and @kain in #general ping @role :kekw: :dance:"


def test_persona_display_name_comes_from_the_build(tmp_path):
    d = _training_dir(tmp_path, stats={"args": {"display_name": "chris", "persona": "mimic_user1"}})

    assert mn.persona_display_name("mimic_user1", d) == "chris"


def test_system_prompt_uses_the_trained_display_name(tmp_path, monkeypatch):
    d = _training_dir(tmp_path, stats={"args": {"display_name": "chris"}})
    monkeypatch.setattr(mn, "MIMIC_TRAINING_DIR", d)

    assert "mimics chris's Discord personality" in config.get_mimic_system_prompt("mimic_user1")


def test_system_prompt_falls_back_to_the_placeholder_before_a_build():
    assert "mimics User1's Discord personality" in config.get_mimic_system_prompt("mimic_user1")
