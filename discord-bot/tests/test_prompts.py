"""
The prompt templates and their assembly.

These are the highest-leverage tests in the suite: the /lore system prompt is
the head of a cached prefix, so an accidental change to it is both a behaviour
change and a performance cliff (a full cold prefill on every turn).
"""

import pytest

import prompt_loader
from lore import prompts


TEMPLATES = [
    "lore_identity",
    "lore_agent",
    "lore_answer_rules",
    "lore_followup",
    "lore_synthesis",
    "lore_compaction_system",
    "lore_compaction_user",
    "mimic_persona",
]


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_template_exists_and_is_non_empty(name):
    assert prompt_loader.load(name).strip()


def test_missing_template_raises_rather_than_rendering_a_hole():
    with pytest.raises(prompt_loader.PromptError):
        prompt_loader.load("no_such_template")


def test_missing_placeholder_raises():
    with pytest.raises(prompt_loader.PromptError) as excinfo:
        prompt_loader.render("mimic_persona", persona="x")  # display_name omitted
    assert "display_name" in str(excinfo.value)


def test_system_prompt_has_no_unrendered_placeholders():
    out = prompts.build_system_prompt(["general", "lore"], now="2026-01-01 00:00 UTC")
    assert "{" not in out and "}" not in out


def test_system_prompt_pins_the_timestamp_it_is_given():
    # A ticking timestamp at the head of the prompt voids the prefix cache.
    out = prompts.build_system_prompt(["general"], now="PINNED")
    assert "CURRENT DATE/TIME: PINNED" in out


def test_channels_are_sorted_and_bulleted():
    out = prompts.build_system_prompt(["zulu", "alpha"], now="X")
    assert "  - alpha\n  - zulu" in out


def test_followup_prompt_is_the_agent_prompt_plus_the_clause():
    base = prompts.build_system_prompt(["general"], now="X")
    followup = prompts.build_lore_followup_prompt(["general"], now="X")
    assert followup.startswith(base)
    assert followup.endswith(prompt_loader.load("lore_followup"))


def test_both_answering_prompts_carry_the_answer_rules():
    # The synthesis turn is the one that writes user-visible prose, so it is the
    # turn that most needs the anti-fabrication and citation rules.
    rules = prompt_loader.load("lore_answer_rules")
    assert rules in prompts.build_system_prompt(["general"], now="X")
    assert rules in prompts.build_synthesis_prompt(now="X")


def test_compaction_messages_carry_the_excerpts():
    msgs = prompts.build_compaction_messages("EXCERPT-BODY")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "EXCERPT-BODY" in msgs[1]["content"]


def test_missing_lore_context_degrades_to_no_background(monkeypatch):
    monkeypatch.setattr(prompts, "_lore_context_cache", None)
    out = prompts.build_system_prompt(["general"], now="X")
    assert "GENERAL BACKGROUND KNOWLEDGE" not in out
