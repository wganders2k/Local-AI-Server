"""
The persona catalog. Serving the wrong adapter under a persona's name puts one
member's voice in another's mouth, so the adapter order is only ever read from
the backend — and a persona is available only when both the roster and the
backend know it.
"""

import json
import os

import pytest

from tests.conftest import PERSONA_KEYS, write_roster
from personas import PersonaCatalog, member_name

MULTI = {
    "id": "mimic-multi",
    "status": {"args": ["--lora", "/srv/models/lora/mimic_user2-lora.gguf,/srv/models/lora/mimic_user1-lora.gguf"]},
}


class FakeProxy:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def models_payload(self):
        self.calls += 1
        return self.payload


async def test_nothing_is_available_until_the_adapter_order_is_discovered(training):
    catalog = PersonaCatalog(training_dir=str(training))

    assert not catalog.ready and catalog.all() == []
    with pytest.raises(LookupError):
        catalog.lora("mimic_user1")

    assert not await catalog.ensure_discovered(FakeProxy([]))
    proxy = FakeProxy([MULTI])
    assert await catalog.ensure_discovered(proxy)
    assert await catalog.ensure_discovered(proxy)
    assert proxy.calls == 1  # cached once discovered


async def test_adapter_ids_follow_the_backend_not_the_roster(training):
    catalog = PersonaCatalog(training_dir=str(training))
    await catalog.discover(FakeProxy([MULTI]))

    # Only the two with adapters, even though the roster lists seven.
    assert catalog.keys() == ["mimic_user1", "mimic_user2"]
    assert catalog.get("mimic_user1").adapter_id == 1
    assert catalog.lora("mimic_user1") == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": catalog.scale}]
    assert all(s["scale"] == 0.0 for s in catalog.lora(None))
    with pytest.raises(KeyError):
        catalog.lora("mimic_user3")


def test_an_adapter_with_no_roster_entry_is_unavailable(tmp_path):
    d = write_roster(tmp_path, {"mimic_user1": {"user_ids": "11", "display_name": "Alice"}})
    catalog = PersonaCatalog(training_dir=str(d), adapters=["mimic_user1", "mimic_ghost"])

    assert catalog.keys() == ["mimic_user1"]
    assert "mimic_ghost" not in catalog


def test_roster_edits_apply_without_a_restart(tmp_path):
    d = write_roster(tmp_path, {"mimic_user1": {"user_ids": "11", "display_name": "Alice"}})
    catalog = PersonaCatalog(training_dir=str(d), adapters=["mimic_user1"])
    assert catalog.get("mimic_user1").display_name == "Alice"

    write_roster(tmp_path, {"mimic_user1": {"user_ids": "11, 12", "display_name": "Alicia"}})
    stat = os.stat(d / "personas.json")
    os.utime(d / "personas.json", (stat.st_atime, stat.st_mtime + 5))

    persona = catalog.get("mimic_user1")
    assert (persona.display_name, persona.user_ids) == ("Alicia", ("11", "12"))


def test_a_broken_roster_means_no_personas_rather_than_a_crash(tmp_path):
    (tmp_path / "personas.json").write_text("{not json")
    assert PersonaCatalog(training_dir=str(tmp_path), adapters=["mimic_user1"]).all() == []
    assert PersonaCatalog(training_dir=str(tmp_path / "missing"), adapters=["mimic_user1"]).all() == []


def test_system_prompt_is_the_trained_one_and_names_no_one(catalog):
    """Keyed by the persona alone: the member's name never enters a prompt, a dataset or an adapter."""
    prompt = catalog.system_prompt(catalog.get("mimic_user1"))
    assert prompt.startswith("You are mimic_user1") and "User1" not in prompt


def test_prompt_shape_comes_from_the_live_adapters_manifest(training, tmp_path):
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "mimic_user1-lora.json").write_text(json.dumps(
        {"source": "/runs/a", "dataset": {"context_n": 3, "merge_seconds": 60, "no_system": True}}))
    (lora / "mimic_user2-lora.json").write_text(json.dumps({"source": "/runs/b", "eval_loss": 3.4}))
    # A rebuilt dataset describes the next adapter, not the live one: never read.
    (training / "mimic_user2").mkdir()
    (training / "mimic_user2" / "stats.json").write_text(json.dumps({"args": {"context_n": 5, "no_system": True}}))
    catalog = PersonaCatalog(training_dir=str(training), lora_dir=str(lora), adapters=list(PERSONA_KEYS))

    recorded, legacy = catalog.get("mimic_user1"), catalog.get("mimic_user2")
    assert (recorded.context_n, recorded.merge_seconds, recorded.no_system) == (3, 60.0, True)
    assert catalog.system_prompt(recorded) is None
    # Exported before the record existed: trained with the system prompt and the build defaults.
    assert (legacy.context_n, legacy.merge_seconds, legacy.no_system) == (24, 600.0, False)
    assert catalog.system_prompt(legacy).startswith("You are mimic_user2")


def test_aliases_follow_the_training_rules(catalog, training):
    (training / "names.json").write_text(json.dumps({
        "_readme": ["notes"],
        "1": {"alias": " erin ", "handle": "erin_h"},
        "2": {"alias": "", "handle": "frank_h"},
    }))
    assert catalog.aliases() == {"1": "erin"}


def test_member_name_prefers_alias_then_handle():
    assert member_name(1, "erin_h", {"1": "erin"}) == "erin"
    assert member_name(2, "frank_h", {"1": "erin"}) == "frank_h"


def test_personas_are_named_by_display_name_or_key(catalog):
    by_name = catalog.by_name()
    assert by_name["user3"] == "mimic_user3" and by_name["mimic_user3"] == "mimic_user3"
