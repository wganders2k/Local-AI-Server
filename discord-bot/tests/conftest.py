"""
Shared fixtures.

DISCORD_TOKEN is set before config is imported anywhere: validate_config() is
only called at startup, but importing config with a real .env present would
otherwise pull the developer's token into the test process.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DISCORD_TOKEN", "test-token")
# The lore context file is host-local and absent in CI. Point at a path that
# does not exist so tests exercise the documented "running without background
# knowledge" path rather than picking up a developer's real alias index.
os.environ["LORE_CONTEXT_PATH"] = "tests/fixtures/absent_context.md"
# Likewise the mimic names and persona stats, which hold real names.
os.environ["MIMIC_TRAINING_DIR"] = "tests/fixtures/absent_training"
# And the live adapters' manifests, so a test never serves by a real export.
os.environ["MIMIC_LORA_DIR"] = "tests/fixtures/absent_lora"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

PERSONA_KEYS = [f"mimic_user{i}" for i in range(1, 8)]


def write_roster(directory: Path, roster: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "personas.json").write_text(json.dumps(roster))
    return directory


@pytest.fixture
def training(tmp_path):
    """
    A training dir holding only a roster: mimic_userN is "UserN", posting from
    account 1N. No names.json, no stats — every persona falls back to
    build_dataset's defaults. Tests add files as they need them.
    """
    return write_roster(tmp_path / "training", {
        key: {"user_ids": str(10 + i), "display_name": f"User{i}"}
        for i, key in enumerate(PERSONA_KEYS, start=1)
    })


@pytest.fixture
def catalog(training):
    """The fixture roster, served by [gemma4-12b-mimic-u] adapters in key order — discovery already done."""
    from personas import PersonaCatalog

    return PersonaCatalog(training_dir=str(training), adapters=list(PERSONA_KEYS))


@pytest.fixture
def fake_rag():
    """A RAG client that returns whatever the test tells it to."""

    class FakeRAG:
        def __init__(self):
            self.retrieve_result = ("", 0)
            self.literal_result = ("", 0)
            self.aggregate_result = ("", 0)
            self.calls: list[tuple[str, dict]] = []

        async def retrieve(self, query, **kw):
            self.calls.append(("retrieve", {"query": query, **kw}))
            return self.retrieve_result

        async def search_literal(self, **kw):
            self.calls.append(("search_literal", kw))
            return self.literal_result

        async def aggregate(self, **kw):
            self.calls.append(("aggregate", kw))
            return self.aggregate_result

    return FakeRAG()
