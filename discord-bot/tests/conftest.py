"""
Shared fixtures.

DISCORD_TOKEN is set before config is imported anywhere: validate_config() is
only called at startup, but importing config with a real .env present would
otherwise pull the developer's token into the test process.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISCORD_TOKEN", "test-token")
# The lore context file is host-local and absent in CI. Point at a path that
# does not exist so tests exercise the documented "running without background
# knowledge" path rather than picking up a developer's real alias index.
os.environ["LORE_CONTEXT_PATH"] = "tests/fixtures/absent_context.md"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


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
