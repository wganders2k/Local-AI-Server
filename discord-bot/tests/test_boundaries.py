"""
The agent and the multi-persona loop import no discord.

That boundary is what lets tests drive a whole /lore run or /mimic-chat burst
against a fake backend, and it has been broken by accident before: lore's
simulate_conversation reached discord through the module that read Discord
history. Checked in a fresh interpreter, where importing discord raises.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

BOT = Path(__file__).resolve().parent.parent

CORE = ["lore.agent", "lore.tools", "lore.compaction", "mimic.engine", "mimic.controller", "personas"]

_PROBE = """
import importlib, sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "discord" or name.startswith("discord."):
            raise ImportError("discord imported by the core: " + name)
sys.meta_path.insert(0, Block())
importlib.import_module(sys.argv[1])
"""


@pytest.mark.parametrize("module", CORE)
def test_core_module_imports_without_discord(module):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(BOT), str(BOT.parent / "shared")]),
        "DISCORD_TOKEN": "test-token",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module], cwd=BOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1]
