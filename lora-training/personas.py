"""
The persona roster: which Discord account(s) each mimic persona is built from.

    {"mimic_user3": {"user_ids": "456…,111…", "display_name": "…"}, …}

It lives beside names.json on the archive disk, not in the repo, for the same
reason: display names are real names. Everything that loops over personas —
build_all_datasets.py, train_queue.sh, export_latest.py — reads it, so adding a
persona is one entry here plus its configs/<persona>.yaml and models.ini section.

CPU only, stdlib only — imported by the torch-free tests.
"""

import json
import os

ROSTER_PATH = os.environ.get("PERSONA_ROSTER", "/archive/training/personas.json")


class RosterError(Exception):
    pass


def load_roster(path: str = ROSTER_PATH) -> dict[str, dict]:
    if not os.path.isfile(path):
        raise RosterError(f"No persona roster at {path}")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    roster = {}
    for persona, entry in raw.items():
        if persona.startswith("_"):
            continue  # _readme and the like
        if not isinstance(entry, dict) or not entry.get("user_ids") or not entry.get("display_name"):
            raise RosterError(f"{path}: {persona} needs both user_ids and display_name")
        roster[persona] = entry
    if not roster:
        raise RosterError(f"{path} lists no personas")
    return roster


def select(roster: dict[str, dict], wanted: str | list[str] | None) -> list[str]:
    """The personas to act on: all of them, or the named ones in the order given."""
    if not wanted:
        return sorted(roster)
    names = wanted.split() if isinstance(wanted, str) else list(wanted)
    unknown = [n for n in names if n not in roster]
    if unknown:
        raise RosterError(
            f"Not in the roster: {', '.join(unknown)}. Known: {', '.join(sorted(roster))}"
        )
    return names
