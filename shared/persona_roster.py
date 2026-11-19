"""
Who the mimic personas are, and which adapter serves each one.

Two facts, each with exactly one source:

* The roster — persona -> Discord account(s) and display name:

      {"mimic_user3": {"user_ids": "456…,111…", "display_name": "…"}, …}

  It lives beside names.json on the archive disk, not in the repo, for the same
  reason: display names are real names. Training reads it to build datasets
  and to know whom to queue and export; the bot reads it to know who it can
  mimic and what their system prompt calls them.

* The adapter order — a persona's adapter id is its position in the `lora`
  list of [gemma4-12b-mimic-u] in models.ini. Nobody copies that list: callers read it
  from the proxy's /v1/models, whose entries carry the argv llama-server
  launches each model with (adapter_order, below). A copy that drifted would
  serve one member's voice under another's name.

So adding a persona is a roster entry, its configs/<persona>.yaml, and its GGUF
on the [gemma4-12b-mimic-u] list.

A third fact rides with each adapter: how its training prompts were shaped
(system prompt or not, context turns, silence gap, merge window). It is the
dataset's build arguments as they were when the run trained, copied by the
worker into the run and by export_latest into the <persona>-lora.json beside
the GGUF (trained_format, below). Serving reads it from there, never from the
persona's current stats.json: a rebuilt dataset describes the next adapter,
not the one live now.

Pure stdlib: copied into the bot's slim image and the trainer image alike.

    python3 persona_roster.py list [--roster PATH]   # keys, for train_queue.sh
"""

import json
import os
import sys
from typing import Iterable, Mapping, Optional

from mimic_format import aliases_from

ROSTER_PATH = os.environ.get("PERSONA_ROSTER", "/archive/training/personas.json")

# The [gemma4-12b-mimic-u] section in models.ini, and how its adapter files are named.
MULTI_MODEL = "gemma4-12b-mimic-u"
ADAPTER_SUFFIX = "-lora.gguf"
# Beside each GGUF: which run it came from, and trained_format's record.
MANIFEST_SUFFIX = "-lora.json"
LORA_DIR = os.environ.get("LORA_GGUF_DIR", "/srv/models/lora")
# The manifest key holding the run's dataset build arguments, and the file in a
# run's output dir the worker copies the dataset's stats.json to.
TRAINED_FORMAT_KEY = "dataset"
RUN_DATASET_STATS = "dataset_stats.json"
# [gemma4-12b-mimic-candidate-u]: one adapter not yet live, for judging against the live one.
CANDIDATE_MODEL = "gemma4-12b-mimic-candidate-u"

# How strongly a persona's adapter is applied when served. Below 1 the base
# model's instruction-following leaks back in, so a steering message lands more
# often, at some cost to the voice; much under 0.7 the base's phrasing shows
# through. Shared so a sampled reply (lora-training/sample_persona.py) is what
# /mimic would post; MIMIC_ADAPTER_SCALE overrides it on both sides.
DEFAULT_ADAPTER_SCALE = 0.85


def lora_scales(adapters: list[str], persona: Optional[str], scale: float) -> list[dict]:
    """A request's per-adapter scales: `persona`'s at `scale`, every other at 0. None = the bare base."""
    if persona is not None and persona not in adapters:
        raise KeyError(f"{persona} has no adapter in {MULTI_MODEL}")
    return [{"id": i, "scale": scale if p == persona else 0.0} for i, p in enumerate(adapters)]


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


def user_ids(entry: Mapping) -> tuple[str, ...]:
    """A roster entry's accounts: comma-separated, old account first."""
    return tuple(u.strip() for u in str(entry.get("user_ids") or "").split(",") if u.strip())


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


def load_aliases(path: Optional[str]) -> dict[str, str]:
    """
    {user_id: alias} from the hand-filled names.json, or {} if there is none.
    Keys starting with "_" are notes, and an empty alias means "keep their handle".
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return aliases_from(entries)


def trained_format(name: str, lora_dir: Optional[str] = None) -> dict:
    """
    The build arguments of the dataset that the adapter now exported as `name`
    (a persona key, or "candidate") was trained on: no_system, context_n,
    gap_minutes, merge_seconds and the rest of build_dataset's args. {} when
    there is no manifest or it predates the record — every such adapter was
    trained with the system prompt and build_dataset's defaults, which is what
    callers fall back to.
    """
    path = os.path.join(lora_dir or LORA_DIR, name + MANIFEST_SUFFIX)
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh).get(TRAINED_FORMAT_KEY)
    except (OSError, ValueError, AttributeError):
        return {}
    return record if isinstance(record, dict) else {}


def adapter_persona(path: str) -> Optional[str]:
    """`/srv/models/lora/mimic_user3-lora.gguf` -> "mimic_user3"; None for any other file."""
    base = os.path.basename(path.strip())
    return base[: -len(ADAPTER_SUFFIX)] if base.endswith(ADAPTER_SUFFIX) else None


def adapter_order(models_payload: Iterable[Mapping], model: str = MULTI_MODEL) -> list[str]:
    """
    The personas `model` preloads, in adapter-id order, from the proxy's
    /v1/models `data` array. In router mode each entry's status.args is the argv
    llama-server would launch it with, e.g.

        [..., "--lora", "/srv/models/lora/a-lora.gguf,/srv/models/lora/b-lora.gguf",
         "--lora-init-without-apply", ...]

    [] when the model is absent or carries no adapters, so a caller can tell
    "not discovered" from a real order. A repeated --lora flag appends, as it
    does for llama-server itself.
    """
    for entry in models_payload:
        if entry.get("id") != model:
            continue
        args = (entry.get("status") or {}).get("args")
        if not isinstance(args, list):
            return []
        paths: list[str] = []
        for i, flag in enumerate(args[:-1]):
            if flag == "--lora":
                paths += [p for p in str(args[i + 1]).split(",") if p.strip()]
        return [adapter_persona(p) or os.path.basename(p) for p in paths]
    return []


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Persona roster")
    ap.add_argument("command", choices=["list"])
    ap.add_argument("--roster", default=ROSTER_PATH)
    args = ap.parse_args(argv)
    try:
        print(" ".join(sorted(load_roster(args.roster))))
    except RosterError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
