"""
Names for /mimic, rendered the way the persona adapters were trained.

lora-training/build_dataset.py labels every speaker `[name]:` and renders every
mention `@name`, where `name` is the member's alias from names.json or else
their Discord handle. An adapter served anything else — a server nickname, raw
`<@123>` markup — is reading a format it never saw. This module applies the
same rules to what /mimic sends, from the same files:

    <MIMIC_TRAINING_DIR>/names.json            {user_id: {"alias": ...}}
    <MIMIC_TRAINING_DIR>/<persona>/stats.json  the build's arguments, including
                                               the display name the persona's
                                               system prompt was rendered with

Both hold real names, so they live beside the training data on the archive
disk and are mounted read-only, never committed. Both are read per request:
they are small, and an edit takes effect without a restart.
"""

import json
import logging
import os
import re
from typing import Callable, Optional

logger = logging.getLogger("mimic-bot.member_names")

MIMIC_TRAINING_DIR: str = os.environ.get("MIMIC_TRAINING_DIR", "/archive/training")

# The markup build_dataset.normalise rewrites, rewritten identically.
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def load_aliases(training_dir: Optional[str] = None) -> dict[str, str]:
    """{user_id: alias}. Same rules as build_dataset.load_aliases: "_" keys are notes, blank keeps the handle."""
    entries = _read_json(os.path.join(training_dir or MIMIC_TRAINING_DIR, "names.json")) or {}
    return {
        uid: entry["alias"].strip()
        for uid, entry in entries.items()
        if not uid.startswith("_") and isinstance(entry, dict) and (entry.get("alias") or "").strip()
    }


def persona_display_name(persona: str, training_dir: Optional[str] = None) -> Optional[str]:
    """The display name this persona's dataset was built with, or None if it has not been built."""
    stats = _read_json(os.path.join(training_dir or MIMIC_TRAINING_DIR, persona, "stats.json")) or {}
    return (stats.get("args") or {}).get("display_name")


def member_name(user_id: int | str, handle: str, aliases: dict[str, str]) -> str:
    return aliases.get(str(user_id)) or handle


def normalise_message(
    text: str,
    name_for_id: Callable[[str], str],
    channel_name_for_id: Callable[[str], str],
) -> str:
    """Raw Discord markup -> what training saw. `name_for_id` should already apply aliases."""
    text = _USER_MENTION_RE.sub(lambda m: "@" + name_for_id(m.group(1)), text)
    text = _CHANNEL_MENTION_RE.sub(lambda m: "#" + channel_name_for_id(m.group(1)), text)
    text = _ROLE_MENTION_RE.sub("@role", text)
    text = _CUSTOM_EMOJI_RE.sub(r":\1:", text)
    return text.strip()
