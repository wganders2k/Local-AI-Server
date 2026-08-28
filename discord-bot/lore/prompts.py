"""
The /lore agent's prompts.

Prose lives in ``prompts/*.md``; this module is the assembly — which template
goes where, and what fills its placeholders. Server-specific background (member
aliases, persona notes) is loaded separately from a host-local file and is
deliberately absent from the repo.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import LORE_CONTEXT_PATH
from prompt_loader import render

logger = logging.getLogger("mimic-bot.lore.prompts")

_lore_context_cache: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_lore_context() -> str:
    """
    Load server-specific background knowledge (member alias index, persona
    notes) from the file at config.LORE_CONTEXT_PATH.

    The file is deliberately not committed to the repo — it holds real names —
    and lives on the bot_context volume alongside the session store. If it is
    absent the agent still works; it just answers without the alias index.

    Returns:
        File contents stripped of surrounding whitespace, or "" if unavailable.
    """
    global _lore_context_cache
    if _lore_context_cache is not None:
        return _lore_context_cache

    path = Path(LORE_CONTEXT_PATH)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path

    try:
        _lore_context_cache = path.read_text(encoding="utf-8").strip()
        logger.info("Loaded lore context from %s (%d chars)", path, len(_lore_context_cache))
    except FileNotFoundError:
        logger.warning(
            "Lore context file not found at %s — running without server-specific "
            "background knowledge. Copy prompts/lore_context.example.md to create it.",
            path,
        )
        _lore_context_cache = ""
    except OSError as e:
        logger.error("Failed to read lore context file %s: %s", path, e)
        _lore_context_cache = ""

    return _lore_context_cache


def _background_block() -> str:
    """Render the server-specific background section, or "" if none is configured."""
    lore_context = load_lore_context()
    if not lore_context:
        return ""
    return f"\nGENERAL BACKGROUND KNOWLEDGE:\n{lore_context}\n"


def build_system_prompt(channel_names: list[str], now: Optional[str] = None) -> str:
    """
    Build the system prompt for the RAG agent.

    Args:
        channel_names: List of available Discord channel names the agent can search.
        now: Pre-rendered timestamp. Pass one to pin it for the life of a
            lore thread session — this string sits at the front of the cached
            prefix, so letting it tick would re-prefill the whole conversation
            on every turn. Defaults to the current time.

    Returns:
        Complete system prompt string.
    """
    return render(
        "lore_agent",
        identity=render("lore_identity"),
        now=now or _now(),
        channels="\n".join(f"  - {ch}" for ch in sorted(channel_names)),
        background=_background_block(),
        answer_rules=render("lore_answer_rules"),
    )


def build_lore_followup_prompt(
    channel_names: list[str],
    now: Optional[str] = None,
) -> str:
    """
    Build the system prompt for follow-up turns in a lore thread.

    Identical to the /lore agent prompt plus the follow-up clause — the clause
    that makes a thread a conversation rather than a series of cold /lore runs.
    Without it the model treats every follow-up as a fresh research task and
    re-searches material already sitting in the thread, which is both slow and
    the fastest way to exhaust the context window.

    Build this ONCE per session and store the result: it is the head of the
    cached prefix, so rebuilding it per turn — and letting its timestamp tick —
    would change the very first tokens and force a full re-prefill.

    Args:
        channel_names: Channels the agent may search.
        now: Pre-rendered timestamp to pin. Defaults to the current time.

    Returns:
        Complete follow-up system prompt string.
    """
    base = build_system_prompt(channel_names, now=now)
    return f"{base}\n\n{render('lore_followup')}"


def build_synthesis_prompt(now: Optional[str] = None) -> str:
    """
    Build the system prompt used when max_rounds is exhausted.

    This replaces the tool-calling prompt so the model stops emitting tool
    calls and simply synthesizes an answer from the tool results already in
    the conversation. It keeps the same background knowledge so member names
    are still resolved correctly in the final answer, and the same answer rules
    the tool-calling prompt carries — this is the turn that actually writes the
    user-facing prose, so it is the turn that most needs them.

    Args:
        now: Pre-rendered timestamp; defaults to the current time. See
            build_system_prompt() for why a caller would pin it.

    Returns:
        Complete synthesis system prompt string.
    """
    return render(
        "lore_synthesis",
        identity=render("lore_identity"),
        now=now or _now(),
        background=_background_block(),
        answer_rules=render("lore_answer_rules"),
    )


def build_compaction_messages(excerpts: str) -> list[dict]:
    """
    The two-message prompt that condenses a session's oldest research.

    Args:
        excerpts: The research blocks to compress, already joined.
    """
    return [
        {"role": "system", "content": render("lore_compaction_system")},
        {"role": "user", "content": render("lore_compaction_user", excerpts=excerpts)},
    ]
