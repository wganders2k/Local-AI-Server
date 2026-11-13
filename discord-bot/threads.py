"""
Every thread the bot answers in, and how — on disk and in memory, in one place.

Three kinds, each answered by one cog:

    chat        /chat: one model, plain history          cogs/chat.py
    lore        /lore follow-ups, from the session store  cogs/lore.py
    mimic_chat  /mimic-chat: personas, from the thread    cogs/mimic_chat.py

The JSON file is the only record of which threads those are — scanning every
channel on Discord would be far slower and is rate-limited — so it is restored
on startup and written whenever a thread opens or closes. Opening and closing
go through here and nowhere else, so the file and the in-memory routing cannot
disagree.

On the bot_context volume, NOT in the repo. It used to live in
discord-bot/data/, which put runtime state inside the working tree — and a
branch switch that crossed the commit removing the (then-tracked) file deleted
it out from under the running container's bind mount, taking every thread
binding with it.

Entry format, unchanged from the registry this replaced:

    {"1234567890": {"model": "...", "name": "...", "created_at": "...",
                    "status": "active", "kind": "chat", ...kind-specific fields}}

Entries written before `kind` existed read back as "chat".
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import discord

from mimic.state import MimicChatState

logger = logging.getLogger("mimic-bot.threads")

CHAT, LORE, MIMIC_CHAT = "chat", "lore", "mimic_chat"


@dataclass
class LiveThread:
    kind: str
    model: str


class ThreadDirectory:
    """
    The registry file plus the routing built from it.

    Attributes:
        mimic_chats: thread_id -> the /mimic-chat thread's personas, starter, length
            and pacing. Its transcript is the thread itself, read back each turn.
        claiming: Offer message ids currently being turned into a lore thread.
            Two people reacting within the same second would otherwise both
            pass the "is there an offer?" check and open a thread each, since
            the first await between check and claim lets the second run.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        self._live: dict[int, LiveThread] = {}
        # One lock per thread. A lore turn reads the session, appends its
        # question, runs for a minute or more, then appends its answer — so two
        # messages close together would interleave those appends and race the
        # save (observed: question, question, answer, answer). A mimic chat
        # serialises its bursts on the same lock.
        self._locks: dict[int, asyncio.Lock] = {}
        self.mimic_chats: dict[int, MimicChatState] = {}
        self.claiming: set[int] = set()
        self._load()

    # ---- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("No thread registry at %s — starting fresh", self._path)
            return
        try:
            with open(self._path) as f:
                self._data = dict(json.load(f))
            logger.info("Loaded %d thread(s) from %s", len(self._data), self._path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning("Failed to load thread registry %s: %s — starting fresh", self._path, e)
            self._data = {}

    def _save(self) -> None:
        """Atomically: temp file + rename, so a crash mid-write cannot truncate the registry."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.error("Failed to save thread registry %s: %s", self._path, e)

    # ---- lifecycle ----------------------------------------------------------

    def open(self, thread_id: int, kind: str, model: str, name: str, **extra: Any) -> None:
        """Start answering in a thread, and remember it across restarts."""
        self._data[str(thread_id)] = {
            "model": model,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "kind": kind,
            **extra,
        }
        self._save()
        self._live[thread_id] = LiveThread(kind, model)
        logger.info("Opened %s thread %d (%s) model=%s", kind, thread_id, name, model)

    def open_mimic_chat(self, thread_id: int, model: str, name: str, state: MimicChatState) -> None:
        self.mimic_chats[thread_id] = state
        self.open(thread_id, MIMIC_CHAT, model, name, **state.to_registry())

    def close(self, thread_id: int) -> bool:
        """Stop answering in a thread and forget it everywhere. Stores of its own (sessions) are the caller's."""
        known = self._data.pop(str(thread_id), None) is not None
        if known:
            self._save()
        self._live.pop(thread_id, None)
        self._locks.pop(thread_id, None)
        state = self.mimic_chats.pop(thread_id, None)
        if state is not None and state.pending is not None:
            state.pending.cancel()
        if known:
            logger.info("Closed thread %d", thread_id)
        return known

    # ---- routing ------------------------------------------------------------

    def kind_of(self, thread_id: int) -> Optional[str]:
        live = self._live.get(thread_id)
        return live.kind if live else None

    def model_of(self, thread_id: int) -> Optional[str]:
        live = self._live.get(thread_id)
        return live.model if live else None

    def lock_for(self, thread_id: int) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def is_busy(self, thread_id: int) -> bool:
        lock = self._locks.get(thread_id)
        return lock is not None and lock.locked()

    # ---- startup ------------------------------------------------------------

    async def restore(self, fetch_channel, has_session: Callable[[int], bool]) -> None:
        """
        Route every registered thread that still exists. Archived threads are
        restored too: Discord unarchives one on a new message, and the bot
        should answer it. A thread that is gone is dropped from the file.

        A lore thread only works if its session survived as well; without one
        there is no transcript to answer from, so it is answered as a plain
        chat thread rather than erroring later.
        """
        if not self._data:
            logger.info("No threads in registry — nothing to restore")
            return
        restored = gone = skipped = 0
        for key, entry in list(self._data.items()):
            thread_id, name = int(key), entry.get("name", "unknown")
            try:
                channel = await fetch_channel(thread_id)
            except discord.NotFound:
                channel = None
            except discord.Forbidden:
                skipped += 1
                logger.warning("Forbidden accessing thread %d (%s) — skipping restore", thread_id, name)
                continue
            except Exception:
                skipped += 1
                logger.exception("Unexpected error restoring thread %d (%s)", thread_id, name)
                continue
            if not isinstance(channel, discord.Thread):
                self.close(thread_id)
                gone += 1
                logger.info("Thread %d (%s) no longer exists — dropped", thread_id, name)
                continue

            kind, model = entry.get("kind", CHAT), entry.get("model", "")
            if kind == MIMIC_CHAT:
                self.mimic_chats[thread_id] = MimicChatState.from_registry(entry)
            elif kind == LORE and not has_session(thread_id):
                logger.warning(
                    "Thread %d (%s) is registered as lore but has no saved session — "
                    "it will behave as a plain chat thread", thread_id, name,
                )
                kind = CHAT
            self._live[thread_id] = LiveThread(kind, model)
            restored += 1
            logger.info(
                "Restored %s thread %d (%s) model=%s%s",
                kind, thread_id, name, model, " (archived)" if channel.archived else "",
            )
        logger.info("Thread restore complete: %d restored, %d gone, %d skipped", restored, gone, skipped)
