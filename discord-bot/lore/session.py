"""
Persistent per-thread state for lore follow-up threads.

A session is the conversation behind one lore thread: the system prompt built
when the thread was created, plus an append-only transcript of questions,
research blocks, and answers.

Append-only is the whole design. llama.cpp keeps a prefix cache keyed on the
longest common prefix of the previous request, so a follow-up that only adds
messages at the tail reuses the ~50k tokens of research already prefilled and
costs a small delta. Inserting or rewriting anything earlier throws that away
and forces a full cold prefill — around 50 seconds on the observed payloads.
Compaction is the one operation allowed to break the rule, and it pays that
cost knowingly.

Research is stored as flat text messages, never as raw ``tool_calls`` /
``role="tool"`` messages: leaving tool markup in context made the model imitate
it and emit tool-call syntax as its answer. See
``_flatten_research_for_synthesis`` in lore/research.py.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mimic-bot.lore-session")


def chunk_key(chunk: str) -> str:
    """
    Stable short identity for one retrieved chunk.

    Hashed rather than stored verbatim so cross-turn dedup state stays small —
    the raw chunks already live in the research blocks.
    """
    return hashlib.sha1(chunk.strip().encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class LoreSession:
    """
    One lore thread's conversation and context accounting.

    A session exists before its thread does: /lore builds one as a pending
    offer keyed by the offer message, and it only acquires a thread_id if
    somebody reacts. See LoreSessionStore.

    Attributes:
        thread_id: Discord thread this session backs, or 0 while it is still a
            pending offer.
        pinned_now: Timestamp rendered once at creation. The system prompt
            embeds it, and the system prompt is the first thing in the cached
            prefix — regenerating it per turn would change token 0 and void the
            cache on every message.
        system_prompt: Stored verbatim rather than rebuilt, so a restart (or a
            change to the guild's channel list) cannot alter the prefix.
        transcript: Every message after the system prompt, in order. Appended
            to and never mutated, except by compaction.
        seen_keys: chunk_key() of every chunk already shown, so a follow-up
            that re-runs a similar search does not re-add text already present.
    """

    thread_id: int
    system_prompt: str
    original_question: str
    pinned_now: str
    # Set while the session is a pending offer waiting on a reaction, and kept
    # afterwards so a thread can be traced back to the message that opened it.
    offer_message_id: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Drives expiry. Bumped on every turn, so a thread stays alive as long as
    # anyone is using it and ages out on a week of silence.
    last_active_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    transcript: List[dict] = field(default_factory=list)
    seen_keys: set[str] = field(default_factory=set)
    turns: int = 0
    searches: int = 0
    compactions: int = 0
    last_context_tokens: int = 0
    peak_context_tokens: int = 0

    # ---- message assembly -------------------------------------------------

    def build_messages(self, extra: Optional[List[dict]] = None) -> List[dict]:
        """
        Full request payload: system prompt, transcript, then any ephemeral
        scaffolding (e.g. a "now write the answer" nudge) that should steer
        this one call without becoming part of the thread's history.
        """
        msgs = [{"role": "system", "content": self.system_prompt}]
        # Project to role/content only. Transcript entries carry a "kind" tag so
        # compaction can find research blocks without pattern-matching their
        # text, and that tag must not reach the backend.
        msgs.extend(
            {"role": m["role"], "content": m["content"]} for m in self.transcript
        )
        if extra:
            msgs.extend(extra)
        return msgs

    def append_question(self, question: str) -> None:
        self.transcript.append(
            {"role": "user", "content": question, "kind": "question"}
        )
        # Touched here as well as on the answer, so a thread whose turns are
        # failing still counts as in use and is not swept mid-conversation.
        self.touch()

    def append_research(self, block: str) -> None:
        """Add one turn's retrieved material as a plain user message."""
        self.transcript.append(
            {"role": "user", "content": block, "kind": "research"}
        )
        self.searches += 1

    def append_answer(self, answer: str) -> None:
        self.transcript.append(
            {"role": "assistant", "content": answer, "kind": "answer"}
        )
        self.turns += 1
        self.touch()

    def touch(self) -> None:
        """Mark the session as used now, resetting its expiry clock."""
        self.last_active_at = datetime.now(timezone.utc).isoformat()

    def idle_seconds(self, now: Optional[datetime] = None) -> float:
        """
        Seconds since the last turn. Infinite when the timestamp is unusable, so
        a corrupt entry is swept rather than pinned in the store forever.
        """
        now = now or datetime.now(timezone.utc)
        try:
            last = datetime.fromisoformat(self.last_active_at)
        except (TypeError, ValueError):
            return float("inf")
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds()

    def research_indices(self) -> List[int]:
        """Positions in the transcript holding retrieved material, oldest first."""
        return [
            i for i, m in enumerate(self.transcript) if m.get("kind") == "research"
        ]

    # ---- context accounting ----------------------------------------------

    def record_context(self, tokens: int) -> None:
        self.last_context_tokens = tokens
        self.peak_context_tokens = max(self.peak_context_tokens, tokens)

    def estimated_tokens(self) -> int:
        """
        Rough size of the session as it stands, from character count.

        Used only to seed the counter before the first follow-up has produced a
        real usage report — at which point record_context() overwrites it.
        Seeding it from the opening run's own peak instead would be wrong by an
        order of magnitude now that the transcript carries a distilled digest
        rather than the raw excerpts, and would trip the compaction threshold on
        an almost-empty thread.

        Uses the same ~4 chars/token conversion as the metrics fallback.
        """
        chars = len(self.system_prompt) + sum(
            len(m.get("content") or "") for m in self.transcript
        )
        return chars // 4

    def pct_of(self, limit: int) -> float:
        return self.last_context_tokens / limit if limit else 0.0

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "thread_id": self.thread_id,
            "system_prompt": self.system_prompt,
            "original_question": self.original_question,
            "pinned_now": self.pinned_now,
            "offer_message_id": self.offer_message_id,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "transcript": self.transcript,
            "seen_keys": sorted(self.seen_keys),
            "turns": self.turns,
            "searches": self.searches,
            "compactions": self.compactions,
            "last_context_tokens": self.last_context_tokens,
            "peak_context_tokens": self.peak_context_tokens,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoreSession":
        return cls(
            thread_id=int(d["thread_id"]),
            system_prompt=d.get("system_prompt", ""),
            original_question=d.get("original_question", ""),
            pinned_now=d.get("pinned_now", ""),
            offer_message_id=int(d.get("offer_message_id") or 0),
            created_at=d.get("created_at", ""),
            # Sessions written before expiry existed fall back to their creation
            # time, so they age from when they were made rather than never.
            last_active_at=d.get("last_active_at") or d.get("created_at", ""),
            transcript=list(d.get("transcript") or []),
            seen_keys=set(d.get("seen_keys") or []),
            turns=int(d.get("turns") or 0),
            searches=int(d.get("searches") or 0),
            compactions=int(d.get("compactions") or 0),
            last_context_tokens=int(d.get("last_context_tokens") or 0),
            peak_context_tokens=int(d.get("peak_context_tokens") or 0),
        )


class LoreSessionStore:
    """
    JSON-backed store for LoreSession objects.

    Holds two populations:

    * **threads** — live sessions, keyed by thread id.
    * **pending** — sessions built by a /lore run whose offer message nobody has
      reacted to yet, keyed by that message's id. A pending session graduates to
      a thread session via claim(), or is swept once it passes its TTL.

    Mirrors ThreadRegistry's atomic temp-file + os.replace save, so a crash
    mid-write cannot leave a truncated file behind.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._sessions: Dict[int, LoreSession] = {}
        self._pending: Dict[int, LoreSession] = {}
        self.load()

    def load(self) -> None:
        if not self._path.exists():
            logger.info("No existing lore sessions at %s — starting fresh", self._path)
            return
        try:
            with open(self._path, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load lore sessions from %s: %s — starting fresh", self._path, e
            )
            return

        raw = raw or {}
        # Files written before pending offers existed are a flat
        # {thread_id: session} mapping with no "threads" key. Read either shape;
        # the next save rewrites in the current one.
        if "threads" in raw or "pending" in raw:
            thread_entries = raw.get("threads") or {}
            pending_entries = raw.get("pending") or {}
        else:
            thread_entries, pending_entries = raw, {}

        for target, entries, what in (
            (self._sessions, thread_entries, "session"),
            (self._pending, pending_entries, "pending offer"),
        ):
            for key, entry in entries.items():
                try:
                    target[int(key)] = LoreSession.from_dict(entry)
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning("Skipping malformed lore %s %s: %s", what, key, e)

        logger.info(
            "Loaded %d lore session(s) and %d pending offer(s) from %s",
            len(self._sessions), len(self._pending), self._path,
        )

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(
                    {
                        "threads": {
                            str(k): v.to_dict() for k, v in self._sessions.items()
                        },
                        "pending": {
                            str(k): v.to_dict() for k, v in self._pending.items()
                        },
                    },
                    f,
                    indent=2,
                )
            os.replace(str(tmp_path), str(self._path))
        except OSError as e:
            logger.error("Failed to save lore sessions to %s: %s", self._path, e)

    def get(self, thread_id: int) -> Optional[LoreSession]:
        return self._sessions.get(thread_id)

    def put(self, session: LoreSession) -> None:
        self._sessions[session.thread_id] = session
        self.save()

    def remove(self, thread_id: int) -> None:
        if thread_id in self._sessions:
            del self._sessions[thread_id]
            self.save()
            logger.info("Removed lore session for thread %d", thread_id)

    def thread_ids(self) -> List[int]:
        return list(self._sessions.keys())

    def expired_threads(self, ttl_seconds: int) -> List[int]:
        """Thread ids whose sessions have been idle longer than ``ttl_seconds``."""
        now = datetime.now(timezone.utc)
        return [
            tid for tid, sess in self._sessions.items()
            if sess.idle_seconds(now) > ttl_seconds
        ]

    # ---- pending offers ---------------------------------------------------

    def offer(self, message_id: int, session: LoreSession, ttl_seconds: int = 0) -> None:
        """
        Park a seeded session against the offer message that can claim it.

        Sweeps stale offers on the way in when given a TTL, so a bot that runs
        for months without a restart still clears them.
        """
        if ttl_seconds:
            self.sweep_offers(ttl_seconds)
        session.offer_message_id = message_id
        self._pending[message_id] = session
        self.save()

    def has_offer(self, message_id: int) -> bool:
        return message_id in self._pending

    def peek_offer(self, message_id: int) -> Optional[LoreSession]:
        """Read a pending offer without claiming it."""
        return self._pending.get(message_id)

    def claim(self, message_id: int, thread_id: int) -> Optional[LoreSession]:
        """
        Convert a pending offer into a live thread session.

        Pops before it does anything else, so two people reacting at nearly the
        same moment cannot both come away with a session — the second call sees
        nothing and its caller declines to open a second thread.

        Returns:
            The now-live session, or None if the offer was never made, was
            already claimed, or has expired.
        """
        session = self._pending.pop(message_id, None)
        if session is None:
            return None
        session.thread_id = thread_id
        self._sessions[thread_id] = session
        self.save()
        return session

    def sweep_offers(self, ttl_seconds: int) -> int:
        """
        Drop pending offers older than ``ttl_seconds``.

        Nothing else removes them: an offer nobody reacts to has no event that
        would ever fire, so without a sweep the file grows for the life of the
        bot.

        Returns:
            How many offers were dropped.
        """
        now = datetime.now(timezone.utc)
        stale: List[int] = []
        for message_id, session in self._pending.items():
            try:
                created = datetime.fromisoformat(session.created_at)
            except (TypeError, ValueError):
                stale.append(message_id)  # Unparseable: no way to age it out later.
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if (now - created).total_seconds() > ttl_seconds:
                stale.append(message_id)

        for message_id in stale:
            del self._pending[message_id]
        if stale:
            self.save()
            logger.info("Swept %d expired lore thread offer(s)", len(stale))
        return len(stale)
