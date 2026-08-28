"""
The bot's shared state, in one place with explicit ownership.

Everything here used to be a module-level global in bot.py assigned inside
on_ready. That had two problems: nothing said which command depended on what,
and on_ready fires again after every gateway re-identify — so a reconnect
rebuilt both HTTP clients without closing the old ones, re-read both stores,
and re-ran the globally rate-limited command sync. Services is built once in
setup_hook instead, which runs exactly once, before login.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from config import (
    LORE_SESSION_PATH,
    RAG_ENABLED,
    RAG_SERVICE_URL,
    THREAD_REGISTRY_PATH,
)
from history import ConversationHistory
from lore.session import LoreSessionStore
from proxy_client import ProxyClient
from rag_client import RAGClient
from rate_limiter import RateLimiter
from thread_registry import ThreadRegistry

logger = logging.getLogger("mimic-bot.services")


@dataclass
class ThreadRouter:
    """
    Which threads the bot answers in, and how.

    Attributes:
        models: thread_id -> model name, for every thread the bot replies in.
        lore: thread_ids answered by the lore follow-up path rather than plain
            chat. A lore thread carries its conversation in the session store
            and needs the research system prompt, so it cannot use the chat path.
        locks: One lock per lore thread. A turn reads the session, appends its
            question, runs for a minute or more, then appends its answer — so
            two messages arriving close together would interleave those appends
            and race the save. Observed in production as a transcript reading
            question, question, answer, answer.
        claiming: Offer message ids currently being turned into a thread. Two
            people reacting within the same second would otherwise both pass
            the "is there an offer?" check and open a thread each, since the
            first await between check and claim lets the second reaction run.
    """

    models: dict[int, str] = field(default_factory=dict)
    lore: set[int] = field(default_factory=set)
    locks: dict[int, asyncio.Lock] = field(default_factory=dict)
    claiming: set[int] = field(default_factory=set)

    def lock_for(self, thread_id: int) -> asyncio.Lock:
        return self.locks.setdefault(thread_id, asyncio.Lock())

    def forget(self, thread_id: int) -> None:
        """Drop every in-memory record of a thread. Stores are the caller's job."""
        self.lore.discard(thread_id)
        self.models.pop(thread_id, None)
        self.locks.pop(thread_id, None)


@dataclass
class Services:
    """Long-lived collaborators, built once and shared by every cog."""

    proxy: ProxyClient
    rag: RAGClient | None
    limiter: RateLimiter
    history: ConversationHistory
    registry: ThreadRegistry
    sessions: LoreSessionStore
    threads: ThreadRouter

    @classmethod
    def create(cls) -> "Services":
        """Build the full set from configuration. Cheap — no I/O beyond the stores."""
        return cls(
            proxy=ProxyClient(),
            rag=RAGClient(RAG_SERVICE_URL) if RAG_ENABLED else None,
            limiter=RateLimiter(),
            history=ConversationHistory(),
            registry=ThreadRegistry(THREAD_REGISTRY_PATH),
            sessions=LoreSessionStore(LORE_SESSION_PATH),
            threads=ThreadRouter(),
        )

    async def aclose(self) -> None:
        """Close both HTTP clients. Safe to call more than once."""
        await self.proxy.close()
        if self.rag is not None:
            await self.rag.close()
        logger.info("Services closed.")
