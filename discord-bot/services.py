"""
The bot's shared state, in one place with explicit ownership.

Everything here used to be a module-level global in bot.py assigned inside
on_ready. That had two problems: nothing said which command depended on what,
and on_ready fires again after every gateway re-identify — so a reconnect
rebuilt both HTTP clients without closing the old ones, re-read both stores,
and re-ran the globally rate-limited command sync. Services is built once in
setup_hook instead, which runs exactly once, before login.
"""

import logging
from dataclasses import dataclass

from config import LORE_SESSION_PATH, PERSONA_NAMES_PATH, RAG_ENABLED, RAG_SERVICE_URL, THREAD_REGISTRY_PATH
from history import ConversationHistory
from lore.session import LoreSessionStore
from persona_posting import PersonaNameRegistry, WebhookPoster
from personas import PersonaCatalog
from proxy_client import ProxyClient
from rag_client import RAGClient
from rate_limiter import RateLimiter
from threads import ThreadDirectory
from vision import Captioner

logger = logging.getLogger("mimic-bot.services")


@dataclass
class Services:
    """Long-lived collaborators, built once and shared by every cog."""

    proxy: ProxyClient
    rag: RAGClient | None
    limiter: RateLimiter
    history: ConversationHistory
    threads: ThreadDirectory
    sessions: LoreSessionStore
    personas: PersonaCatalog
    webhooks: WebhookPoster
    vision: Captioner | None = None

    @classmethod
    def create(cls) -> "Services":
        """Build the full set from configuration. Cheap — no I/O beyond the stores."""
        personas = PersonaCatalog()
        proxy = ProxyClient()
        return cls(
            proxy=proxy,
            rag=RAGClient(RAG_SERVICE_URL) if RAG_ENABLED else None,
            limiter=RateLimiter(),
            history=ConversationHistory(),
            threads=ThreadDirectory(THREAD_REGISTRY_PATH),
            sessions=LoreSessionStore(LORE_SESSION_PATH),
            personas=personas,
            webhooks=WebhookPoster(personas, PersonaNameRegistry(PERSONA_NAMES_PATH)),
            vision=Captioner(proxy),
        )

    async def aclose(self) -> None:
        """Close both HTTP clients. Safe to call more than once."""
        await self.proxy.close()
        if self.rag is not None:
            await self.rag.close()
        logger.info("Services closed.")
