"""
Mimic Discord Bot — entry point.

Sets up the discord.py client and loads the cogs that own each command surface:

  cogs/mimic.py  — /mimic, one-shot persona replies
  cogs/chat.py   — /chat, a thread bound to one model
  cogs/lore.py   — /lore, agentic RAG plus its follow-up threads

Everything shared lives on ``bot.services`` (see services.py), built once in
setup_hook. That matters: on_ready fires again after every gateway re-identify,
so building state there — as this file used to — rebuilt both HTTP clients
without closing the old ones, re-read both JSON stores, and re-ran the globally
rate-limited command sync on every reconnect.
"""

import asyncio
import logging
from logging import StreamHandler

import discord
from discord.ext import commands

from config import DISCORD_TOKEN, MIMIC_PERSONAS, validate_config
from services import Services

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[StreamHandler()],
)
logger = logging.getLogger("mimic-bot")

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("discord.http").setLevel(logging.WARNING)

EXTENSIONS = ("cogs.mimic", "cogs.chat", "cogs.lore")


class MimicBot(commands.Bot):
    """The client, plus the one-time setup its cogs depend on."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.members = True
        intents.message_content = True  # Required to read message content in threads
        super().__init__(
            command_prefix="!",  # Unused prefix — prevents default help command
            intents=intents,
            help_command=None,
        )
        self.services: Services | None = None
        self._threads_restored = False

    async def setup_hook(self) -> None:
        """
        Build shared state and load cogs. Runs exactly once, before login.

        Anything here that fails should fail the startup rather than leave a
        half-configured bot answering commands.
        """
        validate_config()
        self.services = Services.create()

        for extension in EXTENSIONS:
            await self.load_extension(extension)
        logger.info("Loaded extensions: %s", ", ".join(EXTENSIONS))

        try:
            synced = await self.tree.sync()
            logger.info(
                "Synced %d global command(s): %s", len(synced), [c.name for c in synced]
            )
        except Exception:
            logger.exception("Failed to sync application commands")

    async def on_ready(self) -> None:
        """
        Fires on every (re)connect, so it does no setup — only reporting.

        Thread restoration lives here rather than in setup_hook because it calls
        fetch_channel, which needs a live gateway session — but it is guarded to
        run once, since it costs an API call per registered thread and nothing
        can register a thread while the bot is disconnected.
        """
        if not self._threads_restored:
            self._threads_restored = True
            chat_cog = self.get_cog("ChatCog")
            if chat_cog is not None:
                await chat_cog.restore_threads()

        logger.info("Logged in as %s (ID: %s)", self.user.name, self.user.id)
        logger.info("Available personas: %s", sorted(MIMIC_PERSONAS))

    async def close(self) -> None:
        if self.services is not None:
            await self.services.aclose()
        await super().close()


async def main() -> None:
    """Start the bot."""
    bot = MimicBot()
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        await bot.close()
        logger.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
