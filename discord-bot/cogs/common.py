"""
What every command surface checks before it spends GPU time, and the
autocompletes they share.

One copy, so a refusal reads the same wherever it comes from:

* a slash command is refused ephemerally, before it defers;
* a message in a bot thread that is over the rate limit gets one visible reply.
"""

import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import MAX_QUEUE_DEPTH
from proxy_client import ProxyError

logger = logging.getLogger("mimic-bot.cogs.common")

_MAX_CHOICES = 25  # Discord's limit on autocomplete choices
_TOO_FAST = "⚠️ You're sending requests too fast. Slow down a bit."


async def refuse(interaction: discord.Interaction, text: str) -> bool:
    """Answer a not-yet-deferred interaction with a private warning. Always False, for `return await refuse(...)`."""
    await interaction.response.send_message(f"⚠️ {text}", ephemeral=True)
    return False


async def queue_full(services) -> Optional[int]:
    """The backend queue depth if it is at the cap, else None. An unreachable proxy is not a refusal."""
    try:
        depth = await services.proxy.get_queue_depth()
    except ProxyError:
        # The request itself will report the outage properly.
        logger.warning("Could not check queue depth — proceeding with request.")
        return None
    return depth if depth >= MAX_QUEUE_DEPTH else None


async def admit(
    interaction: discord.Interaction,
    services,
    *,
    rate: bool = True,
    queue: bool = True,
    personas: Optional[list[str]] = None,
) -> bool:
    """
    The checks a slash command makes before it defers. Answers the interaction
    itself when one fails.

    personas: also require these to be available, and the channel to be one a
        persona webhook can post in.
    """
    if rate and not services.limiter.is_allowed(interaction.user.id):
        return await refuse(interaction, "You're sending requests too fast. Slow down a bit.")

    if personas is not None:
        if not await services.personas.ensure_discovered(services.proxy):
            return await refuse(interaction, "The personas are unavailable right now. Try again in a moment.")
        unknown = [p for p in personas if p not in services.personas]
        if unknown:
            logger.warning("Unknown persona(s) requested: %s", unknown)
            return await refuse(interaction, f"Unknown persona: {', '.join(unknown)}")
        channel = interaction.channel
        parent = channel.parent if isinstance(channel, discord.Thread) else channel
        if not isinstance(parent, discord.TextChannel):
            return await refuse(interaction, "Personas can only post in text channels and their threads.")

    if queue and (depth := await queue_full(services)) is not None:
        return await refuse(interaction, f"The AI backend is busy ({depth} requests queued). Please try again in a moment.")
    return True


async def admit_message(message: discord.Message, services) -> bool:
    """The rate limit for a message posted in a bot thread. Says so in the thread when it bites."""
    if services.limiter.is_allowed(message.author.id):
        return True
    logger.info("Rate-limited %s in thread %d", message.author.name, message.channel.id)
    await message.channel.send(f"{message.author.mention} {_TOO_FAST}")
    return False


# ---- autocomplete -------------------------------------------------------------

def _matching(options: list[tuple[str, str]], current: str) -> list[app_commands.Choice[str]]:
    """(label, value) pairs whose label or value contains `current`, up to Discord's cap."""
    needle = current.lower()
    return [
        app_commands.Choice(name=label, value=value)
        for label, value in options
        if needle in label.lower() or needle in value.lower()
    ][:_MAX_CHOICES]


async def persona_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    """Available personas, shown by display name; the key is the value."""
    catalog = interaction.client.services.personas
    return _matching([(p.display_name, p.key) for p in catalog.all()], current)


async def model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    """Every model the proxy serves. Degrades to no suggestions when it is unreachable."""
    try:
        models = await interaction.client.services.proxy.list_models()
    except Exception:
        models = []
    return _matching([(m, m) for m in sorted(models)], current)


def handle_resolver(bot: commands.Bot):
    """user id -> Discord handle, from the cache. A persona not built yet has a stand-in id."""
    def handle_for_id(user_id: str) -> Optional[str]:
        user = bot.get_user(int(user_id)) if user_id.isdigit() else None
        return user.name if user else None
    return handle_for_id
