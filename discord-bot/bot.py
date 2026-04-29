"""
Mimic Discord Bot — Entry Point

Initialises the discord.py client and exposes a /mimic slash command
that routes to the appropriate AI persona via the orchestration proxy.

Features:
  - /mimic persona:<name> message:<text> → persona response
  - Autocomplete for persona selection
  - Typing indicator throughout swap + inference latency
  - Per-user rate limiting
  - Per-channel, per-persona conversation history
"""

import asyncio
import logging
from logging import StreamHandler

import discord
from discord import app_commands
from discord.ext import commands
from config import (
    DISCORD_TOKEN,
    LORE_MODEL,
    LORE_SYSTEM_PROMPT,
    MAX_QUEUE_DEPTH,
    MIMIC_PERSONAS,
    MIMIC_SYSTEM_PROMPTS,
    get_mimic_system_prompt,
    validate_config,
)
from formatters import build_lore_embed_discord, format_mimic_response
from history import ConversationHistory
from proxy_client import ProxyClient, ProxyError
from rate_limiter import RateLimiter

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


# ──────────────────────────────────────────────────────────────
# Bot Intents & Client Setup
# ──────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(
    command_prefix="!",  # Unused prefix — prevents default help command
    intents=intents,
    help_command=None,
)

# Shared state — initialised in on_ready
proxy_client: ProxyClient | None = None
rate_limiter: RateLimiter | None = None
history: ConversationHistory | None = None


# ──────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────


@bot.event
async def on_ready():
    """Initialise shared state and sync slash commands when the bot comes online."""
    global proxy_client, rate_limiter, history

    validate_config()

    proxy_client = ProxyClient()
    rate_limiter = RateLimiter()
    history = ConversationHistory()

    # Sync slash commands to Discord
    try:
        synced = await bot.tree.sync()
        logger.info("Synced %d global command(s): %s", len(synced), [c.name for c in synced])
    except Exception:
        logger.exception("Failed to sync application commands")

    logger.info(
        "Logged in as %s (ID: %s)", bot.user.name, bot.user.id
    )
    logger.info("Available personas: %s", sorted(MIMIC_SYSTEM_PROMPTS.keys()))


# ──────────────────────────────────────────────────────────────
# Autocomplete: persona options
# ──────────────────────────────────────────────────────────────

async def persona_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Provide autocomplete suggestions for the persona option."""
    choices = []
    for persona in sorted(MIMIC_SYSTEM_PROMPTS.keys()):
        if current.lower() in persona.lower():
            choices.append(app_commands.Choice(name=persona, value=persona))
        if len(choices) >= 25:  # Discord max choices
            break
    return choices


# ──────────────────────────────────────────────────────────────
# Slash Command: /mimic
# ──────────────────────────────────────────────────────────────

@bot.tree.command(name="mimic", description="Chat with a mimic persona")
@app_commands.describe(
    persona="The persona to chat with",
    message="Your message to the persona",
)
@app_commands.autocomplete(persona=persona_autocomplete)
async def mimic_command(
    interaction: discord.Interaction,
    persona: str,
    message: str,
):
    """Handle the /mimic slash command."""
    # Rate limit check
    if not rate_limiter.is_allowed(interaction.user.id):
        await interaction.response.send_message(
            "⚠️ You're sending requests too fast. Slow down a bit.",
            ephemeral=True,
        )
        return

    # Validate persona
    if persona not in MIMIC_PERSONAS:
        logger.warning("No system prompt configured for persona %s", persona)
        await interaction.response.send_message(
            f"⚠️ Unknown persona: {persona}",
            ephemeral=True,
        )
        return
    system_prompt = get_mimic_system_prompt(persona)

    # Queue depth check — reject if backend is too busy
    try:
        depth = await proxy_client.get_queue_depth()
        if depth >= MAX_QUEUE_DEPTH:
            await interaction.response.send_message(
                f"⚠️ The AI backend is busy ({depth} requests queued). "
                f"Please try again in a moment.",
                ephemeral=True,
            )
            return
    except ProxyError:
        # If we can't reach the proxy, the chat() call below will handle the error.
        logger.warning("Could not check queue depth — proceeding with request.")

    logger.info(
        "Request from %s in %s: persona=%s, message=%r",
        interaction.user.name,
        interaction.channel.name if interaction.channel else "DM",
        persona,
        message[:100],
    )

    # Defer to show typing indicator during inference
    await interaction.response.defer()

    try:
        # Build conversation history
        channel_id = interaction.channel.id if interaction.channel else 0
        conv_history = history.get_history(channel_id, persona)

        # Build full message list
        msgs: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]
        msgs.extend(conv_history)
        msgs.append({"role": "user", "content": message})

        # Use channel.typing() context manager to maintain typing indicator
        # throughout the entire proxy call and response processing
        async with interaction.channel.typing():
            # Call proxy
            raw_response = await proxy_client.chat(persona, msgs)

        # Post-process: strip disclaimers, truncate
        response = format_mimic_response(raw_response)

        # Store turn in history
        history.add_turn(channel_id, persona, message, response)

        # Send response
        await interaction.followup.send(response)

    except ProxyError as e:
        logger.warning("Proxy error: %s", e)
        await interaction.followup.send(
            f"⚠️ {e}",
            ephemeral=True,
        )
    except Exception as e:
        logger.exception("Unexpected error handling persona %s", persona)
        await interaction.followup.send(
            "An unexpected error occurred. Please try again.",
            ephemeral=True,
        )


# ──────────────────────────────────────────────────────────────
# Slash Command: /lore
# ──────────────────────────────────────────────────────────────

@bot.tree.command(name="lore", description="Ask the lore assistant a question")
@app_commands.describe(
    question="Your question about server lore",
)
async def lore_command(
    interaction: discord.Interaction,
    question: str,
):
    """Handle the /lore slash command."""
    # Rate limit check
    if not rate_limiter.is_allowed(interaction.user.id):
        await interaction.response.send_message(
            "⚠️ You're sending requests too fast. Slow down a bit.",
            ephemeral=True,
        )
        return

    # Queue depth check — reject if backend is too busy
    try:
        depth = await proxy_client.get_queue_depth()
        if depth >= MAX_QUEUE_DEPTH:
            await interaction.response.send_message(
                f"⚠️ The AI backend is busy ({depth} requests queued). "
                f"Please try again in a moment.",
                ephemeral=True,
            )
            return
    except ProxyError:
        # If we can't reach the proxy, the chat() call below will handle the error.
        logger.warning("Could not check queue depth — proceeding with request.")

    logger.info(
        "Lore request from %s in %s: question=%r",
        interaction.user.name,
        interaction.channel.name if interaction.channel else "DM",
        question[:100],
    )

    # Defer to show typing indicator during inference
    await interaction.response.defer()

    try:
        messages = [
            {"role": "system", "content": LORE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "No retrieved context available (RAG service not yet configured).\n\n"
                    f"Question: {question}"
                ),
            },
        ]

        # Use channel.typing() context manager to maintain typing indicator
        # throughout the entire proxy call and response processing
        async with interaction.channel.typing():
            lore_text = await proxy_client.chat(LORE_MODEL, messages)

        embed = build_lore_embed_discord(lore_text, chunk_count=0)
        await interaction.followup.send(embed=embed)

    except ProxyError as e:
        logger.warning("Proxy error: %s", e)
        await interaction.followup.send(
            f"⚠️ {e}",
            ephemeral=True,
        )
    except Exception as e:
        logger.exception("Unexpected error handling lore request")
        await interaction.followup.send(
            "An unexpected error occurred. Please try again.",
            ephemeral=True,
        )


# ──────────────────────────────────────────────────────────────
# Admin Commands — DISABLED
# ──────────────────────────────────────────────────────────────
# /admin-clear-history disabled due to lack of authorization checks.
# Re-enable only after adding role/permission verification.

# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

async def main():
    """Start the bot."""
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        if proxy_client:
            await proxy_client.close()
        await bot.close()
        logger.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
