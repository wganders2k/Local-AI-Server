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
  - /chat creates a thread for persistent model-based chat
"""

import asyncio
import logging
from logging import StreamHandler
from typing import Dict

import discord
from discord import app_commands
from discord.ext import commands
from config import (
    DISCORD_TOKEN,
    MAX_MESSAGE_LENGTH,
    MAX_QUEUE_DEPTH,
    MIMIC_PERSONAS,
    MIMIC_SYSTEM_PROMPTS,
    RAG_ENABLED,
    RAG_SERVICE_URL,
    THREAD_REGISTRY_PATH,
    get_mimic_system_prompt,
    validate_config,
)
from formatters import build_lore_embeds, find_split_boundary, format_mimic_response
from history import ConversationHistory
from proxy_client import ProxyClient, ProxyError
from rag_client import RAGClient
from rate_limiter import RateLimiter
from thread_registry import ThreadRegistry
from agent_tools import run_agent_loop

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
intents.message_content = True  # Required to read message content in threads

bot = commands.Bot(
    command_prefix="!",  # Unused prefix — prevents default help command
    intents=intents,
    help_command=None,
)

# Shared state — initialised in on_ready
proxy_client: ProxyClient | None = None
rag_client: RAGClient | None = None
rate_limiter: RateLimiter | None = None
history: ConversationHistory | None = None
thread_registry: ThreadRegistry | None = None
thread_models: Dict[int, str] = {}  # thread_id -> model_name mapping


# ──────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────


@bot.event
async def on_ready():
    """Initialise shared state and sync slash commands when the bot comes online."""
    global proxy_client, rag_client, rate_limiter, history, thread_registry

    validate_config()

    proxy_client = ProxyClient()
    rag_client = RAGClient(RAG_SERVICE_URL) if RAG_ENABLED else None
    rate_limiter = RateLimiter()
    history = ConversationHistory()
    thread_registry = ThreadRegistry(THREAD_REGISTRY_PATH)

    # Restore thread mappings from persistent registry
    await restore_thread_models()

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
# Thread Restoration on Startup
# ──────────────────────────────────────────────────────────────


async def restore_thread_models() -> None:
    """
    Restore thread→model mappings from the persistent registry.

    For each entry in the registry, verify the thread still exists via
    Discord API and update its status accordingly. Only non-deleted
    threads are added to the in-memory thread_models dict.
    """
    if thread_registry is None:
        logger.warning("ThreadRegistry not initialised — skipping restore")
        return

    entries = thread_registry.get_all()
    if not entries:
        logger.info("No threads in registry — nothing to restore")
        return

    restored = 0
    archived = 0
    deleted = 0
    skipped = 0

    for thread_id, entry in entries.items():
        model = entry.get("model")
        name = entry.get("name", "unknown")
        try:
            channel = await bot.fetch_channel(thread_id)

            if isinstance(channel, discord.Thread):
                if channel.archived:
                    thread_registry.update_status(thread_id, "archived")
                    archived += 1
                else:
                    thread_registry.update_status(thread_id, "active")

                # Both active and archived threads are restored so the bot
                # responds when users post (Discord auto-unarchives on new message).
                thread_models[thread_id] = model
                restored += 1
                logger.info(
                    "Restored thread %d (%s) model=%s status=%s",
                    thread_id, name, model, channel.archived and "archived" or "active",
                )
            else:
                # Channel exists but is not a thread — mark deleted.
                thread_registry.update_status(thread_id, "deleted")
                deleted += 1
                logger.warning(
                    "Thread %d (%s) exists but is not a Thread — marked deleted",
                    thread_id, name,
                )

        except discord.NotFound:
            thread_registry.update_status(thread_id, "deleted")
            deleted += 1
            logger.info("Thread %d (%s) no longer exists — marked deleted", thread_id, name)
        except discord.Forbidden:
            skipped += 1
            logger.warning(
                "Forbidden accessing thread %d (%s) — skipping restore",
                thread_id, name,
            )
        except Exception:
            skipped += 1
            logger.exception(
                "Unexpected error restoring thread %d (%s)", thread_id, name
            )

    logger.info(
        "Thread restore complete: %d restored, %d archived, %d deleted, %d skipped",
        restored, archived, deleted, skipped,
    )


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


async def model_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Provide autocomplete suggestions for available models from the API."""
    if not proxy_client:
        return []
    try:
        models = await proxy_client.list_models()
    except Exception:
        models = []
    choices = []
    for model in sorted(models):
        if current.lower() in model.lower():
            choices.append(app_commands.Choice(name=model, value=model))
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
        # Label the user message with the sender's display name so the model
        # can track who is speaking when multiple users share a channel.
        labeled = f"[{interaction.user.display_name}]: {message}"

        msgs: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]
        msgs.extend(conv_history)
        msgs.append({"role": "user", "content": labeled})

        # Stream response from proxy, splitting into ~2000-char chunks
        buffer = ""
        full_response = ""

        async with interaction.channel.typing():
            async for token in proxy_client.chat_stream(persona, msgs):
                buffer += token
                full_response += token

                while len(buffer) >= MAX_MESSAGE_LENGTH:
                    split_at = find_split_boundary(buffer, MAX_MESSAGE_LENGTH)
                    chunk = buffer[:split_at].rstrip()
                    buffer = buffer[split_at:].lstrip("\n")
                    if chunk:
                        await interaction.followup.send(chunk)

        # After stream ends — strip disclaimers on final buffer, send remainder
        buffer = format_mimic_response(buffer).strip()
        if buffer:
            await interaction.followup.send(buffer)

        history.add_turn(channel_id, persona, labeled, full_response)

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
# Slash Command: /chat
# ──────────────────────────────────────────────────────────────

@bot.tree.command(name="chat", description="Create a chat thread with an AI model")
@app_commands.describe(
    model="The model to chat with",
)
@app_commands.autocomplete(model=model_autocomplete)
async def chat_command(
    interaction: discord.Interaction,
    model: str,
):
    """Handle the /chat slash command — creates a thread for persistent chat."""
    # Rate limit check
    if not rate_limiter.is_allowed(interaction.user.id):
        await interaction.response.send_message(
            "⚠️ You're sending requests too fast. Slow down a bit.",
            ephemeral=True,
        )
        return

    # Check that we're in a text channel that supports threads (not DMs)
    channel = interaction.channel
    if not hasattr(channel, "create_thread"):
        await interaction.response.send_message(
            "⚠️ Threads are not supported in DMs. Please use this command in a server channel.",
            ephemeral=True,
        )
        return

    logger.info(
        "Chat thread request from %s in %s: model=%s",
        interaction.user.name,
        channel.name if channel else "DM",
        model,
    )

    await interaction.response.defer(ephemeral=False)

    try:
        # Create a thread in the current channel
        thread = await channel.create_thread(
            name=f"chat-{model}",
            auto_archive_duration=60,  # 1 hour archive duration
            type=discord.ChannelType.public_thread,
        )

        # Register this thread for chat with the specified model
        thread_models[thread.id] = model

        # Persist thread metadata to disk registry
        if thread_registry is not None:
            thread_registry.register(thread.id, model, thread.name)

        # Send greeting in the thread
        await thread.send(
            f"🤖 Chat with **{model}** started. Send your messages here!"
        )

        # Confirm to the user
        await interaction.followup.send(
            f"✅ Chat thread created: {thread.mention}",
            ephemeral=True,
        )

    except discord.Forbidden:
        await interaction.followup.send(
            "⚠️ I don't have permission to create threads in this channel.",
            ephemeral=True,
        )
    except discord.HTTPException as e:
        if e.code == 50035:  # Cannot send messages to this thread
            await interaction.followup.send(
                "⚠️ Failed to create thread. The channel may have thread creation disabled.",
                ephemeral=True,
            )
        else:
            logger.exception("Failed to create chat thread")
            await interaction.followup.send(
                "⚠️ An unexpected error occurred while creating the thread.",
                ephemeral=True,
            )


# ──────────────────────────────────────────────────────────────
# Slash Command: /lore
# ──────────────────────────────────────────────────────────────

@bot.tree.command(name="lore", description="Ask the lore assistant a question")
@app_commands.describe(
    question="Your question about server lore",
    rounds="Maximum search rounds (1-25, default: 10)",
)
async def lore_command(
    interaction: discord.Interaction,
    question: str,
    rounds: int = 10,
):
    """Handle the /lore slash command — agentic RAG with tool calling."""
    # Rate limit check removed — lore queries are long-running and don't benefit from rate limiting
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
        # Collect available channel names from the guild text channels
        channel_names: list[str] = []
        if isinstance(interaction.guild, discord.Guild):
            for ch in interaction.guild.text_channels:
                channel_names.append(ch.name)

        # Run the agentic RAG loop — brain-dense will iteratively call tools
        # to search Discord history and synthesize an answer.
        lore_text = await run_agent_loop(
            user_question=question,
            proxy_client=proxy_client,
            rag_client=rag_client,
            interaction=interaction,
            channel_names=channel_names,
            max_rounds=rounds,
        )

        embeds = build_lore_embeds(lore_text)
        for i, embed in enumerate(embeds):
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
# Thread Chat Handler
# ──────────────────────────────────────────────────────────────

@bot.event
async def on_message(message: discord.Message):
    """Respond to regular messages sent inside chat threads."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only respond in threads that have a registered model
    channel = message.channel
    if not isinstance(channel, discord.Thread):
        return

    model = thread_models.get(channel.id)
    if model is None:
        return

    # Rate limit check
    if not rate_limiter.is_allowed(message.author.id):
        await channel.send(
            f"{message.author.mention} ⚠️ You're sending requests too fast. Slow down a bit.",
        )
        return

    logger.info(
        "Thread chat from %s in thread %s (channel %s): model=%s, message=%r",
        message.author.name,
        channel.name,
        channel.parent.name if hasattr(channel, "parent") and channel.parent else "unknown",
        model,
        message.content[:100],
    )

    try:
        # Build conversation history keyed by thread_id
        conv_history = history.get_history(channel.id, model)

        # Build full message list — no system prompt, just history + user message
        # Label the user message with the sender's display name so the model
        # can track who is speaking when multiple users share a thread.
        labeled = f"[{message.author.display_name}]: {message.content}"

        msgs: list[dict] = []
        msgs.extend(conv_history)
        msgs.append({"role": "user", "content": labeled})

        # Stream response from proxy, splitting into ~2000-char chunks
        buffer = ""
        full_response = ""

        async with channel.typing():
            async for token in proxy_client.chat_stream(model, msgs):
                buffer += token
                full_response += token

                while len(buffer) >= MAX_MESSAGE_LENGTH:
                    split_at = find_split_boundary(buffer, MAX_MESSAGE_LENGTH)
                    chunk = buffer[:split_at].rstrip()
                    buffer = buffer[split_at:].lstrip("\n")
                    if chunk:
                        await channel.send(chunk)

        # After stream ends — strip disclaimers on final buffer, send remainder
        buffer = format_mimic_response(buffer).strip()
        if buffer:
            await channel.send(buffer)

        history.add_turn(channel.id, model, labeled, full_response)

    except ProxyError as e:
        logger.warning("Proxy error: %s", e)
        await channel.send(f"⚠️ {e}")
    except Exception as e:
        logger.exception("Unexpected error handling thread chat for model %s", model)
        await channel.send("An unexpected error occurred. Please try again.")


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
        if rag_client:
            await rag_client.close()
        await bot.close()
        logger.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
