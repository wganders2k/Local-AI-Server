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
from datetime import datetime, timezone
from logging import StreamHandler
from typing import Dict

import discord
from discord import app_commands
from discord.ext import commands, tasks
from config import (
    DISCORD_TOKEN,
    MAX_MESSAGE_LENGTH,
    MAX_QUEUE_DEPTH,
    MIMIC_PERSONAS,
    MIMIC_SYSTEM_PROMPTS,
    AGENT_CTX_LIMIT,
    AGENT_MODEL,
    LORE_OFFER_TTL_SECONDS,
    LORE_SESSION_PATH,
    LORE_SWEEP_INTERVAL_SECONDS,
    LORE_THREAD_EMOJI,
    LORE_THREAD_TTL_SECONDS,
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
from lore_session import LoreSession, LoreSessionStore
from agent_tools import (
    LoreStatus,
    maybe_compact_session,
    run_agent_loop,
    run_lore_turn,
    seed_session_from_run,
)

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
lore_sessions: LoreSessionStore | None = None
thread_models: Dict[int, str] = {}  # thread_id -> model_name mapping
lore_threads: set[int] = set()      # thread_ids answered by the lore follow-up path
# One lock per lore thread. A turn reads the session, appends its question,
# runs for a minute or more, then appends its answer — so two messages arriving
# close together would interleave those appends and race the save. Observed in
# production as a transcript reading question, question, answer, answer.
lore_locks: Dict[int, asyncio.Lock] = {}
# Offer message ids currently being turned into a thread. Two people reacting
# within the same second would otherwise both pass the "is there an offer?"
# check and open a thread each, since the first await between check and claim
# lets the second reaction run.
lore_claiming: set[int] = set()


# ──────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────


@bot.event
async def on_ready():
    """Initialise shared state and sync slash commands when the bot comes online."""
    global proxy_client, rag_client, rate_limiter, history, thread_registry
    global lore_sessions

    validate_config()

    proxy_client = ProxyClient()
    rag_client = RAGClient(RAG_SERVICE_URL) if RAG_ENABLED else None
    rate_limiter = RateLimiter()
    history = ConversationHistory()
    thread_registry = ThreadRegistry(THREAD_REGISTRY_PATH)
    lore_sessions = LoreSessionStore(LORE_SESSION_PATH)

    # Restore thread mappings from persistent registry
    await restore_thread_models()

    # Offers nobody reacted to have no event that would ever fire, so they are
    # only ever removed here and when a new offer is made.
    lore_sessions.sweep_offers(LORE_OFFER_TTL_SECONDS)

    # Retire quiet threads now, then hourly. Doing it at startup means a bot
    # that was down past several expiries catches up immediately.
    if not lore_sweep_loop.is_running():
        lore_sweep_loop.start()
        logger.info(
            "Lore thread sweep running every %ds, retiring threads idle over %.0f day(s)",
            LORE_SWEEP_INTERVAL_SECONDS, LORE_THREAD_TTL_SECONDS / 86400,
        )

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
                # A lore thread only works if its session survived too; without
                # one there is no transcript to answer from, so fall back to
                # treating it as a plain chat thread rather than erroring later.
                if thread_registry.get_kind(thread_id) == "lore":
                    if lore_sessions is not None and lore_sessions.get(thread_id):
                        lore_threads.add(thread_id)
                    else:
                        logger.warning(
                            "Thread %d (%s) is registered as lore but has no saved "
                            "session — it will behave as a plain chat thread",
                            thread_id, name,
                        )
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
        result = await run_agent_loop(
            user_question=question,
            proxy_client=proxy_client,
            rag_client=rag_client,
            interaction=interaction,
            channel_names=channel_names,
            max_rounds=rounds,
        )

        embeds = build_lore_embeds(result.answer)
        for i, embed in enumerate(embeds):
            await interaction.followup.send(embed=embed)

        # Offer a follow-up thread rather than opening one. Most answers are
        # read once and never followed up, and a thread per run buried the
        # channel — so the user decides, after reading, whether they want one.
        if result.ok:
            await offer_lore_thread(interaction, question, channel_names, result)

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
# Lore follow-up threads
# ──────────────────────────────────────────────────────────────


def _lore_footer(session: LoreSession, metrics=None, searched: bool = False) -> str:
    """
    One-line context readout appended to every lore thread answer.

    Uses Discord's subtext marker so it reads as a footnote rather than part of
    the answer.

    ``metrics`` is only read when ``searched``, so the opening message of a
    thread — which did no searching of its own — can omit it.
    """
    used = session.last_context_tokens
    pct = (used / AGENT_CTX_LIMIT * 100) if AGENT_CTX_LIMIT else 0.0
    bits = [f"\U0001f9e0 {used:,}/{AGENT_CTX_LIMIT:,} ({pct:.0f}%)"]
    if searched:
        calls = metrics.total_tool_calls
        bits.append(f"{calls} search{'' if calls == 1 else 'es'}")
        bits.append(f"{metrics.rounds_executed} round{'' if metrics.rounds_executed == 1 else 's'}")
    else:
        bits.append("answered from thread context")
    if session.compactions:
        bits.append(f"{session.compactions} compaction{'' if session.compactions == 1 else 's'}")
    return "-# " + " \u00b7 ".join(bits)


async def post_lore_answer(
    thread: discord.Thread,
    answer: str,
    footer: str,
    edit_first: discord.Message | None = None,
) -> None:
    """
    Post an answer to a lore thread, split at paragraph boundaries.

    Args:
        edit_first: If given, this message is edited to hold the first chunk
            instead of a new one being sent — so the status message turns into
            the answer rather than leaving a stale progress line behind.
    """
    chunks: list[str] = []
    buffer = answer
    while buffer:
        if len(buffer) <= MAX_MESSAGE_LENGTH:
            chunks.append(buffer)
            break
        split_at = find_split_boundary(buffer, MAX_MESSAGE_LENGTH)
        chunks.append(buffer[:split_at].rstrip())
        buffer = buffer[split_at:].lstrip("\n")

    if not chunks:
        chunks = ["(empty answer)"]

    # Ride the footer along on the last chunk when there is room for it.
    tail = "\n" + footer
    footer_attached = len(chunks[-1]) + len(tail) <= MAX_MESSAGE_LENGTH
    if footer_attached:
        chunks[-1] += tail

    for i, chunk in enumerate(chunks):
        if i == 0 and edit_first is not None:
            await edit_first.edit(content=chunk)
        else:
            await thread.send(chunk)

    if not footer_attached:
        await thread.send(footer)


def _lore_thread_name(question: str) -> str:
    """Thread name for a lore follow-up, within Discord's 100-char limit."""
    return f"lore-{question[:60].strip() or 'lore'}"


_OFFER_TEXT = (
    "React to this message to follow up in a thread."
)


async def offer_lore_thread(
    interaction: discord.Interaction,
    question: str,
    channel_names: list[str],
    result,
) -> None:
    """
    Post the follow-up offer for a finished /lore run and park its session.

    The offer is a plain message the bot reacts to with LORE_THREAD_EMOJI;
    anyone adding that same reaction gets a thread. Nothing is created until
    then, and the session is built here so that claiming one is instant.

    Failures are logged and swallowed: the user already has their answer in the
    channel, and losing the offer should not turn a successful /lore into an
    error.
    """
    channel = interaction.channel
    if lore_sessions is None or not hasattr(channel, "create_thread"):
        return  # DMs and misconfiguration: the embed alone is fine.

    try:
        offer = await interaction.followup.send(_OFFER_TEXT, wait=True)
        await offer.add_reaction(LORE_THREAD_EMOJI)
    except (discord.Forbidden, discord.HTTPException):
        logger.warning("Could not post the /lore thread offer", exc_info=True)
        return

    try:
        session = seed_session_from_run(
            thread_id=0,
            channel_names=channel_names,
            question=question,
            result=result,
        )
        lore_sessions.offer(offer.id, session, ttl_seconds=LORE_OFFER_TTL_SECONDS)
        logger.info("Lore thread offer %d is ready to claim", offer.id)
    except Exception:
        logger.exception("Failed to seed lore thread offer %d", offer.id)


async def claim_lore_thread(
    channel: discord.abc.Messageable,
    message_id: int,
    claimant: str,
) -> None:
    """
    Turn a reacted-to offer into a live thread.

    Three things keep a burst of reactions from producing a pile of threads: the
    caller's lore_claiming guard, claim() popping the offer so a second caller
    finds nothing, and Discord itself allowing only one thread per message.
    """
    if lore_sessions is None:
        return

    pending = lore_sessions.peek_offer(message_id)
    if pending is None:
        await channel.send(
            "-# That follow-up offer has expired \u2014 run `/lore` again to ask more."
        )
        return

    try:
        message = await channel.fetch_message(message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        logger.warning("Could not fetch offer message %d", message_id, exc_info=True)
        return

    try:
        thread = await message.create_thread(
            name=_lore_thread_name(pending.original_question),
            auto_archive_duration=1440,  # 24h — research threads are slow-burn
        )
    except (discord.Forbidden, discord.HTTPException):
        logger.warning("Could not open a follow-up thread for /lore", exc_info=True)
        return

    session = lore_sessions.claim(message_id, thread.id)
    if session is None:
        logger.warning("Offer %d vanished between check and claim", message_id)
        return

    try:
        thread_models[thread.id] = AGENT_MODEL
        lore_threads.add(thread.id)
        if thread_registry is not None:
            thread_registry.register(thread.id, AGENT_MODEL, thread.name, kind="lore")

        logger.info(
            "Opened lore thread %d from offer %d for %s",
            thread.id, message_id, claimant,
        )
        await thread.send(
            "Ask a follow-up here \u2014 I already have the research behind that "
            "answer, and I'll only search again if your question needs it.\n"
            + _lore_footer(session)
        )
        # The bot has no manage_messages permission anywhere, so it cannot clear
        # the reaction to show the offer is spent. Editing the message is the
        # only signal available.
        await message.edit(content=f"{_OFFER_TEXT}\n-# \u2192 {thread.mention}")
    except Exception:
        logger.exception("Failed to finish opening lore thread %d", thread.id)


async def handle_lore_thread_message(
    message: discord.Message, thread: discord.Thread
) -> None:
    """Answer a follow-up posted in a lore thread."""
    session = lore_sessions.get(thread.id) if lore_sessions else None
    if session is None:
        logger.warning("No lore session for thread %d — ignoring", thread.id)
        return

    if not rate_limiter.is_allowed(message.author.id):
        await thread.send(
            f"{message.author.mention} \u26a0\ufe0f You're sending requests too fast."
        )
        return

    logger.info(
        "Lore follow-up from %s in %s: %r",
        message.author.name, thread.name, message.content[:100],
    )

    # Same progress vocabulary the slash command shows, reported into the thread.
    status = await LoreStatus.from_thread(thread)

    lock = lore_locks.setdefault(thread.id, asyncio.Lock())
    if lock.locked():
        logger.info("Thread %d is mid-answer — queuing this follow-up", thread.id)
        await status.waiting()

    # Held across the whole turn: the session is re-read inside the lock because
    # a queued turn must see the previous one's appended answer, not the
    # snapshot taken before it waited.
    async with lock:
        session = lore_sessions.get(thread.id) if lore_sessions else None
        if session is None:
            logger.warning("Lore session for thread %d vanished — ignoring", thread.id)
            return
        try:
            async with thread.typing():
                answer, metrics, searched = await run_lore_turn(
                    session=session,
                    question=message.content,
                    proxy_client=proxy_client,
                    rag_client=rag_client,
                    status=status,
                )
        except ProxyError as e:
            logger.warning("Proxy error in lore follow-up: %s", e)
            await status.failed()
            return
        except Exception:
            logger.exception("Unexpected error in lore follow-up")
            await status.failed()
            return

        lore_sessions.put(session)
        await post_lore_answer(
            thread, answer, _lore_footer(session, metrics, searched),
            edit_first=status.message,
        )

        try:
            if await maybe_compact_session(session, proxy_client):
                lore_sessions.put(session)
        except Exception:
            logger.exception("Compaction failed after follow-up")


# ──────────────────────────────────────────────────────────────
# Lore thread expiry
# ──────────────────────────────────────────────────────────────


async def retire_lore_thread(thread_id: int, reason: str) -> None:
    """
    Purge every local record of one lore thread.

    The session store holds the whole research payload verbatim, so this is what
    actually reclaims the disk. The Discord thread itself is left in place —
    the bot has no business deleting a channel people can still read — but with
    its records gone it is no longer routed to the lore handler, and posting in
    it does nothing.

    Ordered so the session goes last: if anything below fails, the thread is
    already unrouted and cannot be answered from a half-purged state.
    """
    lore_threads.discard(thread_id)
    thread_models.pop(thread_id, None)
    lore_locks.pop(thread_id, None)
    if thread_registry is not None:
        thread_registry.remove(thread_id)
    if lore_sessions is not None:
        lore_sessions.remove(thread_id)
    logger.info("Retired lore thread %d (%s)", thread_id, reason)


async def sweep_lore_threads() -> int:
    """
    Retire lore threads that have gone quiet, telling each one it has.

    A thread mid-answer is skipped rather than purged: its lock is held for the
    length of a turn, and pulling the session out from under a running turn
    would strand it.

    Returns:
        How many threads were retired.
    """
    if lore_sessions is None:
        return 0

    expired = lore_sessions.expired_threads(LORE_THREAD_TTL_SECONDS)
    retired = 0
    for thread_id in expired:
        lock = lore_locks.get(thread_id)
        if lock is not None and lock.locked():
            logger.info("Thread %d is mid-answer — deferring expiry", thread_id)
            continue

        days = LORE_THREAD_TTL_SECONDS / 86400
        try:
            thread = bot.get_channel(thread_id) or await bot.fetch_channel(thread_id)
            await thread.send(
                f"-# \U0001f4a4 This thread has gone inactive after {days:.0f} days "
                "without a question, so I've released the research behind it. "
                "Run `/lore` again to start fresh."
            )
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            # Deleted, or the bot lost access. Nothing to notify, but the local
            # records still have to go — that is the whole point of the sweep.
            logger.info(
                "Could not notify thread %d of expiry — purging anyway", thread_id,
                exc_info=True,
            )
        except Exception:
            logger.exception("Unexpected error notifying thread %d of expiry", thread_id)

        await retire_lore_thread(thread_id, "inactive")
        retired += 1

    if retired:
        logger.info("Lore thread sweep retired %d thread(s)", retired)
    return retired


@tasks.loop(seconds=LORE_SWEEP_INTERVAL_SECONDS)
async def lore_sweep_loop() -> None:
    try:
        await sweep_lore_threads()
    except Exception:
        # A raising task loop stops silently, and a stopped sweep means the
        # store grows without bound.
        logger.exception("Lore thread sweep failed")


@lore_sweep_loop.before_loop
async def _before_lore_sweep() -> None:
    await bot.wait_until_ready()


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    """
    Open a lore follow-up thread when someone claims an offer.

    Raw rather than on_reaction_add: the cached variant only fires for messages
    still in the client's message cache, so an offer would silently stop working
    after a restart or once the cache rolled over. The raw event fires from the
    gateway regardless, which is why it deals in ids and needs the channel
    fetched by hand.

    Requires the reactions gateway intent (on by default, and unprivileged) plus
    add_reactions and read_message_history in the channel — verified present
    guild-wide for this bot.
    """
    if payload.user_id == bot.user.id:
        return  # The bot's own seed reaction.
    if str(payload.emoji) != LORE_THREAD_EMOJI:
        return
    # Overwhelmingly the common case — every reaction in every channel lands
    # here, so this stays ahead of any API call.
    if lore_sessions is None or not lore_sessions.has_offer(payload.message_id):
        return

    # Check-and-mark with no await in between, so a second reaction arriving
    # while the first is still opening its thread cannot start a second one.
    if payload.message_id in lore_claiming:
        return
    lore_claiming.add(payload.message_id)
    try:
        channel = bot.get_channel(payload.channel_id) or await bot.fetch_channel(
            payload.channel_id
        )
        who = payload.member.name if payload.member else str(payload.user_id)
        await claim_lore_thread(channel, payload.message_id, who)
    except Exception:
        logger.exception(
            "Failed to handle lore thread claim on message %d", payload.message_id
        )
    finally:
        lore_claiming.discard(payload.message_id)


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

    # Lore threads carry their own conversation in the session store and need
    # the research system prompt, so they cannot use the plain chat path below.
    if channel.id in lore_threads:
        await handle_lore_thread_message(message, channel)
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
