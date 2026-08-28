"""
/lore — agentic RAG over the Discord archive, and the follow-up threads it offers.

Lifecycle of a lore thread:

    /lore  ->  answer embeds  ->  offer message (bot reacts with ❓)
                                       |  somebody adds the same reaction
                                       v
                                  thread opened, session claimed
                                       |  follow-up questions
                                       v
                                  swept after LORE_THREAD_TTL_SECONDS of silence

Nothing is created until somebody reacts: most answers are read once and never
followed up, and a thread per run buried the channel.
"""

import logging

import discord
from discord import app_commands
from discord.ext import commands, tasks

from config import (
    AGENT_CTX_LIMIT,
    AGENT_MODEL,
    LORE_OFFER_TTL_SECONDS,
    LORE_SWEEP_INTERVAL_SECONDS,
    LORE_THREAD_EMOJI,
    LORE_THREAD_TTL_SECONDS,
    MAX_QUEUE_DEPTH,
)
from cogs.lore_status import LoreStatus
from discord_io import post_lore_answer
from formatters import build_lore_embeds
from lore.agent import run_agent_loop, run_lore_turn, seed_session_from_run
from lore.compaction import maybe_compact_session
from lore.session import LoreSession
from proxy_client import ProxyError

logger = logging.getLogger("mimic-bot.cogs.lore")

_OFFER_TEXT = "React to this message to follow up in a thread."


def _lore_thread_name(question: str) -> str:
    """Thread name for a lore follow-up, within Discord's 100-char limit."""
    return f"lore-{question[:60].strip() or 'lore'}"


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
    bits = [f"🧠 {used:,}/{AGENT_CTX_LIMIT:,} ({pct:.0f}%)"]
    if searched:
        calls = metrics.total_tool_calls
        bits.append(f"{calls} search{'' if calls == 1 else 'es'}")
        bits.append(f"{metrics.rounds_executed} round{'' if metrics.rounds_executed == 1 else 's'}")
    else:
        bits.append("answered from thread context")
    if session.compactions:
        bits.append(f"{session.compactions} compaction{'' if session.compactions == 1 else 's'}")
    return "-# " + " · ".join(bits)


class LoreCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def services(self):
        return self.bot.services

    async def cog_load(self) -> None:
        # Offers nobody reacted to have no event that would ever fire, so they
        # are only ever removed here and when a new offer is made.
        self.services.sessions.sweep_offers(LORE_OFFER_TTL_SECONDS)
        # Retire quiet threads now, then hourly. Doing it at startup means a bot
        # that was down past several expiries catches up immediately.
        self.sweep_loop.start()
        logger.info(
            "Lore thread sweep running every %ds, retiring threads idle over %.0f day(s)",
            LORE_SWEEP_INTERVAL_SECONDS, LORE_THREAD_TTL_SECONDS / 86400,
        )

    async def cog_unload(self) -> None:
        self.sweep_loop.cancel()

    # ---- command ----------------------------------------------------------

    @app_commands.command(name="lore", description="Ask the lore assistant a question")
    @app_commands.describe(
        question="Your question about server lore",
        rounds="Maximum search rounds (1-25, default: 10)",
    )
    async def lore(
        self,
        interaction: discord.Interaction,
        question: str,
        rounds: int = 10,
    ):
        """Handle the /lore slash command — agentic RAG with tool calling."""
        services = self.services

        # No rate limit check — lore queries are long-running and don't benefit
        # from it. Queue depth still applies: reject if the backend is busy.
        try:
            depth = await services.proxy.get_queue_depth()
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

        await interaction.response.defer()

        try:
            # The channel list goes into the system prompt, so the agent knows
            # what it may search. DMs have no guild and so no channels.
            guild = interaction.guild
            channel_names = (
                [ch.name for ch in guild.text_channels]
                if isinstance(guild, discord.Guild)
                else []
            )

            # The status message belongs to this layer: the agent reports
            # through it but must not know what an interaction is.
            status = await LoreStatus.from_interaction(interaction)

            result = await run_agent_loop(
                user_question=question,
                proxy_client=services.proxy,
                rag_client=services.rag,
                status=status,
                channel_names=channel_names,
                max_rounds=rounds,
            )

            for embed in build_lore_embeds(result.answer):
                await interaction.followup.send(embed=embed)

            # Offer a follow-up thread rather than opening one.
            if result.ok:
                await self.offer_thread(interaction, question, channel_names, result)

        except ProxyError as e:
            logger.warning("Proxy error: %s", e)
            await interaction.followup.send(f"⚠️ {e}", ephemeral=True)
        except Exception:
            logger.exception("Unexpected error handling lore request")
            await interaction.followup.send(
                "An unexpected error occurred. Please try again.",
                ephemeral=True,
            )

    # ---- offer / claim ----------------------------------------------------

    async def offer_thread(
        self,
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

        Failures are logged and swallowed: the user already has their answer in
        the channel, and losing the offer should not turn a successful /lore
        into an error.
        """
        channel = interaction.channel
        if not hasattr(channel, "create_thread"):
            return  # DMs: the embed alone is fine.

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
            self.services.sessions.offer(
                offer.id, session, ttl_seconds=LORE_OFFER_TTL_SECONDS
            )
            logger.info("Lore thread offer %d is ready to claim", offer.id)
        except Exception:
            logger.exception("Failed to seed lore thread offer %d", offer.id)

    async def claim_thread(
        self,
        channel: discord.abc.Messageable,
        message_id: int,
        claimant: str,
    ) -> None:
        """
        Turn a reacted-to offer into a live thread.

        Three things keep a burst of reactions from producing a pile of threads:
        the caller's `claiming` guard, claim() popping the offer so a second
        caller finds nothing, and Discord itself allowing only one thread per
        message.
        """
        services = self.services

        pending = services.sessions.peek_offer(message_id)
        if pending is None:
            await channel.send(
                "-# That follow-up offer has expired — run `/lore` again to ask more."
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

        session = services.sessions.claim(message_id, thread.id)
        if session is None:
            logger.warning("Offer %d vanished between check and claim", message_id)
            return

        try:
            services.threads.models[thread.id] = AGENT_MODEL
            services.threads.lore.add(thread.id)
            services.registry.register(thread.id, AGENT_MODEL, thread.name, kind="lore")

            logger.info(
                "Opened lore thread %d from offer %d for %s",
                thread.id, message_id, claimant,
            )
            await thread.send(
                "Ask a follow-up here — I already have the research behind that "
                "answer, and I'll only search again if your question needs it.\n"
                + _lore_footer(session)
            )
            # The bot has no manage_messages permission anywhere, so it cannot
            # clear the reaction to show the offer is spent. Editing the message
            # is the only signal available.
            await message.edit(content=f"{_OFFER_TEXT}\n-# → {thread.mention}")
        except Exception:
            logger.exception("Failed to finish opening lore thread %d", thread.id)

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """
        Open a lore follow-up thread when someone claims an offer.

        Raw rather than on_reaction_add: the cached variant only fires for
        messages still in the client's message cache, so an offer would silently
        stop working after a restart or once the cache rolled over. The raw event
        fires from the gateway regardless, which is why it deals in ids and needs
        the channel fetched by hand.

        Requires the reactions gateway intent (on by default, and unprivileged)
        plus add_reactions and read_message_history in the channel — verified
        present guild-wide for this bot.
        """
        if payload.user_id == self.bot.user.id:
            return  # The bot's own seed reaction.
        if str(payload.emoji) != LORE_THREAD_EMOJI:
            return
        # Overwhelmingly the common case — every reaction in every channel lands
        # here, so this stays ahead of any API call.
        services = self.services
        if not services.sessions.has_offer(payload.message_id):
            return

        # Check-and-mark with no await in between, so a second reaction arriving
        # while the first is still opening its thread cannot start a second one.
        if payload.message_id in services.threads.claiming:
            return
        services.threads.claiming.add(payload.message_id)
        try:
            channel = self.bot.get_channel(payload.channel_id) or await self.bot.fetch_channel(
                payload.channel_id
            )
            who = payload.member.name if payload.member else str(payload.user_id)
            await self.claim_thread(channel, payload.message_id, who)
        except Exception:
            logger.exception(
                "Failed to handle lore thread claim on message %d", payload.message_id
            )
        finally:
            services.threads.claiming.discard(payload.message_id)

    # ---- follow-up turns --------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Answer a follow-up posted in a lore thread."""
        if message.author == self.bot.user:
            return
        thread = message.channel
        if not isinstance(thread, discord.Thread):
            return

        services = self.services
        if thread.id not in services.threads.lore:
            return  # ChatCog's problem, or nobody's.

        session = services.sessions.get(thread.id)
        if session is None:
            logger.warning("No lore session for thread %d — ignoring", thread.id)
            return

        if not services.limiter.is_allowed(message.author.id):
            await thread.send(
                f"{message.author.mention} ⚠️ You're sending requests too fast."
            )
            return

        logger.info(
            "Lore follow-up from %s in %s: %r",
            message.author.name, thread.name, message.content[:100],
        )

        # Same progress vocabulary the slash command shows, reported into the thread.
        status = await LoreStatus.from_thread(thread)

        lock = services.threads.lock_for(thread.id)
        if lock.locked():
            logger.info("Thread %d is mid-answer — queuing this follow-up", thread.id)
            await status.waiting()

        # Held across the whole turn: the session is re-read inside the lock
        # because a queued turn must see the previous one's appended answer, not
        # the snapshot taken before it waited.
        async with lock:
            session = services.sessions.get(thread.id)
            if session is None:
                logger.warning("Lore session for thread %d vanished — ignoring", thread.id)
                return
            try:
                async with thread.typing():
                    answer, metrics, searched = await run_lore_turn(
                        session=session,
                        question=message.content,
                        proxy_client=services.proxy,
                        rag_client=services.rag,
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

            services.sessions.put(session)
            await post_lore_answer(
                thread, answer, _lore_footer(session, metrics, searched),
                edit_first=status.message,
            )

            try:
                if await maybe_compact_session(session, services.proxy):
                    services.sessions.put(session)
            except Exception:
                logger.exception("Compaction failed after follow-up")

    # ---- expiry -----------------------------------------------------------

    async def retire_thread(self, thread_id: int, reason: str) -> None:
        """
        Purge every local record of one lore thread.

        The session store holds the whole research payload verbatim, so this is
        what actually reclaims the disk. The Discord thread itself is left in
        place — the bot has no business deleting a channel people can still
        read — but with its records gone it is no longer routed to the lore
        handler, and posting in it does nothing.

        Ordered so the session goes last: if anything below fails, the thread is
        already unrouted and cannot be answered from a half-purged state.
        """
        services = self.services
        services.threads.forget(thread_id)
        services.registry.remove(thread_id)
        services.sessions.remove(thread_id)
        logger.info("Retired lore thread %d (%s)", thread_id, reason)

    async def sweep_threads(self) -> int:
        """
        Retire lore threads that have gone quiet, telling each one it has.

        A thread mid-answer is skipped rather than purged: its lock is held for
        the length of a turn, and pulling the session out from under a running
        turn would strand it.

        Returns:
            How many threads were retired.
        """
        services = self.services
        expired = services.sessions.expired_threads(LORE_THREAD_TTL_SECONDS)
        retired = 0

        for thread_id in expired:
            lock = services.threads.locks.get(thread_id)
            if lock is not None and lock.locked():
                logger.info("Thread %d is mid-answer — deferring expiry", thread_id)
                continue

            days = LORE_THREAD_TTL_SECONDS / 86400
            try:
                thread = self.bot.get_channel(thread_id) or await self.bot.fetch_channel(thread_id)
                await thread.send(
                    f"-# 💤 This thread has gone inactive after {days:.0f} days "
                    "without a question, so I've released the research behind it. "
                    "Run `/lore` again to start fresh."
                )
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                # Deleted, or the bot lost access. Nothing to notify, but the
                # local records still have to go — that is the whole point.
                logger.info(
                    "Could not notify thread %d of expiry — purging anyway", thread_id,
                    exc_info=True,
                )
            except Exception:
                logger.exception("Unexpected error notifying thread %d of expiry", thread_id)

            await self.retire_thread(thread_id, "inactive")
            retired += 1

        if retired:
            logger.info("Lore thread sweep retired %d thread(s)", retired)
        return retired

    @tasks.loop(seconds=LORE_SWEEP_INTERVAL_SECONDS)
    async def sweep_loop(self) -> None:
        try:
            await self.sweep_threads()
        except Exception:
            # A raising task loop stops silently, and a stopped sweep means the
            # store grows without bound.
            logger.exception("Lore thread sweep failed")

    @sweep_loop.before_loop
    async def _before_sweep(self) -> None:
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LoreCog(bot))
