"""
/mimic-chat — a thread where mimic personas talk with each other and with whoever joins in.

The thread opens with a prompt for the starter to say something; the
personas speak only once someone has. The thread is the transcript: every
burst reads it back (persona_posting), so a restart loses nothing but the
in-flight burst. Personas post through the channel's webhook; a human message
stops a running burst, waits MIMIC_CHAT_DEBOUNCE_SECONDS for follow-ups, then
starts the next one.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from cogs.common import admit, handle_resolver, persona_autocomplete
from config import (
    MIMIC_CHAT_DEBOUNCE_SECONDS,
    MIMIC_CHAT_MAX_LENGTH,
    MIMIC_MODEL,
    VISION_MIMIC_ENABLED,
)
from mimic.engine import run_burst, seed_line
from mimic.state import MimicChatState
from personas import member_name
from proxy_client import ProxyError
from threads import MIMIC_CHAT

logger = logging.getLogger("mimic-bot.cogs.mimic_chat")

_THREAD_NAME_MAX = 100  # Discord's limit


class MimicChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def services(self):
        return self.bot.services

    # ---- command ----------------------------------------------------------

    @app_commands.command(
        name="mimic-chat", description="Start a thread where mimic personas chat, with each other and with you"
    )
    @app_commands.describe(
        persona1="A persona in the chat",
        persona2="Another persona (optional)",
        persona3="Another persona (optional)",
        persona4="Another persona (optional)",
        persona5="Another persona (optional)",
        persona6="Another persona (optional)",
        persona7="Another persona (optional)",
        length="Bot turns to run after your first message, without waiting (default: they pause at natural breaks)",
    )
    @app_commands.autocomplete(
        persona1=persona_autocomplete,
        persona2=persona_autocomplete,
        persona3=persona_autocomplete,
        persona4=persona_autocomplete,
        persona5=persona_autocomplete,
        persona6=persona_autocomplete,
        persona7=persona_autocomplete,
    )
    async def mimic_chat(
        self,
        interaction: discord.Interaction,
        persona1: str,
        persona2: Optional[str] = None,
        persona3: Optional[str] = None,
        persona4: Optional[str] = None,
        persona5: Optional[str] = None,
        persona6: Optional[str] = None,
        persona7: Optional[str] = None,
        length: Optional[app_commands.Range[int, 2, MIMIC_CHAT_MAX_LENGTH]] = None,
    ):
        """Handle the /mimic-chat slash command."""
        services = self.services
        # Slash commands have no multi-select, so each persona is its own option.
        personas = list(dict.fromkeys(p for p in (persona1, persona2, persona3, persona4, persona5, persona6, persona7) if p))

        if isinstance(interaction.channel, discord.Thread):
            await interaction.response.send_message(
                "⚠️ Start a mimic chat from a channel, not inside a thread.", ephemeral=True
            )
            return
        if not await admit(interaction, services, personas=personas):
            return
        channel: discord.TextChannel = interaction.channel
        names = [services.personas.get(p).display_name for p in personas]

        logger.info(
            "Mimic chat from %s in #%s: personas=%s length=%s",
            interaction.user.name, channel.name, personas, length,
        )
        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            thread = await channel.create_thread(
                name=f"mimic-chat: {', '.join(names)}"[:_THREAD_NAME_MAX],
                auto_archive_duration=1440,
                type=discord.ChannelType.public_thread,
            )
            user = interaction.user
            state = MimicChatState(
                personas=personas,
                starter_id=str(user.id),
                starter_label=member_name(user.id, user.name, services.personas.aliases()),
                started_at=datetime.now(timezone.utc),
                length=length,
            )
            services.threads.open_mimic_chat(thread.id, MIMIC_MODEL, thread.name, state)

            # Mentioning the starter adds them to the thread, so it is in their list.
            await thread.send(
                f"🎭 **{', '.join(names)}** are here. {user.mention}, say something to get the conversation going.\n"
                "Anyone can join in. The personas pause at natural breaks; send a message to keep it going.",
                allowed_mentions=discord.AllowedMentions(users=[user]),
            )
            await interaction.followup.send(f"✅ Mimic chat started: {thread.mention}", ephemeral=True)
        except discord.Forbidden:
            await interaction.followup.send(
                "⚠️ I don't have permission to create threads in this channel.", ephemeral=True
            )
            return

    # ---- the conversation -------------------------------------------------

    async def _burst(self, thread: discord.Thread, state: MimicChatState) -> None:
        """One stretch of bot talk. Serialised per thread: a new burst waits for the last to stop."""
        services = self.services
        async with services.threads.lock_for(thread.id):
            state.interrupted = False
            try:
                if not await services.personas.ensure_discovered(services.proxy):
                    await thread.send("⚠️ The personas are unavailable right now. Send another message to try again.")
                    return
                reader = services.webhooks.reader(thread.guild, handle_resolver(self.bot))
                lines = await reader.fetch_lines(thread)
                if state.premise:
                    lines.insert(0, seed_line(state.premise, state.starter_id, state.starter_label, state.started_at))
                if not lines:
                    return  # nothing anyone said survived the filters (a command, a bare ">>>")
                # `length` paces the first burst only: the one before any persona has spoken.
                length = state.length if all(line.persona is None for line in lines) else None
                # A persona since dropped from the roster or [mimic-multi] sits this chat out.
                personas = [p for p in state.personas if p in reader.cast.personas]
                if not personas:
                    await thread.send("⚠️ None of this chat's personas are available any more.")
                    return

                async def post(persona: str, text: str) -> None:
                    await services.webhooks.post(persona, text, thread)

                new = await run_burst(
                    services.proxy, reader.cast, personas, state.premise, lines,
                    post=post, interrupted=lambda: state.interrupted, length=length,
                    captioner=services.vision if VISION_MIMIC_ENABLED else None,
                )
                logger.info("Mimic chat %d: burst of %d line(s)", thread.id, len(new))
            except discord.Forbidden:
                logger.warning("Missing permissions in mimic chat %d", thread.id)
                await thread.send("⚠️ I need **Manage Webhooks** in the parent channel to post as personas.")
            except ProxyError as e:
                logger.warning("Proxy error in mimic chat %d: %s", thread.id, e)
                await thread.send(f"⚠️ {e}")
            except Exception:
                logger.exception("Mimic chat burst failed in thread %d", thread.id)
                await thread.send("⚠️ Something went wrong. Send another message to try again.")

    async def _debounced(self, thread: discord.Thread, state: MimicChatState) -> None:
        await asyncio.sleep(MIMIC_CHAT_DEBOUNCE_SECONDS)
        # Past the wait: from here a newer message must not cancel this task —
        # cancelling a burst mid-request abandons a generation the GPU still runs.
        if state.pending is asyncio.current_task():
            state.pending = None
        await self._burst(thread, state)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """A human message in a mimic chat: interrupt the bots, then answer once the human pauses."""
        if message.webhook_id is not None or message.author.bot:
            return
        threads = self.services.threads
        if threads.kind_of(message.channel.id) != MIMIC_CHAT:
            return
        state = threads.mimic_chats[message.channel.id]
        # No rate limit: a message only restarts the debounce, and a burst
        # already running finishes its current line and stops.
        state.interrupted = True
        if state.pending is not None and not state.pending.done():
            state.pending.cancel()
        state.pending = asyncio.create_task(self._debounced(message.channel, state))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MimicChatCog(bot))
