"""
/chat — a thread bound to one model, and the handler that answers in it.

Also owns thread restoration at startup: the registry on disk is the only
record of which threads the bot answers in, since scanning every channel on
Discord would be far slower and is rate-limited.
"""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from discord_io import stream_and_send
from proxy_client import ProxyError

logger = logging.getLogger("mimic-bot.cogs.chat")


class ChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def services(self):
        return self.bot.services

    # ---- startup ----------------------------------------------------------

    async def restore_threads(self) -> None:
        """
        Restore thread→model mappings from the persistent registry.

        For each entry in the registry, verify the thread still exists via
        Discord API and update its status accordingly. Only non-deleted
        threads are added to the in-memory router.
        """
        services = self.services
        entries = services.registry.get_all()
        if not entries:
            logger.info("No threads in registry — nothing to restore")
            return

        restored = archived = deleted = skipped = 0

        for thread_id, entry in entries.items():
            model = entry.get("model")
            name = entry.get("name", "unknown")
            try:
                channel = await self.bot.fetch_channel(thread_id)

                if isinstance(channel, discord.Thread):
                    if channel.archived:
                        services.registry.update_status(thread_id, "archived")
                        archived += 1
                    else:
                        services.registry.update_status(thread_id, "active")

                    # Both active and archived threads are restored so the bot
                    # responds when users post (Discord auto-unarchives on new message).
                    services.threads.models[thread_id] = model
                    # A lore thread only works if its session survived too; without
                    # one there is no transcript to answer from, so fall back to
                    # treating it as a plain chat thread rather than erroring later.
                    if services.registry.get_kind(thread_id) == "lore":
                        if services.sessions.get(thread_id):
                            services.threads.lore.add(thread_id)
                        else:
                            logger.warning(
                                "Thread %d (%s) is registered as lore but has no saved "
                                "session — it will behave as a plain chat thread",
                                thread_id, name,
                            )
                    restored += 1
                    logger.info(
                        "Restored thread %d (%s) model=%s status=%s",
                        thread_id, name, model,
                        "archived" if channel.archived else "active",
                    )
                else:
                    # Channel exists but is not a thread — mark deleted.
                    services.registry.update_status(thread_id, "deleted")
                    deleted += 1
                    logger.warning(
                        "Thread %d (%s) exists but is not a Thread — marked deleted",
                        thread_id, name,
                    )

            except discord.NotFound:
                services.registry.update_status(thread_id, "deleted")
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

    # ---- command ----------------------------------------------------------

    async def model_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        """Provide autocomplete suggestions for available models from the API."""
        try:
            models = await self.services.proxy.list_models()
        except Exception:
            models = []
        choices = []
        for model in sorted(models):
            if current.lower() in model.lower():
                choices.append(app_commands.Choice(name=model, value=model))
            if len(choices) >= 25:  # Discord max choices
                break
        return choices

    @app_commands.command(name="chat", description="Create a chat thread with an AI model")
    @app_commands.describe(model="The model to chat with")
    @app_commands.autocomplete(model=model_autocomplete)
    async def chat(self, interaction: discord.Interaction, model: str):
        """Handle the /chat slash command — creates a thread for persistent chat."""
        services = self.services

        if not services.limiter.is_allowed(interaction.user.id):
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
            thread = await channel.create_thread(
                name=f"chat-{model}",
                auto_archive_duration=60,  # 1 hour archive duration
                type=discord.ChannelType.public_thread,
            )

            services.threads.models[thread.id] = model
            services.registry.register(thread.id, model, thread.name)

            await thread.send(
                f"🤖 Chat with **{model}** started. Send your messages here!"
            )
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

    # ---- thread replies ---------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Respond to regular messages sent inside chat threads."""
        if message.author == self.bot.user:
            return

        channel = message.channel
        if not isinstance(channel, discord.Thread):
            return

        services = self.services
        # Lore threads carry their own conversation in the session store and need
        # the research system prompt, so LoreCog answers those instead.
        if channel.id in services.threads.lore:
            return

        model = services.threads.models.get(channel.id)
        if model is None:
            return

        if not services.limiter.is_allowed(message.author.id):
            await channel.send(
                f"{message.author.mention} ⚠️ You're sending requests too fast. Slow down a bit.",
            )
            return

        logger.info(
            "Thread chat from %s in thread %s (channel %s): model=%s, message=%r",
            message.author.name,
            channel.name,
            channel.parent.name if channel.parent else "unknown",
            model,
            message.content[:100],
        )

        try:
            conv_history = services.history.get_history(channel.id, model)

            # No system prompt, just history + user message. Label the user
            # message with the sender's display name so the model can track who
            # is speaking when multiple users share a thread.
            labeled = f"[{message.author.display_name}]: {message.content}"
            msgs: list[dict] = [*conv_history, {"role": "user", "content": labeled}]

            async with channel.typing():
                full_response = await stream_and_send(
                    channel.send, services.proxy, model, msgs
                )

            services.history.add_turn(channel.id, model, labeled, full_response)

        except ProxyError as e:
            logger.warning("Proxy error: %s", e)
            await channel.send(f"⚠️ {e}")
        except Exception:
            logger.exception("Unexpected error handling thread chat for model %s", model)
            await channel.send("An unexpected error occurred. Please try again.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ChatCog(bot))
