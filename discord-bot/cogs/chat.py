"""/chat — a thread bound to one model, and the handler that answers in it."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from cogs.common import admit, admit_message, model_autocomplete
from discord_io import stream_and_send
from proxy_client import ProxyError
from threads import CHAT

logger = logging.getLogger("mimic-bot.cogs.chat")


class ChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def services(self):
        return self.bot.services

    # ---- command ----------------------------------------------------------

    @app_commands.command(name="chat", description="Create a chat thread with an AI model")
    @app_commands.describe(model="The model to chat with")
    @app_commands.autocomplete(model=model_autocomplete)
    async def chat(self, interaction: discord.Interaction, model: str):
        """Handle the /chat slash command — creates a thread for persistent chat."""
        services = self.services

        # No queue check: creating the thread costs nothing, and each message
        # in it streams through the proxy's own queue.
        if not await admit(interaction, services, queue=False):
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
            services.threads.open(thread.id, CHAT, model, thread.name)

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

        services = self.services
        channel = message.channel
        if services.threads.kind_of(channel.id) != CHAT:
            return
        model = services.threads.model_of(channel.id)

        if not await admit_message(message, services):
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
            # No system prompt, just history + user message. Label the user
            # message with the sender's display name so the model can track who
            # is speaking when multiple users share a thread.
            labeled = f"[{message.author.display_name}]: {message.content}"
            msgs: list[dict] = [*services.history.get_history(channel.id), {"role": "user", "content": labeled}]

            full_response = await stream_and_send(channel.send, services.proxy, model, msgs)

            services.history.add_turn(channel.id, labeled, full_response)

        except ProxyError as e:
            logger.warning("Proxy error: %s", e)
            await channel.send(f"⚠️ {e}")
        except Exception:
            logger.exception("Unexpected error handling thread chat for model %s", model)
            await channel.send("An unexpected error occurred. Please try again.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ChatCog(bot))
