"""/mimic — one-shot persona replies in the channel it was invoked from."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import MAX_QUEUE_DEPTH, MIMIC_PERSONAS, get_mimic_system_prompt
from discord_io import stream_and_send
from proxy_client import ProxyError

logger = logging.getLogger("mimic-bot.cogs.mimic")


class MimicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def services(self):
        return self.bot.services

    async def persona_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        """Provide autocomplete suggestions for the persona option."""
        choices = []
        for persona in sorted(MIMIC_PERSONAS):
            if current.lower() in persona.lower():
                choices.append(app_commands.Choice(name=persona, value=persona))
            if len(choices) >= 25:  # Discord max choices
                break
        return choices

    @app_commands.command(name="mimic", description="Chat with a mimic persona")
    @app_commands.describe(
        persona="The persona to chat with",
        message="Your message to the persona",
    )
    @app_commands.autocomplete(persona=persona_autocomplete)
    async def mimic(
        self,
        interaction: discord.Interaction,
        persona: str,
        message: str,
    ):
        """Handle the /mimic slash command."""
        services = self.services

        if not services.limiter.is_allowed(interaction.user.id):
            await interaction.response.send_message(
                "⚠️ You're sending requests too fast. Slow down a bit.",
                ephemeral=True,
            )
            return

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
            depth = await services.proxy.get_queue_depth()
            if depth >= MAX_QUEUE_DEPTH:
                await interaction.response.send_message(
                    f"⚠️ The AI backend is busy ({depth} requests queued). "
                    f"Please try again in a moment.",
                    ephemeral=True,
                )
                return
        except ProxyError:
            # If we can't reach the proxy, the chat call below will handle the error.
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
            channel_id = interaction.channel.id if interaction.channel else 0
            conv_history = services.history.get_history(channel_id, persona)

            # Label the user message with the sender's display name so the model
            # can track who is speaking when multiple users share a channel.
            labeled = f"[{interaction.user.display_name}]: {message}"

            msgs: list[dict] = [{"role": "system", "content": system_prompt}]
            msgs.extend(conv_history)
            msgs.append({"role": "user", "content": labeled})

            async with interaction.channel.typing():
                full_response = await stream_and_send(
                    interaction.followup.send, services.proxy, persona, msgs
                )

            services.history.add_turn(channel_id, persona, labeled, full_response)

        except ProxyError as e:
            logger.warning("Proxy error: %s", e)
            await interaction.followup.send(f"⚠️ {e}", ephemeral=True)
        except Exception:
            logger.exception("Unexpected error handling persona %s", persona)
            await interaction.followup.send(
                "An unexpected error occurred. Please try again.",
                ephemeral=True,
            )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MimicCog(bot))
