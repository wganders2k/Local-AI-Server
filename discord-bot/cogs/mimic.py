"""/mimic — a persona replies to the channel's recent conversation, in the channel it was invoked from."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from cogs.common import admit, handle_resolver, persona_autocomplete
from config import MIMIC_MODEL
from mimic.transcript import clean_persona_reply
from proxy_client import ProxyError

logger = logging.getLogger("mimic-bot.cogs.mimic")


class MimicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def services(self):
        return self.bot.services

    @app_commands.command(name="mimic", description="Have a mimic persona reply to the conversation here")
    @app_commands.describe(persona="The persona who should chime in")
    @app_commands.autocomplete(persona=persona_autocomplete)
    async def mimic(self, interaction: discord.Interaction, persona: str):
        """Handle the /mimic slash command."""
        services = self.services
        if not await admit(interaction, services, personas=[persona]):
            return
        channel = interaction.channel

        logger.info(
            "Request from %s in %s: persona=%s", interaction.user.name, getattr(channel, "name", "?"), persona
        )

        # Ephemeral: the reply itself comes from the persona's webhook, and the
        # placeholder is deleted once it lands.
        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            reader = services.webhooks.reader(interaction.guild, handle_resolver(self.bot))
            prompt = reader.cast.prompt(persona, await reader.fetch_lines(channel))
            if prompt is None:
                await interaction.followup.send(
                    "⚠️ There's no recent conversation here to reply to.", ephemeral=True
                )
                return
            logger.debug("Prompt for %s:\n%s", persona, prompt[-1]["content"])

            raw = await services.proxy.chat(MIMIC_MODEL, prompt, lora=services.personas.lora(persona))
            reply = clean_persona_reply(raw)
            if not reply:
                raise ProxyError("The model returned an empty response.")

            await services.webhooks.post(persona, reply, channel)
            await interaction.delete_original_response()

        except discord.Forbidden:
            logger.warning("Missing permissions for /mimic in %s", getattr(channel, "name", "?"))
            await interaction.followup.send(
                "⚠️ I need **Manage Webhooks** and **Read Message History** here to post as a persona.",
                ephemeral=True,
            )
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
