"""
The /lore progress message.

Implements lore.progress.ProgressReporter against a real Discord message, so
the agent core can report progress without knowing Discord exists.
"""

import logging
from typing import Optional

import discord

logger = logging.getLogger("mimic-bot.cogs.lore-status")

_STATUS_START = "\U0001f50d Searching Discord history..."


class LoreStatus:
    """
    The single source of /lore's progress vocabulary.

    Both the opening run and thread follow-ups report through this, so the
    indicators a user sees in a thread are literally the same strings the
    slash command shows — there is no second copy to drift.

    Every edit is best-effort: a status update must never be the thing that
    fails a run, so all Discord errors are swallowed. A LoreStatus whose
    message could not be sent is a working no-op, which is why callers never
    have to guard against None.
    """

    def __init__(self, message: Optional[discord.Message] = None):
        self._message = message

    @classmethod
    async def from_interaction(cls, interaction: discord.Interaction) -> "LoreStatus":
        """Open a status message as a followup on a (deferred) interaction."""
        try:
            return cls(await interaction.followup.send(_STATUS_START))
        except Exception:
            # Followup fails if the interaction was never deferred; the run
            # should still proceed, just silently.
            return cls(None)

    @classmethod
    async def from_thread(cls, thread: discord.Thread) -> "LoreStatus":
        """Open a status message inside a thread."""
        try:
            return cls(await thread.send(_STATUS_START))
        except Exception:
            return cls(None)

    @property
    def message(self) -> Optional[discord.Message]:
        """The underlying message, so a caller can edit it into the answer."""
        return self._message

    async def _set(self, content: str) -> None:
        if self._message is None:
            return
        try:
            await self._message.edit(content=content)
        except Exception:
            pass

    async def waiting(self) -> None:
        """Queued behind another turn in the same thread."""
        await self._set("⏳ Waiting for the previous question to finish...")

    async def thinking(self, round_num: int, max_rounds: int) -> None:
        await self._set(f"\U0001f50d Thinking... (round {round_num}/{max_rounds})")

    async def searching(self, round_num: int, max_rounds: int) -> None:
        await self._set(f"\U0001f50d Searching... ({round_num}/{max_rounds})")

    async def analyzing(self, round_num: int, max_rounds: int) -> None:
        await self._set(f"\U0001f4dd Analyzing results... (round {round_num}/{max_rounds})")

    async def writing(self) -> None:
        await self._set("✍️ Writing answer...")

    async def generating(self, chars: Optional[int] = None) -> None:
        if chars is None:
            await self._set("\U0001f4dd Generating final answer...")
        else:
            await self._set(f"\U0001f4dd Generating final answer… ({chars:,} chars)")

    async def complete(self, question: str) -> None:
        preview = question[:100] + ("..." if len(question) > 100 else "")
        await self._set(f"✅ Search complete — \"{preview}\"")

    async def failed(self) -> None:
        await self._set("❌ Search failed")
