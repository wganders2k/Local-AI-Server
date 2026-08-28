"""
The progress vocabulary the agent reports through.

Defined here as a protocol so the agent loop can drive a status display without
importing discord. cogs.lore_status.LoreStatus is the real implementation; a
test can pass anything with these methods, or None.
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    """
    Progress callbacks issued over the life of a /lore turn.

    Every method is best-effort: a status update must never be the thing that
    fails a run, so implementations swallow their own transport errors.
    """

    async def waiting(self) -> None:
        """Queued behind another turn in the same thread."""

    async def thinking(self, round_num: int, max_rounds: int) -> None: ...

    async def searching(self, round_num: int, max_rounds: int) -> None: ...

    async def analyzing(self, round_num: int, max_rounds: int) -> None: ...

    async def writing(self) -> None: ...

    async def generating(self, chars: Optional[int] = None) -> None: ...

    async def complete(self, question: str) -> None: ...

    async def failed(self) -> None: ...
