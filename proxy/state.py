import asyncio
import logging

logger = logging.getLogger(__name__)


class OrchestratorState:
    """
    Tracks which model is currently loaded in the swappable Ollama slot
    and serialises all access to it via a single asyncio.Lock.

    The lock is intentionally coarse — one request at a time through the
    swappable slot. This is correct for a single-GPU setup where VRAM is
    the bottleneck, not CPU or network.
    """

    def __init__(self) -> None:
        self.current_model: str | None = None
        self.lock: asyncio.Lock = asyncio.Lock()
        self._queue_depth: int = 0

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def increment_queue(self) -> None:
        self._queue_depth += 1

    def decrement_queue(self) -> None:
        self._queue_depth = max(0, self._queue_depth - 1)


# Module-level singleton — imported by main.py
state = OrchestratorState()
