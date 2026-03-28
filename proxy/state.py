import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class OrchestratorState:
    """
    Tracks which model is currently loaded in the swappable llama-server slot
    and serialises all access to it via a single asyncio.Lock.

    The lock is intentionally coarse — one request at a time through the
    swappable slot. This is correct for a single-GPU setup where VRAM is
    the bottleneck, not CPU or network.
    """

    def __init__(self) -> None:
        self.current_model: str | None = None
        self.lock: asyncio.Lock = asyncio.Lock()
        self._queue_depth: int = 0
        self.last_swap_at: float | None = None  # epoch seconds of last model switch

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def increment_queue(self) -> None:
        self._queue_depth += 1

    def decrement_queue(self) -> None:
        self._queue_depth = max(0, self._queue_depth - 1)

    def record_swap(self, new_model: str) -> None:
        """Record a model switch and update the swap timestamp."""
        self.last_swap_at = time.monotonic()
        self.current_model = new_model

    @property
    def time_since_swap(self) -> float | None:
        """Seconds since the last model swap, or None if no swap has occurred."""
        if self.last_swap_at is None:
            return None
        return time.monotonic() - self.last_swap_at


# Module-level singleton — imported by main.py
state = OrchestratorState()
