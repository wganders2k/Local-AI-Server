"""
Proxy-side state: which model is in the swappable slot, and how many LLM requests
are in flight.

Everything about *external jobs* used to live here too — priority, a
one-runner-at-a-time lease, headroom gating, a three-state handover machine. It
now lives in the arbiter, which is the only component that can stop a job rather
than ask one. The proxy's remaining interest in the GPU is a single question it
asks over HTTP: may I have it?
"""

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
        self.last_request_at: float | None = None  # monotonic, for idle eviction
        # In-flight requests, which gate when the arbiter is told we are idle.
        # Reaching zero is what lets jobs have the GPU back, so an over-release
        # here would hand it away mid-generation.
        self._llm_inflight: int = 0

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def increment_queue(self) -> None:
        self._queue_depth += 1

    def decrement_queue(self) -> None:
        self._queue_depth = max(0, self._queue_depth - 1)

    def llm_request_started(self) -> None:
        self._llm_inflight += 1

    def llm_request_finished(self) -> None:
        self._llm_inflight = max(0, self._llm_inflight - 1)

    @property
    def llm_inflight(self) -> int:
        return self._llm_inflight

    def record_swap(self, new_model: str) -> None:
        """Record a model switch and update the swap timestamp."""
        self.last_swap_at = time.monotonic()
        self.current_model = new_model

    def record_request(self) -> None:
        """Mark LLM activity — resets the idle-eviction clock."""
        self.last_request_at = time.monotonic()

    @property
    def idle_seconds(self) -> float | None:
        """Seconds since the last LLM request, or None if there has been none."""
        if self.last_request_at is None:
            return None
        return time.monotonic() - self.last_request_at

    @property
    def time_since_swap(self) -> float | None:
        """Seconds since the last model swap, or None if no swap has occurred."""
        if self.last_swap_at is None:
            return None
        return time.monotonic() - self.last_swap_at


state = OrchestratorState()
