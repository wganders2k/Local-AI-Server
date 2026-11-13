"""
Proxy-side state: which model is in the swappable slot, and when the backend
was last used.

Which request runs next, and under which arbiter lease, is dispatcher.py's.
Everything about *external jobs* is the arbiter's: it is the only component
that can stop a job rather than ask one.
"""

import logging
import time

logger = logging.getLogger(__name__)


class OrchestratorState:
    def __init__(self) -> None:
        self.current_model: str | None = None
        self.last_swap_at: float | None = None  # monotonic, last model switch
        self.last_request_at: float | None = None  # monotonic, for idle eviction

    def record_swap(self, new_model: str) -> None:
        """Record a model switch and update the swap timestamp."""
        self.last_swap_at = time.monotonic()
        self.current_model = new_model

    def record_request(self) -> None:
        """Mark backend activity — resets the idle-eviction clock."""
        self.last_request_at = time.monotonic()

    @property
    def idle_seconds(self) -> float | None:
        """Seconds since the last request, or None if there has been none."""
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
