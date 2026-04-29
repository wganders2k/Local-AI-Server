"""
Per-user rate limiting with a sliding time window.

Defaults to 5 requests per user per 60-second window, but parameters
are configurable via config.py.
"""

from collections import defaultdict
from time import monotonic
from typing import Dict, List

from config import RATE_LIMIT_PER_USER, RATE_LIMIT_WINDOW_SECONDS


class RateLimiter:
    """
    Sliding-window rate limiter keyed by Discord user ID.

    Tracks timestamps of each request per user and evicts entries
    outside the current window on every check.
    """

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_PER_USER,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.max_requests = max_requests
        self.window = window_seconds
        self.user_timestamps: Dict[int, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        """
        Check if the user is within their rate limit.

        Returns True if the request is allowed, False if the user
        has exceeded their quota for the current window.
        """
        now = monotonic()
        timestamps = self.user_timestamps[user_id]

        # Evict timestamps outside the window
        cutoff = now - self.window
        self.user_timestamps[user_id] = [t for t in timestamps if t > cutoff]

        if len(self.user_timestamps[user_id]) >= self.max_requests:
            return False

        self.user_timestamps[user_id].append(now)
        return True

    def remaining_requests(self, user_id: int) -> int:
        """Return how many requests the user has left in the current window."""
        now = monotonic()
        cutoff = now - self.window
        active = [t for t in self.user_timestamps[user_id] if t > cutoff]
        return max(0, self.max_requests - len(active))

    def reset(self, user_id: int) -> None:
        """Clear the rate limit history for a specific user."""
        self.user_timestamps[user_id] = []
