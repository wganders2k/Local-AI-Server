"""
The rolling conversation window of each /chat thread.

A /chat thread is bound to one model for its life, so the thread id alone keys
it. In memory only: a restart starts every thread's window afresh.
"""

from collections import defaultdict, deque

from config import HISTORY_MAX_MESSAGES


class ConversationHistory:
    """thread_id -> the last HISTORY_MAX_MESSAGES user/assistant messages."""

    def __init__(self):
        self._store: dict[int, deque[dict]] = defaultdict(lambda: deque(maxlen=HISTORY_MAX_MESSAGES))

    def get_history(self, thread_id: int) -> list[dict]:
        """The window, oldest first, ready to prefix a request."""
        return list(self._store[thread_id])

    def add_turn(self, thread_id: int, user_message: str, assistant_message: str) -> None:
        """Append one exchange; the deque evicts the oldest messages past the cap."""
        self._store[thread_id].append({"role": "user", "content": user_message})
        self._store[thread_id].append({"role": "assistant", "content": assistant_message})

    def forget(self, thread_id: int) -> None:
        self._store.pop(thread_id, None)
