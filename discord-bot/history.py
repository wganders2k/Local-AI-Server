"""
Per-channel, per-persona conversation history management.

Maintains a rolling window of message exchanges (deque) for each
unique (channel_id, model_name) pair. History is in-memory only
and cleared on bot restart.
"""

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from config import create_empty_history


class ConversationHistory:
    """
    Manages rolling conversation windows for mimic personas.

    Key: (channel_id, model_name) → deque of message dicts.
    Lore queries are stateless and do not use this store.
    """

    def __init__(self):
        self._store: Dict[Tuple[int, str], Deque[dict]] = defaultdict(
            lambda: create_empty_history()
        )

    def get_history(
        self, channel_id: int, model_name: str
    ) -> List[dict]:
        """
        Retrieve the current conversation window for a channel/persona pair.

        Returns a list of message dicts (user/assistant turns) suitable
        for injection into the proxy request.
        """
        key = (channel_id, model_name)
        return list(self._store[key])

    def add_turn(
        self,
        channel_id: int,
        model_name: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Append a complete exchange (user + assistant) to the history.

        Both messages are added as a single turn. The deque maxlen
        ensures old turns are automatically evicted.
        """
        key = (channel_id, model_name)
        self._store[key].append({"role": "user", "content": user_message})
        self._store[key].append(
            {"role": "assistant", "content": assistant_message}
        )

    def clear_channel(self, channel_id: int) -> None:
        """Clear all history for a specific channel."""
        keys_to_remove = [
            key for key in self._store if key[0] == channel_id
        ]
        for key in keys_to_remove:
            del self._store[key]

    def clear_persona(self, model_name: str) -> None:
        """Clear all history for a specific persona across all channels."""
        keys_to_remove = [
            key for key in self._store if key[1] == model_name
        ]
        for key in keys_to_remove:
            del self._store[key]

    def clear_all(self) -> None:
        """Clear all conversation history."""
        self._store.clear()
