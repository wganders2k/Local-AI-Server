"""
Persistent thread registry for chat threads created by the bot.

Stores metadata about each /chat thread (model, name, creation time, status)
in a JSON file so the bot can restore its thread mappings on restart without
needing to scan all Discord channels from scratch.

Status values:
  - active:   Thread exists and is currently visible
  - archived: Thread exists but is archived by Discord (still responds if unarchived)
  - deleted:  Thread no longer exists on Discord
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("mimic-bot.registry")


class ThreadRegistry:
    """
    JSON-backed store for chat thread metadata.

    Each entry is keyed by thread_id (stored as string for JSON compatibility):
    {
        "1234567890": {
            "model": "llama3.2",
            "name": "chat-llama3.2",
            "created_at": "2026-05-14T03:00:00Z",
            "status": "active"
        }
    }
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Read registry from disk. Creates file if it does not exist."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    raw = json.load(f)
                self._data = raw
                logger.info("Loaded registry with %d entries from %s", len(self._data), self._path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load registry from %s: %s — starting fresh", self._path, e)
                self._data = {}
        else:
            logger.info("No existing registry at %s — starting fresh", self._path)
            self._data = {}

    def save(self) -> None:
        """Atomically write registry to disk (temp file + rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(str(tmp_path), str(self._path))
        except OSError as e:
            logger.error("Failed to save registry to %s: %s", self._path, e)

    def register(self, thread_id: int, model: str, name: str) -> None:
        """Register a new chat thread with status 'active'."""
        key = str(thread_id)
        self._data[key] = {
            "model": model,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self.save()
        logger.info("Registered thread %d (%s) with model=%s", thread_id, name, model)

    def get_all(self) -> Dict[int, Dict[str, Any]]:
        """Return all entries keyed by integer thread_id."""
        return {int(k): v for k, v in self._data.items()}

    def update_status(self, thread_id: int, status: str) -> None:
        """Update the status of an existing thread entry and persist."""
        key = str(thread_id)
        if key in self._data:
            self._data[key]["status"] = status
            self.save()

    def remove(self, thread_id: int) -> None:
        """Remove a thread entry from the registry and persist."""
        key = str(thread_id)
        if key in self._data:
            del self._data[key]
            self.save()
            logger.info("Removed thread %d from registry", thread_id)

    def get_model(self, thread_id: int) -> Optional[str]:
        """Return the model name for a thread, or None if not registered."""
        key = str(thread_id)
        entry = self._data.get(key)
        if entry and entry.get("status") != "deleted":
            return entry.get("model")
        return None
