"""What a /mimic-chat thread remembers between messages. Its transcript is the thread itself."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MimicChatState:
    personas: list[str]
    # Who ran /mimic-chat, and when.
    starter_id: str
    starter_label: str
    started_at: datetime
    # The opening line /mimic-chat used to take, named after the starter.
    # Only threads started before it was dropped have one; new ones open with
    # whatever the first person says.
    premise: Optional[str] = None
    # Bot turns the first burst runs without waiting for anyone, or None to
    # let the controller pace the chat (capped by MIMIC_CHAT_MAX_BURST).
    length: Optional[int] = None
    # Set by a human message while a burst runs, so the burst stops and the
    # next one answers them.
    interrupted: bool = False
    # The debounce timer for the latest human message; replaced by each new one.
    pending: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_registry(self) -> dict:
        return {
            "personas": self.personas,
            "premise": self.premise,
            "starter_id": self.starter_id,
            "starter_label": self.starter_label,
            "started_at": self.started_at.isoformat(),
            "length": self.length,
        }

    @classmethod
    def from_registry(cls, entry: dict) -> "MimicChatState":
        return cls(
            personas=list(entry["personas"]),
            premise=entry.get("premise"),
            starter_id=entry["starter_id"],
            starter_label=entry["starter_label"],
            started_at=datetime.fromisoformat(entry["started_at"]),
            length=entry.get("length"),
        )
