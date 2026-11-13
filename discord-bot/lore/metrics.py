"""
Timing, token and tool-call accounting for one agent run.

Token counts come from the backend's own usage reports rather than from
character estimates, which drift badly on tool-result text. See record_usage().
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from lore.context_window import limit as ctx_limit

# The char -> token estimates, for the few places that have no usage report to
# hand. One home, so every estimate in a run agrees.
#
# General text: the same ~4 chars/token proxy/main.py's Prometheus fallback uses.
CHARS_PER_TOKEN = 4
# RAG excerpts specifically, counted until the next usage report arrives.
# Measured at ~3.1 chars/token (a 34,088-char result grew the next request by
# 10,901 tokens), so 3 errs toward acting a little early.
TOOL_CHARS_PER_TOKEN = 3


def fraction(used: int, limit: int) -> float:
    """used / limit, or 0 for an unknown limit. Callers format it as a percentage."""
    return used / limit if limit else 0.0


@dataclass
class AgentMetrics:
    """Collects timing and usage metrics for a single agent run."""
    start_time: float = field(default_factory=time.monotonic)
    total_tool_calls: int = 0
    rounds_executed: int = 0
    tool_call_times: list[float] = field(default_factory=list)
    llm_response_times: list[float] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # Token accounting. prompt/completion/cached describe the most recent call;
    # peak_context_tokens is the high-water mark across the whole run, which is
    # the number that matters against the model's context window.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    peak_context_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def total_duration(self) -> float:
        return time.monotonic() - self.start_time

    def record_usage(
        self,
        usage: Optional[dict],
        fallback_prompt_chars: int = 0,
        fallback_completion_chars: int = 0,
    ) -> None:
        """
        Absorb one backend usage dict.

        The backend reports real counts on both the streamed and buffered
        paths, so the character estimate is only a guard against a build that
        drops the field (CHARS_PER_TOKEN).
        """
        if usage:
            self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            self.completion_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            self.cached_tokens = int(details.get("cached_tokens") or 0)
        else:
            self.prompt_tokens = fallback_prompt_chars // CHARS_PER_TOKEN
            self.completion_tokens = fallback_completion_chars // CHARS_PER_TOKEN
            self.cached_tokens = 0
        self.total_completion_tokens += self.completion_tokens
        self.peak_context_tokens = max(
            self.peak_context_tokens, self.prompt_tokens + self.completion_tokens
        )

    @property
    def context_used(self) -> int:
        """Tokens the most recent call actually occupied in the slot."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def context_pct(self) -> float:
        """Peak context occupancy as a fraction of the model's window."""
        return fraction(self.peak_context_tokens, ctx_limit())

    def context_line(self) -> str:
        """One-line context summary for the round log."""
        used, limit = self.context_used, ctx_limit()
        return (
            f"context={used:,}/{limit:,} ({fraction(used, limit):.0%}) "
            f"cached={self.cached_tokens:,}"
        )

    def summary(self) -> str:
        return (
            f"AgentMetrics — duration={self.total_duration:.1f}s, "
            f"rounds={self.rounds_executed}, tool_calls={self.total_tool_calls}, "
            f"tools={self.tools_used!r}, "
            f"avg_llm={sum(self.llm_response_times)/max(len(self.llm_response_times),1):.2f}s, "
            f"avg_tool={sum(self.tool_call_times)/max(len(self.tool_call_times),1):.2f}s, "
            f"peak_context={self.peak_context_tokens:,}/{ctx_limit():,} "
            f"({self.context_pct * 100:.0f}%), "
            f"generated={self.total_completion_tokens:,} tok"
        )
