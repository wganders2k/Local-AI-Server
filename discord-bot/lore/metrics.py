"""
Timing, token and tool-call accounting for one agent run.

Token counts come from the backend's own usage reports rather than from
character estimates, which drift badly on tool-result text. See record_usage().
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from config import LORE_DEFAULT_SPEED
from lore.context_window import window
from lore.models import research_model

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
    # The run's speed, for the log.
    speed: str = LORE_DEFAULT_SPEED
    # The model the most recent call went to: its window is what that call's
    # context is measured against. A run with a separate writer moves from the
    # research model to the writer for its answer.
    model: str = field(default_factory=lambda: research_model(LORE_DEFAULT_SPEED))

    # Token accounting. prompt/completion/cached describe the most recent call;
    # peak_context_tokens is the high-water mark across the whole run, which is
    # the number that matters against the model's context window.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    peak_context_tokens: int = 0
    # Highest occupancy of any one call, as a fraction of the window of the
    # model that served it — tokens alone cannot be compared across models.
    peak_context_fraction: float = 0.0
    total_completion_tokens: int = 0
    # Generation timing as llama-server measured it (its "timings"), summed
    # over every call that reported it: decode only, so prefill and model
    # swaps do not drag the speed down.
    timed_tokens: int = 0
    timed_ms: float = 0.0
    # What the run was configured with, for the completion line.
    research_model: str = ""
    writer_model: str = ""
    max_rounds: int = 0

    @property
    def total_duration(self) -> float:
        return time.monotonic() - self.start_time

    def record_usage(
        self,
        usage: Optional[dict],
        fallback_prompt_chars: int = 0,
        fallback_completion_chars: int = 0,
        model: Optional[str] = None,
    ) -> None:
        """
        Absorb one backend usage dict, from a call to ``model`` (default: the
        model of the previous call).

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
        if model:
            self.model = model
        timings = (usage or {}).get("timings") or {}
        if timings.get("predicted_n") and timings.get("predicted_ms"):
            self.timed_tokens += int(timings["predicted_n"])
            self.timed_ms += float(timings["predicted_ms"])
        self.total_completion_tokens += self.completion_tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self.context_used)
        self.peak_context_fraction = max(
            self.peak_context_fraction, fraction(self.context_used, self.ctx_limit)
        )

    @property
    def ctx_limit(self) -> int:
        """The window of the model the most recent call went to."""
        return window(self.model)

    @property
    def context_used(self) -> int:
        """Tokens the most recent call actually occupied in the slot."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def context_pct(self) -> float:
        """Peak occupancy of any one call, as a fraction of its model's window."""
        return self.peak_context_fraction

    def context_line(self) -> str:
        """One-line context summary for the round log."""
        used, limit = self.context_used, self.ctx_limit
        return (
            f"context={used:,}/{limit:,} ({fraction(used, limit):.0%}) "
            f"cached={self.cached_tokens:,}"
        )

    @property
    def tokens_per_second(self) -> float:
        """Average generation speed across the run's calls, 0 if unmeasured."""
        return self.timed_tokens / self.timed_ms * 1000 if self.timed_ms else 0.0

    def run_line(self) -> str:
        """
        The run at a glance, for the finished status message: which models,
        rounds used of allowed, tool calls, wall time, generation speed.
        """
        if self.writer_model and self.writer_model != self.research_model:
            models = f"🔬 {self.research_model} → ✍️ {self.writer_model}"
        else:
            models = f"🔬 {self.research_model or self.model}"
        secs = int(self.total_duration)
        elapsed = f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"
        rounds = f"{self.rounds_executed}/{self.max_rounds}" if self.max_rounds else str(self.rounds_executed)
        calls = self.total_tool_calls
        bits = [
            models,
            f"{rounds} rounds",
            f"{calls} tool call{'' if calls == 1 else 's'}",
            elapsed,
        ]
        if self.tokens_per_second:
            bits.append(f"{self.tokens_per_second:.0f} tok/s")
        return " · ".join(bits)

    def summary(self) -> str:
        return (
            f"AgentMetrics — duration={self.total_duration:.1f}s, "
            f"rounds={self.rounds_executed}, tool_calls={self.total_tool_calls}, "
            f"tools={self.tools_used!r}, "
            f"avg_llm={sum(self.llm_response_times)/max(len(self.llm_response_times),1):.2f}s, "
            f"avg_tool={sum(self.tool_call_times)/max(len(self.tool_call_times),1):.2f}s, "
            f"peak_context={self.peak_context_tokens:,} "
            f"({self.context_pct * 100:.0f}% of its model's window), "
            f"generated={self.total_completion_tokens:,} tok"
        )
