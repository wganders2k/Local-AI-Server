"""
Timing, token and tool-call accounting for one agent run.

Token counts come from the backend's own usage reports rather than from
character estimates, which drift badly on tool-result text. See record_usage().
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from config import AGENT_CTX_LIMIT


@dataclass
class AgentMetrics:
    """Collects timing and usage metrics for a single agent run."""
    start_time: float = field(default_factory=time.monotonic)
    total_tool_calls: int = 0
    rounds_executed: int = 0
    tool_call_times: list[float] = field(default_factory=list)
    llm_response_times: list[float] = field(default_factory=list)
    rag_query_times: list[float] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # Token accounting. prompt/completion/cached describe the most recent call;
    # peak_context_tokens is the high-water mark across the whole run, which is
    # the number that matters against AGENT_CTX_LIMIT.
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
        drops the field — it uses the same ~4 chars/token conversion as
        proxy/main.py's Prometheus fallback.
        """
        if usage:
            self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            self.completion_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            self.cached_tokens = int(details.get("cached_tokens") or 0)
        else:
            self.prompt_tokens = fallback_prompt_chars // 4
            self.completion_tokens = fallback_completion_chars // 4
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
        return self.peak_context_tokens / AGENT_CTX_LIMIT if AGENT_CTX_LIMIT else 0.0

    def context_line(self) -> str:
        """One-line context summary for the round log."""
        used = self.context_used
        pct = (used / AGENT_CTX_LIMIT * 100) if AGENT_CTX_LIMIT else 0.0
        return (
            f"context={used:,}/{AGENT_CTX_LIMIT:,} ({pct:.0f}%) "
            f"cached={self.cached_tokens:,}"
        )

    def summary(self) -> str:
        return (
            f"AgentMetrics — duration={self.total_duration:.1f}s, "
            f"rounds={self.rounds_executed}, tool_calls={self.total_tool_calls}, "
            f"tools={self.tools_used!r}, "
            f"avg_llm={sum(self.llm_response_times)/max(len(self.llm_response_times),1):.2f}s, "
            f"avg_rag={sum(self.rag_query_times)/max(len(self.rag_query_times),1):.2f}s, "
            f"peak_context={self.peak_context_tokens:,}/{AGENT_CTX_LIMIT:,} "
            f"({self.context_pct * 100:.0f}%), "
            f"generated={self.total_completion_tokens:,} tok"
        )
