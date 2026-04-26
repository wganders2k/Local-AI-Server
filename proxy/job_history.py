import logging
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# How long (seconds) before a task is considered "finalized" if no new
# requests arrive for the same model.
TASK_GROUP_WINDOW = 180

# Maximum number of completed tasks to retain.
MAX_HISTORY = 10


class _Task:
    """Represents a single grouped task (one or more requests to the same model)."""

    def __init__(self, model: str, group_window: float) -> None:
        self.model: str = model
        self.request_count: int = 0
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self._active_requests: int = 0
        self._group_window: float = group_window
        self._last_activity: float = time.time()  # updated on every start/end
        # Accumulators for token tracking
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_generation_time: float = 0.0  # seconds the model was actively generating

    @property
    def is_active(self) -> bool:
        return self._active_requests > 0 or self.end_time is None

    @property
    def avg_tokens_per_second(self) -> Optional[float]:
        if self._total_generation_time <= 0:
            return None
        return self._total_output_tokens / self._total_generation_time

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    def request_start(self) -> None:
        self._active_requests += 1
        self.request_count += 1
        self._last_activity = time.time()

    def request_end(self, input_tokens: int, output_tokens: int, generation_time: float) -> None:
        self._active_requests -= 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_generation_time += generation_time
        self._last_activity = time.time()

    def maybe_finalize(self) -> bool:
        """
        If no active requests and enough time has passed since the last
        activity, mark the task as finalized.

        Returns True if the task was finalized.
        """
        if self.end_time is not None:
            return True  # already finalized
        # Use _last_activity instead of start_time for accurate windowing
        if time.time() - self._last_activity >= self._group_window:
            self.end_time = time.time()
            logger.debug(
                f"Finalized task for '{self.model}' "
                f"({self.request_count} req(s), {self.total_tokens} tokens)"
            )
            return True
        return False

    def to_summary_json(self) -> dict:
        """Return a minimal summary: name (token count) + elapsed time (status)."""
        now = time.time()
        if self.is_active:
            elapsed = now - self.start_time
            status = "active"
        else:
            elapsed = self.end_time - self.start_time if self.end_time else 0
            status = "completed"

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Format token count with thousands separator
        token_str = f"{self.total_tokens:,}"

        return {
            "name": f"{self.model} ({token_str} tk)",
            "description": f"{time_str} ({status})",
        }

    def to_homepage_json(self) -> dict:
        """Return Homepage-compatible widget card data."""
        status = "active" if self.is_active else "success"
        desc_parts = [f"{self.request_count} request{'s' if self.request_count != 1 else ''}"]
        if self.avg_tokens_per_second is not None:
            desc_parts.append(f"{self.avg_tokens_per_second:.1f} tok/s avg")
        desc_parts.append(f"{self.total_tokens} tokens ({self.total_input_tokens} in + {self.total_output_tokens} out)")
        description = " · ".join(desc_parts)

        return {
            "name": self.model,
            "status": status,
            "description": description,
            "url": "",
            "icon": "",
            "widget": {
                "component": "custom",
                "options": {
                    "active": self.is_active,
                    "model": self.model,
                    "request_count": self.request_count,
                    "avg_tokens_per_second": round(self.avg_tokens_per_second, 1) if self.avg_tokens_per_second is not None else None,
                    "total_tokens": self.total_tokens,
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
                    "end_time": datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat() if self.end_time else None,
                    "is_active": self.is_active,
                },
            },
        }


class JobHistory:
    """
    In-memory tracker for LLM proxy job history.

    Groups requests to the same model within a sliding time window into a
    single "task", and keeps the last MAX_HISTORY completed tasks.
    """

    def __init__(self, max_history: int = MAX_HISTORY, group_window: float = TASK_GROUP_WINDOW) -> None:
        self._lock = threading.Lock()
        self._max_history = max_history
        self._group_window = group_window
        # Currently active (unfinalized) tasks, keyed by model name
        self._active: dict[str, _Task] = {}
        # Completed tasks, oldest first
        self._completed: deque[_Task] = deque(maxlen=max_history)

    def request_start(self, model: str) -> str:
        """
        Called when a new request begins. Creates or extends a task.

        Returns a task_id (the model name for now, could be a real UUID later).
        """
        with self._lock:
            task = self._active.get(model)
            if task is None:
                # Check if there's a recent unfinalized task we can revive
                task = self._try_revive(model)
                if task is None:
                    task = _Task(model, self._group_window)
                self._active[model] = task
            task.request_start()
            return model

    def request_end(self, model: str, input_tokens: int, output_tokens: int, generation_time: float) -> None:
        """
        Called when a request completes. Records metrics and may finalize the task.
        """
        with self._lock:
            task = self._active.get(model)
            if task is None:
                # Edge case: task was already finalized between start and end
                logger.warning(f"request_end for '{model}' with no active task — ignoring")
                return
            task.request_end(input_tokens, output_tokens, generation_time)
            if task.maybe_finalize():
                del self._active[model]
                self._completed.append(task)

    def get_summary_json(self) -> list[dict]:
        """
        Return up to MAX_HISTORY tasks as a minimal summary list.
        Active tasks first, then most-recent completed tasks.
        Returns an empty list if no tasks exist.
        """
        with self._lock:
            self._flush_finalizable()

            result = []
            # Active tasks first
            for task in sorted(self._active.values(), key=lambda t: t.start_time, reverse=True):
                result.append(task.to_summary_json())
            # Then completed, most recent first
            for task in reversed(self._completed):
                result.append(task.to_summary_json())

            return result

    def get_homepage_json(self, empty_message: str = "No recent activity") -> list[dict]:
        """
        Return all tasks (active + recent completed) as Homepage-compatible JSON.
        Active tasks first, then most-recent completed tasks.
        If no tasks exist, returns a single placeholder card with empty_message.
        """
        with self._lock:
            # Finalize any stale tasks
            self._flush_finalizable()

            result = []
            # Active tasks first
            for task in sorted(self._active.values(), key=lambda t: t.start_time, reverse=True):
                result.append(task.to_homepage_json())
            # Then completed, most recent first
            for task in reversed(self._completed):
                result.append(task.to_homepage_json())

            if not result:
                result.append({
                    "name": empty_message,
                    "status": "info",
                    "description": "",
                    "url": "",
                    "icon": "",
                    "widget": {
                        "component": "custom",
                        "options": {
                            "active": False,
                            "model": "",
                            "request_count": 0,
                            "avg_tokens_per_second": None,
                            "total_tokens": 0,
                            "start_time": None,
                            "end_time": None,
                            "is_active": False,
                        },
                    },
                })

            return result

    def _try_revive(self, model: str) -> Optional[_Task]:
        """
        Check the completed deque for a recently-finished task to revive
        instead of creating a brand-new one.
        """
        if not self._completed:
            return None
        # Look at the most recent completed task
        last = self._completed[-1]
        if last.model == model and (time.time() - last.start_time) < self._group_window:
            # Revive it
            last.end_time = None
            self._completed.pop()
            return last
        return None

    def _flush_finalizable(self) -> None:
        """Move any finalized active tasks to the completed deque."""
        to_finalize = [
            model for model, task in self._active.items() if task.maybe_finalize()
        ]
        for model in to_finalize:
            task = self._active.pop(model, None)
            if task:
                self._completed.append(task)

    def heartbeat(self) -> int:
        """
        Background heartbeat: finalize any tasks whose group window has elapsed.

        Returns the number of tasks finalized.
        """
        with self._lock:
            self._flush_finalizable()
            return 0


# Module-level singleton
job_history = JobHistory()