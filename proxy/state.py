import asyncio
import logging
import time
import threading

logger = logging.getLogger(__name__)


class ExternalJobState:
    """
    Tracks active external jobs (batch processing, ML inference, etc.)
    and coordinates VRAM yielding with the LLM proxy.

    When an external job is running, the proxy can signal it to yield VRAM
    so that LLM model loading can proceed. The external job yields at natural
    boundaries (between processing batches).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # job_id -> {"status": "running"|"yielding", "registered_at": float}
        self._jobs: dict[str, dict] = {}
        self._yield_event: asyncio.Event | None = None  # set when external job should yield

    @property
    def external_job_running(self) -> bool:
        return len(self._jobs) > 0

    @property
    def active_job_ids(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())

    def register(self, job_id: str) -> dict:
        with self._lock:
            self._jobs[job_id] = {
                "status": "running",
                "registered_at": time.time(),
            }
            logger.info(f"External job registered: {job_id} (active: {len(self._jobs)})")
            return {"message": f"Job {job_id} registered", "job_id": job_id}

    def unregister(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                return {"message": f"Job {job_id} not found"}
            del self._jobs[job_id]
            logger.info(f"External job unregistered: {job_id} (active: {len(self._jobs)})")
            return {"message": f"Job {job_id} unregistered", "job_id": job_id}

    def get_status(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                return {"job_id": job_id, "exists": False, "yield_requested": False}
            job = self._jobs[job_id]
            # Yield is requested when we're trying to load an LLM model
            yield_requested = self._yield_event is not None and self._yield_event.is_set()
            return {
                "job_id": job_id,
                "exists": True,
                "status": job["status"],
                "yield_requested": yield_requested,
            }

    def request_yield(self, job_id: str) -> dict:
        """Signal that an external job should yield VRAM (for LLM model loading)."""
        with self._lock:
            if job_id not in self._jobs:
                return {"message": f"Job {job_id} not found"}
            self._jobs[job_id]["status"] = "yielding"
            logger.info(f"Yield requested for job: {job_id}")
            return {"message": f"Yield requested for job {job_id}", "job_id": job_id}

    async def wait_for_yield(self) -> None:
        """Wait until the external job signals it has yielded (cleared the event)."""
        event = asyncio.Event()
        self._yield_event = event
        try:
            # Wait up to 5 minutes for the external job to yield
            await asyncio.wait_for(event.wait(), timeout=300)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for external job to yield")
        finally:
            self._yield_event = None

    def clear_yield(self, job_id: str) -> dict:
        """Called by external job to signal it has yielded VRAM."""
        with self._lock:
            if job_id not in self._jobs:
                return {"message": f"Job {job_id} not found"}
            self._jobs[job_id]["status"] = "yielding"  # still yielding until resume
            logger.info(f"Job {job_id} has yielded VRAM")
            return {"message": f"Job {job_id} has yielded", "job_id": job_id}

    def resume(self, job_id: str) -> dict:
        """Called by external job to signal it's resuming after yield."""
        with self._lock:
            if job_id not in self._jobs:
                return {"message": f"Job {job_id} not found"}
            self._jobs[job_id]["status"] = "running"
            # Clear the yield event if it exists
            if self._yield_event:
                self._yield_event.clear()
                self._yield_event = None
            logger.info(f"Job {job_id} resuming after yield")
            return {"message": f"Job {job_id} resuming", "job_id": job_id}

    def force_yield_all(self) -> dict:
        """Force all external jobs to yield (called when LLM needs VRAM)."""
        with self._lock:
            if not self._jobs:
                return {"message": "No external jobs running"}
            # Set a global yield event
            loop = asyncio.get_event_loop()
            event = asyncio.Event()
            self._yield_event = event
            for job_id in self._jobs:
                self._jobs[job_id]["status"] = "yielding"
            logger.info(f"Force-yielding {len(self._jobs)} external job(s)")
            return {"message": f"Yield requested for {len(self._jobs)} job(s)"}


class OrchestratorState:
    """
    Tracks which model is currently loaded in the swappable llama-server slot
    and serialises all access to it via a single asyncio.Lock.

    The lock is intentionally coarse — one request at a time through the
    swappable slot. This is correct for a single-GPU setup where VRAM is
    the bottleneck, not CPU or network.
    """

    def __init__(self) -> None:
        self.current_model: str | None = None
        self.lock: asyncio.Lock = asyncio.Lock()
        self._queue_depth: int = 0
        self.last_swap_at: float | None = None  # epoch seconds of last model switch

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def increment_queue(self) -> None:
        self._queue_depth += 1

    def decrement_queue(self) -> None:
        self._queue_depth = max(0, self._queue_depth - 1)

    def record_swap(self, new_model: str) -> None:
        """Record a model switch and update the swap timestamp."""
        self.last_swap_at = time.monotonic()
        self.current_model = new_model

    @property
    def time_since_swap(self) -> float | None:
        """Seconds since the last model swap, or None if no swap has occurred."""
        if self.last_swap_at is None:
            return None
        return time.monotonic() - self.last_swap_at


# Module-level singletons — initialized after class definitions
state = OrchestratorState()
external_jobs = ExternalJobState()
