import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class ExternalJobState:
    """
    Tracks active external jobs (batch processing, ML inference, etc.) and
    coordinates VRAM handover with the LLM proxy.

    Two distinct signals, deliberately kept separate:

      * ``yield_requested`` — the proxy wants VRAM back. Jobs poll this and
        stop working. It also gates whether a *new* job may start at all
        (exposed as ``may_run``).
      * ``_all_yielded`` — every registered job has confirmed it has actually
        released its VRAM. The proxy awaits this before loading a model.

    Conflating the two is what made the previous implementation a no-op: the
    proxy waited on the same flag it was asking the job to observe, so nothing
    ever set it and every handover fell through to the timeout path.

    A job confirms only after its GPU memory is verifiably gone (in practice,
    after the process holding the CUDA context has exited). ``confirm_yield``
    is therefore a statement about the GPU, not an acknowledgement of the
    request.

    All access happens on the proxy's single event loop, so no locking is
    needed; the asyncio.Event does the synchronising.
    """

    def __init__(self) -> None:
        # job_id -> {"status": "running"|"yielding", "registered_at": float}
        self._jobs: dict[str, dict] = {}
        self._yield_requested: bool = False
        self._confirmed: set[str] = set()
        self._all_yielded: asyncio.Event = asyncio.Event()
        self._all_yielded.set()  # vacuously true while nothing is registered
        self._llm_inflight: int = 0

    # -- registration --

    @property
    def external_job_running(self) -> bool:
        return len(self._jobs) > 0

    @property
    def active_job_ids(self) -> list[str]:
        return list(self._jobs.keys())

    @property
    def yield_requested(self) -> bool:
        return self._yield_requested

    def register(self, job_id: str) -> dict:
        self._jobs[job_id] = {
            "status": "yielding" if self._yield_requested else "running",
            "registered_at": time.time(),
        }
        # A job registering while a yield is outstanding has not confirmed yet.
        self._reevaluate()
        logger.info(f"External job registered: {job_id} (active: {len(self._jobs)})")
        return {
            "message": f"Job {job_id} registered",
            "job_id": job_id,
            "may_run": not self._yield_requested,
        }

    def unregister(self, job_id: str) -> dict:
        if job_id not in self._jobs:
            return {"message": f"Job {job_id} not found"}
        del self._jobs[job_id]
        self._confirmed.discard(job_id)
        # A job that has gone away is no longer holding VRAM, so an outstanding
        # handover may now be complete.
        self._reevaluate()
        logger.info(f"External job unregistered: {job_id} (active: {len(self._jobs)})")
        return {"message": f"Job {job_id} unregistered", "job_id": job_id}

    def get_status(self, job_id: str) -> dict:
        if job_id not in self._jobs:
            return {
                "job_id": job_id,
                "exists": False,
                "yield_requested": self._yield_requested,
                "may_run": not self._yield_requested,
            }
        return {
            "job_id": job_id,
            "exists": True,
            "status": self._jobs[job_id]["status"],
            "yield_requested": self._yield_requested,
            "may_run": not self._yield_requested,
            "confirmed": job_id in self._confirmed,
        }

    # -- handover --

    def request_yield_all(self) -> dict:
        """Ask every registered job to release VRAM. Idempotent."""
        if not self._yield_requested:
            self._yield_requested = True
            logger.info(
                f"Yield requested from {len(self._jobs)} external job(s): {self.active_job_ids}"
            )
        for job in self._jobs.values():
            job["status"] = "yielding"
        self._reevaluate()
        return {
            "message": f"Yield requested for {len(self._jobs)} job(s)",
            "jobs": self.active_job_ids,
        }

    def confirm_yield(self, job_id: str) -> dict:
        """
        Called by an external job once its VRAM is verifiably released.

        This is the only thing that can unblock the proxy's handover wait.
        """
        if job_id not in self._jobs:
            return {"message": f"Job {job_id} not found"}
        if not self._yield_requested:
            # A confirmation with no outstanding request is stale — typically a
            # job that released just as the handover completed. Recording it
            # would leave the job pre-confirmed, and the *next* handover would
            # then complete instantly without the job having yielded at all.
            logger.debug(f"Ignoring stale yield confirmation from {job_id}")
            return {"message": "No yield outstanding", "job_id": job_id, "stale": True}
        self._jobs[job_id]["status"] = "yielding"
        self._confirmed.add(job_id)
        logger.info(
            f"Job {job_id} confirmed VRAM release "
            f"({len(self._confirmed)}/{len(self._jobs)} confirmed)"
        )
        self._reevaluate()
        return {"message": f"Job {job_id} confirmed yield", "job_id": job_id}

    def resume(self, job_id: str) -> dict:
        """Called by an external job when it starts working again."""
        if job_id not in self._jobs:
            return {"message": f"Job {job_id} not found"}
        self._jobs[job_id]["status"] = "running"
        self._confirmed.discard(job_id)
        self._reevaluate()
        logger.info(f"Job {job_id} resuming")
        return {"message": f"Job {job_id} resuming", "job_id": job_id}

    def release_yield(self) -> dict:
        """
        Drop the yield request — external jobs may run again.

        Called when the last in-flight LLM request finishes, or when the idle
        evictor has unloaded the resident model.
        """
        if not self._yield_requested:
            return {"message": "No yield outstanding"}
        self._yield_requested = False
        self._confirmed.clear()
        for job in self._jobs.values():
            job["status"] = "running"
        self._reevaluate()
        logger.info("Yield released — external jobs may resume")
        return {"message": "Yield released", "jobs": self.active_job_ids}

    async def wait_for_yield(self) -> None:
        """
        Block until every registered job has confirmed VRAM release.

        Deliberately unbounded — the caller owns the timeout policy, because
        what to do on timeout (fail the request vs. keep waiting) is a proxy
        decision, not a state-tracking one.
        """
        await self._all_yielded.wait()

    # -- LLM request refcount --

    def llm_request_started(self) -> None:
        self._llm_inflight += 1

    def llm_request_finished(self) -> None:
        self._llm_inflight = max(0, self._llm_inflight - 1)

    @property
    def llm_inflight(self) -> int:
        return self._llm_inflight

    # -- helpers --

    def _reevaluate(self) -> None:
        """Set or clear _all_yielded to match the current confirmation state."""
        if not self._yield_requested:
            # Nothing outstanding; keep it set so a waiter never hangs on a
            # handover that was already released.
            self._all_yielded.set()
            return
        outstanding = set(self._jobs) - self._confirmed
        if outstanding:
            self._all_yielded.clear()
        else:
            self._all_yielded.set()

    def snapshot(self) -> dict:
        return {
            "jobs": {jid: dict(j) for jid, j in self._jobs.items()},
            "yield_requested": self._yield_requested,
            "confirmed": sorted(self._confirmed),
            "all_yielded": self._all_yielded.is_set(),
            "llm_inflight": self._llm_inflight,
        }


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
        self.last_request_at: float | None = None  # monotonic, for idle eviction

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

    def record_request(self) -> None:
        """Mark LLM activity — resets the idle-eviction clock."""
        self.last_request_at = time.monotonic()

    @property
    def idle_seconds(self) -> float | None:
        """Seconds since the last LLM request, or None if there has been none."""
        if self.last_request_at is None:
            return None
        return time.monotonic() - self.last_request_at

    @property
    def time_since_swap(self) -> float | None:
        """Seconds since the last model swap, or None if no swap has occurred."""
        if self.last_swap_at is None:
            return None
        return time.monotonic() - self.last_swap_at


# Module-level singletons — initialized after class definitions
state = OrchestratorState()
external_jobs = ExternalJobState()
