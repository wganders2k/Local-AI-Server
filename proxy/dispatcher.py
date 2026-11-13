"""
Which request runs next on the backend, and under which arbiter lease.

Two layers schedule the GPU, and they answer different questions:

- the arbiter decides which *process* holds the card: this proxy, the video
  job, the trainer, the profiler;
- this module decides which *request* runs next inside the one llama-server
  (and image server) the proxy owns.

So the proxy is one tenant. It holds at most one lease at a time, named for
the most important work it has: `llm` while any interactive request is queued
or running, `llm-batch` while it has only batch work. Moving up from
`llm-batch` to `llm` is the proxy reclaiming from itself, which is safe because
the same process carries on and runs one request at a time. `llm-batch` is a
`cooperative` job: while it is held, a watcher waits on the arbiter's
reclaim notice, and when something ranked above batch wants the card the
proxy stops dispatching batch work, lets the running call finish, frees the
GPU and releases. That is a reclaim the arbiter can rely on, which `kind: none`
never was.

Requests are served one at a time, people first:

- an interactive request (chat, /mimic, images) runs as soon as nothing else
  is running, oldest first;
- a batch request (BATCH_MODELS: the judge, the archive captioner) runs only
  when no interactive request is waiting or running and none has finished in
  the last `batch_yield_seconds`, so a person waits for at most the one batch
  call already running;
- a request whose client has gone away is dropped while it waits, so an
  abandoned request never runs;
- a refused interactive request gets an answer at once (the proxy turns it into
  a 503), but a refused batch request waits in the queue and asks again later:
  its client is a background job, and a refusal is not its failure.

The idle evictor takes a turn of its own (`maintenance`), so nothing can start
on the backend while it is unloading models or stopping the image server.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

INTERACTIVE = "interactive"
BATCH = "batch"


class Refused(Exception):
    """The arbiter would not grant the card for an interactive request."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class Abandoned(Exception):
    """The client went away while its request was waiting."""


@dataclass(eq=False)
class Ticket:
    klass: str
    label: str = ""
    request: object = None  # anything with `async is_disconnected()`; None = never abandoned
    queued_at: float = field(default_factory=time.monotonic)


class Lease:
    """
    The one arbiter lease the proxy holds, named for the work it is serving.

    `llm` and `batch` are arbiter clients for the two job names. Every change
    of lease goes through `_lock`, so a release can never land between a
    request deciding it has the card and using it.
    """

    def __init__(self, llm, batch, notice_retry: float = 1.0):
        self.llm = llm
        self.batch = batch
        self.held: Optional[str] = None
        self._lock = asyncio.Lock()
        self._notice_retry = notice_retry
        self._watcher: Optional[asyncio.Task] = None
        # True while the proxy itself is asking for `llm` over its own batch
        # lease: the arbiter then asks the batch holder (us) to let go, and the
        # watcher does so at once, with nothing unloaded.
        self._handing_up = False
        # True while the watcher is freeing the GPU for someone else; a release
        # then waits for it rather than racing it.
        self._reclaiming = False
        # Set by the dispatcher: stop batch work and free the GPU.
        self.on_reclaim: Optional[Callable[[], Awaitable[None]]] = None

    async def ensure(self, klass: str) -> tuple[bool, str]:
        """Hold the lease `klass` needs. Returns the arbiter's (granted, detail)."""
        async with self._lock:
            if self.held == klass:
                return True, "already held"
            if klass == INTERACTIVE:
                if self.held == BATCH:
                    self._handing_up = True
                    try:
                        granted, detail = await self.llm.acquire()
                    finally:
                        self._handing_up = False
                else:
                    granted, detail = await self.llm.acquire()
                if granted:
                    self._stop_watcher()
                    self.held = INTERACTIVE
                return granted, detail
            # Batch never outranks `llm`, so it cannot be asked for over it.
            if self.held == INTERACTIVE:
                await self.llm.release()
                self.held = None
            granted, detail = await self.batch.acquire()
            if granted:
                self.held = BATCH
                self._start_watcher()
            return granted, detail

    async def release_if(self, still_idle: Callable[[], bool], clear_stale: bool = False) -> bool:
        """
        Release whatever is held, if `still_idle()` is true once the lease lock
        is held — the dispatcher's state is read inside the lock so a request
        that has just claimed its turn is never left without a lease.

        `clear_stale` (startup and the idle evictor only): holding nothing,
        release both names anyway, in case the arbiter kept one from before a
        restart.
        """
        async with self._lock:
            if not still_idle():
                return False
            if self.held is None:
                if not clear_stale:
                    return False
                # Nothing we know of — but the arbiter may still hold a lease
                # from before a proxy restart. `llm` is kind: none and is never
                # reaped, so a stale one would outrank every job forever.
                # Releasing a lease that is not held is a no-op there.
                await self.llm.release()
                await self.batch.release()
                return False
            if self.held == BATCH and self._reclaiming:
                return False  # the watcher releases once the GPU is free
            if self.held == INTERACTIVE:
                await self.llm.release()
            else:
                self._stop_watcher()
                await self.batch.release()
            self.held = None
            return True

    # -- the cooperative half: told when the card is wanted back --

    def _start_watcher(self) -> None:
        if self._watcher is None or self._watcher.done():
            self._watcher = asyncio.create_task(self._watch())

    def _stop_watcher(self) -> None:
        w, self._watcher = self._watcher, None
        if w is not None and not w.done() and w is not asyncio.current_task():
            w.cancel()

    async def _watch(self) -> None:
        """Hold the reclaim notice open for as long as the batch lease is ours."""
        while self.held == BATCH:
            wanted = await self.batch.reclaim_notice()
            if wanted is None:  # arbiter unreachable: ask again shortly
                await asyncio.sleep(self._notice_retry)
                continue
            if not wanted or self.held != BATCH:
                continue
            if self._handing_up:
                # Ourselves, moving up to `llm`: same process, nothing to free.
                await self.batch.release()
                self.held = None
                return
            logger.info("Arbiter wants the GPU back from llm-batch — pausing batch work")
            self._reclaiming = True
            try:
                if self.on_reclaim is not None:
                    await self.on_reclaim()
            except Exception as exc:  # pragma: no cover - logged, release still happens
                logger.error(f"Freeing the GPU for a reclaim failed: {exc}")
            finally:
                self._reclaiming = False
            await self.batch.release()
            self.held = None
            return

    async def aclose(self) -> None:
        self._stop_watcher()


class Dispatcher:
    """One request at a time on the backend, interactive before batch."""

    def __init__(
        self,
        lease: Lease,
        batch_yield_seconds: float,
        batch_retry_seconds: float = 5.0,
        poll_seconds: float = 0.5,
        free_gpu: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
        self.lease = lease
        self.batch_yield_seconds = batch_yield_seconds
        self.batch_retry_seconds = batch_retry_seconds
        self.poll_seconds = poll_seconds
        # Unloads models and stops the image server; main.py supplies it.
        self.free_gpu = free_gpu
        self.running: Optional[Ticket] = None
        self._waiting: list[Ticket] = []
        self._cond = asyncio.Condition()
        self.last_interactive_done: Optional[float] = None
        self._batch_retry_at = 0.0
        self._batch_suspended = False
        self._maintenance = False
        lease.on_reclaim = self._on_reclaim

    # -- what /status and the evictor read --

    @property
    def queue_depth(self) -> int:
        """Interactive requests waiting. The bot refuses commands past MAX_QUEUE_DEPTH on this."""
        return sum(1 for t in self._waiting if t.klass == INTERACTIVE)

    @property
    def batch_waiting(self) -> int:
        return sum(1 for t in self._waiting if t.klass == BATCH)

    @property
    def idle(self) -> bool:
        return self.running is None and not self._waiting and not self._maintenance

    # -- eligibility --

    def _batch_may_run(self, now: float) -> bool:
        if self._batch_suspended or now < self._batch_retry_at:
            return False
        last = self.last_interactive_done
        return last is None or now - last >= self.batch_yield_seconds

    def _eligible(self, t: Ticket, now: float) -> bool:
        if self.running is not None or self._maintenance:
            return False
        first_interactive = next((w for w in self._waiting if w.klass == INTERACTIVE), None)
        if t.klass == INTERACTIVE:
            return first_interactive is t
        if first_interactive is not None or not self._batch_may_run(now):
            return False
        return next(w for w in self._waiting if w.klass == BATCH) is t

    # -- taking and giving back a turn --

    async def turn(self, klass: str, request=None, label: str = "") -> Ticket:
        """
        Wait for this request's turn and hold the lease it needs. The caller
        must hand the ticket back with `finish` once the backend is done.

        Raises Refused (interactive only) or Abandoned.
        """
        t = Ticket(klass, label, request)
        while True:
            await self._claim(t)
            granted, detail = await self.lease.ensure(klass)
            if granted:
                return t
            async with self._cond:
                self.running = None
                if klass == INTERACTIVE:
                    self._cond.notify_all()
                    logger.error(f"Arbiter would not hand over the GPU: {detail}")
                    raise Refused(detail)
                # A background request: keep its place and ask again later.
                logger.info(f"Batch request refused by the arbiter ({detail}); retrying")
                self._batch_retry_at = time.monotonic() + self.batch_retry_seconds
                self._waiting.insert(0, t)
                self._cond.notify_all()

    async def _claim(self, t: Ticket) -> None:
        async with self._cond:
            if t not in self._waiting:
                self._waiting.append(t)
                self._cond.notify_all()
            try:
                while not self._eligible(t, time.monotonic()):
                    if t.request is not None and await t.request.is_disconnected():
                        logger.info(f"[{t.label}] client went away while queued — dropped")
                        raise Abandoned()
                    try:
                        await asyncio.wait_for(self._cond.wait(), self.poll_seconds)
                    except asyncio.TimeoutError:
                        pass
                self._waiting.remove(t)
                self.running = t
            except BaseException:
                if t in self._waiting:
                    self._waiting.remove(t)
                    self._cond.notify_all()
                raise

    def finish(self, t: Ticket) -> None:
        """The backend is done with `t`. Synchronous so it can run from any exit path."""
        if self.running is t:
            self.running = None
        if t.klass == INTERACTIVE:
            self.last_interactive_done = time.monotonic()
        asyncio.get_running_loop().create_task(self._after_finish())

    async def _after_finish(self) -> None:
        async with self._cond:
            self._cond.notify_all()
        # Let a waiting request claim its turn before deciding the proxy is idle.
        await asyncio.sleep(0)
        await self.lease.release_if(self._nothing_ready)

    def _nothing_ready(self) -> bool:
        """No request running and none able to start now: the lease can go back."""
        if self.running is not None or self._maintenance:
            return False
        if any(t.klass == INTERACTIVE for t in self._waiting):
            return False
        return not (self.batch_waiting and self._batch_may_run(time.monotonic()))

    # -- the idle evictor's turn --

    @asynccontextmanager
    async def maintenance(self):
        """
        Exclusive use of the backend while nothing else wants it. Yields False
        (and does nothing) when a request is running or waiting.
        """
        async with self._cond:
            if not self.idle:
                ok = False
            else:
                self._maintenance = ok = True
        try:
            yield ok
        finally:
            if ok:
                async with self._cond:
                    self._maintenance = False
                    self._cond.notify_all()

    async def release_lease_if_idle(self) -> bool:
        """For the evictor, inside its maintenance turn, once the GPU is really free."""
        return await self.lease.release_if(lambda: self.running is None, clear_stale=True)

    # -- the arbiter wants the card back from llm-batch --

    async def _on_reclaim(self) -> None:
        async with self._cond:
            self._batch_suspended = True
        try:
            async with self._cond:
                while self.running is not None and self.running.klass == BATCH:
                    try:
                        await asyncio.wait_for(self._cond.wait(), self.poll_seconds)
                    except asyncio.TimeoutError:
                        pass
            if self.free_gpu is not None:
                await self.free_gpu()
        finally:
            async with self._cond:
                self._batch_suspended = False
                self._batch_retry_at = time.monotonic() + self.batch_retry_seconds
                self._cond.notify_all()
