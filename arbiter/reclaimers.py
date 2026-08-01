"""
How the arbiter takes the GPU back.

The arbiter no longer starts anything. Every job runs itself, asks for the card
through /gpu/acquire, and gives it back with /gpu/release — so the only thing
that differs between jobs is *how you take it back from one that will not go on
its own*, and that is all this file is.

    NoneReclaimer       nothing to tear down. The job releases between requests
                        and the lease is the whole of the arbiter's grip. The LLM
                        is this: the proxy owns llama-server's lifecycle, and a
                        second service stopping that container would race it.

    ContainerReclaimer  stop the container. The cgroup goes with it and the
                        kernel returns the VRAM. Nothing is asked and nothing is
                        trusted. What a job should be.

    CooperativeReclaimer  ask the job to stop its own worker, and wait. For a
                        job that is not a container, where reaching in would
                        cost more authority than it is worth.

The asymmetry is the argument for preferring containers. ContainerReclaimer's
reclaim is a call and a wait on process exit, measured at 0.9s on this hardware,
and the cgroup teardown is the proof. CooperativeReclaimer's is a request, a
wait, a timeout policy, and an assertion it cannot verify.
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class Reclaimer:
    """
    A job's grip on the GPU.

    Holding is a *lease the arbiter granted*, not a state inferred from the
    outside world. That is the whole simplification: the previous design asked
    Docker whether a container was up and tried to read intent out of the answer,
    which is how a merely-registered job and a job actually using the card came
    to be two different questions. A lease has one answer, and the arbiter is the
    one who wrote it down.
    """

    def __init__(self, job):
        self.job = job
        self._held = False

    @property
    def name(self) -> str:
        return self.job.name

    @property
    def leased(self) -> bool:
        """The arbiter granted this job the card and has not been told otherwise."""
        return self._held

    async def holds(self) -> bool:
        return self._held

    def grant(self) -> None:
        self._held = True

    def forget(self) -> None:
        """Drop the lease without touching the job. The reaper's tool."""
        self._held = False

    async def reclaim(self, timeout: float) -> bool:
        """Take the card back. True once the job is definitely off it."""
        self._held = False
        return True


class NoneReclaimer(Reclaimer):
    """
    Nothing to tear down; the lease is the entire mechanism.

    Honest about what it does not do: reclaiming here is bookkeeping, not a kill,
    and cannot make the process let go of its memory. It is only correct because
    the sole caller is a higher-priority acquire, and a job ranked below another
    was ranked there deliberately.
    """


class ContainerReclaimer(Reclaimer):
    """
    A job that is a container.

    Thin over the Docker API on purpose: the guarantee is Docker's and the
    kernel's, not ours. When the container is gone so is every process in it, and
    so is its device memory.
    """

    def __init__(self, job, client):
        super().__init__(job)
        self.client = client

    async def holds(self) -> bool:
        # Both halves matter. Without the lease, a container that is up but still
        # waiting for permission would look like a tenant and get killed for the
        # LLM every time. Without the liveness check, a lease outlives the
        # container that took it and blocks the card until something notices.
        return self._held and await self._up()

    async def _up(self) -> bool:
        state = await asyncio.to_thread(self.client.state, self.job.container)
        return bool(state and state.get("Running"))

    async def reclaim(self, timeout: float) -> bool:
        """
        Kill the container and return True once it is gone.

        Signalled rather than stopped, and that is not a stylistic choice: Docker
        suppresses the restart policy for a container stopped through its API, so
        a preempted job would never come back. See ``DockerClient.kill``.

        ``job.stop_timeout`` is the grace between SIGTERM and SIGKILL and belongs
        to the job; ``timeout`` is the arbiter's patience overall. stop_timeout=0
        means SIGKILL outright, which is right for a job that checkpoints on a
        wall clock and expects to die — a grace period there is pure latency on
        an interactive request, buying a graceful exit nothing needs.
        """
        logger.info(
            f"Killing {self.job.name} (grace {self.job.stop_timeout}s) — "
            f"cgroup teardown returns the VRAM"
        )
        try:
            await asyncio.wait_for(self._kill_and_wait(timeout), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"{self.job.name} was still up {timeout:.0f}s after being killed")
            return False
        except Exception as exc:
            logger.error(f"Could not kill {self.job.name}: {exc}")
            return False

        if await self._up():
            return False
        self._held = False
        return True

    async def _kill_and_wait(self, timeout: float) -> None:
        """SIGTERM, a bounded wait, then SIGKILL. Straight to SIGKILL if grace is 0."""
        deadline = time.monotonic() + min(self.job.stop_timeout, timeout)
        if self.job.stop_timeout > 0:
            await asyncio.to_thread(self.client.kill, self.job.container, "SIGTERM")
            while time.monotonic() < deadline:
                if not await self._up():
                    return
                await asyncio.sleep(0.2)
        await asyncio.to_thread(self.client.kill, self.job.container, "SIGKILL")
        while await self._up():
            await asyncio.sleep(0.1)


class CooperativeReclaimer(Reclaimer):
    """
    A job that is not a container, so it has to tear *itself* down.

    It speaks the same lease protocol as everything else — acquire, release —
    plus one call nothing else needs: it holds a blocking GET on
    /gpu/reclaim-notice for as long as it holds the card. Reclaiming is waking
    that call and waiting for the release that should follow.

    Why the job kills its own worker rather than the arbiter killing it: the
    worker is an ordinary child process on the host, and reaching it from a
    container means handing the arbiter the systemd user manager — the authority
    to run anything as that user, for the sake of stopping one process the job
    already supervises. The job's own `terminate`-then-`kill` is the same cgroup
    teardown by a shorter route.

    What is genuinely given up: this cannot be *verified*. If the job says it let
    go and has not, nothing here can tell. That is why a timeout is a refusal —
    making the LLM wait is safe, and loading a model over several GB that may
    still be held is not.

    Liveness is the poll itself. A job holding the card is a job holding a
    connection; if it stops holding one, its process is gone and the lease with
    it. No heartbeat, no registration table, no staleness sweep.
    """

    def __init__(self, job):
        super().__init__(job)
        # Set when the arbiter wants the card back — this is what the blocking
        # GET waits on. asyncio.Event so waking every waiter costs nothing.
        self._wanted = asyncio.Event()
        # Set when the holder releases. reclaim() waits on this.
        self._let_go = asyncio.Event()
        self._last_poll = 0.0

    def grant(self) -> None:
        super().grant()
        self._wanted.clear()
        self._let_go.clear()
        self._last_poll = time.monotonic()

    def forget(self) -> None:
        super().forget()
        self._wanted.clear()
        self._let_go.set()

    async def wait_until_wanted(self, timeout: float) -> bool:
        """
        Block until the card is wanted back. False means the wait timed out.

        The timeout is not a poll interval — nothing is checked when it expires.
        It exists so the connection is renewed periodically, which is what makes
        a vanished holder detectable at all, and so a client blocked on a dead
        arbiter eventually notices.
        """
        self._last_poll = time.monotonic()
        try:
            await asyncio.wait_for(self._wanted.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._last_poll = time.monotonic()

    def seconds_since_poll(self) -> float:
        return time.monotonic() - self._last_poll

    async def reclaim(self, timeout: float) -> bool:
        if not self._held:
            return True
        logger.info(
            f"Asking {self.job.name} to give up the GPU "
            f"(it stops its own worker — this cannot be enforced from here)"
        )
        self._wanted.set()
        try:
            await asyncio.wait_for(self._let_go.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"{self.job.name} did not let go within {timeout:.0f}s. Refusing "
                f"rather than loading a model over memory it may still hold."
            )
            return False
        logger.info(f"{self.job.name} let go of the GPU")
        self._held = False
        return True


def build(job, docker_client):
    if job.kind == "cooperative":
        return CooperativeReclaimer(job)
    if job.kind == "container":
        return ContainerReclaimer(job, docker_client)
    return NoneReclaimer(job)
