"""
Who gets the GPU.

The arbiter is an admission controller. It does not start jobs, does not know
whether a job has work, and does not decide when anything should run — a human
or a job's own supervisor decides that. It answers one question, and takes the
card back when the answer changes:

    acquire(job)  reclaim from everything the caller outranks, verify the driver
                  gave the memory back, and grant the lease
    release(job)  the caller is done; the lease is free
    reap()        housekeeping: clear a lease whose holder died without releasing

There is no privileged tenant in this file. The LLM wins because jobs.yaml gives
it the highest priority, and it is not torn down because its ``kind`` says there
is nothing to tear down — two separate facts, read through the same code path as
every other job. Nothing below knows which entry is interactive, and re-ranking
the tenants means editing one YAML file and restarting nothing else.

That matters because the obvious implementation is a boolean: "the LLM holds the
card". It is smaller, and it is wrong in a specific way — importance and
controllability are separate axes, and a boolean fuses them. A second privileged
tenant would then have had no way to exist.

Two things are worth stating because they are not obvious:

**release() does not mean there is room.** It means no request is in flight. The
model stays resident until the proxy's idle evictor unloads it, so a 16 GB job
that trusted that signal would OOM against an 18 GB model. Callers are expected
to ask, and every grant is checked against a real NVML reading, which is why the
arbiter is the only component that needs to see the card.

**Nothing here is a scheduler.** The previous design picked up the
highest-priority job that fit and started it, which meant a training run began
because free VRAM happened to appear, with no human anywhere in the chain. Work
is submitted, not scheduled. What is left is admission: who may hold the card
right now, and how it is taken back.
"""

import asyncio
import logging
import time

import config
import gpu
import reclaimers

logger = logging.getLogger(__name__)

HOLDER_NONE = "none"


class Scheduler:
    def __init__(self, jobs, docker_client):
        self._docker = docker_client
        self.jobs = list(jobs)
        self.reclaimers = {j.name: reclaimers.build(j, docker_client) for j in self.jobs}
        self.by_name = {j.name: j for j in self.jobs}

        # Serialises acquire against the reaper, so a lease cannot be granted
        # into a window another caller is in the middle of claiming.
        self._lock = asyncio.Lock()
        self._last_reason: str = "startup"
        # Cache for /gpu/status and metrics. The truth is each reclaimer's
        # holds(); this saves the status endpoint a round-trip to Docker.
        self._holder: str | None = None

    # -- admission --

    async def acquire(self, name: str, timeout: float | None = None) -> tuple[bool, str]:
        """
        Grant the GPU to ``name``, reclaiming from whatever it outranks.
        Returns (ok, detail).

        Idempotent: a request arriving while the caller already holds the lease
        is the common case and must not pay for a second teardown.
        """
        timeout = config.ACQUIRE_TIMEOUT if timeout is None else timeout
        job = self._job_for(name)

        async with self._lock:
            if await self.reclaimers[job.name].holds():
                return True, "already held"

            deadline = time.monotonic() + timeout
            stopped = []
            for other, reclaimer in self.reclaimers.items():
                if other == job.name or not await reclaimer.holds():
                    continue

                # Priority decides, not identity. An equal rank does not preempt:
                # ties would otherwise let two callers take the card from each
                # other indefinitely and neither would make progress.
                if self.by_name[other].priority >= job.priority:
                    self._last_reason = f"{other} holds the GPU and {job.name} does not outrank it"
                    return False, self._last_reason

                remaining = max(1.0, deadline - time.monotonic())
                if not await reclaimer.reclaim(remaining):
                    self._last_reason = f"could not reclaim the GPU from {other}"
                    return False, self._last_reason
                stopped.append(other)

            if stopped:
                settle = max(1.0, min(config.VRAM_SETTLE_TIMEOUT, deadline - time.monotonic()))
                if not await self._wait_for_vram_to_settle(settle):
                    self._last_reason = "VRAM did not drop after reclaiming"
                    return False, self._last_reason

            if not self._fits(job):
                return False, self._last_reason

            self.reclaimers[job.name].grant()
            self._holder = job.name
            self._last_reason = f"{job.name} holds the GPU"
            if stopped:
                logger.info(f"GPU granted to {job.name} — reclaimed from {stopped}")
            return True, f"reclaimed from {stopped}" if stopped else "nothing was running"

    async def release(self, name: str) -> None:
        """
        ``name`` is done with the card. Whoever wants it next may ask.

        Deliberately does **not** take the lock, and that is load-bearing rather
        than an optimisation. `acquire` holds the lock while it waits for a
        cooperative job to let go — and letting go *is* a call to this method.
        Taking the lock here would mean the only thing that can end the wait is
        blocked on the thing waiting, and every handover would deadlock until the
        acquire timed out.

        Safe without it because releasing is a one-way transition owned by the
        holder: it gives up a lease it already had, and nothing about it depends
        on the state acquire is reading. Racing a grant to another job is fine —
        that job set its own lease, and this clears a different one.
        """
        job = self._job_for(name)
        reclaimer = self.reclaimers[job.name]
        if not await reclaimer.holds():
            return
        reclaimer.forget()
        if self._holder == job.name:
            self._holder = None
            self._last_reason = "idle"
        logger.info(f"{job.name} released the GPU")

    def _job_for(self, name: str):
        """
        The config entry for a caller, inventing one if it is not in jobs.yaml.

        An unknown caller gets priority 0 — the middle of the range, and
        specifically *below* the LLM.

        This used to grant `max(priority) + 1`, on the reasoning that the
        callers are interactive services and a typo in jobs.yaml must not take
        one offline with 503s. That reasoning inverted the risk it was trying to
        manage. Ranking the unknown caller top does not merely protect it from
        being starved; it lets it starve everything else, the LLM included, on
        the strength of a name nobody configured. Observed: a benchmark script
        run under an unconfigured name held the card at priority 101 against the
        LLM's 100, so an interactive request would have waited on it.

        0 keeps the fallback permissive — the caller is still granted, and can
        still take the card from background work — while making the one job a
        typo must never displace unreachable. A name that genuinely needs to
        outrank the LLM has to say so in jobs.yaml, which is the whole point of
        that file.

        It warns on every call regardless: running on an invented policy is not
        a state to sit in quietly.
        """
        job = self.by_name.get(name)
        if job is not None:
            return job

        logger.warning(
            f"{name!r} asked for the GPU but is not in jobs.yaml — granting it priority "
            f"0, which is below the LLM. Add an entry to make this deliberate."
        )
        job = config.JobConfig({"name": name, "kind": "none", "priority": 0})
        self.by_name[name] = job
        self.jobs.append(job)
        self.jobs.sort(key=lambda j: (-j.priority, j.name))
        self.reclaimers[name] = reclaimers.build(job, self._docker)
        return job

    # -- housekeeping --

    async def reap(self) -> None:
        """
        Correct the state nobody asked the arbiter to correct.

        One thing goes wrong without a caller involved: a holder dies without
        releasing, and its lease outranks everyone after it forever. Not urgent —
        the interval is slow on purpose, because everything that *is* urgent
        happens inside acquire, under the same lock.

        There is no overlap to reconcile any more. Every job asks, acquire runs
        under this lock, and it reclaims from everything junior before granting —
        so two jobs cannot be on the card at once for the arbiter to notice
        afterwards. That reconciler existed for a tenant that resumed on its own
        poll loop without asking, and it went when that did.
        """
        async with self._lock:
            await self._drop_dead_leases()
            self._holder = await self._holding_job()
            self._last_reason = f"{self._holder} holds the GPU" if self._holder else "idle"

    async def run_forever(self) -> None:
        logger.info(
            f"Arbiter admitting {len(self.jobs)} job(s); reaping every "
            f"{config.REAP_INTERVAL:.0f}s. Nothing is started from here."
        )
        while True:
            try:
                await self.reap()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"Reap failed: {exc}")
            await asyncio.sleep(config.REAP_INTERVAL)

    # -- helpers --

    async def _drop_dead_leases(self) -> None:
        """
        Clear a lease whose holder is gone without having released it.

        A job that is killed, crashes, or is removed never gets to call
        /gpu/release. Its lease would otherwise outrank the next caller forever
        — and worse, a container that restarts under Docker's restart policy
        would come back appearing to hold a lease it never asked for.
        """
        for job in self.jobs:
            reclaimer = self.reclaimers[job.name]
            if reclaimer.leased and not await reclaimer.holds():
                logger.info(f"{job.name} is gone but never released — clearing its lease")
                reclaimer.forget()
                continue
            # A cooperative job proves it is alive by holding a reclaim-notice
            # connection. Nothing on that end means the process is gone, and its
            # lease would otherwise outrank every later caller indefinitely.
            if isinstance(reclaimer, reclaimers.CooperativeReclaimer) and reclaimer.leased:
                idle = reclaimer.seconds_since_poll()
                if idle > config.COOPERATIVE_STALE_AFTER:
                    logger.warning(
                        f"{job.name} holds the GPU but has not held a reclaim-notice "
                        f"connection for {idle:.0f}s — assuming it died and clearing its lease"
                    )
                    reclaimer.forget()

    def _fits(self, job) -> bool:
        """
        Is there actually room, once everything junior is gone.

        An empty lease table is not headroom. The proxy releases between requests
        but the model stays resident until the idle evictor unloads it, so a job
        that took an ungated grant would load on top of ~20 GB and OOM — which is
        how the trainer died on this hardware. This is the check that used to gate
        starts, and it belongs wherever the card is handed over.

        ``required_mb`` defaults to 0, so a job that has not stated a requirement
        is never refused on this basis. That is deliberate for the interactive
        caller: refusing it over a reading we cannot act on would turn a transient
        into an outage, and it manages its own memory anyway.
        """
        if not job.required_mb:
            return True
        free = self._free_mb()
        if free is None or free >= job.required_mb:
            return True
        self._last_reason = f"{job.name} needs {job.required_mb} MiB, {free} MiB free"
        logger.warning(self._last_reason)
        return False

    async def _holding_job(self) -> str | None:
        """
        Who holds the card, highest priority first.

        Ordered rather than arbitrary so that if two are somehow up at once — a
        container that outlived a stop, say — the answer names the one that
        matters.
        """
        for job in self.jobs:
            if await self.reclaimers[job.name].holds():
                return job.name
        return None

    def _holder_name(self) -> str:
        return self._holder or HOLDER_NONE

    def _free_mb(self) -> int | None:
        """
        Free VRAM, or None if the card could not be read.

        None means "do not gate on it". Stalling every GPU job because nvidia-smi
        hiccuped is worse than letting a job start and fail, which it recovers
        from — whereas a permanently idle GPU needs a human.
        """
        try:
            return gpu.total_mb() - gpu.used_mb()
        except gpu.NvidiaSmiUnavailable as exc:
            logger.warning(f"Cannot read VRAM ({exc}) — granting without a headroom check")
            return None

    async def _wait_for_vram_to_settle(self, timeout: float) -> bool:
        """
        Wait for the driver to finish giving the memory back.

        Container exit means the processes are gone, but the driver reclaims a
        moment afterwards. The signal is that free VRAM has *stopped climbing* —
        two consecutive readings with no further increase means reclaim is done.

        Deliberately not waiting for a threshold or an empty card. Another
        tenant's allocation is not ours to wait for, and demanding the card be
        quiet is precisely what made every handover time out once a model was
        resident. Usually costs one 0.5s sleep.
        """
        deadline = time.monotonic() + timeout
        previous = self._free_mb()
        if previous is None:
            return True  # cannot measure; do not stall on it

        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            free = self._free_mb()
            if free is None:
                return True
            if free <= previous:
                logger.info(f"VRAM settled — {free} MiB free")
                return True
            previous = free

        logger.warning(f"VRAM was still being reclaimed after {timeout:.0f}s")
        return False

    # -- introspection --

    def snapshot(self) -> dict:
        return {
            "holder": self._holder_name(),
            "reason": self._last_reason,
            "free_mb": self._free_mb(),
            "gpu": gpu.describe(),
            "jobs": [
                {
                    "name": j.name,
                    "kind": j.kind,
                    "priority": j.priority,
                    "required_mb": j.required_mb,
                }
                for j in self.jobs
            ],
        }
