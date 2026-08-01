"""
Admission, and the guarantees the design rests on.

Two properties are being pinned above all else.

**Nothing is started here.** The arbiter used to pick up the highest-priority job
that fitted and `docker start` it, which meant a training run began because free
VRAM happened to appear. `test_nothing_is_ever_started` asserts the negative
directly, and FakeDocker has no ``start`` at all — a call would be an
AttributeError rather than a passing test.

**A grant is enforceable.** Stopping a container is; asking a job was not.
acquire must not report success unless everything junior is actually gone.

The LLM appears only as an ordinary jobs.yaml entry named "llm" with a high
priority. Nothing in the arbiter knows that name, which
`test_priority_alone_decides_who_wins` protects by inverting the ranking.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import asyncio

import pytest

import reclaimers
import scheduler as scheduler_mod
from config import JobConfig
from scheduler import Scheduler


class FakeDocker:
    """
    Records calls and lets a test drive container state directly.

    Deliberately has no ``start``: the arbiter starting a container is the
    behaviour this rewrite removed, so any code path that tries becomes a hard
    failure rather than something a test has to think to assert against.

    And no ``stop``: Docker suppresses a container's restart policy when it is
    stopped through the API, so reclaiming that way leaves a preempted job down
    forever. Only ``kill`` exists here, for the same reason.
    """

    def __init__(self):
        self.containers: dict[str, dict] = {}
        self.calls: list[str] = []
        self.stop_fails: set[str] = set()

    def add(self, name, running=False):
        self.containers[name] = {"Running": running}

    def state(self, container):
        self.calls.append(f"state({container})")
        return self.containers.get(container)

    def kill(self, container, signal="SIGKILL"):
        self.calls.append(f"kill({container},{signal})")
        if container in self.stop_fails:
            raise RuntimeError("docker daemon said no")
        self.containers[container]["Running"] = False


@pytest.fixture
def vram(monkeypatch):
    """Free VRAM the arbiter will see. Set .free to steer a test."""
    class V:
        free = 24576

    v = V()
    monkeypatch.setattr(scheduler_mod.gpu, "total_mb", lambda *a, **k: 24576)
    monkeypatch.setattr(scheduler_mod.gpu, "used_mb", lambda *a, **k: 24576 - v.free)
    monkeypatch.setattr(scheduler_mod.gpu, "describe", lambda *a, **k: "fake gpu")
    return v


@pytest.fixture
def docker():
    return FakeDocker()


def _trainer(**kw):
    return JobConfig({"name": "lora-trainer", "kind": "container",
                      "container": "lora-trainer", "priority": -10,
                      "required_mb": 4096, "stop_timeout": 0} | kw)


def _video(**kw):
    return JobConfig({"name": "video-processing", "kind": "cooperative",
                      "priority": 0, "required_mb": 4096} | kw)


def _llm(**kw):
    return JobConfig({"name": "llm", "kind": "none", "priority": 100} | kw)


def _build(jobs, docker):
    """The llm entry is present unless a test supplies its own."""
    jobs = list(jobs)
    if not any(j.name == "llm" for j in jobs):
        jobs.insert(0, _llm())
    jobs.sort(key=lambda j: (-j.priority, j.name))
    return Scheduler(jobs, docker)


# -- the arbiter never starts anything --

async def test_nothing_is_ever_started(docker, vram):
    """
    The headline negative. Free card, an idle container job that fits, nothing
    holding the lease — the previous design started it here, with no human
    anywhere in the chain. FakeDocker has no ``start``, so a regression raises.
    """
    docker.add("lora-trainer", running=False)
    sched = _build([_video(), _trainer()], docker)

    for _ in range(5):
        await sched.reap()

    assert docker.containers["lora-trainer"]["Running"] is False
    assert sched.snapshot()["holder"] == "none"


async def test_an_idle_card_stays_idle(docker, vram):
    """No job has work by virtue of there being room for it."""
    sched = _build([], docker)
    await sched.reap()
    assert sched.snapshot()["holder"] == "none"


# -- acquire: the guarantee the design exists for --

async def test_acquire_reclaims_from_a_running_job(docker, vram):
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")

    ok, _ = await sched.acquire("llm")

    assert ok
    assert "kill(lora-trainer,SIGKILL)" in docker.calls
    assert docker.containers["lora-trainer"]["Running"] is False


async def test_acquire_is_cheap_when_nothing_holds_the_card(docker, vram):
    docker.add("lora-trainer", running=False)
    sched = _build([_trainer()], docker)

    ok, detail = await sched.acquire("llm")

    assert ok and "nothing was running" in detail
    assert not any(c.startswith("kill(") for c in docker.calls)


async def test_a_container_that_is_up_but_has_not_asked_is_not_a_tenant(docker, vram):
    """
    A trainer waiting for permission is a running container holding no VRAM.
    Reading "container is up" as "holds the card" would kill it on every LLM
    request and it would spend its life restarting.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)

    ok, detail = await sched.acquire("llm")

    assert ok and "nothing was running" in detail
    assert docker.containers["lora-trainer"]["Running"] is True


async def test_acquire_refuses_if_a_job_will_not_stop(docker, vram):
    """
    The safe failure. The proxy turns this into a 503; loading a model over a job
    that may still hold several GB OOMs both workloads.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")
    docker.stop_fails.add("lora-trainer")

    ok, detail = await sched.acquire("llm")

    assert ok is False
    assert "lora-trainer" in detail


async def test_acquire_is_idempotent(docker, vram):
    """An LLM request arriving while we already hold the card is the common case."""
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")

    await sched.acquire("llm")
    calls_after_first = len(docker.calls)
    ok, detail = await sched.acquire("llm")

    assert ok and detail == "already held"
    assert len(docker.calls) == calls_after_first


# -- every job asks, including the ones the arbiter can tear down --

async def test_a_container_job_asks_for_the_gpu_like_anything_else(docker, vram):
    """
    One protocol. The trainer takes the card by asking, exactly as the proxy
    does — the only thing its ``kind`` decides is that reclaim is a docker stop.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)

    ok, _ = await sched.acquire("lora-trainer")

    assert ok
    assert sched.snapshot()["holder"] == "lora-trainer"


async def test_a_junior_caller_is_refused_rather_than_queued(docker, vram):
    """
    Refusal, not a wait. The caller decides what to do about it — the trainer
    sleeps and asks again, holding no VRAM meanwhile, and nothing in the arbiter
    has to model a queue.
    """
    sched = _build([_trainer()], docker)
    docker.add("lora-trainer", running=True)
    await sched.acquire("llm")

    ok, detail = await sched.acquire("lora-trainer")

    assert not ok and "does not outrank" in detail


# -- the LLM is an ordinary entry: priority decides, the code does not --

async def test_priority_alone_decides_who_wins(docker, vram):
    """
    Rank the trainer above the LLM and the LLM loses.

    This is the test the whole generalisation exists for. Nothing may treat
    "llm" as meaning anything — if inverting two numbers in jobs.yaml does not
    invert the outcome, a special case has crept back in.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_llm(priority=-10), _trainer(priority=100)], docker)
    await sched.acquire("lora-trainer")

    ok, detail = await sched.acquire("llm")

    assert not ok and "does not outrank" in detail
    assert not any(c.startswith("kill(") for c in docker.calls)


async def test_an_equal_priority_holder_is_not_preempted(docker, vram):
    """
    Ties do not preempt. If they did, two callers at the same rank would take
    the card from each other on every request and neither would get work done.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_llm(priority=0), _trainer(priority=0)], docker)
    await sched.acquire("lora-trainer")

    ok, _ = await sched.acquire("llm")

    assert not ok
    assert docker.containers["lora-trainer"]["Running"] is True


async def test_an_unknown_caller_is_granted_but_below_the_llm(docker, vram):
    """
    Permissive, but bounded. A typo in jobs.yaml must not take a service offline
    with 503s, so the unknown name is still granted and can still take the card
    from background work.

    What it must *not* do is outrank the LLM. The fallback used to hand out
    max(priority)+1, which meant an unconfigured name — a benchmark script, a
    typo, anything — silently became the most important tenant on the box and
    could make an interactive request wait. Protecting the caller from being
    starved is not worth letting it starve everyone else.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")

    ok, _ = await sched.acquire("some-new-service")

    assert ok, "an unconfigured caller must still be granted, not refused"
    assert docker.containers["lora-trainer"]["Running"] is False
    assert sched.snapshot()["holder"] == "some-new-service"


async def test_an_unknown_caller_cannot_take_the_card_from_the_llm(docker, vram):
    """
    The point of the change. This is the case a top-rank fallback got wrong.
    """
    sched = _build([_llm(), _trainer()], docker)
    await sched.acquire("llm")

    ok, why = await sched.acquire("some-new-service")

    assert not ok
    assert "does not outrank" in why
    assert sched.snapshot()["holder"] == "llm"


async def test_the_llm_can_take_the_card_from_an_unknown_caller(docker, vram):
    """The other direction, which is what makes the LLM's rank mean anything."""
    sched = _build([_llm()], docker)
    await sched.acquire("some-new-service")

    ok, _ = await sched.acquire("llm")

    assert ok
    assert sched.snapshot()["holder"] == "llm"


async def test_the_holder_is_reported_by_name(docker, vram):
    docker.add("lora-trainer", running=False)
    sched = _build([_trainer()], docker)

    assert sched.snapshot()["holder"] == "none"
    await sched.acquire("llm")
    assert sched.snapshot()["holder"] == "llm"
    await sched.release("llm")
    assert sched.snapshot()["holder"] == "none"


# -- headroom: an empty lease table is not room --

async def test_a_job_that_does_not_fit_is_refused(docker, vram):
    """
    Nobody holds the lease and the card is still full. The proxy releases between
    requests but the model stays resident until the idle evictor unloads it — a
    grant on the lease alone is how the trainer died with a CUDA OOM.
    """
    docker.add("lora-trainer", running=True)
    vram.free = 2000
    sched = _build([_trainer(required_mb=16000)], docker)

    ok, detail = await sched.acquire("lora-trainer")

    assert not ok and "needs 16000" in detail


async def test_a_job_without_a_stated_requirement_is_never_refused_on_headroom(docker, vram):
    """
    The interactive caller states no requirement, and refusing it over a reading
    we could not act on anyway would turn a transient into an outage.
    """
    vram.free = 10
    sched = _build([], docker)

    ok, _ = await sched.acquire("llm")

    assert ok


async def test_an_unreadable_card_does_not_block_a_grant(docker, vram, monkeypatch):
    """A permanently idle GPU needs a human; a failed run recovers on its own."""
    def boom(*a, **k):
        raise scheduler_mod.gpu.NvidiaSmiUnavailable("no smi")

    monkeypatch.setattr(scheduler_mod.gpu, "used_mb", boom)
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)

    ok, _ = await sched.acquire("lora-trainer")

    assert ok


# -- the reaper --

async def test_a_lease_whose_container_died_is_cleared(docker, vram):
    """
    A job that is killed, crashes, or is removed never gets to call release. Its
    lease would otherwise outrank the next caller forever.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")

    docker.containers["lora-trainer"]["Running"] = False  # died on its own
    await sched.reap()

    assert sched.snapshot()["holder"] == "none"
    assert sched.reclaimers["lora-trainer"].leased is False


async def test_a_restarted_container_does_not_inherit_its_old_lease(docker, vram):
    """
    The sharp edge of Docker's restart policy: the container comes back on its
    own, and if the lease survived the gap it would be holding the card without
    having asked. It must ask again.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")

    docker.containers["lora-trainer"]["Running"] = False
    await sched.reap()
    docker.containers["lora-trainer"]["Running"] = True  # restart policy
    await sched.reap()

    assert sched.snapshot()["holder"] == "none"


# -- the reclaimers themselves --

def test_every_kind_builds_a_reclaimer(docker):
    """
    A new kind in KINDS with no branch in build() would silently fall through to
    the no-op reclaimer, which never fails and never frees anything.
    """
    import config as config_mod

    expected = {
        "none": reclaimers.NoneReclaimer,
        "container": reclaimers.ContainerReclaimer,
        "cooperative": reclaimers.CooperativeReclaimer,
    }
    assert set(expected) == set(config_mod.KINDS)
    for kind, cls in expected.items():
        job = JobConfig({"name": f"j-{kind}", "kind": kind})
        assert type(reclaimers.build(job, docker)) is cls


# -- the cooperative transport: a job that stops its own worker --

async def _grant(sched, name):
    ok, detail = await sched.acquire(name)
    assert ok, detail


async def test_a_cooperative_job_takes_the_card_by_asking(docker, vram):
    """No registration table, no poll. The same two calls as everything else."""
    sched = _build([_video()], docker)

    await _grant(sched, "video-processing")

    assert sched.snapshot()["holder"] == "video-processing"


async def test_reclaiming_wakes_the_notice_and_waits_for_the_release(docker, vram):
    """
    The whole handover. The arbiter cannot stop this job, so it asks and blocks;
    the job kills its own worker and releases; only then is the card granted on.
    """
    sched = _build([_video()], docker)
    await _grant(sched, "video-processing")
    video = sched.reclaimers["video-processing"]

    notice = asyncio.create_task(video.wait_until_wanted(timeout=5))
    await asyncio.sleep(0.01)
    assert not notice.done(), "nothing wanted the card yet"

    llm = asyncio.create_task(sched.acquire("llm"))
    await asyncio.sleep(0.05)

    assert await notice is True, "the job must be told the card is wanted"
    assert not llm.done(), "must not report success before the job lets go"

    await sched.release("video-processing")
    ok, _ = await llm
    assert ok


async def test_a_job_that_never_lets_go_is_a_refusal(docker, vram):
    """
    The safe failure, and the reason this transport is second-best. Nothing here
    can make the job release, so the honest answer is to make the LLM wait —
    loading a model over several GB that may still be held OOMs both workloads.
    """
    sched = _build([_video()], docker)
    await _grant(sched, "video-processing")

    ok, detail = await sched.acquire("llm", timeout=0.1)

    assert ok is False
    assert "video-processing" in detail


async def test_the_notice_returns_false_rather_than_blocking_forever(docker, vram):
    """
    Not a poll interval — nothing is checked when it expires. It exists so a
    client blocked on an arbiter that has died eventually finds out, and so the
    connection is renewed often enough to be a liveness signal.
    """
    sched = _build([_video()], docker)
    await _grant(sched, "video-processing")

    assert await sched.reclaimers["video-processing"].wait_until_wanted(0.05) is False


async def test_a_cooperative_holder_that_vanishes_loses_its_lease(docker, vram, monkeypatch):
    """
    The connection is the liveness signal. A watcher that is killed never gets to
    release, and its lease would otherwise outrank every later caller forever —
    the stale-registration sweep this replaced existed for exactly this.
    """
    monkeypatch.setattr(scheduler_mod.config, "COOPERATIVE_STALE_AFTER", 0.05)
    sched = _build([_video()], docker)
    await _grant(sched, "video-processing")

    await asyncio.sleep(0.06)  # nobody renewed the notice
    await sched.reap()

    assert sched.snapshot()["holder"] == "none"


async def test_a_live_cooperative_holder_keeps_its_lease(docker, vram, monkeypatch):
    """
    The other direction, and the more dangerous one: dropping a lease out from
    under a job that is still using the card would let a model load on top of it.
    """
    monkeypatch.setattr(scheduler_mod.config, "COOPERATIVE_STALE_AFTER", 0.5)
    sched = _build([_video()], docker)
    await _grant(sched, "video-processing")

    for _ in range(3):
        await sched.reclaimers["video-processing"].wait_until_wanted(0.05)
        await sched.reap()

    assert sched.snapshot()["holder"] == "video-processing"


async def test_a_junior_caller_cannot_take_the_card_from_it(docker, vram):
    """Priority still decides. The trainer waits for video work, not the reverse."""
    docker.add("lora-trainer", running=True)
    sched = _build([_video(), _trainer()], docker)
    await _grant(sched, "video-processing")

    ok, detail = await sched.acquire("lora-trainer")

    assert not ok and "does not outrank" in detail


async def test_a_preempted_container_is_killed_not_stopped(docker, vram):
    """
    The distinction that cost an evening. Docker deliberately suppresses a
    container's restart policy when it is stopped through the API — a manual
    stop means "and stay down" — so a preempted trainer with `restart:
    on-failure` never came back and its run was simply over. A signalled
    container dies like it would from any other cause, the policy applies, and it
    restarts and asks for the card again.
    """
    docker.add("lora-trainer", running=True)
    sched = _build([_trainer()], docker)
    await sched.acquire("lora-trainer")

    await sched.acquire("llm")

    assert "kill(lora-trainer,SIGKILL)" in docker.calls
    assert not any("stop(" in c for c in docker.calls)


async def test_a_grace_period_means_sigterm_first(docker, vram):
    """A job with a critical section gets a chance to leave it. stop_timeout=0
    skips straight to SIGKILL, which is the trainer's whole preemption story."""
    docker.add("indexer", running=True)
    indexer = JobConfig({"name": "indexer", "kind": "container",
                         "container": "indexer", "priority": 5, "stop_timeout": 30})
    sched = _build([indexer], docker)
    await sched.acquire("indexer")

    await sched.acquire("llm")

    assert docker.calls.index("kill(indexer,SIGTERM)") < len(docker.calls)
    assert docker.containers["indexer"]["Running"] is False


async def test_a_holder_that_lost_its_lease_is_told_to_give_the_card_up(docker, vram):
    """
    Found on hardware. Restarting the arbiter to pick up a jobs.yaml change drops
    every lease with the process — and a cooperative holder, which is a host
    daemon that outlives the container, never finds out. It kept a worker on the
    card while the arbiter told the LLM nothing was running.

    The endpoint answers this, but the property worth pinning here is that the
    arbiter genuinely does not know about it: an un-leased job holds nothing as
    far as admission is concerned, so the only way back is for it to re-acquire.
    """
    sched = _build([_video()], docker)
    video = sched.reclaimers["video-processing"]

    assert video.leased is False
    ok, detail = await sched.acquire("llm")
    assert ok and "nothing was running" in detail

    # And re-acquiring is what puts it back on the books.
    await sched.release("llm")
    assert (await sched.acquire("video-processing"))[0]
    assert video.leased is True
