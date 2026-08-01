"""
Arbiter configuration — the single place GPU policy is expressed.

Priority and memory requirements live here rather than in the jobs, because both
are operator decisions. A job declaring its own priority is self-assessment:
nothing stops the next service ranking itself top, and re-ranking two jobs would
otherwise mean editing and redeploying at least one of them. Adding a fourth GPU
workload should be one entry in this file and nothing else anywhere.

Jobs are read from a YAML file (ARBITER_JOBS_PATH). Example:

    jobs:
      - name: llm
        kind: none
        priority: 100
      - name: video-processing
        kind: cooperative
        priority: 0
        required_mb: 4096
      - name: lora-trainer
        kind: container
        container: lora-trainer
        priority: -10
        required_mb: 4096
        stop_timeout: 0        # SIGKILL; unsaved work is discarded by design

There is deliberately no enabled/disabled flag. Whether a job *should* run is an
operator decision made by submitting it — a container that does not exist holds
nothing and asks for nothing, which needs no special case here. This file answers
only "if these jobs contend, who wins".

Every job asks for the card the same way, so ``kind`` no longer says how a job is
run. It says one thing only: **how the card is taken back from it.**

    none         nothing to tear down; dropping the lease is all there is. The
                 LLM is this — the proxy owns which model is loaded in
                 llama-server, and a second service stopping that container would
                 desync them. Correct only because the job outranks its
                 neighbours, so an unenforceable reclaim is never exercised.
    container    ``docker stop``. The cgroup goes with it and the kernel returns
                 the VRAM: nothing asked, nothing trusted. What a job should be.
    cooperative  the job tears down its own worker when told. For a job that is
                 not a container, where reaching in would cost the arbiter more
                 authority over the host than stopping one process is worth.

Note that ``priority`` is what makes the LLM win, not anything in the code — the
arbiter has no idea which entry is interactive. Re-ranking the tenants is an edit
to this file.
"""

import logging
import os

import yaml

logger = logging.getLogger(__name__)

JOBS_PATH = os.environ.get("ARBITER_JOBS_PATH", "/config/jobs.yaml")

# Reached through docker-socket-proxy rather than the real socket. Worth being
# honest about what that does and does not buy: it blocks /build and the
# namespaces this service has no use for, but CONTAINERS+POST still admits
# POST /containers/create, which is enough to create a privileged container with
# the host root bind-mounted. So it narrows the surface, not the privilege — this
# service is root-equivalent on the host either way and must stay on the internal
# compose network.
DOCKER_HOST = os.environ.get("DOCKER_HOST", "tcp://docker-socket-proxy:2375")

# How long to wait for the GPU to come free before giving up and telling the
# caller no. Only ever reached on the cooperative path — a container stop is
# seconds, and a lease with nothing to tear down is instant.
ACQUIRE_TIMEOUT = float(os.environ.get("ARBITER_ACQUIRE_TIMEOUT", "180"))

# How often the arbiter clears a lease whose holder died without releasing.
# Nothing waits on this and no job's latency depends on it — everything that has
# to be prompt happens inside acquire — so it is deliberately unhurried.
REAP_INTERVAL = float(os.environ.get("ARBITER_REAP_INTERVAL", "15"))

# SIGKILL leaves the driver a moment to reclaim; this bounds how long we will
# wait for the reading to actually drop before calling the acquire failed.
VRAM_SETTLE_TIMEOUT = float(os.environ.get("ARBITER_VRAM_SETTLE_TIMEOUT", "30"))

KINDS = ("none", "container", "cooperative")

# A cooperative job proves it is alive by holding a reclaim-notice connection.
# Longer than NOTICE_TIMEOUT by enough that a healthy client always renews
# before this fires — otherwise a lease is dropped out from under a live job.
COOPERATIVE_STALE_AFTER = float(os.environ.get("ARBITER_COOPERATIVE_STALE_AFTER", "120"))

# How long a reclaim-notice call blocks before returning "not yet". Not a poll
# interval — nothing is checked when it expires. It bounds how long a client
# blocked on a dead arbiter waits before noticing.
NOTICE_TIMEOUT = float(os.environ.get("ARBITER_NOTICE_TIMEOUT", "30"))


class JobConfig:
    def __init__(self, raw: dict):
        self.name: str = raw["name"]
        self.kind: str = raw.get("kind", "none")
        if self.kind not in KINDS:
            raise ValueError(f"job {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")

        self.container: str = raw.get("container", self.name)
        self.priority: int = int(raw.get("priority", 0))
        # 0 means "do not gate me on headroom". The default rather than a number,
        # because a wrong requirement refuses a job that would have fitted, and
        # the caller that has not stated one is the interactive one.
        self.required_mb: int = int(raw.get("required_mb", 0))
        # Seconds between SIGTERM and SIGKILL on stop. 0 means SIGKILL outright,
        # which is right for a job that checkpoints and expects to be killed;
        # a job with a short critical section wants a few seconds instead.
        self.stop_timeout: int = int(raw.get("stop_timeout", 10))

    def __repr__(self) -> str:
        return (
            f"JobConfig({self.name!r}, kind={self.kind}, priority={self.priority}, "
            f"required_mb={self.required_mb}, stop_timeout={self.stop_timeout})"
        )


def load_jobs(path: str | None = None) -> list[JobConfig]:
    """
    Read the job table, highest priority first.

    A missing or empty file is not an error: an unknown caller is granted a top
    rank, so the arbiter still hands the card to whoever asks. It logs loudly
    instead, because running every job on an invented priority looks identical to
    a working arbiter right up until two of them contend.
    """
    path = path or JOBS_PATH
    if not os.path.exists(path):
        logger.warning(f"No job config at {path} — every caller will run on an invented priority")
        return []

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    jobs = [JobConfig(entry) for entry in raw.get("jobs", [])]
    names = [j.name for j in jobs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate job names in {path}: {sorted(duplicates)}")

    if not jobs:
        logger.warning(f"{path} defines no jobs — every caller will run on an invented priority")

    jobs.sort(key=lambda j: (-j.priority, j.name))
    logger.info(f"Loaded {len(jobs)} job(s): {jobs}")
    return jobs
