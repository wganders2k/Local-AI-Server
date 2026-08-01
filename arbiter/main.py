"""
GPU arbiter — the only component that decides who holds the card.

Three endpoints, which every job speaks, plus one that only a job the arbiter
cannot tear down has any use for.

    POST /gpu/acquire         {"name": ...} — take the card, preempting what you
                              outrank; refuses rather than risk an OOM
    POST /gpu/release         {"name": ...} — done with it
    GET  /gpu/status          who holds it and why
    GET  /gpu/reclaim-notice  blocks until the card is wanted back. Only a
                              `cooperative` job calls this: it is how a job that
                              stops its own worker finds out it should.

The caller sends its name and nothing else. It does not say how important it is,
how much VRAM it needs, or what should be stopped for it — all of that is
jobs.yaml, and a caller that could state its own priority would be assessing
itself. The LLM is one such caller and gets no special treatment in this file.

Nothing here starts a job. Admission is the whole service: work is submitted by
whoever owns it, and the arbiter says who may hold the card while it runs.

Reaches Docker through docker-socket-proxy rather than /var/run/docker.sock, so
it cannot build images, read secrets, or touch the host filesystem even if
something got in through these endpoints. It must still stay on the internal
compose network and must never be published beyond the LAN.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("arbiter")

import config
import gpu
from docker_client import DockerClient
import reclaimers
from scheduler import Scheduler

docker = DockerClient(config.DOCKER_HOST)
scheduler: Scheduler | None = None

ARBITER_ACQUIRE = Counter("arbiter_acquire_total", "GPU acquisitions", ["job", "result"])
ARBITER_ACQUIRE_SECONDS = Gauge(
    "arbiter_acquire_seconds", "How long the last acquire took — the caller waits on this", ["job"]
)
ARBITER_HOLDER = Gauge(
    "arbiter_gpu_holder", "1 for whoever holds the GPU (a job name, or none)", ["holder"]
)
ARBITER_FREE_MB = Gauge("arbiter_gpu_free_mb", "Free VRAM in MiB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    jobs = config.load_jobs()
    scheduler = Scheduler(jobs, docker)

    if not docker.ping():
        logger.error(
            f"Docker at {config.DOCKER_HOST} is not answering — container jobs cannot be "
            f"started or stopped. Check that docker-socket-proxy is up and allows CONTAINERS+POST."
        )
    try:
        logger.info(f"GPU: {gpu.describe()}")
    except Exception as exc:
        logger.error(
            f"Cannot read the GPU ({exc}) — headroom checks will be skipped. Is this "
            f"container running with runtime: nvidia and NVIDIA_DRIVER_CAPABILITIES=utility?"
        )

    task = asyncio.create_task(scheduler.run_forever())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="GPU Arbiter", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# -- what a job that runs itself calls --

@app.post("/gpu/acquire")
async def gpu_acquire(body: dict) -> Response:
    """
    Take the GPU for ``body["name"]``.

    Blocks until everything that job outranks is stopped and the driver has given
    the memory back. On failure this answers 503 and the caller refuses its own
    request — the safe outcome, because loading a model over a job that may still
    hold several GB OOMs both workloads.
    """
    name = body.get("name", "")
    if not name:
        # Without a name there is no priority, and without a priority there is no
        # basis for stopping anything. Guessing would mean inventing policy.
        return JSONResponse({"granted": False, "reason": "name required"}, status_code=400)

    started = asyncio.get_running_loop().time()
    ok, detail = await scheduler.acquire(name)
    elapsed = asyncio.get_running_loop().time() - started

    ARBITER_ACQUIRE.labels(job=name, result="ok" if ok else "refused").inc()
    ARBITER_ACQUIRE_SECONDS.labels(job=name).set(elapsed)

    if not ok:
        logger.error(f"Refusing the GPU to {name} after {elapsed:.1f}s: {detail}")
        return JSONResponse(
            {"granted": False, "reason": detail, "elapsed_seconds": round(elapsed, 2)},
            status_code=503,
            headers={"Retry-After": "30"},
        )
    logger.info(f"{name} acquired the GPU in {elapsed:.1f}s ({detail})")
    return JSONResponse({"granted": True, "detail": detail, "elapsed_seconds": round(elapsed, 2)})


@app.post("/gpu/release")
async def gpu_release(body: dict) -> Response:
    """
    ``body["name"]`` is done; jobs may run once there is room.

    Does not mean there *is* room — a model stays resident until the proxy's idle
    evictor unloads it, and the scheduler gates every start on a real NVML
    reading for exactly that reason.
    """
    name = body.get("name", "")
    if not name:
        return JSONResponse({"released": False, "reason": "name required"}, status_code=400)
    await scheduler.release(name)
    return JSONResponse({"released": True})


@app.get("/gpu/status")
async def gpu_status() -> dict:
    return scheduler.snapshot()


@app.get("/metrics")
async def metrics() -> Response:
    snap = scheduler.snapshot()

    ARBITER_HOLDER.clear()
    ARBITER_HOLDER.labels(holder=snap["holder"]).set(1)
    if snap["free_mb"] is not None:
        ARBITER_FREE_MB.set(snap["free_mb"])

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -- what a job the arbiter cannot tear down calls --

@app.get("/gpu/reclaim-notice")
async def gpu_reclaim_notice(name: str = "") -> Response:
    """
    Block until the caller should give the card up.

    Only useful to a `cooperative` job — one the arbiter cannot stop itself, so
    it has to be told. Everything else either releases between requests or is
    torn down without being consulted.

    A long block rather than a poll, and the connection is the liveness signal:
    a job holding the card is a job holding one of these. If it stops, its
    process is gone and the reaper takes the lease back. That is the whole of the
    registration, heartbeat and staleness machinery this replaced.

    ``{"reclaim": false}`` means the wait expired with nothing wanted — call
    again. It is not a poll interval; nothing is checked when it fires. It exists
    so a client blocked on an arbiter that has died eventually finds out.
    """
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)

    reclaimer = scheduler.reclaimers.get(name)
    if not isinstance(reclaimer, reclaimers.CooperativeReclaimer):
        # A job that is torn down by other means has no business waiting here,
        # and a name that is not in jobs.yaml would wait forever for a signal
        # nothing sends. Both are configuration errors worth surfacing.
        return JSONResponse(
            {"error": f"{name!r} is not a cooperative job; nothing will ever notify it"},
            status_code=400,
        )

    if not reclaimer.leased:
        # Asking about a lease it does not hold. Either it never acquired, or the
        # arbiter restarted and every lease went with the process — which is
        # exactly what happened on deploy day: a redeploy to pick up jobs.yaml
        # dropped the video job's lease, the watcher never found out, and the LLM
        # went on loading a model on top of a live worker for ten minutes.
        #
        # Answering "yes, give it up" is the recovery: the caller stops its GPU
        # process and asks again, which is the one sequence that gets an
        # un-leased holder off the card. Blocking here instead would leave it
        # using memory nobody has accounted for.
        logger.warning(
            f"{name} asked for a reclaim notice without holding the lease — telling it to "
            f"give the card up and re-acquire. Did the arbiter restart?"
        )
        return JSONResponse({"reclaim": True, "reason": "you do not hold the lease"})

    wanted = await reclaimer.wait_until_wanted(config.NOTICE_TIMEOUT)
    return JSONResponse({"reclaim": wanted})
