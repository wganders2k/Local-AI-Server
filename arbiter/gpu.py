"""
GPU memory verification.

The handover protocol rests on one claim: when a worker process exits, its VRAM
is actually returned to the driver. That claim gets *verified* here rather than
assumed, because the alternative — telling the proxy "VRAM is free" while an
ONNX Runtime CUDA/TensorRT context is still winding down — produces an OOM that
takes down both the LLM and the job.

Uses nvidia-smi rather than pynvml to avoid another dependency in the worker
venv; this is called a handful of times per handover, so the subprocess cost is
irrelevant.

Shared by every participant in the handover — both external-job clients and the
proxy — so that there is one implementation of "how much VRAM is in use" on the
box rather than one per service. The proxy runs it with the NVIDIA `utility`
driver capability only: NVML and nvidia-smi, no CUDA, so it can measure the card
without ever being able to allocate on it.

Per-process attribution (``compute_pids``, ``pid_holds_gpu``) only works for
processes in the caller's own PID namespace — the driver reports ``[N/A]`` for
anything in another container. That is why verifying a release stays with the
client that owns the process, and only the aggregate readings are useful to the
proxy.
"""

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

class NvidiaSmiUnavailable(RuntimeError):
    """nvidia-smi is missing or not answering."""


def _run_smi(query: str, extra: list[str] | None = None) -> list[str]:
    cmd = [
        "nvidia-smi",
        f"--query-{query}",
        "--format=csv,noheader,nounits",
    ] + (extra or [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError as exc:
        raise NvidiaSmiUnavailable("nvidia-smi not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise NvidiaSmiUnavailable("nvidia-smi timed out") from exc
    if out.returncode != 0:
        raise NvidiaSmiUnavailable(f"nvidia-smi failed: {out.stderr.strip()}")
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def used_mb(device: int = 0) -> int:
    """Total VRAM in use on the given device, in MiB."""
    lines = _run_smi("gpu=memory.used", ["-i", str(device)])
    if not lines:
        raise NvidiaSmiUnavailable("nvidia-smi returned no memory reading")
    return int(float(lines[0]))


def total_mb(device: int = 0) -> int:
    lines = _run_smi("gpu=memory.total", ["-i", str(device)])
    if not lines:
        raise NvidiaSmiUnavailable("nvidia-smi returned no memory reading")
    return int(float(lines[0]))


def compute_pids() -> dict[int, int]:
    """PID -> MiB for every process currently holding GPU memory."""
    result: dict[int, int] = {}
    for line in _run_smi("compute-apps=pid,used_memory"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            result[int(parts[0])] = int(float(parts[1]))
        except ValueError:
            # "[N/A]" shows up for processes in other containers/namespaces
            continue
    return result


def pid_holds_gpu(pid: int) -> bool:
    try:
        return pid in compute_pids()
    except NvidiaSmiUnavailable:
        return False


def wait_for_release(
    pid: int | None = None,
    timeout: float = 60.0,
    poll: float = 0.5,
    device: int = 0,
) -> bool:
    """
    Block until *our* GPU memory is released, returning True on success.

    The whole test is ``pid`` leaving the compute list. The driver reclaims a
    process's device memory as part of tearing it down, so its absence is the
    guarantee — there is nothing further to wait for.

    Deliberately NOT also requiring total usage to fall below a threshold. The
    card is shared: an LLM holding 20 GB keeps total usage high no matter what
    we do, and demanding an empty card means waiting forever for memory that was
    never ours to free. That bug made the first handover succeed (empty card) and
    every subsequent one time out once a model was resident.

    ``pid=None`` means no worker ran, so nothing of ours is on the card and there
    is nothing to wait for. This used to fall back to the total-usage check
    above, which reintroduced that exact bug through the back door on any path
    where the PID happened to be unknown.

    Returning False means "could not confirm", and callers must treat that as
    still held.
    """
    if pid is None:
        logger.debug("No worker PID — nothing of ours is on the card")
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if not pid_holds_gpu(pid):
                others = used_mb(device)
                logger.info(
                    f"VRAM release confirmed — PID {pid} is gone "
                    f"({others} MiB still in use by other processes)"
                )
                return True
        except NvidiaSmiUnavailable as exc:
            logger.error(f"Cannot verify VRAM release: {exc}")
            return False
        time.sleep(poll)

    logger.warning(f"PID {pid} still holding GPU memory after {timeout:.0f}s")
    return False


def describe() -> str:
    """One-line GPU state, for logs."""
    try:
        procs = compute_pids()
        return (
            f"{used_mb()}/{total_mb()} MiB used, "
            f"{len(procs)} compute process(es): {sorted(procs)}"
        )
    except NvidiaSmiUnavailable as exc:
        return f"unavailable ({exc})"
