"""
PID 1 in the trainer container. Holds the GPU lease; never touches the GPU.

    ask for the card  ->  spawn train_worker.py  ->  told the card is wanted
          ^                                                   |
          |                                          kill the worker, release
          +---------------------------------------------------+

Two processes rather than one, and the split is not stylistic. A CUDA context is
freed when the process holding it exits, so preemption *requires* the training
process to die. If that process were PID 1, killing it would kill the container —
and Docker suppresses a container's restart policy for any API-initiated stop or
kill, so it would never come back. Measured on this box: both `restart: always`
and `restart: on-failure` leave RestartCount=0 after `docker kill`. A preemption
would have quietly ended the run.

So the container outlives every preemption and the worker does not. This is the
same shape the video-processing watcher has, for the same reason: a long-lived
process that owns the lease, and a short-lived one that owns the card.

Nothing here decides that there is training to do. A human submits a run with
`make train-submit`; this asks whether it may hold the card while doing it. The
arbiter used to make that decision from free VRAM alone, which meant a run began
because the LLM went quiet.

The container's exit code is what a human reads afterwards:

    0      the run finished
    other  it genuinely broke — the worker's own failure, passed through
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arbiter import ArbiterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("supervisor")

CONFIG = os.environ.get("TRAIN_CONFIG", "configs/smoke.yaml")
RETRY_SECONDS = float(os.environ.get("ARBITER_RETRY_SECONDS", "30"))
HERE = os.path.dirname(os.path.abspath(__file__))

# Docker reports a signal death as -N from Popen. SIGKILL is ours and means a
# preemption; anything else came from the run itself.
KILLED_BY_US = -signal.SIGKILL


class Supervisor:
    def __init__(self):
        self.client = ArbiterClient()
        self.child: subprocess.Popen | None = None
        self.stopping = False

    def request_shutdown(self, signum=None, _frame=None):
        """The container is going away. Take the worker with us, promptly."""
        logger.info(f"Shutdown requested (signal {signum})")
        self.stopping = True
        if self.child and self.child.poll() is None:
            self.child.kill()

    def wait_for_gpu(self) -> bool:
        """
        Block until the card is ours. False if we are shutting down instead.

        A refusal means something outranks us right now — which, for the
        lowest-ranked tenant on the box, is most of the day. Waiting costs
        nothing: no model is loaded and no VRAM is held while we ask.
        """
        asked = 0
        while not self.stopping:
            granted, detail = self.client.acquire()
            if granted:
                logger.info(f"GPU granted after {asked} refusal(s): {detail}")
                return True
            asked += 1
            # One line per refusal would be an entry every 30s all day. The first
            # says why we are waiting; the arbiter's own log is the record of who
            # held it and when.
            if asked == 1 or asked % 20 == 0:
                logger.info(f"Waiting for the GPU ({detail}) — asked {asked}x")
            time.sleep(RETRY_SECONDS)
        return False

    def watch_for_reclaim(self, wanted: threading.Event) -> None:
        """
        Hold a blocking call on the arbiter until it wants the card back.

        Runs on its own thread for the lifetime of one worker. It returns the
        instant something outranking us asks, so the delay an interactive request
        sees is a SIGKILL and a process exit — no poll interval in the middle.

        It doubles as the liveness signal: the arbiter takes the lease back from
        a holder that stops holding one of these, so a container that dies
        outright does not strand the GPU.
        """
        while not wanted.is_set() and not self.stopping:
            if self.client.wait_until_wanted():
                logger.info("The arbiter wants the GPU")
                wanted.set()
                return

    def run_worker(self) -> int:
        """
        Run one training attempt. Returns its exit code.

        SIGKILL rather than SIGTERM on a preemption, and no grace period: the
        worker checkpoints on a wall clock and expects to die, so a graceful path
        would save nothing that is not already on disk and would spend the
        seconds an interactive request is waiting on.
        """
        cmd = [sys.executable, os.path.join(HERE, "train_worker.py"), "--config", CONFIG]
        logger.info(f"Starting worker: {CONFIG}")
        self.child = subprocess.Popen(cmd, cwd=HERE)

        wanted = threading.Event()
        notice = threading.Thread(
            target=self.watch_for_reclaim, args=(wanted,),
            name="reclaim-notice", daemon=True,
        )
        notice.start()

        try:
            while self.child.poll() is None:
                if wanted.is_set():
                    logger.info("Preempted — killing the worker to free its CUDA context")
                    self.child.kill()
                    break
                # A local Event, so this is free. It bounds only how fast we
                # notice a worker that finished on its own; a reclaim wakes us
                # through the thread above.
                time.sleep(0.25)
            return self.child.wait()
        finally:
            self.child = None
            wanted.set()  # release the notice thread even on an unexpected exit

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)

        try:
            while not self.stopping:
                if not self.wait_for_gpu():
                    return 0
                code = self.run_worker()
                # After the worker has exited, never before: its CUDA context
                # dies with it, and releasing any earlier would tell the arbiter
                # memory is free that is not.
                self.client.release()

                if code == 0:
                    logger.info("Training finished")
                    return 0
                if code == KILLED_BY_US:
                    logger.info("Worker killed for a preemption — asking for the card again")
                    continue
                if self.stopping:
                    return 0
                logger.error(f"Worker failed with exit {code} — not retrying")
                return code if code > 0 else 1
            return 0
        finally:
            self.client.release()
            self.client.close()


if __name__ == "__main__":
    sys.exit(Supervisor().run())
