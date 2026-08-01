"""
Being preempted, and coming back from it.

The supervisor exists for one reason: a CUDA context is freed when its process
exits, so preemption means killing the training process — and if that were PID 1
the container would die, which Docker will not restart. Measured on this box:
`restart: always` and `restart: on-failure` both leave RestartCount=0 after an
API-initiated kill. A preempted run would simply have been over.

So the loop below is the whole feature: kill the child, release, ask again, spawn
a new child that resumes from the last checkpoint. These tests pin its three
outcomes apart, because conflating any two of them is how a run either stops
silently or retries a genuine failure forever.

    python -m pytest tests/test_supervisor.py -q
"""

import os
import signal
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supervisor as sup


class FakeChild:
    """A worker whose exit code a test dictates."""

    def __init__(self, exit_code, alive_until_killed=False):
        self.exit_code = exit_code
        self.killed = False
        self._alive = alive_until_killed

    def poll(self):
        return None if self._alive else self.exit_code

    def kill(self):
        self.killed = True
        self._alive = False
        self.exit_code = sup.KILLED_BY_US

    def wait(self):
        return self.exit_code


class FakeClient:
    def __init__(self, wanted=()):
        self.events = []
        self._wanted = list(wanted)

    def acquire(self):
        self.events.append("acquire")
        return True, ""

    def release(self):
        self.events.append("release")

    def wait_until_wanted(self):
        return self._wanted.pop(0) if self._wanted else False

    def close(self):
        pass


def _sup(client, children):
    s = object.__new__(sup.Supervisor)
    s.client = client
    s.child = None
    s.stopping = False
    s._queue = list(children)
    return s


def _with_children(s, monkeypatch):
    """Make run_worker hand back the queued children instead of spawning."""
    def fake_popen(cmd, cwd=None):
        return s._queue.pop(0)

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sup.time, "sleep", lambda _: None)


def test_a_finished_run_exits_zero_and_stops(monkeypatch):
    """
    Terminal. The container sits at Exited(0) with its logs — asking again after
    a completed run is the specific bug the old 300s idle backoff produced.
    """
    client = FakeClient()
    s = _sup(client, [FakeChild(0)])
    _with_children(s, monkeypatch)

    assert s.run() == 0
    assert client.events == ["acquire", "release", "release"]


def test_a_preemption_releases_and_asks_again(monkeypatch):
    """
    The headline loop. Being killed is not a failure and not the end of the run:
    the worker resumes from its last complete checkpoint on the next attempt.
    """
    client = FakeClient(wanted=[True])
    preempted = FakeChild(0, alive_until_killed=True)
    s = _sup(client, [preempted, FakeChild(0)])
    _with_children(s, monkeypatch)

    assert s.run() == 0
    assert preempted.killed is True
    # Two full cycles: killed, released, asked again, then finished.
    assert client.events == ["acquire", "release", "acquire", "release", "release"]


def test_the_card_is_released_only_after_the_worker_has_exited(monkeypatch):
    """
    The one ordering that cannot be got wrong. Releasing while the worker is
    alive tells the arbiter memory is free that is not, and the next tenant
    loads a model on top of a live CUDA context.
    """
    order = []

    class Watching(FakeChild):
        def wait(self):
            order.append("worker exited")
            return super().wait()

    class WatchingClient(FakeClient):
        def release(self):
            order.append("release")

    client = WatchingClient(wanted=[True])
    s = _sup(client, [Watching(0, alive_until_killed=True), Watching(0)])
    _with_children(s, monkeypatch)
    s.run()

    assert order[:2] == ["worker exited", "release"]


def test_a_genuine_failure_is_not_retried(monkeypatch):
    """
    A malformed dataset would otherwise loop forever, and each attempt would take
    the card from something that had real work to do. Only *our own* SIGKILL
    means "try again".
    """
    client = FakeClient()
    s = _sup(client, [FakeChild(1)])
    _with_children(s, monkeypatch)

    assert s.run() == 1
    assert client.events == ["acquire", "release", "release"]


def test_our_kill_is_told_apart_from_the_worker_dying_of_a_signal(monkeypatch):
    """
    -9 is us. Any other signal death came from outside — an OOM killer, say —
    and is a failure worth surfacing rather than retrying.
    """
    client = FakeClient()
    s = _sup(client, [FakeChild(-signal.SIGSEGV)])
    _with_children(s, monkeypatch)

    assert s.run() != 0


def test_shutdown_kills_the_worker(monkeypatch):
    """`docker stop` on the container must not leave a GPU process behind."""
    client = FakeClient()
    child = FakeChild(0, alive_until_killed=True)
    s = _sup(client, [child])
    s.child = child

    s.request_shutdown(signal.SIGTERM)

    assert s.stopping is True
    assert child.killed is True


def test_a_refusal_is_waited_out_rather_than_ending_the_run(monkeypatch):
    """
    Something outranks this most of the day, and that is the design working.
    Exiting on a refusal would make "the LLM is busy" indistinguishable from
    "the dataset is malformed" at the container level.
    """
    replies = [(False, "llm holds the GPU"), (False, "video-processing holds the GPU"), (True, "")]
    slept = []

    class Refusing(FakeClient):
        def acquire(self):
            return replies.pop(0)

    s = _sup(Refusing(), [])
    monkeypatch.setattr(sup.time, "sleep", lambda x: slept.append(x))

    assert s.wait_for_gpu() is True
    assert len(slept) == 2
    assert replies == []


def test_waiting_for_the_gpu_gives_up_on_shutdown(monkeypatch):
    """Otherwise `docker stop` hangs until the daemon's own timeout kills it."""
    s = _sup(FakeClient(), [])

    class Refusing(FakeClient):
        def acquire(self):
            s.stopping = True
            return False, "llm holds the GPU"

    s.client = Refusing()
    monkeypatch.setattr(sup.time, "sleep", lambda _: None)

    assert s.wait_for_gpu() is False
