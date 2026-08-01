"""
Asking for the GPU, and — mostly — being told no.

This is the lowest-ranked tenant on the card, so refusal is its normal condition
and the interesting behaviour is all on that path: a refusal must never look like
an error and must never let training start anyway.

What the supervisor *does* with a refusal — wait and ask again rather than exit —
is pinned in test_supervisor.py, which is also where the preempt-and-resume loop
these calls exist for is tested.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import json

import httpx
import pytest

from arbiter import ArbiterClient


def _client(handler, **kw) -> ArbiterClient:
    """An ArbiterClient whose transport is a function, not a socket."""
    c = ArbiterClient(base_url="http://arbiter:11438", job_name="lora-trainer", **kw)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_the_name_is_the_only_thing_the_trainer_says_about_itself():
    """
    Not a priority, not a VRAM figure, not what to stop for it. A job that could
    assert its own importance would be grading its own homework, and being the
    lowest-ranked tenant is no more self-assessed than being the highest.
    """
    seen = []

    def handler(request):
        seen.append((str(request.url), request.read()))
        return httpx.Response(200, json={"detail": "nothing was running"})

    _client(handler).acquire()

    url, body = seen[0]
    assert url.endswith("/gpu/acquire")
    assert json.loads(body) == {"name": "lora-trainer"}


def test_a_refusal_is_reported_with_the_arbiter_s_reason():
    def handler(request):
        return httpx.Response(503, json={"reason": "llm holds the GPU"})

    granted, detail = _client(handler).acquire()

    assert granted is False
    assert detail == "llm holds the GPU"


def test_an_unreachable_arbiter_is_a_refusal():
    """
    Proceeding blind is the one option that can OOM another tenant: it would mean
    loading a model without knowing what else is on the card. This is the job
    that matters least, so it is the one that must never take that chance.
    """
    def handler(request):
        raise httpx.ConnectError("no route to host")

    granted, detail = _client(handler).acquire()

    assert granted is False
    assert "unreachable" in detail


def test_an_unexpected_status_is_a_refusal_too():
    """A 500 or a stray 404 is not a grant. Anything but 200 means we do not have it."""
    def handler(request):
        return httpx.Response(500, text="boom")

    granted, _ = _client(handler).acquire()
    assert granted is False


def test_the_reclaim_notice_reports_that_the_card_is_wanted():
    """
    The one call nothing but this job and the video watcher make. The arbiter
    cannot tear either down, so it has to tell them — and this returns the
    instant it does, rather than on a poll interval.
    """
    def handler(request):
        assert request.url.path == "/gpu/reclaim-notice"
        assert request.url.params["name"] == "lora-trainer"
        return httpx.Response(200, json={"reclaim": True})

    assert _client(handler).wait_until_wanted() is True


def test_a_notice_that_times_out_is_not_permission_to_keep_the_card():
    """
    False means the call came back with nothing wanted — the arbiter's own
    timeout, or a transport error. A dead arbiter and a quiet one look identical
    from here, so the caller must ask again rather than settle in.
    """
    def handler(request):
        raise httpx.ReadTimeout("gone")

    assert _client(handler).wait_until_wanted() is False


def test_release_never_raises():
    """
    Best-effort by design. A failure here only delays the next tenant, which the
    arbiter's reaper corrects once this process is gone — it can never cause an
    OOM, so it must not be allowed to fail a run that has already finished.
    """
    def handler(request):
        raise httpx.ConnectError("gone")

    _client(handler).release()  # no raise


def test_the_job_name_comes_from_the_environment(monkeypatch):
    """
    It has to match arbiter/jobs.yaml. A mismatch does not fail loudly — it falls
    through the arbiter's unknown-caller path, which grants the *top* priority,
    so a typo would silently make background training preempt the LLM.
    """
    monkeypatch.setenv("ARBITER_JOB_NAME", "something-else")
    monkeypatch.setenv("ARBITER_URL", "http://elsewhere:1234/")

    c = ArbiterClient()

    assert c.job_name == "something-else"
    assert c.base_url == "http://elsewhere:1234"
