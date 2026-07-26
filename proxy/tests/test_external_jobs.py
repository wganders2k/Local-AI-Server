"""
Tests for the external-job VRAM handover state machine.

The bug these exist to prevent: the previous implementation used a single
asyncio.Event as both "proxy wants VRAM" and "job released VRAM", so nothing
ever set it, `yield_requested` was permanently False, and every handover fell
through to a timeout that then loaded a model anyway. Each test below pins one
half of that contract.

Run:  python -m pytest proxy/tests -q
"""

import asyncio

import pytest

from state import ExternalJobState


@pytest.fixture
def jobs():
    return ExternalJobState()


# -- the flag the job actually polls --

def test_yield_requested_is_visible_to_the_job(jobs):
    """The regression that made the whole protocol a no-op."""
    jobs.register("job1")
    assert jobs.get_status("job1")["yield_requested"] is False

    jobs.request_yield_all()

    assert jobs.get_status("job1")["yield_requested"] is True
    assert jobs.get_status("job1")["may_run"] is False


def test_may_run_gates_unregistered_jobs_too(jobs):
    """A job must be able to ask 'may I start?' before it registers."""
    jobs.register("job1")
    jobs.request_yield_all()
    assert jobs.get_status("newcomer")["may_run"] is False

    jobs.release_yield()
    assert jobs.get_status("newcomer")["may_run"] is True


# -- the wait the proxy performs --

async def _waits(jobs) -> bool:
    """True if wait_for_yield() is currently blocking."""
    try:
        await asyncio.wait_for(jobs.wait_for_yield(), timeout=0.05)
        return False
    except asyncio.TimeoutError:
        return True


@pytest.mark.asyncio
async def test_wait_blocks_until_confirmation(jobs):
    jobs.register("job1")
    jobs.request_yield_all()

    assert await _waits(jobs), "proxy must block while the job still holds VRAM"

    jobs.confirm_yield("job1")

    assert not await _waits(jobs), "confirmation must unblock the proxy"


@pytest.mark.asyncio
async def test_wait_does_not_block_when_no_jobs_registered(jobs):
    jobs.request_yield_all()
    assert not await _waits(jobs)


@pytest.mark.asyncio
async def test_all_jobs_must_confirm(jobs):
    jobs.register("job1")
    jobs.register("job2")
    jobs.request_yield_all()

    jobs.confirm_yield("job1")
    assert await _waits(jobs), "one confirmation out of two must not unblock"

    jobs.confirm_yield("job2")
    assert not await _waits(jobs)


@pytest.mark.asyncio
async def test_unregister_counts_as_release(jobs):
    """A job that exits has released its VRAM by definition."""
    jobs.register("job1")
    jobs.register("job2")
    jobs.request_yield_all()
    jobs.confirm_yield("job1")

    assert await _waits(jobs)

    jobs.unregister("job2")

    assert not await _waits(jobs)


@pytest.mark.asyncio
async def test_job_registering_mid_handover_reblocks(jobs):
    """A job that starts while a yield is outstanding must not be ignored."""
    jobs.register("job1")
    jobs.request_yield_all()
    jobs.confirm_yield("job1")
    assert not await _waits(jobs)

    jobs.register("latecomer")

    assert await _waits(jobs), "a newly registered job has not confirmed yet"
    assert jobs.get_status("latecomer")["may_run"] is False


@pytest.mark.asyncio
async def test_release_then_request_again(jobs):
    """Confirmations must not carry over from a previous handover."""
    jobs.register("job1")
    jobs.request_yield_all()
    jobs.confirm_yield("job1")
    jobs.release_yield()

    assert jobs.get_status("job1")["yield_requested"] is False
    assert jobs.get_status("job1")["confirmed"] is False

    jobs.request_yield_all()
    assert await _waits(jobs), "stale confirmation must not satisfy a new request"


@pytest.mark.asyncio
async def test_confirmation_arriving_after_release_is_ignored(jobs):
    """
    Found in an end-to-end run: the job confirmed twice, and the second landed
    just after the proxy had released. Recording it left the job permanently
    pre-confirmed, so the next handover completed instantly with the job still
    holding the GPU.
    """
    jobs.register("job1")
    jobs.request_yield_all()
    jobs.confirm_yield("job1")
    jobs.release_yield()

    jobs.confirm_yield("job1")  # stale, arrives after the handover finished

    jobs.request_yield_all()
    assert await _waits(jobs), "a stale confirmation must not satisfy the next handover"


@pytest.mark.asyncio
async def test_resume_withdraws_confirmation(jobs):
    jobs.register("job1")
    jobs.request_yield_all()
    jobs.confirm_yield("job1")
    jobs.release_yield()

    jobs.resume("job1")
    assert jobs.get_status("job1")["status"] == "running"

    jobs.request_yield_all()
    assert await _waits(jobs)


def test_request_yield_all_is_idempotent(jobs):
    jobs.register("job1")
    jobs.request_yield_all()
    jobs.confirm_yield("job1")
    jobs.request_yield_all()  # a second concurrent LLM request

    assert jobs.get_status("job1")["confirmed"] is True, \
        "a repeated request must not discard an existing confirmation"


# -- in-flight refcount --

def test_llm_refcount(jobs):
    jobs.llm_request_started()
    jobs.llm_request_started()
    assert jobs.llm_inflight == 2
    jobs.llm_request_finished()
    assert jobs.llm_inflight == 1
    jobs.llm_request_finished()
    jobs.llm_request_finished()  # over-release must not go negative
    assert jobs.llm_inflight == 0


def test_unknown_job_operations_are_safe(jobs):
    assert "not found" in jobs.confirm_yield("ghost")["message"]
    assert "not found" in jobs.resume("ghost")["message"]
    assert "not found" in jobs.unregister("ghost")["message"]
