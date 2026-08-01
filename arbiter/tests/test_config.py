"""
Job policy parsing.

This file is the only place priority and memory requirements are expressed, so a
mistake in it silently mis-schedules the GPU rather than failing loudly. These
tests pin the failure modes that would be quiet otherwise.
"""

import pytest

from config import JobConfig, load_jobs


def _write(tmp_path, text):
    path = tmp_path / "jobs.yaml"
    path.write_text(text)
    return str(path)


def test_jobs_are_sorted_highest_priority_first(tmp_path):
    path = _write(tmp_path, """
jobs:
  - name: trainer
    priority: -10
  - name: video
    priority: 0
  - name: indexer
    priority: 5
""")
    assert [j.name for j in load_jobs(path)] == ["indexer", "video", "trainer"]


def test_equal_priority_is_ordered_deterministically(tmp_path):
    """Two jobs at the same rank must not swap places between restarts."""
    path = _write(tmp_path, """
jobs:
  - name: zulu
    priority: 0
  - name: alpha
    priority: 0
""")
    assert [j.name for j in load_jobs(path)] == ["alpha", "zulu"]


def test_container_defaults_to_the_job_name(tmp_path):
    path = _write(tmp_path, "jobs:\n  - name: lora-trainer\n")
    assert load_jobs(path)[0].container == "lora-trainer"


def test_a_missing_file_schedules_nothing_rather_than_crashing(tmp_path):
    """
    The arbiter's other job is clearing the GPU for the LLM, and it does that
    correctly with no jobs configured. Refusing to start would take the LLM
    stack down over a background-work config file.
    """
    assert load_jobs(str(tmp_path / "absent.yaml")) == []


def test_duplicate_names_are_rejected(tmp_path):
    """
    Two entries with one name means one of them silently never runs — priority is
    keyed on the name, so this cannot be resolved at scheduling time.
    """
    path = _write(tmp_path, """
jobs:
  - name: trainer
    priority: 0
  - name: trainer
    priority: -10
""")
    with pytest.raises(ValueError, match="duplicate"):
        load_jobs(path)


def test_an_unknown_kind_is_rejected(tmp_path):
    path = _write(tmp_path, "jobs:\n  - name: trainer\n    kind: systemd\n")
    with pytest.raises(ValueError, match="kind must be"):
        load_jobs(path)


def test_stop_timeout_zero_is_preserved(tmp_path):
    """
    0 means SIGKILL outright, which is the trainer's whole preemption story. A
    falsy-default bug here would silently give it a 10s grace and put that on
    every LLM request's latency.
    """
    path = _write(tmp_path, "jobs:\n  - name: trainer\n    stop_timeout: 0\n")
    assert load_jobs(path)[0].stop_timeout == 0


def test_kind_defaults_to_the_one_that_tears_nothing_down(tmp_path):
    """
    An entry that forgot its kind gets the harmless reclaim, not `docker stop` on
    a container name guessed from the job name.
    """
    path = _write(tmp_path, "jobs:\n  - name: something\n")
    assert load_jobs(path)[0].kind == "none"


def test_an_unknown_kind_is_refused(tmp_path):
    """
    Loudly, at load. A kind nothing implements would otherwise fall through to a
    reclaimer that always succeeds and never frees anything.
    """
    path = _write(tmp_path, "jobs:\n  - name: x\n    kind: magic\n")
    with pytest.raises(ValueError, match="kind must be one of"):
        load_jobs(path)


def test_required_mb_defaults_to_no_headroom_check(tmp_path):
    """
    A job that has not stated a requirement is never refused over free VRAM. A
    guessed default would refuse a job that would have fitted, and the caller
    that states nothing is the interactive one.
    """
    path = _write(tmp_path, "jobs:\n  - name: llm\n")
    assert load_jobs(path)[0].required_mb == 0


def test_the_deployed_config_parses():
    """The real jobs.yaml, so a typo in it fails here rather than at 3am."""
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs = load_jobs(os.path.join(here, "jobs.yaml"))

    by_name = {j.name: j for j in jobs}
    assert set(by_name) == {"llm", "video-processing", "lora-trainer"}

    # The ordering the whole design assumes, and the only place it is expressed:
    # interactive answers beat video work, which beats background training.
    assert by_name["llm"].priority > by_name["video-processing"].priority
    assert by_name["video-processing"].priority > by_name["lora-trainer"].priority

    # Cooperative, not container. Killing the container would take its
    # supervisor with it, and Docker suppresses restart policies for an
    # API-initiated kill — so a preempted run would be over rather than resumed.
    assert by_name["lora-trainer"].kind == "cooperative"

    # The name the proxy sends must exist here, or every request falls through to
    # the unknown-caller path and runs on an invented priority.
    assert by_name["llm"].kind == "none"

    # And it must never be gated on headroom: the one caller a refusal turns into
    # a user-visible outage rather than a delay.
    assert by_name["llm"].required_mb == 0
