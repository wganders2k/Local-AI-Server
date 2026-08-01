"""
Checkpoint discovery under an unclean stop.

The worker is SIGKILLed on every handover, so a checkpoint directory existing is
not evidence that a checkpoint finished. Resuming from a half-written adapter is
the worst available failure: it loads, it trains, and the weights are quietly
garbage. Each test below pins one half of the COMPLETE-marker contract.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import os

import pytest

import checkpoints


@pytest.fixture
def out(tmp_path):
    return str(tmp_path)


def _make(out: str, step: int, complete: bool) -> str:
    path = os.path.join(out, f"checkpoint-{step}")
    os.makedirs(path)
    # Stand-in for the adapter weights, so a pruned directory is visibly gone.
    with open(os.path.join(path, "adapter_model.safetensors"), "w") as fh:
        fh.write("weights")
    if complete:
        checkpoints.mark_complete(path)
    return path


# -- discovery --

def test_no_checkpoints_means_start_from_scratch(out):
    assert checkpoints.latest(out) is None


def test_missing_directory_is_not_an_error(tmp_path):
    assert checkpoints.latest(str(tmp_path / "never-created")) is None


def test_latest_picks_the_newest_complete_one(out):
    _make(out, 100, complete=True)
    newest = _make(out, 200, complete=True)

    assert checkpoints.latest(out) == newest


def test_a_killed_save_is_never_resumed_from(out):
    """The whole reason the marker exists."""
    good = _make(out, 100, complete=True)
    _make(out, 200, complete=False)  # SIGKILL landed mid-save

    assert checkpoints.latest(out) == good


def test_steps_sort_numerically_not_lexically(out):
    """checkpoint-90 must not beat checkpoint-100."""
    _make(out, 90, complete=True)
    newest = _make(out, 100, complete=True)

    assert checkpoints.latest(out) == newest


def test_unrelated_directories_are_ignored(out):
    os.makedirs(os.path.join(out, "final"))
    os.makedirs(os.path.join(out, "runs"))
    good = _make(out, 10, complete=True)

    assert checkpoints.latest(out) == good


# -- pruning --

def test_prune_discards_incomplete_checkpoints(out):
    partial = _make(out, 200, complete=False)
    _make(out, 100, complete=True)

    checkpoints.prune(out, keep=2)

    assert not os.path.exists(partial)


def test_prune_keeps_two_complete_checkpoints(out):
    oldest = _make(out, 100, complete=True)
    keep_a = _make(out, 200, complete=True)
    keep_b = _make(out, 300, complete=True)

    checkpoints.prune(out, keep=2)

    assert not os.path.exists(oldest)
    assert os.path.exists(keep_a)
    assert os.path.exists(keep_b)


def test_keeping_two_survives_a_kill_during_the_next_save(out):
    """
    Why keep=2 rather than 1.

    With a single retained checkpoint, pruning before the new one is marked
    leaves a window where a kill loses everything.
    """
    older = _make(out, 100, complete=True)
    newer = _make(out, 200, complete=True)
    checkpoints.prune(out, keep=2)

    killed = _make(out, 300, complete=False)  # SIGKILL landed mid-save
    checkpoints.prune(out, keep=2)

    assert not os.path.exists(killed)
    assert checkpoints.latest(out) == newer
    assert os.path.exists(older), "the fallback must still be there"


def test_prune_on_an_empty_dir_is_a_no_op(out):
    assert checkpoints.prune(out, keep=2) == []


# -- the marker itself --

def test_mark_complete_is_what_flips_it(out):
    path = _make(out, 100, complete=False)
    assert checkpoints.is_complete(path) is False

    checkpoints.mark_complete(path)

    assert checkpoints.is_complete(path) is True
    assert checkpoints.latest(out) == path
