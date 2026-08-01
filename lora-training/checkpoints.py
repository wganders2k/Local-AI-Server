"""
Checkpoint discovery that survives a SIGKILL mid-save.

The worker is killed without warning on every handover — that is the design, not
an error path (see README). So a checkpoint directory on disk is not evidence
that a checkpoint finished: it may be a save that was a hundred milliseconds
from completing when the process died.

The rule here is that a checkpoint counts only once a COMPLETE marker exists
beside it, written after the save call returns. A killed save leaves a directory
with no marker, which `latest` skips and `prune` deletes. Trainer state is
therefore never resumed from a half-written adapter, which fails in the worst
possible way: it loads, it trains, and the weights are quietly garbage.

Keeping two complete checkpoints rather than one is what makes that safe. With a
single retained checkpoint, deleting the old one before the new one is marked
leaves a window where a kill loses everything.
"""

import logging
import os
import re
import shutil
import time

logger = logging.getLogger(__name__)

COMPLETE_MARKER = "COMPLETE"

# HF Trainer's own naming, which we keep so `resume_from_checkpoint` accepts the
# path we hand it without translation.
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _step_of(name: str) -> int | None:
    m = _CHECKPOINT_RE.match(name)
    return int(m.group(1)) if m else None


def is_complete(path: str) -> bool:
    return os.path.exists(os.path.join(path, COMPLETE_MARKER))


def mark_complete(path: str) -> None:
    """
    Declare a checkpoint usable. Call only after the save has returned.

    fsync'd because the marker's whole purpose is to be trustworthy after an
    unclean stop, and a marker sitting in the page cache when the box loses
    power is exactly the lie this is meant to prevent.
    """
    marker = os.path.join(path, COMPLETE_MARKER)
    with open(marker, "w") as fh:
        fh.write(f"{time.time():.0f}\n")
        fh.flush()
        os.fsync(fh.fileno())
    dir_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    logger.info(f"Checkpoint marked complete: {path}")


def all_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    """(step, path) for every checkpoint directory, complete or not, newest last."""
    if not os.path.isdir(output_dir):
        return []
    found = []
    for name in os.listdir(output_dir):
        step = _step_of(name)
        if step is None:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            found.append((step, path))
    return sorted(found)


def latest(output_dir: str) -> str | None:
    """
    Newest checkpoint that finished writing, or None to start from scratch.

    Deliberately picks the newest *complete* one rather than the newest one:
    when a kill interrupts a save, falling back one checkpoint costs an interval
    of training and resuming from the partial one costs the whole run.
    """
    complete = [(step, path) for step, path in all_checkpoints(output_dir) if is_complete(path)]
    if not complete:
        return None
    step, path = complete[-1]
    logger.info(f"Resuming from checkpoint at step {step}: {path}")
    return path


def prune(output_dir: str, keep: int = 2) -> list[str]:
    """
    Delete incomplete checkpoints and all but the newest `keep` complete ones.

    Incomplete directories always go, regardless of `keep` — they are debris
    from a kill, they are never usable, and a 35B adapter checkpoint is large
    enough that leaving them to accumulate fills the disk.
    """
    removed = []
    complete, partial = [], []
    for step, path in all_checkpoints(output_dir):
        (complete if is_complete(path) else partial).append((step, path))

    for _, path in partial:
        logger.info(f"Discarding incomplete checkpoint: {path}")
        removed.append(path)

    for _, path in complete[:-keep] if keep > 0 else complete:
        logger.debug(f"Pruning old checkpoint: {path}")
        removed.append(path)

    for path in removed:
        shutil.rmtree(path, ignore_errors=True)
    return removed
