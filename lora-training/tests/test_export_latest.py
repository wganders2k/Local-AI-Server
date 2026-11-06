"""
Which run goes live. Picking the wrong one silently serves an old or
half-trained persona, so the rules are pinned: only runs with a COMPLETE final
count, newest by when they finished, and a persona's name must match exactly
(mimic_user1 must never pick up mimic_user1_v1 or mimic_user10).

    python -m pytest tests/test_export_latest.py -q
"""

import json
import os

import export_latest as el


def _run(root, name, finished_at=None, eval_losses=()):
    run = root / name
    (run / "final").mkdir(parents=True)
    if finished_at is not None:
        marker = run / "final" / "COMPLETE"
        marker.write_text("ok")
        os.utime(marker, (finished_at, finished_at))
    if eval_losses:
        ckpt = run / "checkpoint-10"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps(
            {"log_history": [{"eval_loss": x} for x in eval_losses]}
        ))
    return str(run)


def test_newest_finished_run_wins(tmp_path):
    _run(tmp_path, "mimic_user1", finished_at=100)
    newest = _run(tmp_path, "mimic_user1-20261105-0400", finished_at=300)
    _run(tmp_path, "mimic_user1-v2", finished_at=200)
    assert el.newest_finished(str(tmp_path), "mimic_user1") == newest


def test_an_unfinished_run_is_never_picked_however_new(tmp_path):
    done = _run(tmp_path, "mimic_user4-a", finished_at=100)
    _run(tmp_path, "mimic_user4-b")  # still training, no marker
    assert el.newest_finished(str(tmp_path), "mimic_user4") == done


def test_other_personas_with_a_shared_prefix_are_ignored(tmp_path):
    _run(tmp_path, "mimic_user1_v1", finished_at=900)
    _run(tmp_path, "mimic_user10", finished_at=900)
    own = _run(tmp_path, "mimic_user1-v3", finished_at=100)
    assert el.newest_finished(str(tmp_path), "mimic_user1") == own


def test_no_finished_run_is_none(tmp_path):
    _run(tmp_path, "mimic_user2-x")
    assert el.newest_finished(str(tmp_path), "mimic_user2") is None


def test_eval_loss_is_the_last_evaluation(tmp_path):
    run = _run(tmp_path, "mimic_user3", finished_at=1, eval_losses=(3.5, 3.4, 3.35))
    assert el.last_eval_loss(run) == 3.35


def test_preset_check(tmp_path):
    ini = tmp_path / "models.ini"
    ini.write_text(
        "[mimic_user3]\nlora = /srv/models/lora/mimic_user3-lora.gguf\n"
        "[mimic_user4]\nlora = /srv/models/lora/other.gguf\n"
    )
    assert el.preset_problem(str(ini), "mimic_user3") is None
    assert "other.gguf" in el.preset_problem(str(ini), "mimic_user4")
    assert "no [mimic_user5]" in el.preset_problem(str(ini), "mimic_user5")
