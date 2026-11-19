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


def test_preset_check_reads_the_multi_adapter_list(tmp_path):
    ini = tmp_path / "models.ini"
    ini.write_text(
        "[gemma4-12b-mimic-u]\nlora = /srv/models/lora/mimic_user3-lora.gguf,/srv/models/lora/mimic_user4-lora.gguf\n"
        "[mimic_user5]\nlora = /srv/models/lora/mimic_user5-lora.gguf\n"
    )
    assert el.preset_problem(str(ini), "mimic_user3") is None
    # A section of its own is not enough: /mimic is served from [gemma4-12b-mimic-u].
    assert "does not list mimic_user5-lora.gguf" in el.preset_problem(str(ini), "mimic_user5")
    assert el.preset_problem(str(tmp_path / "unmounted.ini"), "mimic_user5") is None


def test_a_missing_adapter_file_is_reported(tmp_path, monkeypatch):
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "mimic_user3-lora.gguf").write_text("x")
    monkeypatch.setattr(el, "LORA_DIR", str(lora))
    ini = tmp_path / "models.ini"
    ini.write_text("[gemma4-12b-mimic-u]\nlora = /srv/models/lora/mimic_user3-lora.gguf,/srv/models/lora/mimic_user4-lora.gguf\n")
    assert el.missing_adapters(str(ini)) == ["/srv/models/lora/mimic_user4-lora.gguf"]


def test_candidate_export_never_touches_the_live_adapter(tmp_path, monkeypatch):
    import sys
    import types

    lora = tmp_path / "lora"
    lora.mkdir()
    roster = tmp_path / "personas.json"
    roster.write_text(json.dumps({"mimic_user3": {"user_ids": "1", "display_name": "x"}}))
    run = _run(tmp_path / "outputs", "mimic_user3-new", finished_at=100)
    exported = []
    monkeypatch.setitem(sys.modules, "export_adapter", types.SimpleNamespace(
        export=lambda adapter, name: exported.append((adapter, name)) or 0))
    monkeypatch.setattr(el, "LORA_DIR", str(lora))
    monkeypatch.setattr(el, "OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(sys, "argv", [
        "export_latest.py", "--personas", "mimic_user3", "--candidate",
        "--roster", str(roster), "--models-ini", str(tmp_path / "none.ini"),
    ])

    assert el.main() == 0
    assert exported == [(os.path.join(run, "final"), "candidate")]
    manifest = json.loads((lora / "candidate-lora.json").read_text())
    assert (manifest["source"], manifest["persona"]) == (run, "mimic_user3")
    assert not (lora / "mimic_user3-lora.json").exists()


def test_the_manifest_records_how_the_run_was_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "LORA_DIR", str(tmp_path))
    recorded = _run(tmp_path, "mimic_user1-new", finished_at=1)
    with open(os.path.join(recorded, "dataset_stats.json"), "w") as fh:
        json.dump({"args": {"no_system": True, "context_n": 24}, "counts": {}}, fh)
    legacy = _run(tmp_path, "mimic_user2-old", finished_at=1)

    el.write_manifest("mimic_user1", recorded, 3.4)
    el.write_manifest("mimic_user2", legacy, 3.5)

    assert el.persona_roster.trained_format("mimic_user1", str(tmp_path)) == {"no_system": True, "context_n": 24}
    # A run from before the record: no key at all, so readers assume the legacy format.
    assert el.persona_roster.trained_format("mimic_user2", str(tmp_path)) == {}
    with open(el.manifest_path("mimic_user2")) as fh:
        assert "dataset" not in json.load(fh)
