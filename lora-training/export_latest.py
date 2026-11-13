"""
Put each persona's newest finished run live on /mimic.

    python export_latest.py [--personas "mimic_user3"] [--run /home/peacow/lora-outputs/<dir>] [--force] [--dry-run]
    python export_latest.py --personas mimic_user3 --candidate [--run <dir>]

For every persona in the roster (shared/persona_roster.py), finds the newest complete
`final` adapter among lora-outputs/<persona> and lora-outputs/<persona>-*,
newest by when its COMPLETE marker was written, and converts it to
/srv/models/lora/<persona>-lora.gguf — its entry in [mimic-multi]'s lora list,
which is what serves /mimic.

A <persona>-lora.json beside each GGUF records which run it came from, so a
persona already serving its newest run is skipped (--force re-exports). --run
picks a specific run instead, which is also how to roll back. It also records
the build arguments of the dataset the run trained on (persona_roster's
trained_format), which the bot serves the adapter by.

--candidate exports one persona's run to candidate-lora.gguf instead, served by
[mimic-candidate] and by nothing live, so it can be judged against the live
adapter (make train-sample MODELS="<persona> candidate") before it replaces it.

Nothing here trains or touches the GPU: train_queue.sh deliberately leaves the
new adapters unexported, so a run goes live only after someone has looked at it.
"""

import argparse
import configparser
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checkpoints
import persona_roster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("export-latest")

OUTPUTS_DIR = os.environ.get("LORA_OUTPUTS_DIR", "/home/peacow/lora-outputs")
LORA_DIR = os.environ.get("LORA_GGUF_DIR", "/srv/models/lora")


def candidate_runs(outputs_dir: str, persona: str) -> list[str]:
    """Run dirs for this persona: exactly `<persona>` or `<persona>-<anything>`."""
    runs = []
    for path in glob.glob(os.path.join(outputs_dir, f"{persona}*")):
        base = os.path.basename(path)
        if os.path.isdir(path) and (base == persona or base.startswith(f"{persona}-")):
            runs.append(path)
    return runs


def finished_at(run: str) -> float | None:
    """When the run's final adapter was marked complete, or None if it never was."""
    final = os.path.join(run, "final")
    if not checkpoints.is_complete(final):
        return None
    return os.path.getmtime(os.path.join(final, checkpoints.COMPLETE_MARKER))


def newest_finished(outputs_dir: str, persona: str) -> str | None:
    done = [(t, run) for run in candidate_runs(outputs_dir, persona) if (t := finished_at(run)) is not None]
    return max(done)[1] if done else None


def last_eval_loss(run: str) -> float | None:
    """Final eval loss, from the newest checkpoint's trainer_state (final/ has none)."""
    best = None
    for _step, path in checkpoints.all_checkpoints(run):
        state = os.path.join(path, "trainer_state.json")
        if os.path.isfile(state):
            best = state  # all_checkpoints is ordered by step
    if not best:
        return None
    with open(best) as fh:
        evals = [h["eval_loss"] for h in json.load(fh).get("log_history", []) if "eval_loss" in h]
    return evals[-1] if evals else None


def manifest_path(persona: str) -> str:
    return os.path.join(LORA_DIR, f"{persona}-lora.json")


def exported_from(persona: str) -> str | None:
    try:
        with open(manifest_path(persona)) as fh:
            return json.load(fh).get("source")
    except (OSError, ValueError):
        return None


CANDIDATE = "candidate"


def multi_adapters(models_ini: str) -> list[str] | None:
    """[mimic-multi]'s lora paths in models.ini, in adapter-id order; None if it cannot be read."""
    if not os.path.isfile(models_ini):
        return None  # not mounted; nothing to check against
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.read(models_ini)
    if not ini.has_section(persona_roster.MULTI_MODEL):
        return []
    return [p.strip() for p in ini.get(persona_roster.MULTI_MODEL, "lora", fallback="").split(",") if p.strip()]


def preset_problem(models_ini: str, persona: str) -> str | None:
    """Why /mimic would not serve this persona's GGUF, or None if [mimic-multi] lists it."""
    paths = multi_adapters(models_ini)
    if paths is None:
        return None
    if persona not in {persona_roster.adapter_persona(p) for p in paths}:
        return f"[{persona_roster.MULTI_MODEL}] in models.ini does not list {persona}{persona_roster.ADAPTER_SUFFIX}"
    return None


def missing_adapters(models_ini: str) -> list[str]:
    """
    [mimic-multi] adapter files that do not exist. llama-server waits forever
    on a missing adapter, and they all load together — so one missing file
    hangs /mimic for every persona, not just its own.
    """
    return [p for p in multi_adapters(models_ini) or [] if not os.path.isfile(os.path.join(LORA_DIR, os.path.basename(p)))]


def run_dataset_args(run: str) -> dict | None:
    """The build arguments train_worker recorded for `run`; None for a run trained before it did."""
    try:
        with open(os.path.join(run, persona_roster.RUN_DATASET_STATS)) as fh:
            return json.load(fh).get("args") or None
    except (OSError, ValueError):
        return None


def write_manifest(name: str, run: str, loss: float | None, **extra) -> None:
    manifest = {
        "source": run,
        "eval_loss": loss,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    # Absent for an older run: readers then assume the legacy format it was trained in.
    dataset = run_dataset_args(run)
    if dataset is not None:
        manifest[persona_roster.TRAINED_FORMAT_KEY] = dataset
    with open(manifest_path(name), "w") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export each persona's newest finished adapter")
    ap.add_argument("--personas", default="", help="Space-separated personas; default every one in the roster")
    ap.add_argument("--run", default="", help="Export this run dir instead of the newest (one persona only)")
    ap.add_argument("--force", action="store_true", help="Re-export even if already serving that run")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be exported and stop")
    ap.add_argument(
        "--candidate", action="store_true",
        help="Export one persona's run as candidate-lora.gguf ([mimic-candidate]) instead of putting it live",
    )
    ap.add_argument("--roster", default=persona_roster.ROSTER_PATH)
    ap.add_argument("--models-ini", default="/app/models.ini")
    args = ap.parse_args()

    try:
        roster = persona_roster.load_roster(args.roster)
        names = persona_roster.select(roster, args.personas)
    except persona_roster.RosterError as e:
        logger.error(str(e))
        return 1
    if (args.run or args.candidate) and len(names) != 1:
        logger.error("--run and --candidate need exactly one persona (PERSONA=…)")
        return 1

    rows, failed = [], []
    for name in names:
        run = os.path.abspath(args.run) if args.run else newest_finished(OUTPUTS_DIR, name)
        if run is None:
            rows.append((name, "-", None, "no finished run"))
            continue
        if finished_at(run) is None:
            rows.append((name, os.path.basename(run), None, "FAILED: no complete final/"))
            failed.append(name)
            continue
        loss = last_eval_loss(run)
        target = CANDIDATE if args.candidate else name
        if exported_from(target) == run and not args.force:
            rows.append((name, os.path.basename(run), loss, "already live"))
            continue
        if args.dry_run:
            rows.append((name, os.path.basename(run), loss, f"would export{' as candidate' if args.candidate else ''}"))
            continue

        # Imported here so --dry-run works without the conversion stack.
        from export_adapter import export

        if export(os.path.join(run, "final"), target) != 0:
            rows.append((name, os.path.basename(run), loss, "FAILED: conversion"))
            failed.append(name)
            continue
        if args.candidate:
            write_manifest(CANDIDATE, run, loss, persona=name)
            rows.append((name, os.path.basename(run), loss, "exported as candidate"))
        else:
            write_manifest(name, run, loss)
            rows.append((name, os.path.basename(run), loss, "exported"))

    print()
    print(f"{'persona':<14} {'run':<26} {'eval_loss':>9}  result")
    for name, run, loss, status in rows:
        loss_s = f"{loss:.3f}" if loss is not None else "-"
        print(f"{name:<14} {run:<26} {loss_s:>9}  {status}")
        if not args.candidate and status != "no finished run" and not status.startswith("FAILED"):
            if problem := preset_problem(args.models_ini, name):
                print(f"{'':<14} WARNING: {problem}; /mimic will not use this adapter")
    if missing := missing_adapters(args.models_ini):
        print(f"\nWARNING: [{persona_roster.MULTI_MODEL}] lists adapters that do not exist: {', '.join(missing)}.")
        print("llama-server waits forever on a missing adapter, so /mimic hangs for EVERY persona until")
        print("these are exported or removed from the list.")

    if any(status.startswith("exported") for *_, status in rows):
        print("\nA model that is loaded right now keeps its old adapters until it is next")
        print("loaded: at most 10 idle minutes, or immediately if another model is used.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
