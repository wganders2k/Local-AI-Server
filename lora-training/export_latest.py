"""
Put each persona's newest finished run live on /mimic.

    python export_latest.py [--personas "mimic_user3"] [--run /home/peacow/lora-outputs/<dir>] [--force] [--dry-run]

For every persona in the roster (personas.py), finds the newest complete
`final` adapter among lora-outputs/<persona> and lora-outputs/<persona>-*,
newest by when its COMPLETE marker was written, and converts it to
/srv/models/lora/<persona>-lora.gguf, the file models.ini already points at.

A <persona>-lora.json beside each GGUF records which run it came from, so a
persona already serving its newest run is skipped (--force re-exports). --run
picks a specific run instead, which is also how to roll back.

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
import personas

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


def preset_problem(models_ini: str, persona: str) -> str | None:
    """Why /mimic would not serve this GGUF, or None if models.ini is set up for it."""
    if not os.path.isfile(models_ini):
        return None  # not mounted; nothing to check against
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.read(models_ini)
    if not ini.has_section(persona):
        return f"models.ini has no [{persona}] section"
    lora = ini.get(persona, "lora", fallback="").strip()
    if os.path.basename(lora) != f"{persona}-lora.gguf":
        return f"models.ini [{persona}] lora = {lora or '(unset)'}, not {persona}-lora.gguf"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Export each persona's newest finished adapter")
    ap.add_argument("--personas", default="", help="Space-separated personas; default every one in the roster")
    ap.add_argument("--run", default="", help="Export this run dir instead of the newest (one persona only)")
    ap.add_argument("--force", action="store_true", help="Re-export even if already serving that run")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be exported and stop")
    ap.add_argument("--roster", default=personas.ROSTER_PATH)
    ap.add_argument("--models-ini", default="/app/models.ini")
    args = ap.parse_args()

    try:
        roster = personas.load_roster(args.roster)
        names = personas.select(roster, args.personas)
    except personas.RosterError as e:
        logger.error(str(e))
        return 1
    if args.run and len(names) != 1:
        logger.error("--run needs exactly one persona (PERSONA=…)")
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
        if exported_from(name) == run and not args.force:
            rows.append((name, os.path.basename(run), loss, "already live"))
            continue
        if args.dry_run:
            rows.append((name, os.path.basename(run), loss, "would export"))
            continue

        # Imported here so --dry-run works without the conversion stack.
        from export_adapter import export

        if export(os.path.join(run, "final"), name) != 0:
            rows.append((name, os.path.basename(run), loss, "FAILED: conversion"))
            failed.append(name)
            continue
        with open(manifest_path(name), "w") as fh:
            json.dump({
                "source": run,
                "eval_loss": loss,
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, fh, indent=2)
        rows.append((name, os.path.basename(run), loss, "exported"))

    print()
    print(f"{'persona':<14} {'run':<26} {'eval_loss':>9}  result")
    for name, run, loss, status in rows:
        loss_s = f"{loss:.3f}" if loss is not None else "-"
        print(f"{name:<14} {run:<26} {loss_s:>9}  {status}")
        if status not in ("no finished run",) and not status.startswith("FAILED"):
            if problem := preset_problem(args.models_ini, name):
                print(f"{'':<14} WARNING: {problem}; /mimic will not use this adapter")

    if any(status == "exported" for *_, status in rows):
        print("\nA persona that is loaded right now keeps its old adapter until it is next")
        print("loaded: at most 10 idle minutes, or immediately if another model is used.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
