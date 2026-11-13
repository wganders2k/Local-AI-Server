"""
Rebuild every persona's training set (or just some) from the archive.

    python build_all_datasets.py [--personas "mimic_user3 mimic_user5"] [--exclude-channels …] [--exclude-dates …] [-- <build_dataset args>]

Runs build_dataset.py once per persona in the roster (shared/persona_roster.py), with the
user ids and display name recorded there, so a rebuild never depends on anyone
remembering who mimic_user6 is. Arguments after `--` are passed to every build
unchanged, e.g. `-- --context-n 24`.

CPU only, no GPU lease.
"""

import argparse
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persona_roster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build-all-datasets")

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description="Rebuild persona training sets from the archive")
    ap.add_argument("--personas", default="", help="Space-separated personas; default every one in the roster")
    ap.add_argument("--roster", default=persona_roster.ROSTER_PATH)
    ap.add_argument("--exclude-channels", default=os.environ.get("EXCLUDED_CHANNELS", ""))
    ap.add_argument("--exclude-dates", default="")
    ap.add_argument("--out-root", default="/archive/training", help="Each persona is written to <out-root>/<persona>")
    args = ap.parse_args(argv)

    try:
        roster = persona_roster.load_roster(args.roster)
        names = persona_roster.select(roster, args.personas)
    except persona_roster.RosterError as e:
        logger.error(str(e))
        return 1

    results = []
    for name in names:
        entry = roster[name]
        out = os.path.join(args.out_root, name)
        cmd = [
            sys.executable, os.path.join(HERE, "build_dataset.py"),
            "--user-id", entry["user_ids"],
            "--persona", name,
            "--display-name", entry["display_name"],
            "--exclude-channels", args.exclude_channels,
            "--exclude-dates", args.exclude_dates,
            "--out", out,
            *passthrough,
        ]
        logger.info(f"=== {name} ===")
        code = subprocess.run(cmd, cwd=HERE).returncode
        counts = {}
        if code == 0:
            for split in ("train", "eval"):
                with open(os.path.join(out, f"{split}.jsonl"), encoding="utf-8") as fh:
                    counts[split] = sum(1 for _ in fh)
        results.append((name, code, counts))

    print()
    print(f"{'persona':<14} {'train':>7} {'eval':>6}  result")
    for name, code, counts in results:
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"{name:<14} {counts.get('train', '-'):>7} {counts.get('eval', '-'):>6}  {status}")
    failed = [n for n, code, _ in results if code != 0]
    if failed:
        logger.error(f"{len(failed)} build(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
