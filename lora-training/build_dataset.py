"""
Tier 3: turn the per-user archive into one member's mimic training set.

The archive (history-service, Tier 2) is one JSONL per *author*, unfiltered.
A mimic adapter needs something else: the member's replies, each paired with
the conversation they were replying to. That conversation was written by other
people and lives in other people's files, so the first thing this does is put
every file back together into per-channel timelines.

    archive/<user>.jsonl x N  ->  channel timelines  ->  turns  ->  examples

Each example is TRL's conversational prompt/completion shape, so loss falls on
the reply alone and never on the context the model is only meant to read:

    {"prompt":     [{"role": "user", "content": "[a]: ...\\n[b]: ..."}],
     "completion": [{"role": "assistant", "content": "<the member's reply>"}],
     "context_turns": ["[a]: ...", "[b]: ..."]}

context_turns is the same context one rendered turn per entry, oldest first;
the trainer ignores it. A turn can span lines, and a line can start with
"[name]: " inside a message, so the rendered string cannot be split back
reliably — anything that cuts context after the build (cut_context.py) uses
this.

No system prompt by default (since 2026-11-11): an experiment on mimic_user2
trained with and without it to the same eval loss (3.403 / 3.403), so it was
~110 tokens per example and per request for nothing. --with-system restores it
for comparison.

The user turn labels each line `[name]: text`. The rules that get a message
there — normalising, merging turns, windowing context — are in
shared/mimic_format.py, which the bot renders live channel history with too
(discord-bot/mimic_context.py). The system prompt is rendered by the same
function the bot serves it with (mimic_format.render_persona_prompt), when
there is one.

Captions never enter. history-service marks every one
`caption_excluded_from_training`: a synthetic description is not anyone's
voice. Attachments appear in context only as a bare `[attachment]` marker.

CPU only, no torch — it runs in the trainer image without a GPU lease, and its
tests run in the torch-free test venv.

    python build_dataset.py --user-id 123 --persona mimic_user1 --display-name Alice
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from mimic_format import (
    CONTEXT_N,
    GAP_MINUTES,
    LONG_NUMBER,
    MERGE_SECONDS,
    Msg,
    Turn,
    context_window,
    context_line,
    junk_reason,
    is_command,
    is_conversation_type,
    merge_turns,
    normalise,
    render_context,
    render_persona_prompt,
    without_urls,
)
from persona_roster import load_aliases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build-dataset")


# -- text normalisation --
#
# The rules every message passes through, live or archived, are in
# shared/mimic_format.py. What is left here only the archive needs.

def resolve_mentions(record: dict) -> str:
    """
    `content` with each "@Nickname" history-service could identify turned back
    into Discord's "<@id>", so normalise renders it as "@username" — the same
    name the speaker labels use, which lets the model connect a mention to a
    speaker. A nickname never could: "@Sir Lags-a-Lot (AFK)" and "[erin_h]"
    are one person. Records archived before history-service kept `mentions`
    pass through unchanged.
    """
    text = record.get("content") or ""
    for m in sorted(record.get("mentions") or [], key=lambda m: len(m["display"]), reverse=True):
        text = text.replace("@" + m["display"], f"<@{m['user_id']}>")
    return text


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# -- loading --

def load_archive(archive_dir: str) -> list[dict]:
    """Every record from every per-user file. Order is restored later, per channel."""
    records = []
    for path in sorted(glob.glob(os.path.join(archive_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {archive_dir}")
    return records


# -- building --

@dataclass
class Options:
    # Every account the member has posted from: someone whose account was
    # deleted and who rejoined has two ids, and both are their voice.
    user_ids: list[str]
    context_n: int = CONTEXT_N
    gap_minutes: float = GAP_MINUTES
    merge_seconds: float = MERGE_SECONDS
    # The CLI defaults to 24 (see main); 8 here keeps the library's own filter
    # minimal, which is what the tests build on.
    min_chars: int = 8
    exclude_channels: list[str] = field(default_factory=list)
    # UTC days (YYYY-MM-DD) left out entirely, for one-off events that are
    # nobody's normal voice — e.g. an evening of pasted chatbot output.
    exclude_dates: list[str] = field(default_factory=list)
    # user id -> the name to use for them in place of their Discord handle
    aliases: dict[str, str] = field(default_factory=dict)


def _channel_timelines(records: list[dict], opts: Options, stats: Counter) -> dict[str, list[dict]]:
    excluded = set(opts.exclude_channels)
    excluded_dates = set(opts.exclude_dates)
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("is_bot"):
            stats["dropped_bot_message"] += 1
            continue
        if r.get("channel_id") in excluded or r.get("channel_name") in excluded:
            stats["dropped_excluded_channel"] += 1
            continue
        if parse_ts(r["timestamp"]).date().isoformat() in excluded_dates:
            stats["dropped_excluded_date"] += 1
            continue
        content = (r.get("content") or "").strip()
        if is_command(content):
            stats["dropped_command"] += 1
            continue
        if not is_conversation_type(r.get("type")):
            stats["dropped_system_message"] += 1
            continue
        if not content and not r.get("attachments"):
            stats["dropped_empty"] += 1
            continue
        by_channel[r["channel_id"]].append(r)
    for msgs in by_channel.values():
        msgs.sort(key=lambda r: (r["timestamp"], r["message_id"]))
    return by_channel


# A long message posted word for word this many times across the archive is a
# copypasta, not anyone's voice. Needs the whole archive, so the bot cannot
# apply it; in context such a message is only capped (mimic_format.context_text).
COPYPASTA_MIN_CHARS = 200
COPYPASTA_MIN_COUNT = 3


def copypastas(turns: Iterable[Turn]) -> set[str]:
    counts = Counter(t for turn in turns for t in turn.texts if len(t) >= COPYPASTA_MIN_CHARS)
    return {t for t, n in counts.items() if n >= COPYPASTA_MIN_COUNT}


def _junk(text: str, pastas: set[str]) -> Optional[str]:
    return junk_reason(text) or ("copypasta" if text in pastas else None)


def _turns(msgs: list[dict], usernames: dict, channels: dict, merge_seconds: float) -> list[Turn]:
    return merge_turns(
        (
            Msg(
                user_id=r["user_id"],
                username=usernames.get(r["user_id"]) or r["user_id"],
                ts=parse_ts(r["timestamp"]),
                text=normalise(resolve_mentions(r), usernames, channels),
                has_attachments=bool(r.get("attachments")),
            )
            for r in msgs
        ),
        merge_seconds,
    )


def timelines(records: list[dict], opts: Options, stats: Counter) -> tuple[list[list[Turn]], set[str]]:
    """
    Every channel's merged turns, oldest first, as the dataset sees them — and
    the archive's copypastas. Shared with topic_eval.py, which draws its probe
    messages from the same timelines the training examples come from.
    """
    usernames = {m["user_id"]: m["username"] for r in records for m in r.get("mentions") or [] if m.get("username")}
    usernames.update({r["user_id"]: r.get("username") or r["user_id"] for r in records})
    # One name per person everywhere: speaker labels and resolved mentions both
    # come from this map, so "[erin]:" and "@erin" are recognisably the same.
    usernames.update(opts.aliases)
    channels = {r["channel_id"]: r.get("channel_name") or r["channel_id"] for r in records}
    turns_by_channel = [
        _turns(msgs, usernames, channels, opts.merge_seconds)
        for msgs in _channel_timelines(records, opts, stats).values()
    ]
    return turns_by_channel, copypastas(t for turns in turns_by_channel for t in turns)


def is_junk(text: str, pastas: set[str]) -> bool:
    return _junk(text, pastas) is not None


def build_examples(records: list[dict], system_prompt: str | None, opts: Options) -> tuple[list[dict], Counter]:
    """
    (examples sorted oldest first, counts of what was kept and why things were dropped).

    A target turn becomes an example only if it has at least one turn of context
    before it inside the gap. A message that opens a conversation was not a
    reply to anything, and an empty prompt teaches the model to talk unprompted.
    """
    stats: Counter = Counter()
    gap = opts.gap_minutes * 60.0
    turns_by_channel, pastas = timelines(records, opts, stats)

    examples = []
    for turns in turns_by_channel:
        for i, turn in enumerate(turns):
            if turn.user_id not in opts.user_ids:
                continue
            stats["target_turns"] += 1
            if turn.attachment_only:
                stats["dropped_attachment_only"] += 1
                continue
            # Junk messages are cut out of the reply, so one spam line does not
            # cost the member's real lines merged around it. If cutting them
            # leaves too little, the junk is why the reply is dropped.
            junk = [r for r in (_junk(t, pastas) for t in turn.texts if t) if r]
            reply = "\n".join(t for t in turn.texts if t and not _junk(t, pastas))
            if junk:
                stats["trimmed_junk_messages"] += len(junk)
            if not without_urls(reply):
                stats[f"dropped_{junk[0]}" if junk else "dropped_link_only"] += 1
                continue
            if len(without_urls(reply)) < opts.min_chars:
                stats[f"dropped_{junk[0]}" if junk else "dropped_too_short"] += 1
                continue
            # Fine to read in context, but as a target it teaches the adapter to
            # emit a placeholder — or, before replace_links, the digits themselves.
            if LONG_NUMBER in reply:
                stats["dropped_long_number"] += 1
                continue

            context = context_window(turns[max(0, i - opts.context_n):i], opts.context_n, gap, anchor=turn.start)
            if not context:
                stats["dropped_no_context"] += 1
                continue

            prompt = []
            if system_prompt is not None:
                prompt.append({"role": "system", "content": system_prompt})
            prompt.append({"role": "user", "content": render_context(context)})
            examples.append({
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": reply}],
                "context_turns": [context_line(t) for t in context],
                # Kept for the time split and for tracing an example back; the
                # trainer ignores columns it does not know.
                "timestamp": turn.start.isoformat(),
            })
            stats["examples"] += 1

    examples.sort(key=lambda e: e["timestamp"])
    return examples, stats


def split_by_time(examples: list[dict], eval_fraction: float) -> tuple[list[dict], list[dict]]:
    """
    Newest slice is eval. Random splitting would put a message's neighbours —
    near-duplicates in a fast-moving channel — on both sides, and the eval loss
    would measure memory rather than voice.
    """
    if not examples or eval_fraction <= 0:
        return examples, []
    n_eval = max(1, int(len(examples) * eval_fraction))
    return examples[:-n_eval], examples[-n_eval:]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one member's mimic dataset from the archive")
    ap.add_argument(
        "--user-id", required=True,
        help="Discord user id of the member to mimic; comma-separate several for one member's old and new accounts",
    )
    ap.add_argument("--persona", required=True, help="Alias the adapter is served as, e.g. mimic_user1")
    ap.add_argument("--display-name", required=True, help="Name the persona prompt uses for them")
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE_DIR", "/archive/archive"))
    ap.add_argument("--out", default=None, help="Default: /archive/training/<persona>")
    # Stored as no_system in stats.json's args, the name serving reads it by.
    ap.add_argument(
        "--with-system", dest="no_system", action="store_false",
        help="Prefix every prompt with the persona system prompt (the format before 2026-11-11)",
    )
    ap.add_argument("--context-n", type=int, default=CONTEXT_N, help="Max context turns before each reply")
    ap.add_argument("--gap-minutes", type=float, default=GAP_MINUTES, help="Silence that ends a conversation")
    ap.add_argument(
        "--merge-seconds", type=float, default=MERGE_SECONDS, help="Same-author messages this close are one turn"
    )
    # 24, not Options' 8: below it a reply is mostly "lmao same", and ~30% of
    # targets being that teaches the adapter to answer in two words.
    ap.add_argument("--min-chars", type=int, default=24, help="Shorter replies are dropped as targets")
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument(
        "--names",
        default="/archive/training/names.json",
        help="Hand-filled {user_id: {alias}} file; members without an alias keep their handle",
    )
    ap.add_argument(
        "--exclude-dates", default="",
        help="Comma-separated UTC days (YYYY-MM-DD) to leave out entirely",
    )
    ap.add_argument(
        "--exclude-channels",
        default=os.environ.get("EXCLUDED_CHANNELS", ""),
        help="Comma-separated channel ids or names (same rule as history-service)",
    )
    args = ap.parse_args()

    opts = Options(
        user_ids=[u.strip() for u in args.user_id.split(",") if u.strip()],
        context_n=args.context_n,
        gap_minutes=args.gap_minutes,
        merge_seconds=args.merge_seconds,
        min_chars=args.min_chars,
        exclude_channels=[c.strip() for c in args.exclude_channels.split(",") if c.strip()],
        exclude_dates=[d.strip() for d in args.exclude_dates.split(",") if d.strip()],
        aliases=load_aliases(args.names),
    )
    logger.info(f"Aliases: {len(opts.aliases)} member(s) renamed from {args.names}")
    system_prompt = None if args.no_system else render_persona_prompt(args.persona, args.display_name)

    examples, stats = build_examples(load_archive(args.archive), system_prompt, opts)
    if not examples:
        logger.error(f"No examples for user {args.user_id}: {dict(stats)}")
        return 1
    train, evaluation = split_by_time(examples, args.eval_fraction)

    # Keyed on the alias, not the member's id, so the committed configs that
    # point here never name a real person.
    out = args.out or os.path.join("/archive/training", args.persona)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "train.jsonl"), train)
    write_jsonl(os.path.join(out, "eval.jsonl"), evaluation)
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump({"args": vars(args), "counts": dict(stats), "train": len(train), "eval": len(evaluation)}, fh, indent=2)

    for key, value in sorted(stats.items()):
        logger.info(f"  {key:28s} {value}")
    logger.info(f"Wrote {len(train)} train / {len(evaluation)} eval examples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
