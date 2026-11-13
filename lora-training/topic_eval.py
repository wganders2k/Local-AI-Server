"""
Does a persona keep up when the conversation changes under it? Eval loss cannot
say: it rewards using every turn of context, which is the opposite of letting go
of a stale topic. So this asks directly, with two probes on the same held-out
contexts (the persona's eval.jsonl):

  switch  a real conversation opener by another member — a message that started
          a conversation after a long silence — appended as the newest turn.
          Someone sincerely brings up something new. Wanted: the reply takes it
          up ("new"), alone or alongside the old topic ("both").
  chaos   a real message from the middle of some other conversation, appended
          out of nowhere: a non-sequitur. Wanted: the reply passes over it and
          carries on ("ignores"), or treats it as the non-sequitur it is ("wtf")
          — not as if it made sense ("engages").

    python topic_eval.py build --persona mimic_user2 [-k 50] [--seed 0] [--force]
    python topic_eval.py run   --persona mimic_user2 --models mimic_user2 candidate --tag randctx [--no-system MODEL ...]

build writes /archive/training/topic_eval/<persona>.jsonl once. Rebuilding the
persona's dataset leaves it alone, so every adapter is judged on the same
probes; --force replaces it, after which earlier results no longer compare.

run generates each model's replies through the proxy as train-sample does (same
model resolution, same per-adapter prompt shape), then the [judge] preset labels
each reply against a fixed schema. Replies, verdicts and the judge's one-line
reason go to lora-outputs/topic_eval/<persona>-<tag>.jsonl for reading back.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import timedelta

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_dataset as bd
import sample_persona as sp
from mimic_format import LONG_NUMBER, Turn, context_line, without_urls
from persona_roster import load_aliases

PROBE_DIR = os.environ.get("TOPIC_EVAL_DIR", "/archive/training/topic_eval")
RESULTS_DIR = os.environ.get("TOPIC_EVAL_RESULTS", "/home/peacow/lora-outputs/topic_eval")
JUDGE_MODEL = "judge"
# What serving sends at most (the persona's context_n); a probe never exceeds it.
MAX_TURNS = 24
# A probe needs an existing conversation to move away from.
MIN_BASE_TURNS = 4

VERDICTS = {
    "switch": ["new", "old", "both", "neither"],
    "chaos": ["ignores", "wtf", "engages", "neither"],
}
WANTED = {"switch": {"new", "both"}, "chaos": {"ignores", "wtf"}}

_NEXT_SPEAKER_RE = re.compile(r"\n\[[^\]\n]{1,40}\]:")
_LEADING_LABEL_RE = re.compile(r"^\[[^\]\n]{1,40}\]:\s*")


# -- choosing probe messages --

def usable(turn: Turn, pastas: set[str], lo: int, hi: int) -> bool:
    """A turn that reads as one person saying something: text, not a link or spam, a sensible length."""
    if turn.attachment_only or LONG_NUMBER in turn.text:
        return False
    if any(bd.is_junk(t, pastas) for t in turn.texts if t):
        return False
    body = without_urls(turn.text)
    return lo <= len(body) <= hi and sum(c.isalpha() for c in body) >= 3


def openers(turns: list[Turn], gap: timedelta, exclude: set[str], pastas: set[str]) -> list[Turn]:
    """
    Turns that started a conversation: first after a silence longer than `gap`,
    and answered by someone else inside it. The answer is what makes it a real
    opener rather than a message nobody picked up.
    """
    out = []
    for i, t in enumerate(turns[:-1]):
        nxt = turns[i + 1]
        started = i == 0 or t.start - turns[i - 1].end > gap
        answered = nxt.user_id != t.user_id and nxt.start - t.end <= gap
        if started and answered and t.user_id not in exclude and usable(t, pastas, 20, 300):
            out.append(t)
    return out


def mid_conversation(turns: list[Turn], exclude: set[str], pastas: set[str]) -> list[Turn]:
    """Turns deep inside a conversation (a reply within 5 minutes of the last): out of it, a non-sequitur."""
    out = []
    for i, t in enumerate(turns[1:], start=1):
        if t.start - turns[i - 1].end <= timedelta(minutes=5) and t.user_id not in exclude \
                and usable(t, pastas, 25, 200):
            out.append(t)
    return out


def probe_context(base_turns: list[str], appended: str) -> list[str]:
    """The base conversation plus the probe as its newest turn, trimmed from the oldest end to what serving sends."""
    return (base_turns + [appended])[-MAX_TURNS:]


def far_from(pool: list[Turn], when, rng: random.Random, days: int = 7) -> Turn:
    """A random turn from `pool` at least `days` from `when`, so it cannot belong to the same conversation."""
    for _ in range(1000):
        t = rng.choice(pool)
        if abs(t.start - when) > timedelta(days=days):
            return t
    raise RuntimeError("no probe message far enough from the base conversation")


def build(args) -> int:
    path = os.path.join(PROBE_DIR, f"{args.persona}.jsonl")
    if os.path.exists(path) and not args.force:
        print(f"{path} exists — probes are built once so results stay comparable (--force replaces them)")
        return 1
    stats = json.load(open(os.path.join("/archive/training", args.persona, "stats.json")))["args"]
    split = lambda v: [x.strip() for x in (v or "").split(",") if x.strip()]  # noqa: E731
    opts = bd.Options(
        user_ids=split(stats["user_id"]),
        context_n=stats["context_n"],
        gap_minutes=stats["gap_minutes"],
        merge_seconds=stats["merge_seconds"],
        exclude_channels=split(stats.get("exclude_channels")),
        exclude_dates=split(stats.get("exclude_dates")),
        aliases=load_aliases(stats.get("names")),
    )
    turns_by_channel, pastas = bd.timelines(bd.load_archive(stats["archive"]), opts, Counter())
    gap = timedelta(minutes=opts.gap_minutes)
    persona_ids = set(opts.user_ids)
    switch_pool = [t for turns in turns_by_channel for t in openers(turns, gap, persona_ids, pastas)]
    chaos_pool = [t for turns in turns_by_channel for t in mid_conversation(turns, persona_ids, pastas)]
    print(f"pools: {len(switch_pool)} openers, {len(chaos_pool)} mid-conversation messages", file=sys.stderr)

    with open(os.path.join("/archive/training", args.persona, "eval.jsonl"), encoding="utf-8") as fh:
        bases = [json.loads(line) for line in fh]
    bases = [b for b in bases if len(b["context_turns"]) >= MIN_BASE_TURNS]
    rng = random.Random(args.seed)
    bases = rng.sample(bases, min(args.k, len(bases)))

    rows = []
    for n, base in enumerate(bases):
        when = bd.parse_ts(base["timestamp"])
        for kind, pool in (("switch", switch_pool), ("chaos", chaos_pool)):
            probe = far_from(pool, when, rng)
            rows.append({
                "id": f"{n:03d}-{kind}",
                "kind": kind,
                "base_timestamp": base["timestamp"],
                "probe_timestamp": probe.start.isoformat(),
                "context_turns": probe_context(base["context_turns"], context_line(probe)),
                "real_reply": base["completion"][0]["content"],  # what they said to the base, for reference
            })
    os.makedirs(PROBE_DIR, exist_ok=True)
    bd.write_jsonl(path, rows)
    print(f"Wrote {len(rows)} probes ({len(bases)} contexts × switch/chaos) to {path}")
    return 0


# -- judging --

JUDGE_INTRO = (
    "You are labelling one reply in a Discord chat log. The chat is crude and casual; judge only what the "
    "reply is responding to, never its tone or content.\n\n"
)
JUDGE_TASK = {
    "switch": (
        "The NEWEST MESSAGE starts a new subject. Label the REPLY:\n"
        '- "new": it responds to the newest message\'s subject\n'
        '- "old": it carries on the earlier conversation, ignoring the newest message\n'
        '- "both": it responds to both\n'
        '- "neither": neither, or you cannot tell\n'
    ),
    "chaos": (
        "The NEWEST MESSAGE is out of place: it was pasted in from an unrelated conversation. Label the REPLY:\n"
        '- "ignores": it carries on the earlier conversation, passing over the out-of-place message\n'
        '- "wtf": it reacts to the message as odd, confusing or out of nowhere\n'
        '- "engages": it treats the message as normal and goes along with its subject\n'
        '- "neither": none of these, or you cannot tell\n'
    ),
}


def judge_messages(kind: str, turns: list[str], reply: str) -> list[dict]:
    earlier, newest = "\n".join(turns[:-1]), turns[-1]
    text = (
        f"{JUDGE_INTRO}EARLIER CONVERSATION:\n{earlier}\n\nNEWEST MESSAGE:\n{newest}\n\n"
        f"REPLY (from another participant, written right after the newest message):\n{reply}\n\n{JUDGE_TASK[kind]}"
        "\nGive your reason in one or two complete sentences.\n"
    )
    return [{"role": "user", "content": text}]


def judge_schema(kind: str) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "verdict", "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS[kind]},
            "why": {"type": "string", "maxLength": 800},  # enforced mid-sentence: keep it roomy
        },
        "required": ["verdict", "why"],
    }}}


def judge(client: httpx.Client, kind: str, turns: list[str], reply: str) -> dict:
    body = {
        "model": JUDGE_MODEL, "messages": judge_messages(kind, turns, reply),
        "response_format": judge_schema(kind), "stream": False, "max_tokens": 400,
    }
    resp = client.post(f"{sp.PROXY_URL}/v1/chat/completions", json=body)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def clean_reply(text: str) -> str:
    """As the bot posts it (discord-bot's clean_persona_reply): no leading label, nothing past a second speaker."""
    text = _LEADING_LABEL_RE.sub("", text.strip())
    return _NEXT_SPEAKER_RE.split(text, maxsplit=1)[0].strip()


def summarise(rows: list[dict], models: list[str]) -> str:
    lines = []
    for kind in VERDICTS:
        lines.append(f"\n{kind:<7} " + "".join(f"{v:>9}" for v in VERDICTS[kind]) + "   wanted")
        for m in models:
            got = [r["verdict"] for r in rows if r["model"] == m and r["kind"] == kind]
            if not got:
                continue
            c = Counter(got)
            wanted = sum(c[v] for v in WANTED[kind]) / len(got)
            lines.append(f"  {m:<14}" + "".join(f"{c[v] / len(got):>9.0%}" for v in VERDICTS[kind]) + f"   {wanted:>5.0%}")
    return "\n".join(lines)


def run(args) -> int:
    with open(os.path.join(PROBE_DIR, f"{args.persona}.jsonl"), encoding="utf-8") as fh:
        probes = [json.loads(line) for line in fh]
    if args.limit:
        probes = probes[: args.limit * 2]
    display_name = sp.persona_roster.load_roster()[args.persona]["display_name"]

    replies: dict[str, list[str]] = {}
    # Judge calls are batch traffic the proxy holds back while /mimic is busy: wait, never resend.
    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=6 * 3600.0, write=60.0, pool=None)) as client:
        adapters = sp.live_adapters(client)
        # Every model's replies first, then all judging: one model swap per model, not per probe.
        for spec in args.models:
            model, lora = sp.resolve(spec, adapters)
            system = sp.system_for(spec, adapters, args.persona, display_name, args.no_system)
            print(f"Generating {len(probes)} replies from {spec} "
                  f"({'with' if system else 'without'} the system prompt)...", file=sys.stderr)
            replies[spec] = [
                clean_reply(sp.generate(
                    client, model, lora,
                    sp.shaped([{"role": "user", "content": "\n".join(p["context_turns"])}], system),
                    args.max_tokens,
                ))
                for p in probes
            ]
        rows = []
        print(f"Judging {len(probes) * len(args.models)} replies with [{JUDGE_MODEL}]...", file=sys.stderr)
        for spec in args.models:
            for p, reply in zip(probes, replies[spec]):
                v = judge(client, p["kind"], p["context_turns"], reply)
                rows.append({"id": p["id"], "kind": p["kind"], "model": spec, "reply": reply, **v,
                             "wanted": v["verdict"] in WANTED[p["kind"]]})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{args.persona}-{args.tag}.jsonl")
    bd.write_jsonl(out, rows)
    print(summarise(rows, args.models))
    print(f"\nReplies and verdicts: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Topic-switch and chaos probes for persona adapters")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Write the persona's probe set (once)")
    b.add_argument("--persona", required=True)
    b.add_argument("-k", type=int, default=50, help="Held-out contexts; each gets one switch and one chaos probe")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--force", action="store_true")
    r = sub.add_parser("run", help="Generate and judge replies")
    r.add_argument("--persona", required=True)
    r.add_argument("--models", nargs="+", required=True, help="base, candidate, persona keys or preset names")
    r.add_argument("--tag", required=True, help="Names the results file")
    r.add_argument("--no-system", nargs="+", default=[], metavar="MODEL",
                   help="Send these without the system prompt whatever their manifest says")
    r.add_argument("--max-tokens", type=int, default=200)
    r.add_argument("--limit", type=int, default=0, help="Only the first N contexts (a quick look)")
    args = ap.parse_args()
    return build(args) if args.cmd == "build" else run(args)


if __name__ == "__main__":
    sys.exit(main())
