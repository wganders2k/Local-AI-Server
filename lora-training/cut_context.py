"""
Cut each training example's context to what its reply actually responds to.

A persona's examples carry up to 24 turns of context, cut only at a 2-hour
silence, so most of them hold earlier topics the reply has nothing to do with.
Here the [qwen3.6-27b-judge-u] preset reads each example — context and the member's real
reply — and picks how many of the most recent turns the reply needs; the
example is rewritten with only those. Because the judge sees the reply, a reply
that deliberately returns to an earlier topic keeps that topic in its context.

The judge is asked to cut generously: the first prompt asked for what the reply
"responds to", and the judge kept only literal antecedents — half the pilot cut
to a single turn, dropping conversation that plainly set the reply up. And no
cut leaves fewer than --min-turns (default 4, the bot's own serving floor), so
a reply that connects to nothing — keyboard mash, a non-sequitur — still keeps
some conversation around it; examples that short are never sent to the judge.

    python cut_context.py --persona mimic_user2 [--limit N] [--uncut-share 0.2] [--explain]
    python cut_context.py --persona mimic_user2 --agreement 300
    python cut_context.py --persona mimic_user2 --review 50      # interactive, after an --explain pilot

Reads /archive/training/<persona>/{train,eval}.jsonl (built with context_turns)
and writes /archive/training/<persona>-cut/: the cut train.jsonl, eval.jsonl
unchanged (full contexts, as serving sends them, so eval loss compares with
uncut runs), and stats.json — the persona's, so the run records the right
prompt shape, plus a "cut" section with the label distribution.

Labels are cached in /archive/training/cut_labels/<persona>.jsonl, keyed by the
prompt version, the judge model, the context and the reply. A run that is
killed, preempted or re-run after a dataset rebuild only labels what is new;
changing PROMPT_VERSION relabels everything. Examples with one turn of context
have nothing to cut and are never sent.

--uncut-share keeps that fraction of examples whole (chosen by a hash of the
example, so the same ones every run): a hedge in case an adapter trained only on
relevant context latches onto stale context when serving. --agreement N labels N
examples twice at temperature 0.7 and reports how often the labels agree, which
says how decisive the judge is before any GPU time goes into training on it.
--review N shows N cached labels that carry the judge's reason (an --explain
pilot) one at a time — the context with the cut marked, the reply, the reason —
and records whether the cut is right, too short or too long. It runs in the
terminal rather than as a page because the conversations hold real names.

Hours for a whole roster — run it under systemd-run (make train-cut does).
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_dataset as bd
from persona_roster import load_aliases

PROXY_URL = os.environ.get("PROXY_URL", "http://proxy:11436")
TRAINING_DIR = os.environ.get("TRAINING_DIR", "/archive/training")
JUDGE_MODEL = "qwen3.6-27b-judge-u"
# Bump when the prompt or schema changes: every cached label then misses.
PROMPT_VERSION = 2
# No cut leaves fewer turns than this (or than the example has). The bot serves
# at least this many (discord-bot MIMIC_MIN_CONTEXT_TURNS), so training never
# shows the adapter less context than it will be given.
MIN_TURNS = 4
REVIEW_LOG = os.environ.get("CUT_REVIEW_LOG", "/home/peacow/lora-outputs/cut_review.jsonl")

PROMPT = """You are preparing training data from a Discord chat log. Below are the most recent turns of a \
conversation, numbered from the newest (1) back to the oldest ({n}), then the REPLY {name} actually wrote next.

{turns}

REPLY by {name}:
{reply}

Where should the conversation start so that it still holds everything behind this reply? Count back from \
turn 1 to the oldest turn that belongs with it: anything the reply responds to, answers, continues or refers \
to, including a topic it deliberately returns to — and generously include any turns that may be tangentially \
related or may have informed the reply: the setup of a joke, the thread a remark is part of, the mood or \
subject it is riffing on. When unsure whether a turn belongs, include it. Only cut where the conversation was \
clearly about something else that the reply has nothing to do with. If even the newest turn has nothing to do \
with the reply, the answer is 1. The chat is crude and casual; judge only what the reply connects to, never \
its tone or content."""


def numbered(turns: list[str]) -> str:
    """Oldest first, as the conversation reads, each labelled with how far back it is."""
    n = len(turns)
    return "\n".join(f"({n - i}) {t}" for i, t in enumerate(turns))


def prompt_for(turns: list[str], reply: str, name: str) -> list[dict]:
    # The same with or without --explain: an explained label is cached and reused
    # for training, so asking for a reason must not change the prompt it answers.
    return [{"role": "user", "content": PROMPT.format(n=len(turns), name=name, turns=numbered(turns), reply=reply)}]


def schema(n: int, explain: bool) -> dict:
    props = {"turns": {"type": "integer", "minimum": 1, "maximum": n}}
    if explain:
        # Generous: llama-server enforces maxLength by stopping mid-sentence, and at
        # 160 nearly every reason was cut off. "turns" comes first, so it is decided
        # before the reason is written and does not depend on it.
        props["why"] = {"type": "string", "maxLength": 800}
    return {"type": "json_schema", "json_schema": {"name": "cut", "schema": {
        "type": "object", "properties": props, "required": list(props),
    }}}


def cache_key(turns: list[str], reply: str) -> str:
    raw = json.dumps([PROMPT_VERSION, JUDGE_MODEL, turns, reply], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def turns_to_keep(label: int, available: int, min_turns: int = MIN_TURNS) -> int:
    """The judge's label, raised to the floor (which an example shorter than it cannot reach)."""
    return min(available, max(label, min_turns))


def keep_whole(key: str, share: float) -> bool:
    """Stable per example: the same examples stay uncut on every run at the same share."""
    return share > 0 and int(key[:8], 16) / 0xFFFFFFFF < share


def cut(example: dict, turns_kept: int) -> dict:
    """The example with only its last `turns_kept` context turns, in the shape build_dataset writes."""
    kept = example["context_turns"][-turns_kept:]
    prompt = [m for m in example["prompt"] if m["role"] != "user"]
    prompt.append({"role": "user", "content": "\n".join(kept)})
    return {**example, "prompt": prompt, "context_turns": kept, "cut_from": len(example["context_turns"])}


# The grammar allows whitespace between tokens, and now and then the judge spends
# its whole budget on it after a complete label: '{\n  "turns": 11\n' and no brace.
# A number followed by whitespace is whole, so that label stands; a reason is lost.
CUT_SHORT = re.compile(r'\{\s*"turns"\s*:\s*(\d+)\s')


def parse_label(content: str) -> dict:
    try:
        return json.loads(content)
    except ValueError:
        if m := CUT_SHORT.match(content):
            return {"turns": int(m.group(1)), "cut_short": True}
        raise


def label(client: httpx.Client, turns: list[str], reply: str, name: str,
          explain: bool = False, temperature: float | None = None) -> dict:
    body = {"model": JUDGE_MODEL, "messages": prompt_for(turns, reply, name),
            "response_format": schema(len(turns), explain), "stream": False,
            # A label is ~10 tokens; a reason up to 800 characters. Cut short, the JSON does not parse.
            "max_tokens": 400 if explain else 32}
    if temperature is not None:
        body["temperature"] = temperature
    failures = 0
    while True:
        try:
            resp = client.post(f"{PROXY_URL}/v1/chat/completions", json=body)
            if resp.status_code == 503:
                # The arbiter gave the card to something that outranks batch work
                # (the profiler, say). Not a failure: wait as long as it takes.
                print("  GPU busy with higher-priority work — waiting", file=sys.stderr)
                time.sleep(30)
                continue
            resp.raise_for_status()
            out = parse_label(resp.json()["choices"][0]["message"]["content"])
            out["turns"] = max(1, min(len(turns), int(out["turns"])))
            return out
        except (httpx.HTTPError, ValueError, KeyError) as e:
            failures += 1
            if failures == 5:
                if isinstance(e, ValueError):
                    # The same unreadable answer five times (temperature 0): keep the
                    # whole context, the generous call, rather than stop the queue.
                    print(f"  unreadable label after 5 tries, keeping all {len(turns)} turns", file=sys.stderr)
                    return {"turns": len(turns), "unreadable": True}
                raise
            print(f"  retrying after {e!r}", file=sys.stderr)
            time.sleep(10 * failures)


# The proxy holds a judge request back while anyone is using /mimic (llm-batch in
# arbiter/jobs.yaml), so one can wait as long as a busy chat lasts. A timeout would
# only resend it and leave the first still queued, so reads wait for hours.
CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=6 * 3600.0, write=60.0, pool=None)


def bucket(n: int) -> str:
    return "1" if n == 1 else "2-4" if n <= 4 else "5-12" if n <= 12 else "13+"


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def speaker_name(stats_args: dict) -> str:
    """How the member appears in their own context: alias from names.json, else their display name."""
    aliases = load_aliases(stats_args.get("names"))
    first_id = stats_args["user_id"].split(",")[0].strip()
    return aliases.get(first_id) or stats_args["display_name"]


def agreement(client: httpx.Client, examples: list[dict], name: str, n: int) -> int:
    pool = [e for e in examples if len(e["context_turns"]) > MIN_TURNS]
    sample = random.Random(0).sample(pool, min(n, len(pool)))
    pairs = []
    for i, e in enumerate(sample, 1):
        reply = e["completion"][0]["content"]
        a = label(client, e["context_turns"], reply, name, temperature=0.7)["turns"]
        b = label(client, e["context_turns"], reply, name, temperature=0.7)["turns"]
        pairs.append((a, b))
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}", file=sys.stderr)
    exact = sum(a == b for a, b in pairs) / len(pairs)
    near = sum(abs(a - b) <= 2 for a, b in pairs) / len(pairs)
    same_bucket = sum(bucket(a) == bucket(b) for a, b in pairs) / len(pairs)
    print(f"{len(pairs)} examples labelled twice at temperature 0.7: exact {exact:.0%}, "
          f"within 2 turns {near:.0%}, same bucket (1 / 2-4 / 5-12 / 13+) {same_bucket:.0%}")
    return 0


def review(examples: list[dict], cache: dict, persona: str, n: int, min_turns: int) -> int:
    """
    Hand-check cuts as they would be written — the judge's label raised to the
    floor: right, too short (the reply needed more) or too long (kept turns it
    does not use). Only labels from the current prompt are shown.
    """
    by_key = {cache_key(e["context_turns"], e["completion"][0]["content"]): e for e in examples}
    pool = [(by_key[k], r) for k, r in cache.items() if "why" in r and k in by_key]
    if not pool:
        print("No explained labels to review — run a pilot with --explain first")
        return 1
    picks = random.Random(0).sample(pool, min(n, len(pool)))
    tally = Counter()
    for i, (e, r) in enumerate(picks, 1):
        turns = e["context_turns"]
        kept = turns_to_keep(r["turns"], len(turns), min_turns)
        floor = f" (judge said {r['turns']}, raised to the floor)" if kept != r["turns"] else ""
        print("\n" + "=" * 72 + f"\n{i}/{len(picks)}  kept {kept} of {len(turns)} turns{floor}\n")
        for j, t in enumerate(turns):
            back = len(turns) - j
            if back == kept:
                print("──────── cut: everything above is dropped ────────")
            print(f"({back}) {t}")
        print(f"\nREPLY: {e['completion'][0]['content']}\nJUDGE: {r['why']}")
        choice = ""
        while choice not in ("r", "s", "l", "q"):
            choice = input("Cut is r(ight) / s(hort, needed more) / l(ong, kept too much) / q(uit): ").strip().lower()
        if choice == "q":
            break
        verdict = {"r": "right", "s": "too_short", "l": "too_long"}[choice]
        tally[verdict] += 1
        with open(REVIEW_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"persona": persona, "key": r["key"], "label": r["turns"], "kept": kept,
                                 "of": len(turns), "min_turns": min_turns, "prompt_version": PROMPT_VERSION,
                                 "verdict": verdict}) + "\n")
    total = sum(tally.values()) or 1
    print("\n" + "  ".join(f"{k}: {v} ({v / total:.0%})" for k, v in tally.most_common()) + f"  (logged to {REVIEW_LOG})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Cut training contexts to what each reply responds to")
    ap.add_argument("--persona", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N train examples (a pilot)")
    ap.add_argument("--uncut-share", type=float, default=0.0, help="Fraction of examples kept whole")
    ap.add_argument("--min-turns", type=int, default=MIN_TURNS, help="No cut leaves fewer turns than this")
    ap.add_argument("--explain", action="store_true", help="Also ask the judge why (for hand review; slower)")
    ap.add_argument("--relabel", action="store_true",
                    help="Label again even when cached (e.g. to get reasons for labels made without --explain)")
    ap.add_argument("--agreement", type=int, default=0, metavar="N",
                    help="Label N examples twice at temperature 0.7, report agreement, write nothing")
    ap.add_argument("--review", type=int, default=0, metavar="N", help="Hand-check N explained labels")
    ap.add_argument("--out", default=None, help="Default: <training>/<persona>-cut")
    args = ap.parse_args()

    src = os.path.join(TRAINING_DIR, args.persona)
    stats = json.load(open(os.path.join(src, "stats.json")))
    train = read_jsonl(os.path.join(src, "train.jsonl"))
    if train and "context_turns" not in train[0]:
        print(f"{src} predates context_turns — rebuild it (make train-datasets PERSONA={args.persona})")
        return 1
    name = speaker_name(stats["args"])
    if args.limit:
        train = train[: args.limit]

    cache_path = os.path.join(TRAINING_DIR, "cut_labels", f"{args.persona}.jsonl")
    cache = {r["key"]: r for r in read_jsonl(cache_path)} if os.path.exists(cache_path) else {}
    if args.review:
        return review(train, cache, args.persona, args.review, args.min_turns)

    with httpx.Client(timeout=CLIENT_TIMEOUT) as client:
        if args.agreement:
            return agreement(client, train, name, args.agreement)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        todo = sum(1 for e in train if len(e["context_turns"]) > args.min_turns
                   and (args.relabel or cache_key(e["context_turns"], e["completion"][0]["content"]) not in cache))
        print(f"{args.persona}: {len(train)} examples, {todo} to label ({len(cache)} cached)", file=sys.stderr)

        out_rows, kept, whole, started, done = [], Counter(), 0, time.time(), 0
        with open(cache_path, "a", encoding="utf-8") as cache_fh:
            for e in train:
                turns, reply = e["context_turns"], e["completion"][0]["content"]
                if len(turns) <= args.min_turns:
                    n = len(turns)  # nothing the floor would let a cut remove
                else:
                    key = cache_key(turns, reply)
                    if args.relabel or key not in cache:
                        result = label(client, turns, reply, name, explain=args.explain)
                        cache[key] = {"key": key, **result, "of": len(turns), "timestamp": e["timestamp"]}
                        cache_fh.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
                        cache_fh.flush()
                        done += 1
                        if done % 100 == 0:
                            rate = done / (time.time() - started)
                            print(f"  {done}/{todo} labelled, {rate:.2f}/s, ~{(todo - done) / rate / 60:.0f} min left",
                                  file=sys.stderr)
                    n = turns_to_keep(cache[key]["turns"], len(turns), args.min_turns)
                    if keep_whole(key, args.uncut_share):
                        n, whole = len(turns), whole + 1
                kept[bucket(n)] += 1
                out_rows.append(cut(e, n))

    out = args.out or src + "-cut"
    os.makedirs(out, exist_ok=True)
    bd.write_jsonl(os.path.join(out, "train.jsonl"), out_rows)
    bd.write_jsonl(os.path.join(out, "eval.jsonl"), read_jsonl(os.path.join(src, "eval.jsonl")))
    before = sum(len(e["context_turns"]) for e in train) / max(len(train), 1)
    after = sum(len(e["context_turns"]) for e in out_rows) / max(len(out_rows), 1)
    stats["args"]["out"] = out
    stats["train"] = len(out_rows)
    stats["cut"] = {
        "source": src, "judge": JUDGE_MODEL, "prompt_version": PROMPT_VERSION, "min_turns": args.min_turns,
        "uncut_share": args.uncut_share,
        "kept_whole": whole, "turns_kept": dict(kept), "mean_turns_before": round(before, 2),
        "mean_turns_after": round(after, 2), "limit": args.limit or None,
    }
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Wrote {len(out_rows)} examples to {out}: mean context {before:.1f} -> {after:.1f} turns; "
          f"kept {dict(sorted(kept.items()))}" + (f", {whole} kept whole" if whole else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
