"""
Read what an adapter actually says, on conversations it never trained on.

Eval loss says the adapter predicts held-out replies better; it does not say
the replies are any good — a model that hedges everything scores well on loss.
So the deciding signal is a person reading outputs, and this makes that cheap:

    # side by side: base vs persona vs the real reply
    python sample_persona.py --persona mimic_user1 --models mimic-base mimic_user1

    # blind A/B between two adapters; your picks go to experiments.jsonl
    python sample_persona.py --persona mimic_user1 --models mimic_user1 mimic_user1_r64 --blind

Prompts come from the persona's eval.jsonl — the newest slice, held out by time
in build_dataset.py — with a fixed seed, so every run is judged on the same set.
Requests go through the proxy like any other client, so the arbiter preempts
a training run for them exactly as it would for /mimic.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import httpx

PROXY_URL = os.environ.get("PROXY_URL", "http://proxy:11436")
RESULTS = os.environ.get("EXPERIMENTS_LOG", "/home/peacow/lora-outputs/experiments.jsonl")


def pick_prompts(eval_path: str, k: int, seed: int) -> list[dict]:
    with open(eval_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(seed)
    return rng.sample(rows, min(k, len(rows)))


def generate(client: httpx.Client, model: str, messages: list[dict], max_tokens: int) -> str:
    resp = client.post(
        f"{PROXY_URL}/v1/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def blind_order(models: list[str], rng: random.Random) -> list[str]:
    """Which model is shown as option 1. Shuffled per prompt so position cannot leak."""
    order = list(models)
    rng.shuffle(order)
    return order


def record(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample persona outputs on held-out prompts")
    ap.add_argument("--persona", required=True, help="Training set to draw prompts from")
    ap.add_argument("--models", nargs="+", required=True, help="Aliases to compare (models.ini sections)")
    ap.add_argument("--blind", action="store_true", help="Blind A/B between exactly two models; records picks")
    ap.add_argument("-k", type=int, default=30, help="Number of held-out prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--eval-path", default=None, help="Default: /archive/training/<persona>/eval.jsonl")
    args = ap.parse_args()

    if args.blind and len(args.models) != 2:
        ap.error("--blind compares exactly two models")

    eval_path = args.eval_path or os.path.join("/archive/training", args.persona, "eval.jsonl")
    prompts = pick_prompts(eval_path, args.k, args.seed)

    # One model at a time over every prompt, not alternating per prompt: the
    # swappable slot holds one preset, and alternating would reload it 2k times.
    outputs: dict[str, list[str]] = {}
    with httpx.Client(timeout=300.0) as client:
        for model in args.models:
            print(f"Generating {len(prompts)} replies from {model}...", file=sys.stderr)
            outputs[model] = [generate(client, model, p["prompt"], args.max_tokens) for p in prompts]

    rng = random.Random(args.seed)
    tally = {m: 0 for m in args.models}
    for i, p in enumerate(prompts):
        print("\n" + "=" * 72)
        print(p["prompt"][-1]["content"])
        print("-" * 72)
        if not args.blind:
            for model in args.models:
                print(f"[{model}] {outputs[model][i]}")
            print(f"[real] {p['completion'][0]['content']}")
            continue

        order = blind_order(args.models, rng)
        for n, model in enumerate(order, 1):
            print(f"  {n}) {outputs[model][i]}")
        choice = ""
        while choice not in ("1", "2", "t", "s"):
            choice = input("More like them? 1 / 2 / t(ie) / s(kip): ").strip().lower()
        print(f"  real: {p['completion'][0]['content']}")
        if choice == "s":
            continue
        winner = "tie" if choice == "t" else order[int(choice) - 1]
        if winner in tally:
            tally[winner] += 1
        record(RESULTS, {
            "at": datetime.now(timezone.utc).isoformat(),
            "persona": args.persona,
            "models": args.models,
            "seed": args.seed,
            "prompt_timestamp": p.get("timestamp"),
            "winner": winner,
        })

    if args.blind:
        print("\n" + "  ".join(f"{m}: {n}" for m, n in tally.items()) + f"  (logged to {RESULTS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
