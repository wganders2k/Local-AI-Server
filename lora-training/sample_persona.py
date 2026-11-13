"""
Read what an adapter actually says, on conversations it never trained on.

Eval loss says the adapter predicts held-out replies better; it does not say
the replies are any good — a model that hedges everything scores well on loss.
So the deciding signal is a person reading outputs, and this makes that cheap:

    # side by side: base vs persona vs the real reply
    python sample_persona.py --persona mimic_user1 --models base mimic_user1

    # blind A/B: the live adapter against a new run exported as the candidate
    # (make train-export-latest PERSONA=mimic_user1 CANDIDATE=1); picks go to experiments.jsonl
    python sample_persona.py --persona mimic_user1 --models mimic_user1 candidate --blind

    # a run trained without the system prompt before runs recorded it (see below)
    python sample_persona.py --persona mimic_user1 --models mimic_user1 candidate --blind --no-system candidate

A model is one of:
    base        the bare base: [mimic-multi] with every adapter at 0
    <persona>   that persona's live adapter on [mimic-multi], exactly as /mimic serves it
    candidate   [mimic-candidate], the one adapter export_latest --candidate wrote
    anything else is sent as a models.ini preset name, unchanged

Prompts come from the persona's eval.jsonl — the newest slice, held out by time
in build_dataset.py — with a fixed seed, so every run is judged on the same set.
Each model gets them in the shape its adapter was trained on: with or without
the persona system prompt, by the adapter's manifest (persona_roster's
trained_format), whatever the current eval.jsonl was built with. --no-system
overrides that for a run exported before runs recorded it.
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

import mimic_format
import persona_roster

PROXY_URL = os.environ.get("PROXY_URL", "http://proxy:11436")
RESULTS = os.environ.get("EXPERIMENTS_LOG", "/home/peacow/lora-outputs/experiments.jsonl")
SCALE = float(os.environ.get("MIMIC_ADAPTER_SCALE", persona_roster.DEFAULT_ADAPTER_SCALE))
LORA_DIR = os.environ.get("LORA_GGUF_DIR", "/srv/models/lora")


def resolve(spec: str, adapters: list[str], scale: float = SCALE) -> tuple[str, list[dict] | None]:
    """A --models entry -> (the model to request, its per-request adapter scales or None)."""
    if spec == "base":
        return persona_roster.MULTI_MODEL, persona_roster.lora_scales(adapters, None, scale)
    if spec == "candidate":
        return persona_roster.CANDIDATE_MODEL, [{"id": 0, "scale": scale}]
    if spec in adapters:
        return persona_roster.MULTI_MODEL, persona_roster.lora_scales(adapters, spec, scale)
    return spec, None


def live_adapters(client: httpx.Client) -> list[str]:
    """[mimic-multi]'s adapter order, from the proxy — never a copy that could drift."""
    resp = client.get(f"{PROXY_URL}/v1/models")
    resp.raise_for_status()
    return persona_roster.adapter_order(resp.json().get("data") or [])


def pick_prompts(eval_path: str, k: int, seed: int) -> list[dict]:
    with open(eval_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(seed)
    return rng.sample(rows, min(k, len(rows)))


def manifest_name(spec: str, adapters: list[str]) -> str | None:
    """Whose manifest says how `spec` was trained: a persona's, the candidate's, or nobody's (base, presets)."""
    return "candidate" if spec == "candidate" else spec if spec in adapters else None


def system_for(spec: str, adapters: list[str], persona: str, no_system: list[str]) -> str | None:
    """The system prompt `spec` was trained with, by its manifest, or None (overridden by --no-system)."""
    owner = manifest_name(spec, adapters)
    if owner is None or spec in no_system or persona_roster.trained_format(owner, LORA_DIR).get("no_system"):
        return None
    return mimic_format.render_persona_prompt(persona)


def shaped(messages: list[dict], system: str | None) -> list[dict]:
    """The same turns, with `system` as the system message, or with none."""
    turns = [m for m in messages if m["role"] != "system"]
    return ([{"role": "system", "content": system}] if system is not None else []) + turns


def generate(client: httpx.Client, model: str, lora: list[dict] | None, messages: list[dict], max_tokens: int) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if lora is not None:
        body["lora"] = lora
    resp = client.post(f"{PROXY_URL}/v1/chat/completions", json=body)
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
    ap.add_argument("--models", nargs="+", required=True, help="base, candidate, persona keys or preset names")
    ap.add_argument("--blind", action="store_true", help="Blind A/B between exactly two models; records picks")
    ap.add_argument("-k", type=int, default=30, help="Number of held-out prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--eval-path", default=None, help="Default: /archive/training/<persona>/eval.jsonl")
    ap.add_argument("--no-system", nargs="+", default=[], metavar="MODEL",
                    help="Send these without the system prompt whatever their manifest says (runs from before the record)")
    args = ap.parse_args()

    if args.blind and len(args.models) != 2:
        ap.error("--blind compares exactly two models")
    if not set(args.no_system) <= set(args.models):
        ap.error("--no-system names a model not in --models")
    # llama-server waits forever on a missing adapter: refuse rather than hang.
    if "candidate" in args.models and not os.path.isfile(os.path.join(LORA_DIR, "candidate-lora.gguf")):
        ap.error("no candidate exported yet — make train-export-latest PERSONA=<persona> CANDIDATE=1")

    eval_path = args.eval_path or os.path.join("/archive/training", args.persona, "eval.jsonl")
    prompts = pick_prompts(eval_path, args.k, args.seed)

    # One model at a time over every prompt, not alternating per prompt: the
    # swappable slot holds one preset, and alternating base/persona against a
    # candidate would reload it 2k times. (base and personas share [mimic-multi].)
    outputs: dict[str, list[str]] = {}
    with httpx.Client(timeout=300.0) as client:
        adapters = live_adapters(client) if any(m != "candidate" for m in args.models) else []
        for spec in args.models:
            model, lora = resolve(spec, adapters)
            print(f"Generating {len(prompts)} replies from {spec} ({model})...", file=sys.stderr)
            system = system_for(spec, adapters, args.persona, args.no_system)
            print(f"  ...{'with' if system else 'without'} the system prompt", file=sys.stderr)
            outputs[spec] = [
                generate(client, model, lora, shaped(p["prompt"], system), args.max_tokens) for p in prompts
            ]

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
            "no_system": args.no_system,
            "seed": args.seed,
            "prompt_timestamp": p.get("timestamp"),
            "winner": winner,
        })

    if args.blind:
        print("\n" + "  ".join(f"{m}: {n}" for m, n in tally.items()) + f"  (logged to {RESULTS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
