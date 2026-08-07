# LoRA Training — a container that gets killed a lot

Fine-tuning that waits for the GPU, takes it when nothing else wants it, and is
killed the moment something does.

## A human starts a run. Nothing else does.

```bash
make train-submit CONFIG=configs/smoke.yaml
```

That is the only thing that begins a training run, and it is worth stating
because it used to be otherwise: the arbiter would pick this job up whenever free
VRAM appeared and `docker start` it. A run therefore began because the LLM went
quiet — nobody decided to train. Preparing a dataset, tweaking hyperparameters
and then committing to hours of GPU time is a human decision, and there is now no
path by which it happens without one.

The arbiter answers a narrower question — *may this hold the card right now* —
and knows nothing about whether a run is wanted. With no container submitted,
nothing holds its lease, which needs no special case anywhere.

## What this has to do

Almost nothing. There is no client library here, no registration, no polling, no
yield protocol, no priority to declare, and no supervisor process. Two HTTP calls
in [`arbiter.py`](arbiter.py) and a checkpoint discipline:

| requirement | where |
|---|---|
| ask for the card before allocating anything | `supervisor.py` |
| wait, rather than fail, when refused | `supervisor.py` |
| kill the training process when the card is wanted | `supervisor.py` |
| resume from the last complete checkpoint on start | `train_worker.py` |
| checkpoint often enough that being killed is cheap | `checkpoints.py` |

`arbiter.py` is a deliberate copy of the calls in `proxy/arbiter.py`, not a
shared library. The previous design vendored a client into three repos and every
protocol change became a three-repo change; forty duplicated lines are cheaper
than coupling two images' build contexts.

## Two processes, and why

`supervisor.py` is PID 1 and holds the lease. `train_worker.py` is its child and
holds the card. The split is forced, not stylistic:

- a CUDA context is freed when its process **exits**, so preemption means killing
  the training process
- if that were PID 1, killing it kills the container
- **Docker suppresses a container's restart policy for any API-initiated stop or
  kill.** Measured on this box: `RestartCount=0` after `docker kill` for both
  `restart: always` and `restart: on-failure`

So a preempted run would simply have been over. With the supervisor, the
container outlives every handover and the worker does not — the same shape the
video-processing watcher has, for the same underlying reason.

The container's exit code is what a human reads afterwards: **0** for a finished
run, anything else for a genuine failure. Preemption never appears in it.

A refusal is *waited out* rather than exited on. Exiting would make "the LLM is
busy" indistinguishable from "the dataset is malformed". Waiting costs nothing:
no model is loaded and no VRAM is held while it asks.

## Being killed is the normal path

`supervisor.py` sends the worker SIGKILL, not SIGTERM, and gives it no grace
period. That is deliberate. Unsaved work is discarded either way, so a graceful
shutdown would buy nothing and spend the only budget that matters: the LLM's
time to first token. A training step on a 35B can be five to ten seconds away;
SIGKILL bounds the handover to process teardown.

Worth being precise about where that lives, because it is easy to look for in
the wrong file. `stop_timeout` in [arbiter/jobs.yaml](../arbiter/jobs.yaml) is
the SIGTERM→SIGKILL grace for a `kind: container` job, and this job is
`cooperative` — the arbiter never signals it at all. It is told, and it kills its
own worker. So the no-grace decision is `self.child.kill()` in this directory,
not policy over there, and setting `stop_timeout` on this job's entry would do
nothing.

What makes that safe is `checkpoints.py`. A checkpoint counts only once a
`COMPLETE` marker sits beside it, written after the save returns. A killed save
leaves a directory with no marker, which `latest()` skips and `prune()` deletes.
Two complete checkpoints are kept, so falling back one is always possible.

### Checkpoint cadence is wall-clock, not steps

The quantity to bound is *lost work*, and a step is not a fixed amount of time —
the same `save_steps` that means four minutes on a 0.6B means over an hour on a
35B. `checkpoint_minutes` fires a save on a wall-clock interval instead, so the
worst case is the same number regardless of the model.

## Running

```bash
make train-submit CONFIG=configs/smoke.yaml   # start a run
make train-logs                               # watch it
make train-status                             # container state + what the arbiter thinks
make train-cancel                             # stop and remove it
```

Never with `docker compose up`. The service sits behind the `managed` profile
precisely so a plain `up` skips it — otherwise it would take 16 GB behind the
arbiter's back.

`configs/` is bind-mounted, so editing a YAML or a dataset JSONL between runs
needs no rebuild of a multi-GB CUDA image. `TRAIN_CONFIG` picks which one; the
Makefile sets it from `CONFIG=`. To change *when* this runs relative to other GPU
work, edit `arbiter/jobs.yaml` — nothing in this directory expresses priority.

Foreground, taking the card without asking, for debugging a config on a machine
where nothing else wants it:

```bash
python train_worker.py --config configs/smoke.yaml
```

## Layout

```
supervisor.py       PID 1: asks for the card, spawns and kills the worker
train_worker.py     load, train, checkpoint, exit. Knows nothing about the GPU
                    lease and talks to nothing over the network
arbiter.py          the GPU calls — acquire, release, wait_until_wanted
checkpoints.py      COMPLETE-marker discovery and pruning
configs/smoke.yaml  Qwen3-0.6B — validates the handover in minutes
tests/              checkpoint safety under an unclean stop; refusal handling
```

`requirements.txt` is pinned because this stack renames constructor arguments
between minor releases (`torch_dtype` → `dtype`, `max_seq_length` → `max_length`)
and `train_worker.py` passes both by name. The Dockerfile skips the pinned
`torch` line — it comes from the CUDA base image, and installing it again would
pull a second copy.

## The window this actually gets

Measured on the deployed box: llama-server holds ~20 GB of a 24 GB card for as
long as a model is resident, leaving ~4.4 GB. That is under this job's
`required_mb`, so **a submitted run only gets the card once the proxy's idle
evictor has unloaded the model** — i.e. after `IDLE_EVICT_SECONDS` (600s) with no
LLM traffic. Any request restarts that clock. The video-processing job outranks
this too, so a full inbox delays it further.

This is the design working, not a fault, but it sets the expectation: training
happens in long idle stretches, not continuously, and a submitted run may sit
waiting for hours. If it is not progressing, check free VRAM and the holder
before suspecting anything — `make train-status` reports both.

`required_mb` in [arbiter/jobs.yaml](../arbiter/jobs.yaml) must be the *peak* need,
not the steady state. The first hardware run OOMed at 4478 MiB free against a
declared 4096: the check passed, then another tenant arrived. The headroom gate
can only be as honest as that number.

## Verifying preemption

```bash
# with a run holding the card
time curl -s localhost:11436/v1/chat/completions \
  -d '{"model":"brain-dense","messages":[{"role":"user","content":"hi"}]}'

docker inspect lora-trainer --format '{{.State.Status}}'  # still "running"
docker logs lora-trainer | grep -i 'preempted\|resum'     # killed, then not step 0
```

Measured end to end at **0.96s** on this hardware: notice, SIGKILL, process exit,
driver reclaim. The other things worth checking are that the container is still
`running` afterwards — it must survive the preemption — and that a finished run
sits at `Exited (0)` and stays there.

## Phase 2 — the mimic personas

Not built. The harness is model-agnostic; the persona work is a config plus a
merge step:

- `configs/mimic.yaml` — QLoRA on the 35B-A3B base against the history-service
  per-user JSONL. Turn `load_in_4bit` on and raise `required_mb` to ~16000 in
  `arbiter/jobs.yaml`. Swap in Unsloth here, where the memory saving decides
  whether the run fits at all. **Not approved** — training a model on per-user
  message history has not been discussed as a decision, only inherited from an
  earlier design sketch.
- `merge.py` — merge the adapter, export GGUF, write a `[mimic_<user>]` section
  into `models.ini`. None exist yet, despite `mimic_user1..6` already being
  listed in `proxy/config.py`.

From the original design: 1–2 epochs, and always train from the fresh base
checkpoint — never on top of a previously merged model, which compounds drift
across versions.
