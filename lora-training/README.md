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
build_dataset.py    archive -> one persona's train/eval JSONL (CPU, no lease)
export_adapter.py   finished adapter -> GGUF LoRA for llama-server (CPU, no lease)
personas.py         the roster: persona -> Discord account(s) + display name
build_all_datasets.py  build_dataset.py for every persona in the roster
train_queue.sh      host-side: one run after another (make train-queue)
export_latest.py    newest finished run per persona -> live GGUF (CPU, no lease)
sample_persona.py   held-out outputs side by side, or a recorded blind A/B
configs/smoke.yaml  Qwen3-0.6B — validates the handover in minutes
configs/mimic_gemma4_12b.yaml   the persona run, one copy per member
tests/              checkpoint safety under an unclean stop; refusal handling;
                    every dataset filter
```

`requirements.txt` is pinned because this stack renames constructor arguments
between minor releases (`torch_dtype` → `dtype`, `max_seq_length` → `max_length`,
`warmup_ratio` → a float `warmup_steps`) and `train_worker.py` passes them by
name. Gemma 4 forced the jump to transformers 5.x / TRL 1.x; re-run the smoke
config after any re-pin. The Dockerfile skips the pinned `torch` line — it comes
from the CUDA base image, and installing it again would pull a second copy.

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

**Decided** (2026-11-04): per-member adapters, with the members' agreement, on
an abliterated **Gemma 4 12B** first. The 35B-A3B comes later, once the pipeline
is proven on a model that fits with room to spare.

```
make train-fetch   MODEL=culturerevolt/gemma-4-12b-heretic-abliterated   # once
make train-dataset USER_ID=<discord id> PERSONA=mimic_user1 NAME=<display name>
make train-submit  CONFIG=configs/mimic_gemma4_12b.yaml
make train-export  CONFIG=configs/mimic_gemma4_12b.yaml PERSONA=mimic_user1
make train-sample  PERSONA=mimic_user1 MODELS="mimic-base mimic_user1"
```

None of these but `train-submit` asks for the GPU: they are one-off containers
from the same image, safe beside live inference and beside a waiting run.

### The data (Tier 3)

`build_dataset.py` reassembles the per-author archive into channel timelines and
emits one example per reply: the preceding turns (≤12, broken by a 30-minute
silence) as a `[name]: text` user turn, the member's reply as the completion.
TRL's prompt/completion format puts the loss on the reply only. The system
prompt is rendered from `discord-bot/prompts/mimic_persona.md` — mounted, not
copied — so the adapter trains on the prompt `/mimic` sends.

Dropped as targets: attachment-only, link-only, under 8 characters, bare pings
(under 8 characters once mentions are removed — identified ones from the
archive's `mentions`, plus any whole message of the form `@…` that recurs, which
catches role pings);
all still kept as context. Also conversation openers with no context. Dropped
everywhere: bot messages, `/` and `!` commands (on this server, overwhelmingly
music-bot `!p`/`!play`), every message `type` but `Default` and `Reply` (pin
events, thread creation, poll results — the archive keeps them, tagged),
`EXCLUDED_CHANNELS` and the Makefile's `TRAIN_EXCLUDED_CHANNELS` (channels kept
in the archive for lore but not trained on, e.g. #grant-chronicle). A member who
has posted from several accounts is built from all of them:
`USER_ID=<old id>,<new id>`.

Mentions the archive could identify (its `mentions` field) render as "@name",
using the same name as the speaker labels, so "[will]:" and "@will" read as one
person. That name is the member's Discord handle unless
`/archive/training/names.json` gives them an alias — a hand-filled
`{user_id: {"alias": ...}}` file kept beside the data, not in the repo, because
it holds real names. Blank aliases keep the handle. Emails, phone numbers and street addresses are scrubbed from
both sides. **Captions never enter** — history-service marks them
`caption_excluded_from_training`; an attachment shows up as `[attachment]`.
The newest 5% is held out as eval, by time rather than at random, so
near-duplicate neighbours cannot sit on both sides.

Output lands in `/mnt/storage_cold/array/DiscordArchive/training/<persona>/`,
keyed on the alias so committed configs never name a member, and on the archive
disk rather than in the repo because it is real messages.

### The base, and why the GGUF must match

`culturerevolt/gemma-4-12b-heretic-abliterated` is trained on (BF16 safetensors,
4-bit at load); the same author's Q4_K_M GGUF is what llama-server serves. A LoRA
is a delta against specific weights: over any other base it loads without
complaint and produces mush. The model is multimodal, so `target_modules` is a
regex scoped to `language_model.layers` — a bare suffix list would also catch
projections outside the text decoder.

### Serving: adapters, not merges

`export_adapter.py` runs llama.cpp's `convert_lora_to_gguf.py` (tag pinned in the
Dockerfile) and writes `/srv/models/lora/<persona>-lora.gguf` — tens of MB. Each
persona is a `models.ini` section pointing the shared base at its adapter, so six
personas cost one base on disk, not six re-quantised models.

### Judging a run

Eval loss falls for a model that learns to hedge, so it is not the verdict.
`sample_persona.py --blind` shows two models' replies to the same held-out
conversations in shuffled order and logs your pick to
`/home/peacow/lora-outputs/experiments.jsonl`. Change one variable per run
against the baseline — data shape (context length, merge window, min length,
system prompt or not), then learning rate/epochs, rank and target modules,
rsLoRA/DoRA, dropout/NEFTune — each in its own config under
`configs/experiments/` with its own `output_dir`. The config keys for all of
these pass straight through `train_worker.py`.

From the original design: 1–2 epochs, and always train from the fresh base
checkpoint — never on top of a previously merged model, which compounds drift
across versions.

### Retraining personas

Three steps, none of which needs anyone to remember who each persona is. That
lives in the roster, `/mnt/storage_cold/array/DiscordArchive/training/personas.json`
(`/archive/training/personas.json` in the container), beside `names.json` and
out of the repo for the same reason: it holds real names.

```bash
make train-datasets [PERSONA=mimic_user3] [REFRESH=1]   # 1. archive -> train/eval JSONL
make train-queue [PERSONAS="mimic_user3 mimic_user5"]  # 2. retrain, in the background
make train-queue-status                                 #    progress; train-queue-stop to stop
make train-export-latest [PERSONA=mimic_user3]          # 3. put the newest runs live on /mimic
```

- **Datasets.** `REFRESH=1` first pulls new Discord history into the archive
  (history-service `/evaluate`, which can take hours). Don't rebuild while a queue
  is training: the queue reads these files.
- **Queue.** Runs as the systemd user unit `train-queue`, so it survives logout
  and any terminal or Claude session that started it. Largest dataset first. Each
  run trains into `lora-outputs/<persona>-<job>`, a fresh directory, because the
  worker resumes from whatever is in its output dir. `JOB=<id>` re-runs a job: done
  personas are skipped, a half-finished one resumes. Log:
  `~/.local/share/train-queue/queue-<job>.log`. The queue **does not export**.
- **Export.** Picks each persona's newest run with a complete `final/`, writes
  `/srv/models/lora/<persona>-lora.gguf` (atomically) and a `<persona>-lora.json`
  saying which run it came from, and skips personas already serving it. Check the
  run first (`make train-sample … ARGS=--blind`). `DRY_RUN=1` shows the picks;
  `PERSONA=… RUN=<dir>` exports a specific run, which is also the rollback.

A new persona needs a roster entry, a `configs/<persona>.yaml` (copy one) and a
`models.ini` section whose `lora` points at `<persona>-lora.gguf`. Export warns
if that section is missing. Add the section only once the GGUF exists:
llama-swappable waits forever on a missing adapter, so `/mimic` hangs until it
times out.
