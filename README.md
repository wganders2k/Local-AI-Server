# Local AI Server

(Disclaimer: documentation generated with AI assistance)

Monorepo for a personal AI server stack running on an Ubuntu machine with an RTX 3090. Hosts a Discord bot (mimic personas, thread chat, and an agentic lore assistant), a self-hosted chat UI (Open WebUI), and a VS Code coding assistant — all fronted by a FastAPI proxy, with a single arbiter deciding who holds the GPU.

## Architecture

One card, several tenants. The proxy owns *which model answers a request*; the arbiter owns *who may be on the GPU at all*. They are separate services because they answer separate questions.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Ubuntu Server (RTX 3090)                        │
│                                                                        │
│   ┌──────────────────────┐              ┌───────────────────────────┐  │
│   │  Proxy :11436        │─── acquire ─▶│  Arbiter :11438           │  │
│   │  FastAPI             │◀── granted ──│  the only component that  │  │
│   │  routes + serialises │              │  can see the card         │  │
│   └──────────────────────┘              └───────────────────────────┘  │
│        │              │                        │            │          │
│        ▼              ▼                        │            │          │
│  ┌─────────────┐  ┌──────────────────────┐     │            │          │
│  │ llama       │  │ llama-swappable      │◀────┘            ▼          │
│  │ -permanent  │  │ :11434               │           ┌──────────────┐  │
│  │ :11435      │  │ router — one model   │           │ lora-trainer │  │
│  │ autocomplete│  │ resident at a time   │           │ preemptible  │  │
│  │ (profile)   │  │ (models.ini presets) │           │ (profile)    │  │
│  └─────────────┘  └──────────────────────┘           └──────────────┘  │
│                                                                        │
│   CPU-only: rag-service ─▶ chromadb   history-service ─▶ DCE           │
│   Observability: prometheus :9091 ─▶ grafana :3002                     │
└────────────────────────────────────────────────────────────────────────┘
        ▲                     ▲                    ▲
    VS Code /             Open WebUI           Discord Bot
    Cursor                :3000
```

- [`arbiter/README.md`](arbiter/README.md) — GPU admission control, the lease model, and why there is no privileged tenant.
- [`lora-training/README.md`](lora-training/README.md) — the preemptible trainer and its checkpoint discipline.
- [`proxy/README.md`](proxy/README.md), [`discord-bot/README.md`](discord-bot/README.md), [`history-service/README.md`](history-service/README.md), [`rag/README.md`](rag/README.md).

> **Note on the design docs.** [`Design.md`](Design.md) and [`DiscordBot-Design.md`](DiscordBot-Design.md) are the v3.0 design and predate the GPU arbiter entirely — neither mentions it, and their VRAM-handover sections describe a scheme that has since been replaced. Read them for intent and model-selection rationale, not for current behaviour. The per-service READMEs above are closer to the code.

## GPU arbitration

Everything that wants the card asks [`arbiter`](arbiter/) for it and is granted or refused. The arbiter starts nothing — work is submitted by whoever owns it, and the arbiter only decides who may hold the card while it runs.

Policy lives entirely in [`arbiter/jobs.yaml`](arbiter/jobs.yaml). Priority alone decides contention:

| job | priority | reclaim | notes |
|---|---|---|---|
| `llm` | 100 | `none` | the proxy; wins every contention |
| `profile-segment` | 50 | `none` | LongSafeVision profiler, run by hand |
| `video-processing` | 0 | `cooperative` | host job, not a container |
| `lora-trainer` | −10 | `cooperative` | SIGKILLed on demand; resumes from checkpoint |

A caller sends only its name — never its own priority or VRAM figure. An unknown name gets priority 0 and is logged loudly.

Because llama-server keeps a model resident once loaded (~20 GB of 24), a background job only gets a real window after the proxy's idle evictor unloads it — `IDLE_EVICT_SECONDS`, default 600s with no LLM traffic. Training happens in long idle stretches, not continuously.

```bash
make gpu-status     # who holds the card, why, and how much is free
```

## Repository Structure

```
local-ai-server/
├── docker-compose.yml      # All services
├── Makefile                # Common server operations (make help)
├── models.ini              # llama-server presets for the swappable slot
├── system_prompts.ini      # Per-alias system prompts, injected by the proxy
├── prometheus.yml          # Scrape config
├── dce-compose.yml         # One-shot DiscordChatExporter definition
├── .env.example            # Environment template — copy to .env
│
├── arbiter/                # GPU admission control (:11438) — jobs.yaml is the policy
├── proxy/                  # FastAPI orchestration middleware (:11436)
├── discord-bot/            # discord.py bot — /mimic, /chat, /lore
├── rag/                    # Ingest + retrieval over ChromaDB (CPU-only)
├── history-service/        # DCE orchestration, JSONL archive, image captioning (:11437)
├── lora-training/          # Preemptible QLoRA trainer (supervisor + worker)
├── scripts/                # download_models.py — pulls all GGUFs from HuggingFace
├── systemd/                # Boot unit; recreates containers after driver upgrades
├── plans/                  # Design/refactor plans, historical
├── dce_exports/            # Local DCE scratch (gitignored contents)
│
├── Design.md               # v3.0 design document (pre-arbiter — see note above)
└── DiscordBot-Design.md    # Discord bot design document (pre-arbiter)
```

GGUF files live on the host at `$MODELS_DIR/<publisher>/<model>/file.gguf` (default `/srv/models`) and are bind-mounted read-only. They are not in a named volume, so `make nuke` does not touch them.

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/wganders2k/Local-AI-Server.git
cd Local-AI-Server
cp .env.example .env
```

Set at minimum `DISCORD_TOKEN` and `DISCORD_GUILD_ID`. If you run Open WebUI or Grafana behind authentik SSO, also set `OPENWEBUI_CLIENT_ID` / `OPENWEBUI_CLIENT_SECRET`, `GRAFANA_CLIENT_ID` / `GRAFANA_CLIENT_SECRET`, and `GRAFANA_ADMIN_PASSWORD` — these are read by `docker-compose.yml` but are **not yet listed in `.env.example`**.

The compose file bind-mounts several configs by absolute path under `/home/peacow/local-ai-server` (`models.ini`, `system_prompts.ini`, `arbiter/jobs.yaml`, `prometheus.yml`, `dce-compose.yml`, `dce.env`). Deploying elsewhere means editing those paths.

### 2. Download model files

```bash
make models-download          # all slots; skips files already present
make models-download DRY_RUN=1
```

Creates a venv in `scripts/.venv/` on first run. Set `HF_TOKEN` for gated repos.

### 3. Start the stack

```bash
make up          # everything except the profiled services
make status
make check-gpu
```

`make up` deliberately skips three profiled services: `llama-permanent` (`autocomplete`), `lora-trainer` (`managed`), and `discord-chat-exporter` (`manual`). The trainer in particular would take 16 GB behind the arbiter's back if a plain `up` started it.

### 4. Optional: enable IDE autocomplete

```bash
make enable-autocomplete      # starts llama-permanent on :11435
```

The permanent slot currently serves `qwen2.5-coder-1.5b-q8_0` (~1.65 GB VRAM); the Qwen3.5-2B line is commented out in `docker-compose.yml`. With autocomplete disabled, requests for the `autocomplete` alias get a `503` and everything else is unaffected. The setting persists to `.env` as `AUTOCOMPLETE_ENABLED`.

## Common Commands

`make help` prints the full list.

| Command | Description |
|---|---|
| `make up` / `make down` / `make restart` | Lifecycle (volumes preserved) |
| `make pull` | `git pull` + rebuild changed images + restart |
| `make build` | Rebuild custom images without starting |
| `make logs` | Tail all logs (`logs-proxy`, `logs-bot`, `logs-llama`, `logs-rag`, `logs-history`, `logs-openwebui`) |
| `make status` | Container state and health |
| `make check-gpu` | Verify the GPU is visible inside the llama-server containers |
| `make llama-ps` | Models currently loaded in both slots |
| `make restart-bot` / `restart-proxy` / `restart-rag` / `restart-openwebui` | Per-service restart |
| `make restart-llama-swappable` | Pick up `models.ini` changes |
| `make shell-bot` / `shell-proxy` / `shell-rag` | Shell into a container |
| `make gpu-status` | Arbiter view: holder, reason, free VRAM |
| `make train-submit [CONFIG=…]` | Submit a training run — the only thing that starts one |
| `make train-status` / `train-logs` / `train-cancel` | Manage the run |
| `make rag-ingest` / `rag-status` | Ingest lore into ChromaDB; show collection counts |
| `make dce-evaluate` (alias `history-refresh`) | Trigger channel evaluation + targeted exports |
| `make dce-export-full` / `dce-export-guild` / `dce-export-channel` | Manual DCE exports |
| `make enable-autocomplete` / `disable-autocomplete` / `status-autocomplete` | Permanent-slot toggle |
| `make nuke` | ⚠️ Wipe all named volumes (GGUFs in `$MODELS_DIR` survive) |

## Services

| Service | Port | Profile | Description |
|---|---|---|---|
| `llama-swappable` | `:11434` | — | Router mode, `--models-max 1`; loads a preset on demand |
| `llama-permanent` | `:11435` | `autocomplete` | Permanent autocomplete model |
| `proxy` | `:11436` | — | FastAPI orchestration proxy |
| `arbiter` | `127.0.0.1:11438` | — | GPU admission control; NVML-only, no CUDA |
| `history-service` | `127.0.0.1:11437` | — | DCE orchestration, archive merge, image captioning |
| `discord-bot` | — | — | discord.py bot |
| `open-webui` | `:3000` | — | Chat UI (authentik SSO; `chat.wganderbox.ca`) |
| `rag-service` | — | — | Ingest + retrieval, CPU-only |
| `chromadb` | — | — | Vector store |
| `prometheus` | `:9091` | — | Scrapes the proxy and glances |
| `grafana` | `:3002` | — | Dashboards (authentik SSO; `grafana.wganderbox.ca`) |
| `docker-socket-proxy` | `127.0.0.1:2375` | — | Restricted Docker API for arbiter + history-service |
| `lora-trainer` | — | `managed` | Preemptible QLoRA trainer |
| `discord-chat-exporter` | — | `manual` | One-shot export CLI |

### Key endpoints

```
proxy    GET  /health  /status  /metrics        POST /v1/* (OpenAI-compatible, forwarded)
arbiter  POST /gpu/acquire  /gpu/release        GET  /gpu/status  /gpu/reclaim-notice  /metrics
rag      GET  /health                           POST /retrieve  /ingest
history  GET  /health  /status                  POST /evaluate  /reparse  /clear
```

`proxy /history` and `/history/summary` are stubs returning `[]` — that data moved to Prometheus.

## Model Management

- **`models.ini`** — every swappable-slot preset. Each `[section]` is a named alias with its GGUF path and inference parameters.
- **`docker-compose.yml`** — the permanent-slot model, via the `--model` flag on `llama-permanent`.
- **`system_prompts.ini`** — per-alias system prompts, injected by the proxy.

Currently enabled aliases in `models.ini`:

| Alias | Base | Used by |
|---|---|---|
| `brain` | Qwen3.6-35B-A3B UD-IQ4_NL | coding assistant |
| `brain-dense` | Qwen3.6-27B Q4_K_M | coding assistant |
| `brain-dense-heretic` | Qwen3.6-27B Heretic (uncensored) | `/lore` agent (`AGENT_MODEL`) |
| `gemma-brain-dense` | gemma-4-31B-it Q4_K_M | coding assistant |
| `lore` | gemma-4-26B-A4B-it UD-Q4_K_XL | direct lore routing (not `/lore`) |
| `chat-liberal` | gemma-4-26B-A4B-it UD-Q4_K_XL | Open WebUI |
| `chat-chinese` | Qwen3.6-35B-A3B UD-IQ4_NL | Open WebUI |
| `bomb-you` | Qwen3.5-35B-A3B-Uncensored IQ4_XS | Open WebUI |

**Commented out and therefore unavailable:** `mimic_user1`–`mimic_user6` and `image-caption`. They are still listed in `proxy/config.py`'s `SWAPPABLE_MODELS` and `system_prompts.ini`, and `/mimic` still offers them via autocomplete — but a request reaches llama-server and finds no such preset. Re-enable the sections in `models.ini` before using `/mimic` or image captioning.

`SWAPPABLE_MODELS` in `proxy/config.py` is advisory rather than a gate: the proxy forwards *any* non-autocomplete model name to the swappable slot and lets llama-server decide. It is out of sync with `models.ini` in both directions.

### Switching a model or quant

1. Edit the preset in `models.ini` (or the `--model` flag in `docker-compose.yml` for the permanent slot).
2. Add or update the entry in `scripts/download_models.py`.
3. `make models-download`
4. `make restart-llama-swappable` (or `restart-llama-permanent`).

## Discord Bot

| Command | Description |
|---|---|
| `/mimic <persona> <message>` | One-shot reply from a mimic persona (see the caveat above) |
| `/chat <model>` | Creates a thread; every subsequent message in it is streamed to that model. Model list is autocompleted live from the proxy's `/v1/models`. Threads survive restarts via `discord-bot/data/threads.json` |
| `/lore <question> [rounds]` | Agentic RAG. The agent iteratively calls `search_discord_history`, `search_channel_history`, and `summarize_channel` against the RAG service before synthesising an answer. Default 10 rounds, hard cap 25 |

`/lore` injects server-specific background (member alias index, persona notes) from `discord-bot/prompts/lore_context.md`, which is gitignored because it holds real names:

```bash
cp discord-bot/prompts/lore_context.example.md discord-bot/prompts/lore_context.md
```

Missing, the bot still starts and `/lore` still answers — it just logs a warning.

`/admin-clear-history` is disabled: it had no authorization check.

## History & RAG Pipeline

```
make dce-evaluate  →  history-service reads channel_state.json
                   →  runs DiscordChatExporter per stale channel (via docker-socket-proxy)
                   →  merges raw JSON into per-user JSONL at /archive/archive/<user_id>.jsonl
                   →  optional image captioning (IMAGE_CAPTION_ENABLED, 3–6 AM window)

make rag-ingest    →  rag-service chunks the JSONL by conversation, embeds with
                      ibm-granite/granite-embedding-311m-multilingual-r2, upserts to
                      ChromaDB collection `lore`  (idempotent)
```

Archive tiers live on cold storage at `/mnt/storage_cold/array/DiscordArchive/{raw,archive,state}` — bind mounts, so they survive `make nuke`.

Automatic retraining is **not wired up**. `history-service` still calls `_notify_lora_training()` after a merge, which POSTs to `http://lora-training:11438/notify` — a service that does not exist. It fails harmlessly and is logged at warning level. A training run starts only when a human runs `make train-submit`.

## LoRA Training

```bash
make train-submit CONFIG=configs/smoke.yaml
make train-logs
make train-status
make train-cancel
```

The container is PID-1 `supervisor.py` holding the arbiter lease, with `train_worker.py` as a child holding the CUDA context. Preemption SIGKILLs the worker; the container survives and resumes from the last checkpoint carrying a `COMPLETE` marker. Being killed is the normal path — see [`lora-training/README.md`](lora-training/README.md) for why the split is forced and why the cadence is wall-clock rather than step-based.

Per-persona mimic training is not built. `configs/smoke.yaml` (Qwen3-0.6B) exists to validate the handover in minutes.

## Monitoring

Prometheus (`:9091`) scrapes `proxy:11436/metrics` and a glances exporter every 5s; Grafana is on `:3002`. The proxy exports request counts, token counts, queue depth, request duration, in-flight requests, idle seconds, and current-model/model-age gauges. GPU metrics come from the arbiter — it is the only component that reads the card, so there is one answer rather than two.

## Boot

```bash
sudo bash systemd/install.sh
```

Installs a oneshot unit that runs `docker compose up -d --remove-orphans`, falling back to `--force-recreate`, and pins the NVIDIA driver against unattended-upgrades.

The fallback matters: the NVIDIA container runtime bakes versioned driver library paths into a container's mount spec at *create* time, so an unattended driver upgrade leaves every pre-existing GPU container unable to start. Restart policies don't rescue it — the failure is during container init rather than an exit, so Docker gives up and the stack stays down until someone recreates it.

## Tests

Per-service pytest suites, run from each service directory. `arbiter`, `proxy`, and `lora-training` each keep a gitignored `.venv-test/` holding just the test dependencies — the trainer's deliberately excludes `torch`, so the suite runs on a machine with no GPU and no multi-GB install:

```bash
cd arbiter       && .venv-test/bin/python -m pytest tests -q   # scheduler, config, docker client
cd proxy         && .venv-test/bin/python -m pytest tests -q   # GPU gate, idle evictor
cd lora-training && .venv-test/bin/python -m pytest tests -q   # arbiter client, checkpoints, supervisor
cd history-service && python -m pytest tests -q                # channel state, DCE parser, JSONL store
```

`arbiter/tests/test_scheduler.py` is the one worth knowing about: it pins the guarantees the design rests on — that nothing is ever started, that `acquire` never reports success while a junior tenant survives, that a lease never outlives its holder, and that priority alone decides who wins.

There is no repo-wide test runner and no CI. `discord-bot` and `rag` have no tests.

## Known gaps

Things the docs describe accurately but that are broken, inert, or unfinished in the code — tracked in full in [`TODO.md`](TODO.md):

- **`/mimic` has no backing preset.** The `mimic_user*` sections are commented out in `models.ini` while the bot still offers them via autocomplete.
- **The proxy's system-prompt loader is dead code.** `system_prompts.ini` is bind-mounted and editing it does nothing.
- **`history-service` notifies a service that doesn't exist.** `_notify_lora_training()` POSTs to `http://lora-training:11438/notify`; it fails and logs a warning.
- **`EXCLUDED_CHANNELS` never reaches its container** — read by `history-service/config.py`, not passed by `docker-compose.yml`.
- **Model config is spread across six files** and has drifted in both directions between `models.ini` and `proxy/config.py`.

The documents in [`plans/`](plans/) are point-in-time design records, not current-state docs, and are left as written.
