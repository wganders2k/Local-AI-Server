
---

# Local AI Server — Design Document v4.0

**Revision history:**

| Version | Change |
|---|---|
| v4.0 | GPU arbitration extracted into a standalone `arbiter` service. The proxy no longer negotiates VRAM handover with external jobs; it asks for a lease like every other tenant. Preemptible LoRA trainer added. `/lore` became an agentic tool-calling loop; `/chat` threads added. Prometheus/Grafana replaced the in-proxy job-history endpoints. Open WebUI and Grafana moved behind authentik SSO. Model lineup moved to the Qwen3.6 / gemma-4 generation. |
| v3.0 | Inference backend migrated from Ollama to llama.cpp (`llama-server`). Router mode (`--models-preset models.ini`) provides swap-on-demand loading natively. OpenAI-compatible API throughout. |

**Scope:** this is the system architecture and rationale document — *why* the pieces are shaped the way they are. Operational detail (commands, ports, env vars) lives in [`README.md`](README.md), and the components with the most subtle reasoning have their own documents: [`arbiter/README.md`](arbiter/README.md) and [`lora-training/README.md`](lora-training/README.md) are the authoritative accounts of GPU arbitration and preemptible training respectively, and are worth reading before changing anything in that area.

---

## 1. Guiding Principles

- **One physical GPU, one tenant at a time.** CUDA has no memory preemption, so occupancy is arbitrated centrally rather than negotiated between peers. See §4b.
- **A job never states its own importance.** Callers send a name; priority and VRAM requirements are operator policy in [`arbiter/jobs.yaml`](arbiter/jobs.yaml). A job that could declare its own priority is grading its own homework.
- **Two distinct model personalities.** Mimic personas use an uncensored base with no content refusals. The lore assistant uses a sterile, instruction-following base. Neither bleeds into the other.
- **Nothing starts work on its own.** The arbiter grants occupancy but starts no containers; a training run begins because a human ran `make train-submit`. An earlier design started one whenever free VRAM appeared, which meant runs began because the LLM went quiet rather than because anyone decided to train.
- **The proxy is content-blind.** It routes by model alias and serialises access. It does not inspect, rewrite, or inject into request bodies — prompt construction belongs to the client that owns the conversation.
- **Open WebUI is a first-class consumer.** It routes through the same proxy as Discord and VS Code, competing for the swappable slot under the same lock. When using Claude via API key it bypasses the proxy entirely — zero VRAM impact.

---

## 2. Hardware & VRAM Budget

**RTX 3090 — 24 GB VRAM (usable: ~24,300 MB)**

The swappable slot runs `--models-max 1`: exactly one model resident at a time, whichever was requested last. The practical consequence is that a large model occupies roughly 20 GB for as long as it stays loaded, and the ~4 GB remainder is the only thing a background job can fit in.

| Slot | Purpose | Model | Quant | VRAM |
|---|---|---|---|---|
| Permanent (`:11435`) | Autocomplete | `qwen2.5-coder-1.5b` | Q8_0 | ~1.65 GB |
| Swappable (`:11434`) | Brain (coding) | Qwen3.6-35B-A3B | UD-IQ4_NL | ~18 GB |
| Swappable (`:11434`) | Brain dense | Qwen3.6-27B | Q4_K_M | ~16.8 GB |
| Swappable (`:11434`) | Lore agent | Qwen3.6-27B Heretic | Q4_K_M | ~16.9 GB |
| Swappable (`:11434`) | Lore / Open WebUI chat | gemma-4-26B-A4B-it | UD-Q4_K_XL | ~17.6 GB |
| Swappable (`:11434`) | Mimic personas (×6) | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB (presets currently disabled) |

The permanent slot is optional and sits behind the `autocomplete` compose profile. Disabling it (`make disable-autocomplete`) frees its footprint and returns `503` for the `autocomplete` alias; nothing else is affected.

**Why the headroom figure matters.** llama-server keeps a model resident once loaded — it does not release VRAM between requests. So "no request in flight" is not the same as "the card is free", and a background job that trusted an idle proxy would OOM against a still-resident 18 GB model. Two mechanisms address this: every grant is checked against a live NVML reading (§4b), and the proxy unloads the resident model after `IDLE_EVICT_SECONDS` of quiet (§6.2).

---

## 3. Model Selection Rationale

### 3.1 Mimic base: `HauhauCS/Qwen3.5-35B-A3B-Uncensored`

The core requirement for mimic personas is **zero refusals on crude, raunchy, or dark humour** — the kind that characterises tight gaming communities. A standard instruction-tuned model sanitises this, adds disclaimers, and breaks character at exactly the wrong moment.

This is Qwen3.5-35B-A3B with refusals removed and no other changes to datasets or capabilities. The 35B-A3B Mixture-of-Experts architecture delivers strong personality capture at a footprint comparable to a dense 9B model.

It may still occasionally append a short disclaimer ("This is general information…") baked into base training. That is not a refusal — the content is generated in full — and it is handled two ways: a system prompt instruction, and regex post-processing in the bot (`DISCLAIMER_PATTERNS` in [`discord-bot/config.py`](discord-bot/config.py)).

**Current status:** the `mimic_user1`–`mimic_user6` presets are commented out in [`models.ini`](models.ini). The base GGUF is still used by the `bomb-you` alias. `/mimic` therefore has no backing preset and will fail at llama-server until those sections are re-enabled.

### 3.2 Lore: instruction-tuned, not uncensored

The lore assistant needs to be reliable, structured, and citation-aware — the opposite of the mimic's job. Keeping it on a standard instruction-tuned base means it won't inject personality where none belongs or hallucinate lore to fill gaps.

Two models serve this area, and the split is deliberate:

- **`lore`** (gemma-4-26B-A4B-it) — the direct alias, for plain retrieval-augmented answering.
- **`brain-dense-heretic`** (Qwen3.6-27B Heretic) — what the `/lore` *agent* actually runs on, set as `AGENT_MODEL`. The agentic loop (§9.2) needs reliable tool-calling, and it reads raw Discord history as tool output — content a censored model will refuse to process even when the question about it is innocuous.

### 3.3 Why llama.cpp, not vLLM or Ollama

**Not vLLM**, on two independent grounds. The mimic base is a sparse MoE, and vLLM's support for quantised GGUF MoEs is unoptimised compared to llama.cpp's tuned MoE CUDA kernels. Separately, the uncensored base is GGUF-only — no safetensors release exists — and vLLM's GGUF support is experimental. The entire model stack is GGUF, making vLLM a non-starter.

**Not Ollama**, which wraps llama.cpp anyway. Using `llama-server` directly gives full control over every inference parameter, `--models-preset` provides the same swap-on-demand behaviour without the abstraction, and llama-server speaks OpenAI format natively rather than translating.

---

## 4. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Ubuntu Server (RTX 3090)                        │
│                                                                        │
│   ┌──────────────────────┐              ┌───────────────────────────┐  │
│   │  Proxy :11436        │─── acquire ─▶│  Arbiter :11438           │  │
│   │  routes by alias     │◀── granted ──│  leases the card;         │  │
│   │  serialises via lock │              │  reads NVML; starts       │  │
│   │  idle-evicts         │              │  nothing                  │  │
│   └──────────────────────┘              └───────────────────────────┘  │
│        │              │                        ▲            ▲          │
│        ▼              ▼                        │            │          │
│  ┌─────────────┐  ┌──────────────────────┐     │            │          │
│  │ llama       │  │ llama-swappable      │     │            │          │
│  │ -permanent  │  │ :11434  router mode  │     │            │          │
│  │ :11435      │  │ --models-max 1       │     │            │          │
│  └─────────────┘  └──────────────────────┘     │            │          │
│                                                 │            │          │
│                               ┌─────────────────┘     ┌──────┴───────┐ │
│                               │ video-processing      │ lora-trainer │ │
│                               │ (host job, not a      │ preemptible  │ │
│                               │  container)           │ supervisor + │ │
│                               └───────────────────────│ worker       │ │
│                                                       └──────────────┘ │
│   CPU-only: rag-service ─▶ chromadb    history-service ─▶ DCE          │
│   Observability: prometheus :9091 ─▶ grafana :3002                     │
└────────────────────────────────────────────────────────────────────────┘
        ▲                     ▲                    ▲
    VS Code /             Open WebUI           Discord Bot
    Cursor                :3000 (SSO)          /mimic /chat /lore
```

### 4a. Inference backend

Two `llama-server` instances share `NVIDIA_VISIBLE_DEVICES=0`:

| Instance | Port | Role | Config |
|---|---|---|---|
| `llama-permanent` | `:11435` | Autocomplete — loaded at startup, never evicted | `--model` flag in `docker-compose.yml`; `autocomplete` profile |
| `llama-swappable` | `:11434` | Everything else — router mode | `--models-preset /models.ini --models-max 1` |

The router loads a model on first request and evicts the previous one; there is no explicit load API to call. The proxy enforces that only one swappable request is in flight at a time via `asyncio.Lock`.

### 4b. GPU arbitration

**The problem.** One card, several tenants: llama-server, a host video-processing job, and a LoRA trainer. CUDA has no memory preemption, so something must arbitrate — and a containerised proxy has no way to verify that a job which claimed to release actually did.

**The previous design did it by asking.** Each job was told "please release" over HTTP and replied "done", and the proxy believed it. That produced roughly 2,900 lines: a three-state handover machine, stale-registration sweeps, a 180s timeout, a 409 refusal path, 1/sec polling from every client, and a client library vendored into three repos.

**What replaced it is a lease.** A job asks; it is granted or refused; when something that outranks it asks, it is told and gives the card back. There is no registration table, no parking state, no polling, and no shared client library.

```
acquire(job)   reclaim from everything the caller outranks, wait for the driver,
               grant the lease
release(job)   the caller is done; the lease is free
reap()         housekeeping — clear a lease whose holder died without releasing
```

**It is an admission controller, not a scheduler.** It starts nothing. Work is submitted by whoever owns it; the arbiter only answers *who may hold the card while it runs*, which is the one question that genuinely needs a central answer.

**There is no privileged tenant.** The LLM is a row in `jobs.yaml` like everything else — it wins because its priority is 100, and nothing in the codebase contains the string `llm` as a special case. Two facts that a boolean would have fused are kept independent: how important a job is (`priority`) and how controllable it is (`kind`).

| job | priority | kind | reclaim mechanism |
|---|---|---|---|
| `llm` | 100 | `none` | drop the lease; the proxy already starts/stops llama-server for model swaps, and two services driving one container would race |
| `profile-segment` | 50 | `none` | hand-run pipeline profiler |
| `video-processing` | 0 | `cooperative` | holds a blocking `GET /gpu/reclaim-notice`; stops its own worker when woken |
| `lora-trainer` | −10 | `cooperative` | SIGKILLs its own training process; the container survives |

An unknown caller name gets priority 0 and is logged loudly — permissive on purpose, because the alternative is a YAML typo taking the LLM offline with 503s.

**Why `cooperative` for both real tenants:** in each case the process holding the CUDA context is not a process the arbiter can reach. For the video job it is a child of a host daemon. For the trainer it is a child of the container's PID 1 — and killing the container instead does not work, because Docker suppresses restart policies for any API-initiated stop or kill, so a preempted run would never come back (measured: `RestartCount=0` after `docker kill` for both `always` and `on-failure`). The cost is verification: a cooperative job asserts it let go and nothing can check. That is bounded by a timeout, and a timeout is a refusal.

**Privilege.** The arbiter reaches Docker through `docker-socket-proxy` rather than the real socket. Be clear about what that buys: `CONTAINERS` + `POST` still admits `/containers/create`, and a container created with `Binds: ["/:/host"]` is root on the host. The proxy narrows the *surface*, not the *privilege* — this service is root-equivalent either way and must stay on the internal compose network. Its GPU access is `NVIDIA_DRIVER_CAPABILITIES=utility`: NVML and `nvidia-smi`, no CUDA, so the component deciding whether anyone else has room is structurally incapable of allocating any itself.

Full reasoning, including the two non-obvious invariants ("an empty lease table is not headroom", "a running container is not a tenant"), is in [`arbiter/README.md`](arbiter/README.md).

---

## 5. Model Registry

All swappable models are defined in [`models.ini`](models.ini). The proxy references models by alias only — it does not know or care what GGUF is behind them, which is what makes swapping a prototype persona for a LoRA-merged model a config edit rather than a code change.

**Currently enabled aliases:**

| Alias | GGUF | Purpose |
|---|---|---|
| `brain` | Qwen3.6-35B-A3B UD-IQ4_NL | Coding assistant, 128k ctx |
| `brain-dense` | Qwen3.6-27B Q4_K_M | Coding assistant, 96k ctx |
| `brain-dense-heretic` | Qwen3.6-27B Heretic Q4_K_M | `/lore` agent orchestrator |
| `gemma-brain-dense` | gemma-4-31B-it Q4_K_M | Coding assistant, 64k ctx |
| `lore` | gemma-4-26B-A4B-it UD-Q4_K_XL | Direct lore routing |
| `chat-liberal` | gemma-4-26B-A4B-it UD-Q4_K_XL | Open WebUI |
| `chat-chinese` | Qwen3.6-35B-A3B UD-IQ4_NL | Open WebUI |
| `bomb-you` | Qwen3.5-35B-A3B-Uncensored IQ4_XS | Open WebUI, uncensored |

**Commented out:** `mimic_user1`–`mimic_user6` and `image-caption`. Both are still referenced elsewhere — in `proxy/config.py`'s `SWAPPABLE_MODELS`, in `system_prompts.ini`, and by the bot's `/mimic` autocomplete — so those paths reach llama-server and find nothing. Re-enable the sections before using `/mimic` or image captioning.

### 5.1 Configuration is not single-source

Model identity is currently spread across `models.ini` (presets), `docker-compose.yml` (permanent slot), `proxy/config.py` (`SWAPPABLE_MODELS`), `discord-bot/config.py` (`MIMIC_PERSONAS`, `AGENT_MODEL`), `system_prompts.ini`, and `scripts/download_models.py`. These drift — the sets above disagree in both directions today. Consolidating them is tracked in [`TODO.md`](TODO.md).

Mitigating this: `SWAPPABLE_MODELS` is advisory rather than a gate. The proxy forwards any non-autocomplete model name to the swappable slot and lets llama-server decide, so a preset missing from that set still works. The set only affects swap logging.

### 5.2 Adding a mimic persona

1. Add (or uncomment) a `[mimic_<member>]` section in `models.ini` — all personas share one GGUF, so no download is needed.
2. Add the persona to `MIMIC_PERSONAS` in [`discord-bot/config.py`](discord-bot/config.py); the system prompt is generated from `MIMIC_PROMPT_TEMPLATE`.
3. `make restart-llama-swappable` and `make restart-bot`.

---

## 6. Proxy Behaviour

The proxy is **model-aware** but **content-blind**. It knows which model is loaded and serialises access; it never inspects or modifies request content.

### 6.1 Routing

```python
model = extract_model_from_body(request)

if model in AUTOCOMPLETE_MODELS:          # fast path — no lock, no arbiter
    return forward(LLAMA_PERMANENT)       # 503 if AUTOCOMPLETE_ENABLED is false

if model is not None:                     # every other named model
    granted = arbiter.acquire()           # blocks until junior tenants are gone
    if not granted:
        return 503, Retry-After: 30
    async with state.lock:                # one swappable request at a time
        return forward(LLAMA_SWAPPABLE, on_complete=release_if_idle)

return forward(LLAMA_SWAPPABLE)           # bodyless requests, e.g. GET /v1/models
```

`arbiter.release()` fires only when in-flight count and queue depth both reach zero — not after each request — so a burst of Discord traffic does not hand the card away between messages.

### 6.2 Idle eviction

llama-server keeps a model resident indefinitely once loaded, so without intervention a background job would never see a VRAM window on a busy day. After `IDLE_EVICT_SECONDS` (default 600) with no LLM request, the proxy asks the router to unload and then **verifies** the unload by re-reading `/v1/models` — the point is to guarantee free memory, not to trust a response.

This lives in the proxy rather than the arbiter deliberately: nothing else may unload llama-server's models, and moving it would give two services authority over one container's state.

### 6.3 What the proxy does not do

**It does not inject system prompts.** `proxy/system_prompts.py` and `system_prompts.ini` exist and the INI is bind-mounted, but `main.py` never imports the loader — the code is inert. Every system prompt in use is constructed client-side: mimic prompts from `MIMIC_PROMPT_TEMPLATE` in the bot's config, the `/lore` agent's prompts in `agent_tools.py`, and Open WebUI's per-model prompts in its own UI. `/chat` threads send no system prompt at all.

Either delete the dead loader or wire it up; leaving it in place invites edits to `system_prompts.ini` that silently do nothing.

**It does not report GPU state.** `/status` returns proxy state only — current model, queue depth, in-flight count, model age. What is on the card and why is `/gpu/status` on the arbiter. Asking two services the same question is how they end up disagreeing.

**It no longer stores job history.** `/history` and `/history/summary` are stubs returning `[]`, kept for Homepage compatibility. That data moved to Prometheus (§11).

---

## 7. Container Layout

See [`docker-compose.yml`](docker-compose.yml) for full definitions.

| Service | Image / build | Port | Profile | Notes |
|---|---|---|---|---|
| `llama-swappable` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | `:11434` | — | Router mode, `--models-max 1`, `--jinja` |
| `llama-permanent` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | `:11435` | `autocomplete` | Single model, `--parallel 3` |
| `proxy` | `./proxy` | `:11436` | — | No GPU access at all |
| `arbiter` | `./arbiter` | `127.0.0.1:11438` | — | NVML only (`utility`), no CUDA |
| `history-service` | `./history-service` | `127.0.0.1:11437` | — | Background; APScheduler |
| `discord-bot` | `./discord-bot` | — | — | Depends on proxy + rag-service |
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | `:3000` | — | SQLite built-in; authentik SSO |
| `rag-service` | `./rag` | — | — | CPU-only |
| `chromadb` | `chromadb/chroma:latest` | — | — | Vector store |
| `prometheus` | `prom/prometheus:latest` | `:9091` | — | 9090 was taken |
| `grafana` | `grafana/grafana:latest` | `:3002` | — | 3000/3001 taken; authentik SSO |
| `docker-socket-proxy` | `tecnativa/docker-socket-proxy` | `127.0.0.1:2375` | — | Restricted Docker API |
| `lora-trainer` | `./lora-training` | — | `managed` | Never auto-starts |
| `discord-chat-exporter` | `tyrrrz/discordchatexporter` | — | `manual` | One-shot CLI |

**Profiles are load-bearing, not organisational.** `lora-trainer` sits behind `managed` specifically so a plain `docker compose up` skips it — otherwise it would take 16 GB behind the arbiter's back on every restart. `discord-chat-exporter` is behind `manual` because it is a one-shot CLI, not a service. `llama-permanent` is behind `autocomplete` so the toggle can start and stop it independently.

**Model files** live on the host at `MODELS_DIR` (default `/srv/models`) and are bind-mounted read-only into both llama-server containers. They are not in named volumes, so `make nuke` does not delete downloaded weights.

**Absolute host paths.** Several configs are bind-mounted by absolute path under `/home/peacow/local-ai-server` (`models.ini`, `system_prompts.ini`, `arbiter/jobs.yaml`, `prometheus.yml`, `dce-compose.yml`, `dce.env`). Deploying elsewhere means editing those.

### 7.1 Surviving driver upgrades

The NVIDIA container runtime resolves driver libraries at container *create* time and bakes their exact versioned paths into the OCI mount spec. When unattended-upgrades replaces the driver, those paths vanish and every GPU container created beforehand fails at init.

`restart: unless-stopped` does not rescue this — restart policies apply to container *exits*, and this is a failure during init, so Docker gives up and the container stays down. The stack silently stayed dead for four days after the 2026-07-22 reboot for exactly this reason.

[`systemd/local-ai-server.service`](systemd/local-ai-server.service) tries a normal `up` and falls back to `--force-recreate`, which regenerates the mount spec against the installed driver. `systemd/install.sh` also pins the driver against unattended-upgrades, on the grounds that driver upgrades on a GPU box should be deliberate.

---

## 8. Discord Request Flows

The bot exposes three slash commands. Mention-based routing was removed in favour of slash commands (see [`DiscordBot-Design.md`](DiscordBot-Design.md) §1), and compound lore+mimic chains are not supported.

### 8.1 `/mimic persona message`

```
/mimic persona:mimic_user3 message:rate my strats
  → rate limit check (5/user/min)
  → bot builds messages: [system: persona prompt] + history + user
  → proxy: arbiter.acquire → lock → llama-swappable
  → router loads the persona GGUF if not current
  → typing indicator held throughout
  → response: disclaimer patterns stripped, history updated
```

Blocked today: the persona presets are commented out in `models.ini` (§5).

### 8.2 `/chat model` — thread-based conversation

Creates a public thread bound to a model alias. Every subsequent message in that thread is streamed to that model, with no system prompt and no slash command. The model list is autocompleted live from the proxy's `/v1/models`, so any alias llama-server knows about is selectable — including ones absent from `SWAPPABLE_MODELS`.

Messages are labelled with the sender's display name (`[Alice]: …`) so the model can track who is speaking when several people share a thread. Responses stream and are split at sentence boundaries to fit Discord's 2000-character limit.

Thread bindings persist to `discord-bot/data/threads.json` and are restored on startup; threads that were archived or deleted while the bot was down are reconciled then.

This is the reason the bot needs the `MESSAGE_CONTENT` privileged intent — slash commands alone would not.

### 8.3 `/lore question [rounds]` — agentic RAG

Not a single retrieval. The agent (`AGENT_MODEL` = `brain-dense-heretic`) runs a tool-calling loop against the RAG service, deciding for itself what to search and when it has enough:

```
question
  → agent receives TOOLS + server-specific background (lore_context.md)
  → loop, up to `rounds` iterations (default 10, hard cap 25):
      search_discord_history(query, start_date?, end_date?, top_k=10)
      search_channel_history(query, channel_name, …)
      summarize_channel(channel_name, …, top_k=20)
    each call → rag-service /retrieve → ChromaDB → chunks returned as tool output
  → synthesis prompt → final answer → Discord embeds
```

Queue depth is checked before starting; per-user rate limiting is deliberately *not* applied, because a lore query is long-running and multi-round, and the rate limiter's assumptions do not fit it.

`AgentMetrics` records rounds, tool calls, and per-stage latency for each run and logs a summary line.

---

## 9. RAG Pipeline

**Stack:** ChromaDB + `ibm-granite/granite-embedding-311m-multilingual-r2` embeddings, both CPU-only. Zero VRAM impact — this is why the lore path can run while the GPU is busy with something else.

### 9.1 Ingestion

```
per-user JSONL archive → conversation chunking → embed → ChromaDB collection "lore"
```

Chunking is **temporal, not per-message** ([`rag/ingest.py`](rag/ingest.py)): messages are grouped into conversations, a new chunk starts when the channel changes or after a 60-minute gap in the conversation, and chunks above a 512-token soft cap are split at a message boundary. Grouping this way keeps a joke and its punchline in the same chunk, which per-message embedding would separate.

Chunk metadata:

```python
{
    "chunk_id": str,
    "channel_id": str,
    "channel_name": str,
    "timestamp_start": int,   # Unix epoch
    "timestamp_end": int,     # Unix epoch
    "authors": str,           # comma-joined
    "message_count": str,
    "message_ids": str,       # comma-joined
}
```

Timestamps are epoch **integers**, not ISO strings, so that ChromaDB's `$gte`/`$lte` operators work — the agent's date-range filters depend on it. ChromaDB type-infers a field on first insert, so a collection that ever saw a string timestamp will raise `ValueError` on numeric comparison forever after; changing this means re-ingesting from scratch.

Ingestion is idempotent (upsert by `chunk_id`) and triggered with `make rag-ingest`.

### 9.2 Retrieval

Retrieval is exposed as `POST /retrieve` on the RAG service and driven by the agent loop (§8.3) rather than being a single prepended block of context. The service supports channel and date-range filtering, which is what makes the agent's second and third tools meaningful.

---

## 9a. History Pipeline (`history-service`)

A background maintenance process, not in the inference hot path.

### 9a.1 Three-tier data architecture

| Tier | Location | Format | Owner |
|---|---|---|---|
| **1: Raw DCE exports** | `/mnt/storage_cold/array/DiscordArchive/raw/` | DCE native JSON | `discord-chat-exporter` |
| **2: Per-user JSONL archive** | `…/DiscordArchive/archive/` | JSONL, one line per message | `history-service` |
| **3: Filtered training dataset** | not built | JSONL chat format | training pipeline |

**Key design principle:** raw data is preserved and Tier 2 is **unfiltered** — every message is retained regardless of length or content. Filtering belongs at training time, so that filtering rules can be revised without re-pulling from Discord. Tier 3 does not exist yet; nothing currently consumes Tier 2 for training.

### 9a.2 Data flow

```
make dce-evaluate  →  POST /evaluate on :11437
    → read channel_state.json for last export per channel
    → for each channel with activity since: invoke DCE via the Docker socket proxy
    → DCE writes raw JSON to Tier 1
    → parse and merge into per-user JSONL (Tier 2), deduplicated by message_id
    → update channel_state.json
```

Endpoints: `GET /health`, `GET /status`, `POST /evaluate`, `POST /reparse` (rebuild Tier 2 from existing Tier 1 without re-pulling), `POST /clear`, `GET /archive/{user_id}/count`.

### 9a.3 Retraining is not wired up

Earlier revisions of this document described a three-path training trigger (threshold check, training-window scheduler, force-all) backed by `training_trigger.py`, `training_state.py`, and `llama_registrar.py`. **None of those files exist.** No `training_state.json` is written, and there is no training window.

What remains is a vestige: after a merge, `main.py` calls `_notify_lora_training()`, which POSTs to `http://lora-training:11438/notify` — a service that does not exist, on a port belonging to the arbiter. It fails and logs a warning.

Training is started by a human running `make train-submit` and by nothing else (§10). Either delete `_notify_lora_training()` or point it somewhere real.

### 9a.4 Why not merge into the RAG service

The RAG service is a **live inference dependency** — the bot calls it synchronously during every `/lore` round, so it must stay fast and available. The history service is a **background process** that shells out to Docker, performs heavy I/O, and runs on a schedule. Keeping them separate means RAG can restart without interrupting collection, and an export failure cannot affect lore retrieval.

---

## 9b. Image Captioning

Discord messages are full of memes and screenshots. Without captions that content is invisible to lore retrieval, which can only index text.

[`history-service/image_captioner.py`](history-service/image_captioner.py) scans Tier 2 for attachments with `caption_status == "pending"`, downloads each below `IMAGE_CAPTION_MAX_FILE_SIZE_MB`, and sends it base64-inlined to the proxy as an OpenAI-format image message under the `image-caption` alias. Results are written back into the JSONL record.

`caption_status` transitions: `pending` → `done`, or `skipped` (download failed, unsupported format, oversized) / `error` (model failure, retried next run).

Every caption carries `caption_excluded_from_training: true`. The distinction it encodes: a caption is **read-only context**. RAG ingestion appends it to the message's effective text so image content is searchable; a future training exporter must never treat it as training data, because a synthetic description is not the user's voice and would degrade persona quality.

**Status:** disabled by default (`IMAGE_CAPTION_ENABLED=false`), and the `image-caption` preset is commented out in `models.ini` — so enabling the flag alone is not sufficient. The captioner runs on an APScheduler job inside a configurable window (default 3–6 AM) and checks proxy queue depth before each batch.

Note that captioning is a *GPU* consumer using the swappable slot at ~18 GB. It competes under the same lock and the same arbiter lease as everything else.

---

## 10. LoRA Training

Fine-tuning that waits for the GPU, takes it when nothing else wants it, and is killed the moment something does. Full reasoning in [`lora-training/README.md`](lora-training/README.md).

**A human starts a run and nothing else does:**

```bash
make train-submit CONFIG=configs/smoke.yaml
```

**Two processes, and the split is forced.** `supervisor.py` is PID 1 and holds the arbiter lease; `train_worker.py` is its child and holds the CUDA context. A CUDA context is freed when its process exits, so preemption means killing the training process — and if that were PID 1 it would kill the container, which Docker would not restart (§4b). The container therefore outlives every handover and the worker does not.

**Being killed is the normal path.** `stop_timeout: 0` — SIGKILL, no grace period. Unsaved work is discarded either way, so a graceful shutdown would buy nothing and spend the only budget that matters: the LLM's time to first token.

**What makes that safe is checkpoint discipline.** A checkpoint counts only once a `COMPLETE` marker is written beside it, after the save returns. A killed save leaves a directory with no marker, which discovery skips and pruning deletes. Cadence is **wall-clock, not steps** — the quantity to bound is lost work, and the same `save_steps` that means four minutes on a 0.6B means over an hour on a 35B.

**The window this actually gets.** llama-server holds ~20 GB for as long as a model is resident, leaving ~4.4 GB — under the trainer's `required_mb`. So a submitted run only gets the card after the proxy's idle evictor has unloaded the model, i.e. 600s with no LLM traffic, and any request restarts that clock. Training happens in long idle stretches; a submitted run may sit waiting for hours. That is the design working.

`required_mb` must be the *peak* need, not the steady state — the first hardware run OOMed at 4478 MiB free against a declared 4096.

**Not built:** per-persona mimic training. `configs/smoke.yaml` (Qwen3-0.6B) exists to validate the handover in minutes. A mimic config plus a `merge.py` (adapter merge → GGUF export → `models.ini` registration) is the remaining work, and training a model on per-user message history has not been agreed as a decision — it is inherited from an earlier design sketch, not chosen.

---

## 11. Observability

Prometheus (`:9091`) scrapes the proxy and a glances exporter every 5s; Grafana (`:3002`) visualises. This replaced an in-proxy job-history store, whose endpoints remain as `[]` stubs.

Proxy metrics: `proxy_requests_total{model,status}`, `proxy_active_requests`, `proxy_tokens_total{model,token_type}`, `proxy_queue_depth`, `proxy_request_duration_seconds`, `proxy_current_model_info{model}`, `proxy_model_age_seconds`, `proxy_llm_requests_inflight`, `proxy_llm_idle_seconds`.

GPU metrics come from the arbiter's `/metrics` — it is the only component that reads the card, so there is one source rather than two that can disagree.

---

## 12. Access & Authentication

Open WebUI and Grafana are reachable from outside the LAN and sit behind authentik SSO:

| Service | Public hostname | Authorisation |
|---|---|---|
| Open WebUI | `chat.wganderbox.ca` | OIDC signup enabled, but `DEFAULT_USER_ROLE=pending` — a new account cannot use anything until an admin promotes it |
| Grafana | `grafana.wganderbox.ca` | authentik group `grafana-admins` → Admin, everyone else → Viewer |
| authentik | `auth.wganderbox.ca` | — |

Two details worth preserving:

- **`OAUTH_MERGE_ACCOUNTS_BY_EMAIL=false`** on Open WebUI. It was enabled for exactly one login so the SSO identity could adopt the pre-existing local admin account with the same address; open-webui then wrote authentik's `sub` into the user record and matches on it directly. Left on, a self-signup or federated source in authentik could claim an existing account by asserting its email.
- **Grafana's local admin password** is a generated value in `.env`, not `admin`. It is break-glass access for when authentik is down.

Everything else — the proxy, arbiter, history-service, ChromaDB — is unauthenticated and bound either to the compose network or to `127.0.0.1`. That is the whole access control model for those services, so exposing any of them means adding authentication first.

---

## 13. Key Inference Parameters

Set per-alias in [`models.ini`](models.ini). Representative values:

| Parameter | `brain` | `brain-dense` | `lore` / `chat-liberal` | `chat-chinese` | `bomb-you` |
|---|---|---|---|---|---|
| temperature | 0.65 | 0.65 | 1.0 | 1.0 | 0.6 |
| top-k | 20 | 20 | 64 | 40 | 40 |
| top-p | 0.95 | 0.95 | 0.95 | 0.95 | 0.9 |
| repeat-penalty | 1.05 | 1.05 | 1.0 | 1.0 | 1.15 |
| ctx-size | 131072 | 96000 | 65536 | 131072 | 32768 |
| n-predict | −1 | −1 | −1 | −1 | −1 |
| cache-type-k/v | q8_0 | q8_0 | q8_0 | q8_0 | q8_0 |

Global flags on the swappable server: `-fa on --parallel 1 --jinja --models-max 1`.

The Qwen3.6 presets set `chat-template-kwargs = {"preserve_thinking": true}` rather than suppressing reasoning — a reversal from v3.0, which disabled chain-of-thought everywhere via `reasoning-format = none`. That flag survives only on `bomb-you`, where snappy replies matter more than reasoning quality.

Note that the parameters in this file are the *server's* defaults for an alias; a client may override sampling per request. The `/lore` agent does exactly that, forcing `temperature=0.1` for deterministic tool-calling.

---

## 14. Risk Table

| Risk | Likelihood | Mitigation |
|---|---|---|
| A cooperative job lies about releasing the card | Low | Bounded by timeout; a timeout is a refusal. Genuinely unverifiable — the known cost of not holding systemd's user manager |
| `required_mb` understates a job's peak, causing OOM | Medium | Declare peak, not steady state; already caused one failure at 4478 MiB free vs 4096 declared |
| Config drift between `models.ini`, `proxy/config.py`, and the bot | **Occurring now** | `SWAPPABLE_MODELS` is advisory so drift degrades rather than breaks; consolidation tracked in `TODO.md` |
| Driver upgrade kills every GPU container | Medium | systemd unit recreates on failed start; unattended-upgrades pinned. Has happened once, cost four days |
| A training run never gets the card | Medium | By design — it needs a 600s idle window. `make train-status` reports holder and free VRAM before suspecting a fault |
| Baked-in disclaimer appears in mimic output | Medium | `DISCLAIMER_PATTERNS` regex strip in the bot, plus system prompt instruction |
| ChromaDB returns irrelevant lore | Medium | Agentic loop can re-query with different terms rather than being stuck with one retrieval; synthesis prompt instructs admitting ignorance |
| Re-ingesting with string timestamps breaks date filters permanently | Low | Epoch integers enforced in `_build_metadata`; a collection that saw strings must be rebuilt |
| Open WebUI account claimed via email assertion | Low | `OAUTH_MERGE_ACCOUNTS_BY_EMAIL=false`; `DEFAULT_USER_ROLE=pending` as second gate |
| Arbiter compromise is host compromise | Low | Root-equivalent via `CONTAINERS`+`POST`. Must stay on the internal network; tightening the socket proxy, or dropping Docker entirely once `kind: container` has no user, would remove it |
| Uncensored model generates something harmful | Low | Private server, membership controlled by admin; system prompt boundaries |
| llama.cpp regression | Low | Image tag is `:server-cuda` (unpinned) — pinning it is unaddressed |
| Open WebUI history lost | Low | SQLite persists in `open_webui_data`; `make nuke` destroys it |
