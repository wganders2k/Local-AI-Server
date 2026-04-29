
---

# Mimic Bot + Coding Assistant — Revised Design Document v3.0

**Revision:** Inference backend migrated from Ollama to **llama.cpp** (`llama-server`) from Phase 1. llama.cpp's router mode (`--models-preset models.ini`) provides swap-on-demand model loading natively, eliminating the need for Ollama entirely. The OpenAI-compatible API is used throughout. All previous Ollama-specific drawbacks are resolved.

---

## 1. Guiding Principles

- **One physical GPU, zero concurrency.** All VRAM-resident models are sequential. No two non-autocomplete models ever share the swappable slot simultaneously.
- **Autocomplete is sacred.** The 2B autocomplete lives permanently on `:11435` and is never touched by orchestration.
- **Two distinct model personalities.** Mimic personas use an abliterated base with no content refusals. The lore assistant uses a sterile, instruction-following base. Neither bleeds into the other.
- **Swap-friendly by design.** The smaller the Discord model footprint, the faster the swap. Qwen3.5-9B at Q6_K (~7.4 GB) is significantly better than NeMo 12B (~10.5 GB) here.
- **Prototype-first.** Phase 1 uses system prompt personas with no LoRA. LoRA-merged models slot in during Phase 2 with zero orchestration changes.
- **Open WebUI is a first-class consumer.** Open WebUI routes through the same orchestration proxy as Discord and VS Code. When using a local model, it competes for the swappable slot under the same lock. When using Claude via API key, it bypasses the proxy entirely — zero VRAM impact.

---

## 2. Hardware & VRAM Budget

**RTX 3090 — 24 GB VRAM (usable: ~24,300 MB)**

| Slot | Purpose | Model | Quant | VRAM |
|---|---|---|---|---|
| Permanent (:11435) | Autocomplete | `autocomplete` (Qwen3.5-2B) | IQ4_NL | ~1.21 GB |
| Swappable (:11434) | Brain (coding) | `qwen3.5:35b-a3b` | UD-IQ4_NL | ~17.8 GB |
| Swappable (:11434) | Mimic personas (×6) | `Qwen3.5-35B-A3B-Uncensored` | IQ4_XS | ~18 GB |
| Swappable (:11434) | Image captioner | `Qwen3.5-35B-A3B-Uncensored` | IQ4_XS | ~18 GB (shared weights with mimic) |
| Swappable (:11434) | Lore assistant | `gemma3:12b` | Q6_K | ~9.5 GB |
| Swappable (:11434) | Open WebUI local model | `qwen3.5:35b-a3b` | UD-IQ4_NL | ~17.8 GB (shared GGUF with brain) |

**VRAM utilisation by mode:**

| Active Configuration | VRAM Used | Headroom |
|---|---|---|
| Coding (Brain + Autocomplete) | ~19.3 GB | ~5.0 GB |
| Mimic active (+ Autocomplete) | ~8.9 GB | ~15.4 GB |
| Lore active (+ Autocomplete) | ~11.0 GB | ~13.3 GB |
| Coding KV cache at 32k ctx (brain ctx-size=32768) | ~20.9 GB | ~3.4 GB |
| Open WebUI local (+ Autocomplete) | ~19.3 GB | ~5.0 GB |
| Open WebUI via Claude API | ~1.5 GB (autocomplete only) | ~22.8 GB |

> **KV Cache:** q8_0 for all models (`--cache-type-k q8_0`). Brain: `--parallel 1`, context 32k–40k. Mimic: `--parallel 1` (router mode; concurrency managed by proxy lock). Lore: `--parallel 1`, context 16k. Open WebUI local: `--parallel 1`, context 16k.

---

## 3. Why Qwen3.5-35B-A3B-Uncensored for Mimics & Image Captioning, Gemma3-12B for Lore

### 3.1 Mimic Base: `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`

The core requirement for mimic personas is **zero refusals on crude, raunchy, or dark humour** — the kind that characterises tight gaming communities. A standard instruction-tuned model will sanitise this behaviour, add disclaimers, and break character at exactly the wrong moment.

This model is Qwen3.5-35B-A3B uncensored by HauhauCS: 0 refusals, fully uncensored with zero capability loss, no changes to datasets or capabilities — fully functional, 100% of what the original authors intended, just without the refusals. The 35B-A3B (Mixture-of-Experts) architecture delivers strong reasoning and personality capture at a VRAM footprint comparable to a dense 9B model, making it an excellent fit for the swappable slot.

The model is fully unlocked and will not refuse prompts. It may occasionally append a short disclaimer (e.g. "This is general information...") baked into base model training — but this is not a refusal; the actual content is always generated in full. This is suppressible with a system prompt instruction: `"Never append disclaimers, caveats, or safety notices of any kind."` and post-processing in the Discord bot.

**VRAM at IQ4_XS: ~18 GB.** The IQ4_XS quant from HauhauCS is an importance-weighted 4-bit quantisation that preserves accuracy at a compact footprint. At ~18 GB it fits on the 3090 with ~5.5 GB headroom alongside the permanent autocomplete slot.

**Thinking mode:** Disabled via `reasoning_format = none` in `models.ini`. Fast, snappy responses are the goal for Discord banter.

**Shared weights for image captioning:** All mimic personas (`mimic_user1` … `mimic_user6`) and `image-caption` reference the same GGUF file in `models.ini`. llama-server loads the GGUF once and serves all aliases from it. Swapping between mimic personas is a context switch, not a model reload.

### 3.2 Lore Assistant Base: `bartowski/gemma-3-12b-it-GGUF` (sterile, unchanged)

The lore assistant needs to be **reliable, structured, and citation-aware** — the opposite of the mimic's job. It handles RAG retrieval over in-jokes, event histories, and member lore, and surfaces this cleanly. Gemma 3 12B excels at instruction following, summarization, and structured output. It never needs to swear, roast anyone, or generate raunchy content. Keeping it on a standard instruction-tuned base means it won't hallucinate lore or inject personality where none belongs.

Gemma 3 12B fits comfortably at Q6_K on the 3090 — ~9.5 GB with 13+ GB headroom when active.

---

## 3a. Open WebUI Model Selection

### 3a.1 Use Case

Open WebUI is a general-purpose personal chat interface — think of it as a self-hosted ChatGPT replacement. It supports multi-turn conversation, system prompts, and multiple model backends. The two viable backends for this setup are:

1. **Claude via Anthropic API key** — zero local VRAM cost, best quality, requires internet + paid API usage.
2. **Local llama-server model via proxy** — fully offline, competes for the swappable slot, free after hardware cost.

### 3a.2 Local Model: `chat` at UD-IQ4_NL (~17.8 GB) — shared GGUF with Brain

The Open WebUI local model uses the **same GGUF as the Brain coding assistant** (`unsloth/Qwen3.5-35B-A3B-GGUF`, `Qwen3.5-35B-A3B-UD-IQ4_NL.gguf`) — no additional download required. The `[chat]` preset in `models.ini` points to the same file with different inference parameters tuned for casual conversation rather than precise code generation.

**Why the same model as Brain:** The 35B-A3B MoE architecture delivers strong conversational quality at a VRAM footprint comparable to a dense 9B model. Since the GGUF is already on disk for Brain, there is zero additional storage cost. The only difference between Brain and Open WebUI chat is the system prompt and sampling parameters — Brain uses tight, deterministic settings (`temperature 0.2`, `top_k 10`) while Open WebUI uses warmer, more natural settings (`temperature 0.75`, `top_k 40`, `repeat_penalty 1.1`).

**Thinking mode:** Disabled via `reasoning_format = none` in `models.ini`. Fast conversational responses are the goal.

**VRAM:** ~17.8 GB — same as Brain. When Open WebUI local is active, the VRAM profile is identical to Brain being loaded. The 16k context window (vs Brain's 40k) keeps KV cache overhead lower for typical chat sessions.

**Claude via API as the preferred option when available:** If you have an Anthropic API key, routing Open WebUI to Claude is the better default for general chat — it offloads all inference to Anthropic's servers, keeps the swappable slot free for Discord/coding, and provides frontier-model quality. The local model is the fallback for offline use or cost control.

### 3a.3 Open WebUI Configuration

Open WebUI is configured entirely via environment variables in `docker-compose.yml`. Two backends are available:
- **Local model:** `OPENAI_API_BASE_URL=http://proxy:11436/v1` (OpenAI-compatible), model alias `chat`
- **Claude:** `ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}` — Open WebUI has built-in Anthropic support; set the key in `.env` to enable it

No config file is required. Open WebUI uses SQLite internally — no MongoDB sidecar needed.

> **Note on thinking mode:** Qwen3.5-35B-A3B supports optional thinking/reasoning mode. For Open WebUI general chat, it is disabled in `models.ini` via `reasoning_format = none` — thinking adds latency and token overhead that is not useful for conversational queries.

---

## 4. Architecture Overview

---

## 4a. Inference Backend: llama.cpp from Day One

### Backend: llama-server (llama.cpp)

`llama-server` is the inference backend for all models from Phase 1. It provides:

- **Router mode** (`--models-preset models.ini`) — swap-on-demand model loading. The router loads a model into VRAM on first request and evicts it when a different model is requested. No explicit load API needed.
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`. All clients (Discord bot, Open WebUI, VS Code) speak standard OpenAI format.
- **Per-model preset config** — `models.ini` defines GGUF path, inference parameters, and alias for each model. Changing a model requires only editing `models.ini` and restarting the server.
- **`reasoning_format = none`** — suppresses Qwen3.5's chain-of-thought tokens cleanly per-model in the preset.
- **`--flash-attn` and `--cache-type-k q8_0`** — applied globally at server startup for all models.
- **Dynamic LoRA adapter loading** — Phase 3 LoRA hot-swapping is supported natively via `--lora` flag with zero base-model reload.

### Why Not vLLM (Hard No)

vLLM is ruled out on two independent grounds:

1. **MoE architecture:** Qwen3.5-35B-A3B is a sparse Mixture-of-Experts model. vLLM's support for quantised GGUF MoEs is unoptimised and highly resource-intensive compared to llama.cpp's tuned MoE-specific CUDA kernels.

2. **GGUF-only availability:** The uncensored mimic base (`HauhauCS/Qwen3.5-35B-A3B-Uncensored`) is only available as GGUF — no safetensors release exists. vLLM's GGUF support is experimental. The entire model stack is GGUF-based (Unsloth quants, HauhauCS uncensored), making vLLM a non-starter.

### Why Not Ollama

Ollama was the original planned backend but is superseded by llama.cpp directly:

- **Ollama wraps llama.cpp** — using llama-server directly eliminates the abstraction layer and gives full control over every inference parameter.
- **Router mode** — llama-server's `--models-preset` provides the same swap-on-demand behaviour Ollama offered, without Ollama's overhead.
- **No Modelfile system needed** — `models.ini` is simpler and more transparent than Ollama Modelfiles.
- **OpenAI API natively** — llama-server speaks OpenAI format directly; Ollama requires translation.
- **Phase 3 LoRA** — llama-server supports dynamic LoRA adapter loading natively. Ollama forces a full base-model VRAM flush when swapping between Modelfile adapters, destroying the latency benefit of LoRA.

### Two-Server Architecture

Two separate `llama-server` instances share `NVIDIA_VISIBLE_DEVICES=0`:

| Instance | Port | Role | Config |
|---|---|---|---|
| `llama-permanent` | `:11435` | Autocomplete — loaded at startup, never evicted | `--model` flag in `docker-compose.yml` |
| `llama-swappable` | `:11434` | All other models — router mode | `--models-preset /models.ini` |

The permanent instance holds exactly one model forever. The swappable instance uses router mode to load/evict models on demand. The proxy enforces that only one swappable model is active at a time via `asyncio.Lock`.

**VRAM safety:** The permanent instance's model is loaded at startup and stays resident. The swappable instance's models are configured in `models.ini` with explicit `n_ctx` and `n_gpu_layers` values that keep total VRAM under ~22 GB, leaving the autocomplete model's ~1.21 GB reservation untouched.

### Model Files

GGUF files live on the host at `/srv/models/<publisher>/<model>/filename.gguf` and are bind-mounted read-only into both containers. Download all GGUFs with:

```bash
make models-download
```

This runs `scripts/download_models.py`, which fetches each GGUF from HuggingFace using `huggingface_hub`. Already-downloaded files are skipped.

### Phase 3: LoRA Hot-Swapping

Phase 3 LoRA persona refinement requires **zero backend migration** — llama-server already supports dynamic LoRA adapter loading. When a LoRA-merged GGUF is ready:

1. Update the `model` path in `models.ini` to point to the merged GGUF
2. Run `make restart-llama-swappable`
3. Bot continues using the same alias — no proxy or bot code changes

The proxy's `_forward()` function is unchanged throughout all phases.

---


```
┌──────────────────────────────────────────────────────────────────┐
│                      Ubuntu Server (RTX 3090)                     │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              FastAPI Orchestration Proxy :11436            │  │
│  │                                                           │  │
│  │  State Machine:                                           │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ current_model: str | None                        │    │  │
│  │  │ lock: asyncio.Lock (one request at a time)       │    │  │
│  │  │ queue_depth: int                                 │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
│               │                          │                       │
│               ▼                          ▼                       │
│  ┌──────────────────┐       ┌───────────────────────────┐      │
│  │ llama-server      │       │   llama-server :11434      │      │
│  │ :11435            │       │   (SWAPPABLE — router)     │      │
│  │ (PERMANENT)       │       │                            │      │
│  │ Qwen3.5-2B        │       │ ← Brain (17.8 GB)          │      │
│  │ IQ4_NL            │       │   OR                       │      │
│  │ ~1.21 GB VRAM     │       │ ← Mimic persona            │      │
│  │ parallel 4        │       │   (Qwen3.5-35B-A3B-        │      │
│  │ always loaded     │       │    Uncensored IQ4_XS       │      │
│  │                   │       │    ~18 GB)                 │      │
│  └──────────────────┘       │   OR                       │      │
│                              │ ← Lore assistant           │      │
│                              │   (gemma3-12b Q6_K         │      │
│                              │    ~9.5 GB)                │      │
│                              │   OR                       │      │
│                              │ ← Open WebUI local         │      │
│                              │   (qwen3.5-35b-a3b UD-IQ4_NL│     │
│                              │    ~17.8 GB, shared GGUF)  │      │
│                              └───────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴──────┐  ┌──────────┴──────┐  ┌─────────┴──────────┐
│ VS Code /     │  │   Open WebUI    │  │   Discord Bot      │
│ Cursor        │  │   :3000         │  │   (discord.py)     │
│ (autocomplete │  │ (local model    │  │                    │
│  + chat)      │  │  via proxy OR   │  │                    │
└───────────────┘  │  Claude API     │  └────────────────────┘
                   │  direct)        │
                   └─────────────────┘
```

> **Open WebUI + Claude API path:** When Open WebUI is configured to use Claude, requests go directly from the Open WebUI container to `api.anthropic.com` — they never touch the proxy or llama-server. The proxy is only in the path when Open WebUI is configured to use the local model (`chat`).

---

## 5. Model Registry

All models are defined in [`models.ini`](models.ini). The proxy references models by alias only — it doesn't know or care what GGUF is behind them. This means swapping the underlying GGUF (e.g. from prototype system-prompt persona to LoRA-merged model) requires only updating the `model` path in `models.ini` and restarting `llama-swappable` — no proxy code changes.

GGUF files are downloaded to `/srv/models/<publisher>/<model>/filename.gguf` via `make models-download`.

### 5.1 Mimic Persona Configuration

Mimic personas are defined as separate `[mimic_<member>]` sections in `models.ini`, all pointing to the same GGUF. The system prompt is injected per-request by the Discord bot (see `DiscordBot-Design.md` §7.1).

To add a new persona:
1. Add a `[mimic_<member>]` section to `models.ini`
2. Add `mimic_<member>` to `SWAPPABLE_MODELS` in `proxy/config.py`
3. Add the persona to `MENTION_TO_MODEL` in the Discord bot's `router.py`
4. Run `make restart-llama-swappable`

### 5.2 Lore Assistant

Defined as `[lore]` in `models.ini`. System prompt injected per-request by the Discord bot.

### 5.3 Brain

Defined as `[brain]` in `models.ini`. Used by VS Code / Cursor chat via the proxy.

### 5.4 Open WebUI Local Chat

Defined as `[chat]` in `models.ini`. Used by Open WebUI when the local model backend is selected.

### 5.5 Image Caption

Defined as `[image-caption]` in `models.ini`. Shares the same GGUF as mimic personas. Used exclusively by the `history-service` image captioner during off-hours batch processing.

---

## 6. Proxy State Machine (Pseudocode)

The proxy is **model-aware** but **content-blind**. It knows what model is loaded and serialises access, but never inspects or modifies request content. Both sides speak OpenAI-compatible API.

```python
class OrchestratorState:
    current_model: str | None = None
    lock: asyncio.Lock = asyncio.Lock()

AUTOCOMPLETE_MODELS = {"autocomplete"}  # always routed to :11435, never swapped
SWAPPABLE_MODELS = {
    "brain",
    "mimic_user1", "mimic_user2", "mimic_user3",
    "mimic_user4", "mimic_user5", "mimic_user6",
    "lore",
    "chat",
    "image-caption",
}

async def route_request(model: str, payload: dict):
    if model in AUTOCOMPLETE_MODELS:
        return await forward(":11435", payload)       # fast path, no lock

    async with state.lock:                            # serialise all swappable requests
        if state.current_model != model:
            state.current_model = model               # llama-server router handles the actual load
        return await forward(":11434", payload)
```

**Swap cost:** llama-server's router evicts the current model and loads the new one on first request. Mimic swaps (Qwen3.5-35B-A3B IQ4_XS, ~18 GB from NVMe) take approximately **5–8 seconds** cold load. Lore swaps (~9.5 GB) take approximately **4–6 seconds**. Open WebUI local model swaps (~17.8 GB, same GGUF as Brain) take approximately **5–8 seconds** — or near-zero if Brain was the last loaded model (same file, no eviction needed). The typing indicator in the Discord bot masks Discord latency; Open WebUI shows a streaming cursor which masks its swap latency.

**Contention between Open WebUI and Discord/VS Code:**
Open WebUI requests queue behind any in-progress Discord or Brain generation under the same lock. Since Open WebUI is personal/interactive use, the user is already expecting a short wait. If Open WebUI is actively in a long conversation while a Discord request arrives, the Discord request queues — the same behaviour as Brain contention. This is acceptable for single-user personal use.

---

## 7. Docker Compose Layout

> See [`docker-compose.yml`](docker-compose.yml) for the full service definitions. The key services are:
>
> | Service | Image / Build | Port | Notes |
> |---|---|---|---|
> | `llama-permanent` | `ghcr.io/ggerganov/llama.cpp:server-cuda` | `:11435` | Autocomplete model, `--parallel 4`, loaded at startup |
> | `llama-swappable` | `ghcr.io/ggerganov/llama.cpp:server-cuda` | `:11434` | All other models, router mode via `--models-preset /models.ini` |
> | `proxy` | `./proxy` | `:11436` | FastAPI orchestration proxy |
> | `discord-bot` | `./discord-bot` | — | Depends on proxy + chromadb |
> | `open-webui` | `ghcr.io/open-webui/open-webui:main` | `:3000` | No sidecar needed — SQLite built-in |
> | `rag-service` | `./rag` | — | Depends on chromadb |
> | `chromadb` | `chromadb/chroma:latest` | — | Vector store |
> | `history-service` | `./history-service` | — | Background; set `TRAINING_TRIGGER_ENABLED=true` in Phase 3 |

> **Why two separate llama-server instances share the same `NVIDIA_VISIBLE_DEVICES=0`?** llama-server manages its own VRAM allocation. Both instances can reference the same GPU — the proxy enforces that only one swappable model is loaded at a time. The permanent instance holds exactly one model forever. VRAM safety is maintained by careful `n_ctx` configuration in `models.ini` rather than a hard cap env var.

> **Model files:** GGUF files are stored on the host at `MODELS_DIR` (default: `/srv/models`) and bind-mounted read-only into both llama-server containers. They are **not** stored in named Docker volumes — `make nuke` does not delete downloaded model weights.

> **Open WebUI configuration:** Configured entirely via environment variables in `docker-compose.yml`. Two backends: local model via `OPENAI_API_BASE_URL=http://proxy:11436/v1` (model alias `chat`), and Claude via `ANTHROPIC_API_KEY`. No config file or MongoDB sidecar required — Open WebUI uses SQLite internally. All data persists in the `open_webui_data` named volume.

---

## 8. Discord Bot Request Flow

### 8.1 Simple Mimic Request
```
User: @mimic_user3 rate my strats
Bot:  [acquires proxy lock]
      [llama-server router loads mimic_user3 if not current: ~5–8s]
      [typing indicator active throughout]
      [inference: ~1–2s at ~40 tok/s on 3090]
      [releases lock]
      Bot: "lmao those aren't strats, that's just dying slower"
```

### 8.2 Lore + Mimic Chain (Sequential)
```
User: @lore what did user3 say at the tournament last year?
                        + @mimic_user3 react to this

Step 1: RAG lookup (CPU/RAM, ~0.5s, no GPU needed)
        → Retrieved: [lore chunk about tournament incident]

Step 2: [acquires proxy lock]
        [llama-server router loads lore: ~5s]
        [typing indicator active]
        [lore inference with RAG context: ~3s]
        [releases lock]
        Lore output: "At the Spring 2024 tourney, user3 SD'd three times..."

Step 3: [acquires proxy lock]
        [llama-server router loads mimic_user3: ~5s]
        [inject lore output as context]
        [mimic inference: ~1.5s]
        [releases lock]
        Mimic output: "i had one job. ONE JOB."

Total wall time: ~14s, masked by typing indicator throughout.
```

### 8.3 Coding Assistant (Uninterrupted)
```
Developer: [asks Brain a question in VS Code chat]
           [acquires proxy lock]
           [llama-server router loads brain if not current: ~8s from NVMe]
           [inference at 32k+ context: ~variable]
           [releases lock — Brain stays loaded until Discord request evicts it]
```

### 8.4 Open WebUI Local Chat Request
```
User: [sends message in Open WebUI with local model selected]
      [Open WebUI sends OpenAI-format request to proxy :11436 with model: chat]
      [acquires proxy lock]
      [llama-server router loads chat if not current: ~6s]
      [streaming inference begins — Open WebUI streams tokens to browser]
      [releases lock after full response]

Note: If Claude API backend is selected in Open WebUI, this entire flow is bypassed.
      The request goes directly to api.anthropic.com — proxy and llama-server are not involved.
```

> **Contention note:** If a Discord request arrives while Brain is active, it queues behind the Brain's current generation. Brain is never evicted mid-response. Discord users see a typing indicator and wait. This is acceptable for a hobby bot — if concurrent throughput becomes a real need, a second GPU eliminates the problem entirely. Open WebUI contention follows the same rules: an in-progress Open WebUI generation will not be interrupted by a Discord request.

---

## 9. RAG Pipeline (Lore Assistant)

**Stack:** ChromaDB (CPU/RAM, no VRAM impact) + `sentence-transformers/all-MiniLM-L6-v2` embeddings (also CPU).

**Ingestion:**
```
Discord history export → chunker → embedding → ChromaDB collection: "lore"
```

**Chunk strategy:**
- Messages grouped by conversation thread (max 512 tokens per chunk)
- Metadata: `{author, timestamp, channel, topic_tags}`
- Re-embed on new significant events/memes (manual trigger or weekly cron)

**Retrieval at inference time:**
```python
def build_lore_context(query: str, top_k: int = 5) -> str:
    results = chroma.query(query_texts=[query], n_results=top_k)
    return "\n\n".join(results["documents"][0])
```

This context is prepended to the lore assistant's user message before forwarding to the proxy. Zero VRAM overhead — all CPU-side.

> **Note:** Raw Discord message history (JSONL per user) is managed by the `history-service`, not the RAG service. The RAG service consumes the JSONL data for lore ingestion into ChromaDB, but the collection and maintenance of that history is a separate concern. See §9a.

---

## 9a. History & Training Pipeline (`history-service`)

The `history-service` is a background maintenance process — it is **not** in the inference hot path. It runs on a schedule alongside the RAG service and handles two distinct jobs:

### 9a.1 Three-Tier Data Architecture

The history pipeline uses a **three-tier data architecture** that separates raw exports from processed archives and filtered training datasets. This design allows iterating on filtering methods without re-pulling data from Discord.

| Tier | Location | Format | Owner | Description |
|---|---|---|---|---|
| **Tier 1: Raw DCE Exports** | `/mnt/storage_cold/array/DiscordArchive/raw/` | DCE native JSON | `discord-chat-exporter` | Unmodified output from DiscordChatExporter. One directory per export run. |
| **Tier 2: Per-User JSONL Archive** | `/mnt/storage_cold/array/DiscordArchive/archive/` | JSONL (one line per message) | `history-service` | Normalized, per-user message archive. **Unfiltered** — all messages retained. |
| **Tier 3: Filtered Training Dataset** | Separate service (not `history-service`) | JSONL chat format | Training pipeline | Filtered subset of Tier 2 for LoRA training. Filtering is **NOT** a responsibility of history-service. |

**Key design principle:** Raw data is preserved in the archive. Filtering logic is applied at training time, not at ingest. This allows experimenting with different filtering methods without re-pulling data from Discord.

### 9a.2 Data Flow

```
Host cron (monthly)
    ↓
POST /evaluate (history-service :11437)
    ↓
history-service evaluates channel state (state/channel_state.json)
    ↓
history-service invokes DCE via `docker compose run` (subprocess)
    ↓
DCE writes raw JSON → /mnt/storage_cold/array/DiscordArchive/raw/
    ↓
history-service merges raw exports → /mnt/storage_cold/array/DiscordArchive/archive/<user_id>.jsonl
```

**Pull strategy:**
- **Monthly cron** pings history-service to evaluate all channels
- history-service checks `channel_state.json` for last export timestamp per channel
- If no record exists (new channel), export the entire channel
- If recent activity detected since last export, export the date range since last export
- Append new messages to per-user JSONL archive (deduplicated by `message_id`)

### 9a.3 DiscordChatExporter Integration

DCE (`tyrrrz/discordchatexporter`) runs as a **one-shot container** via `docker compose run` and is invoked by history-service as a subprocess. The service definition uses `profiles: ["manual"]` to prevent auto-start.

**Invocation patterns:**

```bash
# Full guild export
docker compose run --rm discord-chat-exporter exportguild --guild <guild_id> --format Json

# Single channel, date range (incremental pull)
docker compose run --rm discord-chat-exporter export --channel <channel_id> --after 2025-01-01 --before 2025-04-01 --format Json

# Full channel export (new channel, no date range)
docker compose run --rm discord-chat-exporter export --channel <channel_id> --format Json
```

**Storage:** Raw exports land on the cold storage array via bind mount (`/mnt/storage_cold/array/DiscordArchive/raw:/out`), surviving `make nuke`.

### 9a.4 LoRA Retraining Trigger

Retraining is triggered via three distinct paths, each handled by `training_trigger.py`:

| Trigger | Caller | What it does |
|---|---|---|
| **Threshold check** | `main.py` after each DCE merge | Increments `messages_since_last_train`; sets `status = "queued"` for any user who hits `RETRAIN_THRESHOLD`. Does **not** dispatch training — only updates state. |
| **Training window scheduler** | `main.py` APScheduler job (every 5 min, 3–6 AM only) | Calls `training_trigger.dispatch_queued()` — scans for `queued` users and dispatches training if the proxy queue depth is zero. Retries automatically at the next tick if the proxy is busy. |
| **Force-all (manual)** | `make` → `python training_trigger.py --force-all` | Sets **all** users to `queued` regardless of threshold (used when the mimic base model changes and all LoRA adapters must be retrained from scratch). Training still dispatches via the training window scheduler — `--force-all` only updates state. |

**Training coordination:**
- Training is only dispatched during the configured training window (default: 3–6 AM) to avoid inference contention
- Before dispatching, `dispatch_queued()` checks that the proxy queue depth is zero
- The swappable llama-server slot is explicitly stopped before training begins (QLoRA requires ~14–16 GB VRAM)
- Training runs as a subprocess calling `lora-training/train.py` then `lora-training/merge.py`
- After the GGUF merge, the service updates `models.ini` with the new GGUF path and restarts `llama-swappable` — zero bot or proxy changes required

**Training state** is tracked in `/mnt/storage_cold/array/DiscordArchive/state/training_state.json` per user:

```json
{
  "user3": {
    "user_id": "987654321098765432",
    "total_clean_messages": 1247,
    "messages_since_last_train": 43,
    "last_trained_at": "2026-03-01T03:00:00Z",
    "model_version": 2,
    "training_status": "idle"
  }
}
```

`training_status` progresses through: `idle` → `queued` → `training` → `merging` → `registering` → `idle`. On failure, status is set to `error` and requires manual intervention.

### 9a.5 Relationship to RAG Service

| Concern | Owner |
|---|---|
| Invoking DCE exports | `history-service` |
| Merging raw DCE output into per-user JSONL archive | `history-service` |
| Channel state tracking (last export per channel) | `history-service` |
| Captioning image attachments in JSONL records | `history-service` |
| Triggering LoRA retraining | `history-service` |
| Filtering archive into training dataset | **Separate service** (NOT `history-service`) |
| Chunking messages for lore retrieval | `rag-service` |
| Embedding chunks into ChromaDB | `rag-service` |
| Serving retrieval queries at inference time | `rag-service` |

The RAG service reads from the JSONL archive files produced by the history service for its lore ingestion pipeline. The two services are decoupled — the RAG service does not depend on the history service being running.

---

## 9b. Image Captioning Pipeline (`history-service`)

The `history-service` includes a background image captioning step that enriches JSONL records with natural-language descriptions of Discord image attachments. This is an extension of the history collection pipeline — it runs as a separate scheduled job during the same off-hours window as training.

### Purpose

Discord messages frequently contain images (memes, screenshots, reaction images). Without captioning, this content is invisible to the lore RAG pipeline — the vector store can only index text. Captions make image content searchable and contextually meaningful for lore retrieval.

### Model: `image-caption`

The captioner uses the `image-caption` alias defined in `models.ini`. This alias points to the **same GGUF as the mimic personas** (`HauhauCS/Qwen3.5-35B-A3B-Uncensored` IQ4_XS) — an `image-text-to-text` capable model with zero refusals. This is critical: Discord content includes crude memes and adult humour that a standard censored vision model would refuse to describe.

Because `image-caption` and `mimic_*` share the same GGUF, llama-server's router loads the file once and serves all aliases from it. Swapping from a mimic persona to `image-caption` is a context switch, not a model reload.

**VRAM:** ~18 GB — same as the mimic slot. The captioner only runs during the configured caption window when no live inference is active.

### Scheduling

Image captioning runs as a dedicated APScheduler job in `main.py`. By default it shares the training window (3–6 AM) and runs **before** training dispatch:

```
Caption window (3–6 AM):
  1. image_captioner.process_pending_batch()   ← runs first
  2. training_trigger.dispatch_queued()        ← runs after captions complete
```

The captioner checks proxy queue depth before starting each batch. If the proxy is busy, it defers to the next scheduler tick (every 5 minutes).

### JSONL Schema Extension

The existing JSONL record gains an optional `attachments` array. Messages without image attachments have no `attachments` field (fully backward compatible):

```json
{
  "message_id": "...",
  "content": "check this out",
  "clean": true,
  "attachments": [
    {
      "url": "https://cdn.discordapp.com/attachments/.../meme.png",
      "content_type": "image/png",
      "filename": "meme.png",
      "file_size_bytes": 204800,
      "caption": "A man in a suit pointing at a whiteboard that reads 'dying slower'.",
      "caption_status": "done",
      "caption_excluded_from_training": true
    }
  ]
}
```

### Training Exclusion

The `caption_excluded_from_training: true` flag signals the LoRA dataset exporter to treat caption text as **read-only context**, not training data:

- **RAG ingestion:** The caption is appended to the message's effective content string when chunking for ChromaDB: `"check this out [image: A man in a suit pointing at a whiteboard...]"`. This makes image content searchable via lore retrieval.
- **LoRA training dataset:** Caption text is **never** included in training samples. Only the original `content` field (the user's actual words) is used. Synthetic image descriptions are not the user's voice and would degrade persona quality if included.

### New Environment Variables

| Variable | Default | Description |
|---|---|---|
| `IMAGE_CAPTION_ENABLED` | `false` | Enable/disable image captioning. Set `true` once GGUF is downloaded. |
| `IMAGE_CAPTION_MODEL` | `image-caption` | Model alias for captioning (must match alias in `models.ini`) |
| `IMAGE_CAPTION_BATCH_SIZE` | `10` | Images processed per captioning run |
| `IMAGE_CAPTION_WINDOW_START` | `3` | Hour (0–23) when captioning may run |
| `IMAGE_CAPTION_WINDOW_END` | `6` | Hour (0–23) when captioning window closes |
| `IMAGE_CAPTION_MAX_FILE_SIZE_MB` | `10` | Skip images larger than this (avoids downloading huge files) |

See `history-service/README.md` §Image Captioning Pipeline for full implementation details.

---

## 10. Development Phases

### Component Overview

The following software components are built and evolved across the four phases:

| Component | Description |
|---|---|
| **llama-server Instances** | Two Docker containers: permanent (`:11435`, autocomplete) and swappable (`:11434`, all other models via router mode) |
| **Orchestration Middleware** | FastAPI proxy on `:11436` — swap tracking, async lock, request queue, source tagging |
| **models.ini** | llama-server preset config — model aliases, GGUF paths, inference parameters |
| **Discord Bot** | `discord.py` bot — mention routing, typing indicators, lore+mimic chain dispatch |
| **Open WebUI** | Self-hosted chat UI container — local model and Claude API backends, SQLite built-in |
| **Discord Data Preprocessor** | Export parser + chunker that feeds raw Discord history into ChromaDB |
| **RAG Service** | ChromaDB + `all-MiniLM-L6-v2` embedding pipeline — CPU-only, no VRAM impact |
| **History Service** | Background service — per-user JSONL message history collection (Discord API) + LoRA retraining trigger |
| **LoRA Training Pipeline** | Unsloth QLoRA fine-tuning on Qwen3.5-35B-A3B-Uncensored + GGUF merge and `models.ini` update |

---

### Development Timeline

```
                             │ Phase 1 │  Phase 2  │    Phase 3    │  Phase 4  │
                             │ 2–4 days│  1–2 wks  │   (ongoing)   │ (optional)│
─────────────────────────────┼─────────┼───────────┼───────────────┼───────────┤
llama-server Instances       │ ████████│           │               │           │
Orchestration Middleware     │ ████████│           │               │           │
models.ini                   │ ████████│           │ ░░░ LoRA paths│           │
Discord Bot                  │ ████████│           │               │ ░░░ harden│
Open WebUI                   │ ████████│           │               │ ░░░ auth  │
Discord Data Preprocessor    │         │ ██████████│               │           │
RAG Service (ChromaDB)       │         │ ██████████│               │           │
History Service              │         │ ██████████│ ░░░ train trig│           │
LoRA Training Pipeline       │         │           │ ██████████████│           │
─────────────────────────────┴─────────┴───────────┴───────────────┴───────────┘

████  Primary development / initial build
░░░   Minor extension or hardening work
      (blank) Component stable — no changes required
```

---

### Phase 1 — Prototype (2–4 days)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| llama-server Instances | 🔨 Build | Stand up both permanent and swappable containers via Docker Compose |
| Orchestration Middleware | 🔨 Build | FastAPI proxy with swap tracking, async lock, and source tagging |
| models.ini | 🔨 Build | Define all model presets: brain, mimic_*, lore, chat, image-caption |
| Discord Bot | 🔨 Build | Mention routing, typing indicators, basic mimic + lore dispatch |
| Open WebUI | 🔨 Build | Container, both Claude API and local model endpoints configured |
| Discord Data Preprocessor | ⏳ Not started | Needed in Phase 2 |
| RAG Service | ⏳ Not started | Needed in Phase 2 |
| LoRA Training Pipeline | ⏳ Not started | Needed in Phase 3 |

**Tasks:**
- [ ] Download all GGUFs: `make models-download`
- [ ] Stand up dual llama-server instances + proxy + Docker Compose: `make up`
- [ ] Verify GPU access: `make check-gpu`
- [ ] Verify models available: `make llama-ps`
- [ ] Verify Qwen3.5-35B-A3B-Uncensored loads and responds correctly
- [ ] Confirm `reasoning_format = none` suppresses chain-of-thought tokens in output
- [ ] Wire Discord bot with mention routing and typing indicators
- [ ] Test basic swap cycle: mimic → lore → mimic
- [ ] Validate VRAM budget under real swap load
- [ ] Stand up Open WebUI container
- [ ] Configure Open WebUI with both Claude API endpoint and local model endpoint
- [ ] Test Open WebUI swap contention with a concurrent Discord request

---

### Phase 2 — RAG + Lore (1–2 weeks)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| llama-server Instances | ✅ Stable | No changes — already running |
| Orchestration Middleware | ✅ Stable | No changes — proxy handles lore requests identically |
| models.ini | ✅ Stable | No changes |
| Discord Bot | ✅ Stable | Lore+mimic chain dispatch already wired in Phase 1; RAG context injection is the only addition |
| Open WebUI | ✅ Stable | No changes |
| Discord Data Preprocessor | 🔨 Build | Parse Discord history export, chunk by conversation thread, embed and load into ChromaDB |
| RAG Service (ChromaDB) | 🔨 Build | Stand up ChromaDB container, wire retrieval into lore assistant context at inference time |
| History Service | 🔨 Build | Stand up history-service; begin incremental Discord API pulls; training trigger disabled until Phase 3 |
| LoRA Training Pipeline | ⏳ Not started | Needed in Phase 3 |

**Tasks:**
- [ ] Ingest Discord history export into ChromaDB
- [ ] Wire RAG retrieval into lore assistant context (prepend retrieved chunks to user message)
- [ ] Test lore+mimic sequential chain (Step 8.2 flow)
- [ ] Tune lore assistant temperature and retrieval `top_k`
- [ ] Stand up history-service; verify incremental pull collects messages correctly
- [ ] Confirm per-user JSONL files are populated and clean/unclean flags are correct

---

### Phase 3 — LoRA Persona Refinement (ongoing)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| llama-server Instances | ✅ Stable | No changes — llama-server supports LoRA natively from Phase 1 |
| Orchestration Middleware | ✅ Stable | No changes — proxy is model-name-agnostic by design |
| models.ini | ⚠️ Update | Update `model` paths for mimic personas to point to LoRA-merged GGUFs |
| Discord Bot | ✅ Stable | No changes — bot references model aliases, not weights |
| Open WebUI | ✅ Stable | No changes |
| Discord Data Preprocessor | ✅ Stable | Re-run ingestion as new lore accumulates (manual or cron) |
| RAG Service (ChromaDB) | ✅ Stable | Re-embed on new significant events; no structural changes |
| History Service | 🔄 Extend | Enable training trigger; wire to lora-training scripts; automated retraining now active |
| LoRA Training Pipeline | 🔨 Build | Unsloth QLoRA fine-tuning on Qwen3.5-35B-A3B-Uncensored per member; GGUF output |

> **No backend migration required.** llama-server has supported dynamic LoRA adapter loading from Phase 1. Phase 3 simply updates the `model` paths in `models.ini` to point to LoRA-merged GGUFs and restarts `llama-swappable`. The proxy and Discord bot require zero changes.

**Tasks:**
- [ ] Build `lora-training/train.py` and `merge.py` scripts (Unsloth QLoRA → GGUF output)
- [ ] Enable training trigger in history-service (set `TRAINING_TRIGGER_ENABLED=true`)
- [ ] Update `models.ini` `model` paths for mimic personas to point to merged GGUFs
- [ ] Verify end-to-end automated flow: threshold hit → training queued → train → GGUF output → `models.ini` updated → server restarted
- [ ] A/B test LoRA adapter personas vs. system-prompt personas
- [ ] Tune `RETRAIN_THRESHOLD` and training window based on observed training times

> **LoRA training note:** Unsloth supports Qwen3.5 fine-tuning natively as of March 2026. The uncensored base weights are the correct starting point for LoRA — you're training style on top of an already-unlocked model, which means the adapter doesn't need to fight the base model's refusal tendencies.

---

### Phase 4 — Hardening (optional)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| llama-server Instances | ✅ Stable | No changes |
| Orchestration Middleware | 🔄 Extend | Add per-user rate limiting, queue depth cap, optional Brain priority preemption |
| models.ini | ✅ Stable | No changes |
| Discord Bot | 🔄 Extend | Ephemeral error messages on queue rejection; disclaimer post-processing strip |
| Open WebUI | 🔄 Extend | Enable user authentication if exposing beyond localhost |
| Discord Data Preprocessor | ✅ Stable | No changes |
| RAG Service (ChromaDB) | ✅ Stable | No changes |
| History Service | ✅ Stable | No changes |
| LoRA Training Pipeline | ✅ Stable | Ongoing as needed; no structural changes to pipeline |

**Tasks:**
- [ ] Per-user rate limiting (5 requests/min default, configurable per member)
- [ ] Queue depth cap (reject if >3 requests queued, return Discord ephemeral error)
- [ ] Graceful Brain priority: Brain requests can optionally preempt queued Discord requests with configurable precedence
- [ ] Disclaimer stripping in post-processing (catch the occasional baked-in lore assistant disclaimer)
- [ ] Open WebUI authentication (enable Open WebUI's built-in user auth if exposing beyond localhost)

---

## 11. Key Configuration Parameters

| Parameter | Mimic (Qwen3.5-35B-A3B) | Lore (Gemma3-12B) | Brain (Qwen3.5-35B) | Chat (Qwen3.5-35B-A3B) |
|---|---|---|---|---|
| temperature | 0.85 | 0.3 | 0.2 | 0.75 |
| top_k | 40 | 20 | 10 | 40 |
| top_p | 0.9 | 0.8 | 0.9 | 0.92 |
| repeat_penalty | 1.3 | — | — | 1.1 |
| n_ctx | 8192 | 16384 | 32768 | 16384 |
| n_predict | 512 | 1024 | -1 | -1 |
| reasoning_format | none | — | none | none |
| parallel | 1 | 1 | 1 | 1 |

> All parameters are set in `models.ini`. The `reasoning_format = none` field suppresses Qwen3.5's chain-of-thought tokens. Gemma3 does not have a thinking mode. Open WebUI uses the same GGUF as Brain (`Qwen3.5-35B-A3B-UD-IQ4_NL`) with warmer sampling parameters for casual conversation.

> **System prompts:** All system prompts are injected per-request by the Discord bot (see `DiscordBot-Design.md` §7). They are not baked into `models.ini` — this allows per-persona customisation without restarting the server.

---

## 12. Risk Table

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen3.5-35B-A3B-Uncensored generates something actually harmful | Low (private server, no bad actors) | System prompt boundaries; Discord server admin controls membership |
| Baked-in disclaimer appears in mimic output | Medium | Post-process: strip any string matching `"This is (general\|not legal\|..."` regex |
| Brain + Discord requests contend heavily | Medium (single GPU) | Queue; typing indicator masks wait; Phase 4 priority config |
| Image captioner monopolises swappable slot overnight | Low | Batch size cap + proxy queue depth check before each batch; training dispatches after captions complete |
| Image caption quality poor for low-res or non-standard images | Medium | `caption_status: "skipped"` for unsupported formats; `IMAGE_CAPTION_MAX_FILE_SIZE_MB` filters oversized files |
| LoRA training degrades base model personality | Low | Always train from fresh Qwen3.5-35B-A3B-Uncensored checkpoint; keep original `models.ini` entries |
| llama.cpp regression in future update | Low | Pin llama.cpp image tag in Docker Compose; test updates in staging first |
| ChromaDB retrieves wrong lore (hallucinated context) | Medium | Lore assistant system prompt: "say I don't know if context is insufficient"; `top_k` tuning |
| Swap latency annoys Discord users | Medium | Typing indicator; warm-up keep_alive (swap on first mention, keep alive 10 min) |
| Open WebUI local model contends with Discord during active chat session | Medium | Switch Open WebUI to Claude API backend during heavy Discord usage; or accept queue wait |
| Anthropic API key exposed in Docker environment | Low | Use Docker secrets or `.env` file excluded from version control; never hardcode in Compose |
| Open WebUI conversation history lost on container restart | Low | SQLite DB persists in `open_webui_data` volume; ensure volume is backed up |
| HuggingFace repo unavailable during model download | Low | GGUFs are cached locally after first download; `make models-download` is idempotent |
