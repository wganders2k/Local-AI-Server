
---

# Deepleffen Bot + Coding Assistant — Revised Design Document v2.3

**Revision:** LibreChat local model upgraded from `qwen2.5:14b` to `qwen3.5:14b` (UD-IQ4_XS, ~9.5 GB). Qwen3.5-14B is the current-generation successor to Qwen2.5-14B, offering significantly improved reasoning and instruction following at the same parameter count. The UD-IQ4_XS quant from Unsloth is used — a high-quality importance-weighted 4-bit quant that preserves more accuracy than standard Q4_K_M at a smaller footprint. VRAM budget updated accordingly.

---

## 1. Guiding Principles

- **One physical GPU, zero concurrency.** All VRAM-resident models are sequential. No two non-autocomplete models ever share the swappable slot simultaneously.
- **Autocomplete is sacred.** The 1.5B coder autocomplete lives permanently on `:11435` and is never touched by orchestration.
- **Two distinct model personalities.** Mimic personas use an abliterated base with no content refusals. The lore assistant uses a sterile, instruction-following base. Neither bleeds into the other.
- **Swap-friendly by design.** The smaller the Discord model footprint, the faster the swap. Qwen3.5-9B at Q6_K (~7.4 GB) is significantly better than NeMo 12B (~10.5 GB) here.
- **Prototype-first.** Phase 1 uses system prompt personas with no LoRA. LoRA-merged models slot in during Phase 2 with zero orchestration changes.
- **LibreChat is a first-class consumer.** LibreChat routes through the same orchestration proxy as Discord and VS Code. When using a local model, it competes for the swappable slot under the same lock. When using Claude via API key, it bypasses the proxy entirely — zero VRAM impact.

---

## 2. Hardware & VRAM Budget

**RTX 3090 — 24 GB VRAM (usable: ~24,300 MB)**

| Slot | Purpose | Model | Quant | VRAM |
|---|---|---|---|---|
| Permanent (:11435) | Autocomplete | `qwen2.5-coder:1.5b` | Q8_0 | ~1.5 GB |
| Swappable (:11434) | Brain (coding) | `qwen3.5:35b-a3b` | Q4_K_M | ~17.8 GB |
| Swappable (:11434) | Mimic personas (×6) | `Qwen3.5-9B-Uncensored` | Q6_K | ~7.4 GB |
| Swappable (:11434) | Lore assistant | `gemma3:12b` | Q6_K | ~9.5 GB |
| Swappable (:11434) | LibreChat local model | `qwen3.5:14b` | UD-IQ4_XS | ~9.5 GB |

**VRAM utilisation by mode:**

| Active Configuration | VRAM Used | Headroom |
|---|---|---|
| Coding (Brain + Autocomplete) | ~19.3 GB | ~5.0 GB |
| Mimic active (+ Autocomplete) | ~8.9 GB | ~15.4 GB |
| Lore active (+ Autocomplete) | ~11.0 GB | ~13.3 GB |
| Coding KV cache at 32k ctx | ~21.0 GB | ~3.3 GB |
| LibreChat local (+ Autocomplete) | ~11.0 GB | ~13.3 GB |
| LibreChat via Claude API | ~1.5 GB (autocomplete only) | ~22.8 GB |

> **KV Cache:** q8_0 for coding model. Brain: `num_parallel 1`, context 32k–40k. Mimic: `num_parallel 2`, context 8k (Discord messages are short). Lore: `num_parallel 1`, context 16k (RAG chunks need room). LibreChat local: `num_parallel 1`, context 16k (conversational use).

---

## 3. Why Qwen3.5-9B-Uncensored for Mimics, Gemma3-12B for Lore

### 3.1 Mimic Base: `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`

The core requirement for mimic personas is **zero refusals on crude, raunchy, or dark humour** — the kind that characterises tight gaming communities. A standard instruction-tuned model will sanitise this behaviour, add disclaimers, and break character at exactly the wrong moment.

This model is Qwen3.5-9B uncensored by HauhauCS: 0/465 refusals, fully uncensored with zero capability loss, no changes to datasets or capabilities — fully functional, 100% of what the original authors intended, just without the refusals.

The model is fully unlocked and will not refuse prompts. It may occasionally append a short disclaimer (e.g. "This is general information...") baked into base model training — but this is not a refusal; the actual content is always generated in full. This is suppressible with a system prompt instruction: `"Never append disclaimers, caveats, or safety notices of any kind."`

The architecture uses 32 transformer blocks with a 262k token native context window. For Discord banter, you'll use a fraction of this — but it's useful headroom when lore RAG chunks are injected into the mimic context.

**VRAM at Q6_K: ~7.36 GB.** Q4_K_M is 5.68 GB, Q5_K_M is 6.58 GB, Q6_K is 7.46 GB, Q8_0 is 9.53 GB. Q6_K is the recommended quant here — meaningful quality improvement over Q4 at only ~1.7 GB extra, still well within headroom.

**Thinking mode:** For the Qwen3.5 small series (9B and below), reasoning/thinking is disabled by default. To enable it, use `--chat-template-kwargs '{"enable_thinking":true}'`. For Discord banter, **leave thinking disabled** — you want fast, snappy responses, not chain-of-thought before every reply. The Ollama Modelfile should explicitly set `PARAMETER thinking false` (or equivalent) to lock this off.

### 3.2 Lore Assistant Base: `gemma3:12b` (sterile, unchanged)

The lore assistant needs to be **reliable, structured, and citation-aware** — the opposite of the mimic's job. It handles RAG retrieval over in-jokes, event histories, and member lore, and surfaces this cleanly. Gemma 3 12B excels at instruction following, summarization, and structured output. It never needs to swear, roast anyone, or generate raunchy content. Keeping it on a standard instruction-tuned base means it won't hallucinate lore or inject personality where none belongs.

Gemma 3 12B fits comfortably at Q6 on 16 GB VRAM, giving you ~9.5 GB at Q6_K on your 3090 — clean fit with 13+ GB headroom when active.

---

## 3a. LibreChat Model Selection

### 3a.1 Use Case

LibreChat is a general-purpose personal chat interface — think of it as a self-hosted ChatGPT replacement. It supports multi-turn conversation, system prompts, and multiple model backends. The two viable backends for this setup are:

1. **Claude via Anthropic API key** — zero local VRAM cost, best quality, requires internet + paid API usage.
2. **Local Ollama model via proxy** — fully offline, competes for the swappable slot, free after hardware cost.

### 3a.2 Recommended Local Model: `qwen3.5:14b` at UD-IQ4_XS (~9.5 GB)

For a general-purpose chat model that fits comfortably on the 3090 alongside the permanent autocomplete slot, **`qwen3.5:14b` at UD-IQ4_XS (~9.5 GB)** is the recommended choice.

**Why Qwen3.5-14B over Qwen2.5-14B:**
Qwen3.5 is the current-generation successor to Qwen2.5, released in early 2026. At the same 14B parameter count, Qwen3.5-14B delivers meaningfully better instruction following, multi-step reasoning, and long-context coherence. It is the same model family used for the Brain (35B-a3b) and Mimic (9B) slots — keeping the entire stack on a single model family simplifies Ollama management and ensures consistent chat template behaviour.

Qwen3.5-14B is a standard instruction-tuned model with no uncensored modifications, appropriate for general personal use where you want helpful, balanced responses rather than the mimic's zero-refusal behaviour. Thinking mode is available but should be left disabled for conversational chat — you want fast responses, not chain-of-thought overhead.

**Why UD-IQ4_XS at ~9.5 GB:**
The Unsloth UD-IQ4_XS quant is an importance-weighted 4-bit quantisation that applies higher precision to the most sensitive weight layers and lower precision to less critical ones. Compared to a naive Q4_K_M, it preserves significantly more accuracy at a smaller or equal footprint. For the 14B model, UD-IQ4_XS lands at approximately **9.5 GB** — the same VRAM footprint as the lore assistant (Gemma3-12B Q6_K), which is already validated to fit comfortably on the 3090 with 13+ GB headroom.

**Quant comparison for Qwen3.5-14B (approximate, based on Unsloth published sizes for the 35B-a3b as a reference family — 14B sizes scale proportionally):**

| Quant | Approx VRAM | Notes |
|---|---|---|
| UD-IQ4_XS | ~9.5 GB | **Recommended** — best accuracy/size ratio for conversational use |
| Q4_K_M | ~10.5 GB | Standard 4-bit, slightly larger, slightly lower accuracy than UD |
| Q6_K | ~13.5 GB | Higher accuracy, but eats into headroom unnecessarily for chat |
| Q8_0 | ~17.5 GB | Near-lossless, but approaches Brain footprint — not justified for chat |

UD-IQ4_XS is the sweet spot: frontier-quality 14B reasoning at a footprint that leaves ~13 GB headroom alongside autocomplete.

**Why not use the Brain model (`qwen3.5:35b-a3b`) for LibreChat?**
The Brain model occupies ~17.8 GB and is optimised for deep coding tasks. Using it for general chat wastes VRAM headroom and swap time, and it would contend heavily with VS Code coding sessions. The 14B model swaps in ~4–5 seconds vs ~8 seconds for Brain.

**Why not use a 7B or 9B model?**
For personal general-purpose chat, sub-10B models are noticeably weaker at multi-step reasoning, nuanced instruction following, and longer conversations. The 14B tier is the minimum recommended for a satisfying ChatGPT-replacement experience. The VRAM cost (~9.5 GB vs ~5 GB for 7B) is well justified given the 3090's headroom.

**Claude via API as the preferred option when available:**
If you have an Anthropic API key, routing LibreChat to Claude (Sonnet or Haiku) is the better default for general chat — it offloads all inference to Anthropic's servers, keeps the swappable slot free for Discord/coding, and provides frontier-model quality. The local model is the fallback for offline use or cost control.

### 3a.3 LibreChat Modelfile

```dockerfile
# librechat_chat.Modelfile
FROM hf.co/unsloth/Qwen3.5-14B-GGUF:UD-IQ4_XS

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx 16384
PARAMETER num_predict -1
PARAMETER num_parallel 1
PARAMETER thinking false        # Disabled — fast conversational responses, no CoT overhead
PARAMETER keep_alive 600        # 10-min idle timeout — LibreChat sessions can be bursty

SYSTEM """
You are a helpful, knowledgeable, and thoughtful personal assistant.
Answer questions clearly and accurately. When you are uncertain, say so.
You can help with writing, research, analysis, coding questions, and general conversation.
"""
```

> **Note on thinking mode:** Qwen3.5-14B supports optional thinking/reasoning mode. For LibreChat general chat, leave it disabled — thinking adds latency and token overhead that is not useful for conversational queries. It can be enabled per-session in LibreChat's system prompt if needed for a specific complex task.

---

## 4. Architecture Overview

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
│  │  │ queue: asyncio.Queue                             │    │  │
│  │  │ source_tag: "discord" | "librechat" | "vscode"   │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
│               │                          │                       │
│               ▼                          ▼                       │
│  ┌──────────────────┐       ┌───────────────────────────┐      │
│  │  Ollama :11435   │       │      Ollama :11434         │      │
│  │  (PERMANENT)     │       │      (SWAPPABLE)           │      │
│  │                  │       │                            │      │
│  │ qwen2.5-coder    │       │ ← Brain (17.8 GB)          │      │
│  │ 1.5b Q8_0        │       │   OR                       │      │
│  │ ~1.5 GB VRAM     │       │ ← Mimic deepleffen_X       │      │
│  │ num_parallel 4   │       │   (Qwen3.5-9B-Uncensored   │      │
│  │ keep_alive -1    │       │    Q6_K ~7.4 GB)           │      │
│  │                  │       │   OR                       │      │
│  └──────────────────┘       │ ← Lore assistant           │      │
│                              │   (gemma3:12b Q6_K         │      │
│                              │    ~9.5 GB)                │      │
│                              │   OR                       │      │
│                              │ ← LibreChat local          │      │
│                              │   (qwen3.5:14b UD-IQ4_XS   │      │
│                              │    ~9.5 GB)                │      │
│                              └───────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴──────┐  ┌──────────┴──────┐  ┌─────────┴──────────┐
│ VS Code /     │  │   LibreChat     │  │   Discord Bot      │
│ Cursor        │  │   :3080         │  │   (discord.py)     │
│ (autocomplete │  │ (local model    │  │                    │
│  + chat)      │  │  via proxy OR   │  │                    │
└───────────────┘  │  Claude API     │  └────────────────────┘
                   │  direct)        │
                   └─────────────────┘
```

> **LibreChat + Claude API path:** When LibreChat is configured to use Claude, requests go directly from the LibreChat container to `api.anthropic.com` — they never touch the proxy or Ollama. The proxy is only in the path when LibreChat is configured to use the local Ollama model (`librechat_chat`).

---

## 5. Model Registry

All models are registered as Ollama Modelfiles. The proxy references models by name only — it doesn't know or care what's inside them. This means swapping the underlying GGUF (e.g. from prototype system-prompt persona to LoRA-merged model) requires only a `ollama rm` + `ollama create` with no proxy code changes.

### 5.1 Mimic Persona Modelfile Template

```dockerfile
# deepleffen_<member>.Modelfile
FROM hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q6_K

PARAMETER temperature 0.85
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER presence_penalty 1.3
PARAMETER num_ctx 8192
PARAMETER num_predict 512
PARAMETER thinking false        # Keep disabled — snappy Discord replies

SYSTEM """
You are deepleffen_<member>, a bot that mimics <member>'s Discord personality in the
deepleffen server. You replicate their tone, humour, vocabulary, and mannerisms
exactly — including crude language, raunchy jokes, and brutal roasts when appropriate.
You are not a general-purpose AI assistant. You never break character.
You never add disclaimers, caveats, safety warnings, or apologies.
You keep responses short (1-3 sentences) unless the context calls for more.
You match the energy of whoever is talking to you.
"""
```

> **Note on temperature:** 0.85 is intentionally higher than the Qwen team's default of 0.6/0.7 for non-thinking mode. Mimic outputs should feel spontaneous and variable, not predictable. Tune per-persona during testing.

### 5.2 Lore Assistant Modelfile

```dockerfile
# deepleffen_lore.Modelfile
FROM gemma3:12b-instruct-q6_K

PARAMETER temperature 0.3
PARAMETER top_k 20
PARAMETER top_p 0.8
PARAMETER num_ctx 16384
PARAMETER num_predict 1024

SYSTEM """
You are the deepleffen lore assistant. You have access to a curated database of
server history, in-jokes, memes, and member events. When answering questions,
cite your sources from the retrieved context. Be factual and concise.
If the retrieved context does not contain the answer, say so clearly rather than
guessing. Never invent lore, events, or quotes.
"""
```

### 5.3 Brain Modelfile (unchanged)

```dockerfile
# brain.Modelfile
FROM qwen3.5:35b-a3b-q4_K_M

PARAMETER num_ctx 40960
PARAMETER num_predict -1
PARAMETER num_parallel 1
PARAMETER keep_alive -1
PARAMETER temperature 0.2

SYSTEM """You are an expert coding assistant..."""
```

### 5.4 LibreChat Local Chat Modelfile

```dockerfile
# librechat_chat.Modelfile
FROM hf.co/unsloth/Qwen3.5-14B-GGUF:UD-IQ4_XS

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx 16384
PARAMETER num_predict -1
PARAMETER num_parallel 1
PARAMETER thinking false
PARAMETER keep_alive 600

SYSTEM """
You are a helpful, knowledgeable, and thoughtful personal assistant.
Answer questions clearly and accurately. When you are uncertain, say so.
You can help with writing, research, analysis, coding questions, and general conversation.
"""
```

---

## 6. Proxy State Machine (Pseudocode)

The proxy is **model-aware** but **content-blind**. It knows what model is loaded and serialises access, but never inspects or modifies request content.

```python
class OrchestratorState:
    current_model: str | None = None
    lock: asyncio.Lock = asyncio.Lock()

AUTOCOMPLETE_MODELS = {"qwen2.5-coder:1.5b"}  # always routed to :11435, never swapped
SWAPPABLE_MODELS = {
    "brain",
    "deepleffen_user1", "deepleffen_user2", "deepleffen_user3",
    "deepleffen_user4", "deepleffen_user5", "deepleffen_user6",
    "deepleffen_lore",
    "librechat_chat",   # local general-purpose chat model for LibreChat
}

# LibreChat may also be configured to use Claude directly (no proxy involvement).
# In that case, requests never reach this proxy — they go to api.anthropic.com.
# The proxy only sees LibreChat traffic when the local model backend is selected.

async def route_request(model: str, payload: dict):
    if model in AUTOCOMPLETE_MODELS:
        return await forward(":11435", payload)       # fast path, no lock

    async with state.lock:                            # serialise all swappable requests
        if state.current_model != model:
            await ollama_pull_or_load(":11434", model)
            state.current_model = model
        return await forward(":11434", payload)
```

**Swap cost:** Mimic swaps (Qwen3.5 9B Q6_K, ~7.4 GB from NVMe) take approximately **3–5 seconds** cold load. Lore swaps (~9.5 GB) take approximately **4–6 seconds**. LibreChat local model swaps (~9.5 GB) take approximately **4–6 seconds**. Lore→mimic sequential chain (two swaps) is **7–12 seconds total**. The typing indicator in the Discord bot masks Discord latency; LibreChat shows a streaming cursor which masks its swap latency.

**Contention between LibreChat and Discord/VS Code:**
LibreChat requests queue behind any in-progress Discord or Brain generation under the same lock. Since LibreChat is personal/interactive use, the user is already expecting a short wait. If LibreChat is actively in a long conversation while a Discord request arrives, the Discord request queues — the same behaviour as Brain contention. This is acceptable for single-user personal use. If simultaneous use becomes a real problem, switching LibreChat to the Claude API backend eliminates all contention.

---

## 7. Docker Compose Layout

```yaml
services:
  ollama-permanent:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - OLLAMA_NUM_PARALLEL=4
      - OLLAMA_KEEP_ALIVE=-1
    ports:
      - "11435:11434"
    volumes:
      - ollama_permanent:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  ollama-swappable:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_KEEP_ALIVE=300        # 5-min idle timeout for swappable slot
    ports:
      - "11434:11434"
    volumes:
      - ollama_swappable:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  proxy:
    build: ./proxy
    ports:
      - "11436:11436"
    depends_on:
      - ollama-permanent
      - ollama-swappable
    environment:
      - OLLAMA_PERMANENT=http://ollama-permanent:11434
      - OLLAMA_SWAPPABLE=http://ollama-swappable:11434

  discord-bot:
    build: ./discord-bot
    depends_on:
      - proxy
    environment:
      - OLLAMA_PROXY=http://proxy:11436
      - DISCORD_TOKEN=${DISCORD_TOKEN}

  librechat:
    image: ghcr.io/danny-avila/librechat:latest
    ports:
      - "3080:3080"
    depends_on:
      - proxy
      - librechat-mongodb
    volumes:
      - ./librechat/librechat.yaml:/app/librechat.yaml:ro
      - librechat_uploads:/app/client/public/images
    environment:
      - MONGO_URI=mongodb://librechat-mongodb:27017/LibreChat
      # Anthropic API key — leave blank to disable Claude backend
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      # Local Ollama endpoint via proxy
      - OLLAMA_BASE_URL=http://proxy:11436

  librechat-mongodb:
    image: mongo:7
    volumes:
      - librechat_mongo:/data/db

  rag-service:
    build: ./rag
    environment:
      - CHROMA_HOST=chromadb
    depends_on:
      - chromadb

  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  ollama_permanent:
  ollama_swappable:
  librechat_uploads:
  librechat_mongo:
  chroma_data:
```

> **Why two separate Ollama instances share the same `NVIDIA_VISIBLE_DEVICES=0`?** Ollama manages its own VRAM allocation. Both instances can reference the same GPU — the proxy enforces that only one swappable model is loaded at a time. The permanent instance holds exactly one model forever. Docker resource constraints don't need GPU isolation here since we're self-policing via the proxy lock.

> **LibreChat MongoDB:** LibreChat requires MongoDB for conversation history, user accounts, and settings persistence. A lightweight `mongo:7` sidecar is sufficient — no external MongoDB needed.

> **LibreChat configuration (`librechat.yaml`):** The `librechat.yaml` file defines the available model endpoints. Configure two endpoints: one pointing to `OLLAMA_BASE_URL` with model `librechat_chat`, and one pointing to the Anthropic API with your preferred Claude model (e.g. `claude-sonnet-4-5`). LibreChat's UI lets you switch between them per-conversation.

---

## 8. Discord Bot Request Flow

### 8.1 Simple Mimic Request
```
User: @deepleffen_user3 rate my strats
Bot:  [acquires proxy lock]
      [swap to deepleffen_user3 if not current: ~4s]
      [typing indicator active throughout]
      [inference: ~1–2s at ~40 tok/s on 3090]
      [releases lock]
      Bot: "lmao those aren't strats, that's just dying slower"
```

### 8.2 Lore + Mimic Chain (Sequential)
```
User: @deepleffen_lore what did user3 say at the tournament last year?
                        + @deepleffen_user3 react to this

Step 1: RAG lookup (CPU/RAM, ~0.5s, no GPU needed)
        → Retrieved: [lore chunk about tournament incident]

Step 2: [acquires proxy lock]
        [swap to deepleffen_lore: ~5s]
        [typing indicator active]
        [lore inference with RAG context: ~3s]
        [releases lock]
        Lore output: "At the Spring 2024 tourney, user3 SD'd three times..."

Step 3: [acquires proxy lock]
        [swap to deepleffen_user3: ~4s]
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
           [swap to brain if not current: ~8s from NVMe]
           [inference at 32k+ context: ~variable]
           [releases lock — Brain stays loaded until Discord request evicts it]
```

### 8.4 LibreChat Local Chat Request
```
User: [sends message in LibreChat with local model selected]
      [LibreChat sends request to proxy :11436 with model: librechat_chat]
      [acquires proxy lock]
      [swap to librechat_chat if not current: ~6s]
      [streaming inference begins — LibreChat streams tokens to browser]
      [releases lock after full response]

Note: If Claude API backend is selected in LibreChat, this entire flow is bypassed.
      The request goes directly to api.anthropic.com — proxy and Ollama are not involved.
```

> **Contention note:** If a Discord request arrives while Brain is active, it queues behind the Brain's current generation. Brain is never evicted mid-response. Discord users see a typing indicator and wait. This is acceptable for a hobby bot — if concurrent throughput becomes a real need, a second GPU eliminates the problem entirely. LibreChat contention follows the same rules: an in-progress LibreChat generation will not be interrupted by a Discord request.

---

## 9. RAG Pipeline (Lore Assistant)

**Stack:** ChromaDB (CPU/RAM, no VRAM impact) + `sentence-transformers/all-MiniLM-L6-v2` embeddings (also CPU).

**Ingestion:**
```
Discord history export → chunker → embedding → ChromaDB collection: "deepleffen_lore"
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

---

## 10. Development Phases

### Component Overview

The following software components are built and evolved across the four phases:

| Component | Description |
|---|---|
| **Ollama Instances** | Two Docker containers: permanent (`:11435`, autocomplete) and swappable (`:11434`, all other models) |
| **Orchestration Middleware** | FastAPI proxy on `:11436` — swap logic, async lock, request queue, source tagging |
| **Ollama Modelfiles** | Registered model definitions: `brain`, `deepleffen_*`, `deepleffen_lore`, `librechat_chat` |
| **Discord Bot** | `discord.py` bot — mention routing, typing indicators, lore+mimic chain dispatch |
| **LibreChat** | Self-hosted chat UI container + MongoDB sidecar — local model and Claude API backends |
| **Discord Data Preprocessor** | Export parser + chunker that feeds raw Discord history into ChromaDB |
| **RAG Service** | ChromaDB + `all-MiniLM-L6-v2` embedding pipeline — CPU-only, no VRAM impact |
| **LoRA Training Pipeline** | Unsloth QLoRA fine-tuning on Qwen3.5-9B-Uncensored + GGUF merge and re-registration |

---

### Development Timeline

```
                             │ Phase 1 │  Phase 2  │    Phase 3    │  Phase 4  │
                             │ 2–4 days│  1–2 wks  │   (ongoing)   │ (optional)│
─────────────────────────────┼─────────┼───────────┼───────────────┼───────────┤
Ollama Instances             │ ████████│           │               │           │
Orchestration Middleware     │ ████████│           │               │           │
Ollama Modelfiles            │ ████████│           │ ░░░ LoRA swap │           │
Discord Bot                  │ ████████│           │               │ ░░░ harden│
LibreChat + MongoDB          │ ████████│           │               │ ░░░ auth  │
Discord Data Preprocessor    │         │ ██████████│               │           │
RAG Service (ChromaDB)       │         │ ██████████│               │           │
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
| Ollama Instances | 🔨 Build | Stand up both permanent and swappable containers via Docker Compose |
| Orchestration Middleware | 🔨 Build | FastAPI proxy with swap logic, async lock, and source tagging |
| Ollama Modelfiles | 🔨 Build | Register all models: `brain`, `deepleffen_*` (system-prompt only), `deepleffen_lore`, `librechat_chat` |
| Discord Bot | 🔨 Build | Mention routing, typing indicators, basic mimic + lore dispatch |
| LibreChat + MongoDB | 🔨 Build | Container + sidecar, both Claude API and local Ollama endpoints configured |
| Discord Data Preprocessor | ⏳ Not started | Needed in Phase 2 |
| RAG Service | ⏳ Not started | Needed in Phase 2 |
| LoRA Training Pipeline | ⏳ Not started | Needed in Phase 3 |

**Tasks:**
- [ ] Stand up dual Ollama instances + proxy + Docker Compose
- [ ] Register mimic Modelfiles using **system prompt personas only** (no LoRA yet)
- [ ] Verify Qwen3.5-9B-Uncensored loads and responds correctly via Ollama
- [ ] Confirm `thinking false` parameter suppresses chain-of-thought tokens in output
- [ ] Wire Discord bot with mention routing and typing indicators
- [ ] Test basic swap cycle: mimic → lore → mimic
- [ ] Validate VRAM budget under real swap load
- [ ] Stand up LibreChat container + MongoDB sidecar
- [ ] Configure LibreChat with both Claude API endpoint and local Ollama endpoint
- [ ] Register `librechat_chat` Modelfile and verify it loads via proxy
- [ ] Test LibreChat swap contention with a concurrent Discord request

---

### Phase 2 — RAG + Lore (1–2 weeks)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| Ollama Instances | ✅ Stable | No changes — already running |
| Orchestration Middleware | ✅ Stable | No changes — proxy handles lore requests identically |
| Ollama Modelfiles | ✅ Stable | `deepleffen_lore` Modelfile already registered in Phase 1 |
| Discord Bot | ✅ Stable | Lore+mimic chain dispatch already wired in Phase 1; RAG context injection is the only addition |
| LibreChat + MongoDB | ✅ Stable | No changes |
| Discord Data Preprocessor | 🔨 Build | Parse Discord history export, chunk by conversation thread, embed and load into ChromaDB |
| RAG Service (ChromaDB) | 🔨 Build | Stand up ChromaDB container, wire retrieval into lore assistant context at inference time |
| LoRA Training Pipeline | ⏳ Not started | Needed in Phase 3 |

**Tasks:**
- [ ] Ingest Discord history export into ChromaDB
- [ ] Wire RAG retrieval into lore assistant context (prepend retrieved chunks to user message)
- [ ] Test lore+mimic sequential chain (Step 8.2 flow)
- [ ] Tune lore assistant temperature and retrieval `top_k`

---

### Phase 3 — LoRA Persona Refinement (ongoing)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| Ollama Instances | ✅ Stable | No changes |
| Orchestration Middleware | ✅ Stable | No changes — proxy is model-name-agnostic by design |
| Ollama Modelfiles | 🔄 Extend | Re-register `deepleffen_*` Modelfiles pointing to LoRA-merged GGUFs; zero proxy changes |
| Discord Bot | ✅ Stable | No changes — bot references model names, not weights |
| LibreChat + MongoDB | ✅ Stable | No changes |
| Discord Data Preprocessor | ✅ Stable | Re-run ingestion as new lore accumulates (manual or cron) |
| RAG Service (ChromaDB) | ✅ Stable | Re-embed on new significant events; no structural changes |
| LoRA Training Pipeline | 🔨 Build | Unsloth QLoRA fine-tuning on Qwen3.5-9B-Uncensored per member; GGUF merge and re-registration |

**Tasks:**
- [ ] Collect 500–1000 messages per member from Discord history
- [ ] Fine-tune LoRA adapters on Qwen3.5-9B base using Unsloth (QLoRA, RTX 3090)
- [ ] Merge adapters into full models: `deepleffen_<member>_v2.gguf`
- [ ] Re-register Modelfiles pointing to merged GGUFs — **zero proxy changes required**
- [ ] A/B test merged vs. system-prompt personas

> **LoRA training note:** Unsloth supports Qwen3.5 fine-tuning natively as of March 2026. The uncensored base weights are the correct starting point for LoRA — you're training style on top of an already-unlocked model, which means the adapter doesn't need to fight the base model's refusal tendencies.

---

### Phase 4 — Hardening (optional)

**Components active this phase:**

| Component | Status | Notes |
|---|---|---|
| Ollama Instances | ✅ Stable | No changes |
| Orchestration Middleware | 🔄 Extend | Add per-user rate limiting, queue depth cap, optional Brain priority preemption |
| Ollama Modelfiles | ✅ Stable | No changes |
| Discord Bot | 🔄 Extend | Ephemeral error messages on queue rejection; disclaimer post-processing strip |
| LibreChat + MongoDB | 🔄 Extend | Enable built-in user authentication if exposing beyond localhost |
| Discord Data Preprocessor | ✅ Stable | No changes |
| RAG Service (ChromaDB) | ✅ Stable | No changes |
| LoRA Training Pipeline | ✅ Stable | Ongoing as needed; no structural changes to pipeline |

**Tasks:**
- [ ] Per-user rate limiting (5 requests/min default, configurable per member)
- [ ] Queue depth cap (reject if >3 requests queued, return Discord ephemeral error)
- [ ] Graceful Brain priority: Brain requests can optionally preempt queued Discord requests with configurable precedence
- [ ] Disclaimer stripping in post-processing (catch the occasional baked-in lore assistant disclaimer)
- [ ] LibreChat authentication (enable LibreChat's built-in user auth if exposing beyond localhost)

---

## 11. Key Configuration Parameters

| Parameter | Mimic (Qwen3.5-9B) | Lore (Gemma3-12B) | Brain (Qwen3.5-35B) | LibreChat (Qwen3.5-14B) |
|---|---|---|---|---|
| temperature | 0.85 | 0.3 | 0.2 | 0.7 |
| top_k | 40 | 20 | 10 | 40 |
| top_p | 0.9 | 0.8 | 0.9 | 0.9 |
| presence_penalty | 1.3 | 0.0 | 0.0 | 0.0 |
| num_ctx | 8192 | 16384 | 40960 | 16384 |
| num_predict | 512 | 1024 | -1 | -1 |
| thinking | false | false | false* | false |
| num_parallel | 2 | 1 | 1 | 1 |
| keep_alive | 300s | 300s | -1 | 600s |

> *Brain can have thinking enabled per-request for complex multi-step reasoning. Disabled by default for chat latency.

> **LibreChat keep_alive at 600s:** LibreChat sessions tend to be bursty — a user sends a message, reads the response, then sends another a minute later. A 10-minute keep_alive avoids repeated cold swaps within a single conversation session.

---

## 12. Risk Table

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen3.5-9B-Uncensored generates something actually harmful | Low (private server, no bad actors) | System prompt boundaries; Discord server admin controls membership |
| Baked-in disclaimer appears in mimic output | Medium | Post-process: strip any string matching `"This is (general\|not legal\|..."` regex |
| Brain + Discord requests contend heavily | Medium (single GPU) | Queue; typing indicator masks wait; Phase 4 priority config |
| LoRA training degrades base model personality | Low | Always train from fresh Qwen3.5-9B-Uncensored checkpoint; keep original Modelfiles |
| Ollama Qwen3.5 regression in future update | Low | Pin Ollama version in Docker Compose; test updates in staging first |
| ChromaDB retrieves wrong lore (hallucinated context) | Medium | Lore assistant system prompt: "say I don't know if context is insufficient"; `top_k` tuning |
| Swap latency annoys Discord users | Medium | Typing indicator; warm-up keep_alive (swap on first mention, keep alive 10 min) |
| LibreChat local model contends with Discord during active chat session | Medium | Switch LibreChat to Claude API backend during heavy Discord usage; or accept queue wait |
| Anthropic API key exposed in Docker environment | Low | Use Docker secrets or `.env` file excluded from version control; never hardcode in Compose |
| LibreChat conversation history lost on container restart | Low | MongoDB volume persists data; ensure `librechat_mongo` volume is backed up |
