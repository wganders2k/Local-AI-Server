Great news to lead with — the blocker is **gone**. And yes, this is absolutely the right call. Let me confirm the facts and deliver the updated design document.

**Ollama status update:** The Qwen 3.5 small model series is now available in Ollama — `ollama run qwen3.5:9b`, supporting native tool calling, thinking, and multimodal capabilities. Ollama has also already shipped fixes including a crash fix when the model is split over GPU & CPU, and a fix for Qwen 3.5 models repeating themselves due to missing presence penalty. The Unsloth docs warning about the mmproj incompatibility is now outdated. The HauhauCS 9B Aggressive variant is fully viable.

**Confirmed VRAM numbers for the Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:** Q4_K_M is **5.63 GB**, Q6_K is **7.36 GB**, Q8_0 is **9.53 GB**. This is a meaningful upgrade from Mistral NeMo (~10.5 GB) — much more headroom.

---

# Deepleffen Bot + Coding Assistant — Revised Design Document v2.1

**Revision:** Mimic persona base model changed from `mistral-nemo:12b` to `HauhauCS/Qwen3.5-9B-Uncensored-Aggressive`. Lore assistant unchanged. Ollama compatibility confirmed.

---

## 1. Guiding Principles

- **One physical GPU, zero concurrency.** All VRAM-resident models are sequential. No two non-autocomplete models ever share the swappable slot simultaneously.
- **Autocomplete is sacred.** The 1.5B coder autocomplete lives permanently on `:11435` and is never touched by orchestration.
- **Two distinct model personalities.** Mimic personas use an abliterated base with no content refusals. The lore assistant uses a sterile, instruction-following base. Neither bleeds into the other.
- **Swap-friendly by design.** The smaller the Discord model footprint, the faster the swap. Qwen3.5-9B at Q6_K (~7.4 GB) is significantly better than NeMo 12B (~10.5 GB) here.
- **Prototype-first.** Phase 1 uses system prompt personas with no LoRA. LoRA-merged models slot in during Phase 2 with zero orchestration changes.

---

## 2. Hardware & VRAM Budget

**RTX 3090 — 24 GB VRAM (usable: ~24,300 MB)**

| Slot | Purpose | Model | Quant | VRAM |
|---|---|---|---|---|
| Permanent (:11435) | Autocomplete | `qwen2.5-coder:1.5b` | Q8_0 | ~1.5 GB |
| Swappable (:11434) | Brain (coding) | `qwen3.5:35b-a3b` | Q4_K_M | ~17.8 GB |
| Swappable (:11434) | Mimic personas (×6) | `Qwen3.5-9B-Uncensored` | Q6_K | ~7.4 GB |
| Swappable (:11434) | Lore assistant | `gemma3:12b` | Q6_K | ~9.5 GB |

**VRAM utilisation by mode:**

| Active Configuration | VRAM Used | Headroom |
|---|---|---|
| Coding (Brain + Autocomplete) | ~19.3 GB | ~5.0 GB |
| Mimic active (+ Autocomplete) | ~8.9 GB | ~15.4 GB |
| Lore active (+ Autocomplete) | ~11.0 GB | ~13.3 GB |
| Coding KV cache at 32k ctx | ~21.0 GB | ~3.3 GB |

> **KV Cache:** q8_0 for all models. Brain: `num_parallel 1`, context 32k–40k. Mimic: `num_parallel 2`, context 8k (Discord messages are short). Lore: `num_parallel 1`, context 16k (RAG chunks need room).

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

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Ubuntu Server (RTX 3090)                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           FastAPI Orchestration Proxy :11436          │  │
│  │                                                      │  │
│  │  State Machine:                                      │  │
│  │  ┌─────────────────────────────────────────────┐    │  │
│  │  │ current_model: str | None                   │    │  │
│  │  │ lock: asyncio.Lock (one request at a time)  │    │  │
│  │  │ queue: asyncio.Queue                        │    │  │
│  │  └─────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────┘  │
│            │                          │                     │
│            ▼                          ▼                     │
│  ┌─────────────────┐       ┌──────────────────────────┐   │
│  │  Ollama :11435  │       │      Ollama :11434        │   │
│  │  (PERMANENT)    │       │      (SWAPPABLE)          │   │
│  │                 │       │                           │   │
│  │ qwen2.5-coder   │       │ ← Brain (17.8 GB)         │   │
│  │ 1.5b Q8_0       │       │   OR                      │   │
│  │ ~1.5 GB VRAM    │       │ ← Mimic deepleffen_X      │   │
│  │ num_parallel 4  │       │   (Qwen3.5-9B-Uncensored  │   │
│  │ keep_alive -1   │       │    Q6_K ~7.4 GB)          │   │
│  │                 │       │   OR                      │   │
│  └─────────────────┘       │ ← Lore assistant          │   │
│                             │   (gemma3:12b Q6_K        │   │
│                             │    ~9.5 GB)               │   │
│                             └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │                                    │
┌────────┴──────────┐             ┌───────────┴──────────┐
│  VS Code / Cursor │             │    Discord Bot        │
│  (autocomplete    │             │    (discord.py)       │
│   + chat)         │             │                       │
└───────────────────┘             └──────────────────────┘
```

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
}

async def route_request(model: str, payload: dict):
    if model in AUTOCOMPLETE_MODELS:
        return await forward(":11435", payload)       # fast path, no lock

    async with state.lock:                            # serialise all swappable requests
        if state.current_model != model:
            await ollama_pull_or_load(":11434", model)
            state.current_model = model
        return await forward(":11434", payload)
```

**Swap cost:** Mimic swaps (Qwen3.5 9B Q6_K, ~7.4 GB from NVMe) take approximately **3–5 seconds** cold load. Lore swaps (~9.5 GB) take approximately **4–6 seconds**. Lore→mimic sequential chain (two swaps) is **7–12 seconds total**, down from the 10–15s estimate with NeMo. The typing indicator in the Discord bot masks this latency.

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
  chroma_data:
```

> **Why two separate Ollama instances share the same `NVIDIA_VISIBLE_DEVICES=0`?** Ollama manages its own VRAM allocation. Both instances can reference the same GPU — the proxy enforces that only one swappable model is loaded at a time. The permanent instance holds exactly one model forever. Docker resource constraints don't need GPU isolation here since we're self-policing via the proxy lock.

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

> **Contention note:** If a Discord request arrives while Brain is active, it queues behind the Brain's current generation. Brain is never evicted mid-response. Discord users see a typing indicator and wait. This is acceptable for a hobby bot — if concurrent throughput becomes a real need, a second GPU eliminates the problem entirely.

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

### Phase 1 — Prototype (2–4 days)
- [ ] Stand up dual Ollama instances + proxy + Docker Compose
- [ ] Register mimic Modelfiles using **system prompt personas only** (no LoRA yet)
- [ ] Verify Qwen3.5-9B-Uncensored loads and responds correctly via Ollama
- [ ] Confirm `thinking false` parameter suppresses chain-of-thought tokens in output
- [ ] Wire Discord bot with mention routing and typing indicators
- [ ] Test basic swap cycle: mimic → lore → mimic
- [ ] Validate VRAM budget under real swap load

### Phase 2 — RAG + Lore (1–2 weeks)
- [ ] Ingest Discord history export into ChromaDB
- [ ] Wire RAG retrieval into lore assistant Modelfile context
- [ ] Test lore+mimic sequential chain (Step 8.2 flow)
- [ ] Tune lore assistant temperature and retrieval `top_k`

### Phase 3 — LoRA Persona Refinement (ongoing)
- [ ] Collect 500–1000 messages per member from Discord history
- [ ] Fine-tune LoRA adapters on Qwen3.5-9B base using Unsloth (QLoRA, RTX 3090)
- [ ] Merge adapters into full models: `deepleffen_<member>_v2.gguf`
- [ ] Re-register Modelfiles pointing to merged GGUFs — **zero proxy changes required**
- [ ] A/B test merged vs. system-prompt personas

> **LoRA training note:** Unsloth supports Qwen3.5 fine-tuning natively as of March 2026. The uncensored base weights are the correct starting point for LoRA — you're training style on top of an already-unlocked model, which means the adapter doesn't need to fight the base model's refusal tendencies.

### Phase 4 — Hardening (optional)
- [ ] Per-user rate limiting (5 requests/min default, configurable per member)
- [ ] Queue depth cap (reject if >3 requests queued, return Discord ephemeral error)
- [ ] Graceful Brain priority: Brain requests can optionally preempt queued Discord requests with configurable precedence
- [ ] Disclaimer stripping in post-processing (catch the occasional baked-in lore assistant disclaimer)

---

## 11. Key Configuration Parameters

| Parameter | Mimic (Qwen3.5-9B) | Lore (Gemma3-12B) | Brain (Qwen3.5-35B) |
|---|---|---|---|
| temperature | 0.85 | 0.3 | 0.2 |
| top_k | 40 | 20 | 10 |
| top_p | 0.9 | 0.8 | 0.9 |
| presence_penalty | 1.3 | 0.0 | 0.0 |
| num_ctx | 8192 | 16384 | 40960 |
| num_predict | 512 | 1024 | -1 |
| thinking | false | false | false* |
| num_parallel | 2 | 1 | 1 |

> *Brain can have thinking enabled per-request for complex multi-step reasoning. Disabled by default for chat latency.

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

---

## 13. What Changed from v2.0

| Item | v2.0 | v2.1 |
|---|---|---|
| Mimic base model | `mistral-nemo:12b` Q6_K (~10.5 GB) | `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` Q6_K (~7.4 GB) |
| Mimic VRAM | ~10.5 GB | ~7.4 GB |
| Mimic context window | 128k | 262k (capped to 8k in Modelfile) |
| Content filtering | Standard instruction-tuned (soft refusals) | Fully abliterated (0/465 refusals) |
| Ollama compatibility | ✅ | ✅ (confirmed as of recent Ollama release) |
| Mimic swap time (est.) | ~4–6s | ~3–5s |
| Lore assistant | `gemma3:12b` Q6_K (~9.5 GB) | Unchanged |
| Brain | `qwen3.5:35b-a3b` Q4_K_M (~17.8 GB) | Unchanged |
| Autocomplete | `qwen2.5-coder:1.5b` Q8_0 (~1.5 GB) | Unchanged |