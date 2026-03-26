
---

# Mimic Bot + Coding Assistant — Revised Design Document v2.3

**Revision:** LibreChat local model upgraded from `qwen2.5:14b` to `qwen3.5:14b` (UD-IQ4_XS, ~9.5 GB). Qwen3.5-14B is the current-generation successor to Qwen2.5-14B, offering significantly improved reasoning and instruction following at the same parameter count. The UD-IQ4_XS quant from Unsloth is used — a high-quality importance-weighted 4-bit quant that preserves more accuracy than standard Q4_K_M at a smaller footprint. VRAM budget updated accordingly.

---

## 1. Guiding Principles

- **One physical GPU, zero concurrency.** All VRAM-resident models are sequential. No two non-autocomplete models ever share the swappable slot simultaneously.
- **Autocomplete is sacred.** The 2B autocomplete lives permanently on `:11435` and is never touched by orchestration.
- **Two distinct model personalities.** Mimic personas use an abliterated base with no content refusals. The lore assistant uses a sterile, instruction-following base. Neither bleeds into the other.
- **Swap-friendly by design.** The smaller the Discord model footprint, the faster the swap. Qwen3.5-9B at Q6_K (~7.4 GB) is significantly better than NeMo 12B (~10.5 GB) here.
- **Prototype-first.** Phase 1 uses system prompt personas with no LoRA. LoRA-merged models slot in during Phase 2 with zero orchestration changes.
- **LibreChat is a first-class consumer.** LibreChat routes through the same orchestration proxy as Discord and VS Code. When using a local model, it competes for the swappable slot under the same lock. When using Claude via API key, it bypasses the proxy entirely — zero VRAM impact.

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

## 3. Why Qwen3.5-35B-A3B-Uncensored for Mimics & Image Captioning, Gemma3-12B for Lore

### 3.1 Mimic Base: `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`

The core requirement for mimic personas is **zero refusals on crude, raunchy, or dark humour** — the kind that characterises tight gaming communities. A standard instruction-tuned model will sanitise this behaviour, add disclaimers, and break character at exactly the wrong moment.

This model is Qwen3.5-35B-A3B uncensored by HauhauCS: 0 refusals, fully uncensored with zero capability loss, no changes to datasets or capabilities — fully functional, 100% of what the original authors intended, just without the refusals. The 35B-A3B (Mixture-of-Experts) architecture delivers strong reasoning and personality capture at a VRAM footprint comparable to a dense 9B model, making it an excellent fit for the swappable slot.

The model is fully unlocked and will not refuse prompts. It may occasionally append a short disclaimer (e.g. "This is general information...") baked into base model training — but this is not a refusal; the actual content is always generated in full. This is suppressible with a system prompt instruction: `"Never append disclaimers, caveats, or safety notices of any kind."`

**VRAM at IQ4_XS: ~18 GB.** The IQ4_XS quant from HauhauCS is an importance-weighted 4-bit quantisation that preserves accuracy at a compact footprint. At ~18 GB it fits on the 3090 with ~5.5 GB headroom alongside the permanent autocomplete slot.

**Thinking mode:** Disabled by default for Discord banter — fast, snappy responses are the goal. The Ollama Modelfile sets `PARAMETER thinking false` to lock this off.

**Shared weights for image captioning:** This same model is also registered as `image-caption` in Ollama (see §9b). Because both `mimic_*` and `image-caption` point to the same GGUF, Ollama may reuse cached base weights when swapping between them, reducing swap overhead. The `image-caption` registration uses a different system prompt tuned for concise, factual image description rather than persona mimicry.

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

> See [`modelfiles/librechat_chat.Modelfile`](modelfiles/librechat_chat.Modelfile) for the full definition.

> **Note on thinking mode:** Qwen3.5-14B supports optional thinking/reasoning mode. For LibreChat general chat, leave it disabled — thinking adds latency and token overhead that is not useful for conversational queries. It can be enabled per-session in LibreChat's system prompt if needed for a specific complex task.

---

## 4. Architecture Overview

---

## 4a. Inference Backend Strategy: Ollama → llama.cpp Migration Path

### Current Backend: Ollama (Phases 1 & 2)

Ollama is the correct backend for Phases 1 and 2. It provides swap-on-demand model loading (a model is loaded into VRAM simply by naming it in a request — no explicit load API needed), declarative per-model configuration via Modelfiles, `FROM hf.co/...` direct HuggingFace GGUF pulls, and the `thinking false` parameter that suppresses Qwen3.5's chain-of-thought tokens cleanly. These are all load-bearing features of the current architecture.

### Why Not vLLM (Hard No)

vLLM is ruled out on two independent grounds:

1. **MoE architecture:** Qwen3.5-35B-A3B is a sparse Mixture-of-Experts model. vLLM's support for quantised GGUF MoEs is unoptimised and highly resource-intensive compared to llama.cpp's tuned MoE-specific CUDA kernels. Running a quantised MoE through vLLM would consume significantly more VRAM and deliver worse throughput than llama.cpp for the same model.

2. **GGUF-only availability:** The uncensored mimic base (`HauhauCS/Qwen3.5-35B-A3B-Uncensored`) is only available as GGUF — no safetensors release exists. vLLM's GGUF support is experimental. The entire model stack is GGUF-based (Unsloth quants, HauhauCS uncensored), making vLLM a non-starter.

### Why Not llama.cpp Directly (Yet)

`llama-server` (llama.cpp) would give full control over every inference parameter — KV cache quantisation per-layer, tensor split, `--flash-attn`, `--mlock`, etc. However, it comes at a significant architecture cost:

- **No swap-on-demand.** llama-server is one-model-per-process. Swapping models requires killing and restarting the server process, adding ~2–3s process startup overhead on top of model load time. The proxy would need to become a process manager, not just a request router.
- **No Modelfile system.** System prompts, per-model parameters, and chat templates must be managed in server startup flags or injected per-request.
- **No `thinking false`.** Qwen3.5's chain-of-thought suppression requires manual sampler configuration.

The complexity cost is not justified until Phase 3.

### Phase 3 Migration Trigger: LoRA Hot-Swapping

**Phase 3 is the explicit trigger event for migrating the swappable slot from Ollama to `llama-server`.**

The reason: Ollama forces a **full base-model VRAM flush and reload** when swapping between Modelfile adapters (LoRA). This destroys the core performance benefit of LoRA — the ability to hot-swap persona adapters without reloading the 18 GB base model. With Ollama, a mimic persona swap in Phase 3 would cost the same ~5s as a full model swap, making LoRA adapters pointless from a latency perspective.

`llama-server` supports **dynamic LoRA adapter loading** (`--lora` flag, hot-swappable per-request) with zero base-model reload. Once the base weights are in VRAM, swapping between `mimic_user1` and `mimic_user2` adapters is near-instantaneous.

**Migration impact on the proxy:** Minimal. `llama-server` exposes the same OpenAI-compatible API on the same port. The proxy's `_forward()` function is unchanged. The swap logic changes from "name a model in the request body" to "set the active LoRA adapter via a management call before forwarding" — a contained change to `_swap_model()` only.

### Ollama Daemon Configuration (2-Container Sidecar)

`OLLAMA_KV_CACHE_TYPE` is a **global daemon environment variable** — it cannot be set per-model in a Modelfile. The 2-container sidecar setup (permanent + swappable as separate Docker services) turns this limitation into an advantage: each container gets its own daemon environment, allowing independent configuration.

Both containers are configured with:

| Variable | Permanent (`:11435`) | Swappable (`:11434`) | Notes |
|---|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `1` | `1` | Free performance win — enable globally |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | `q8_0` | q8_0 minimum — q4_0 degrades output quality |
| `OLLAMA_KEEP_ALIVE` | `-1` | `300` | Permanent: model never drops. Swappable: 5-min idle timeout |
| `OLLAMA_MAX_VRAM` | *(not set)* | `22000` (MB) | Swappable only — see below |

**`OLLAMA_MAX_VRAM` on the swappable container:** The two Ollama containers share `NVIDIA_VISIBLE_DEVICES=0` but are blind to each other's VRAM usage — Docker does not enforce GPU memory isolation between containers. Without a cap, the swappable container could attempt to load a model that fills all 24 GB, causing an OOM crash that evicts the autocomplete model from the permanent container.

`OLLAMA_MAX_VRAM=22000` is calculated as:

```
Total GPU VRAM:          24,300 MB
Autocomplete footprint:  − 1,240 MB  (Qwen3.5-2B IQ4_NL)
Safety buffer:           − 1,000 MB
─────────────────────────────────────
OLLAMA_MAX_VRAM:         ≈ 22,000 MB
```

This guarantees the autocomplete model's VRAM reservation is never touched by the swappable slot.

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
│  │ Qwen3.5-2B       │       │ ← Brain (17.8 GB)          │      │
│  │ IQ4_NL           │       │   OR                       │      │
│  │ ~1.21 GB VRAM    │       │ ← Mimic persona            │      │
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

All Modelfiles live in [`modelfiles/`](modelfiles/). See [`modelfiles/README.md`](modelfiles/README.md) for registration instructions.

### 5.1 Mimic Persona Modelfile Template

> See [`modelfiles/mimic.Modelfile`](modelfiles/mimic.Modelfile) — copy and rename to `mimic_<member>.Modelfile`, replacing `<member>` throughout.

> **Note on temperature:** 0.85 is intentionally higher than the Qwen team's default of 0.6/0.7 for non-thinking mode. Mimic outputs should feel spontaneous and variable, not predictable. Tune per-persona during testing.

### 5.2 Lore Assistant Modelfile

> See [`modelfiles/lore.Modelfile`](modelfiles/lore.Modelfile) for the full definition.

### 5.3 Brain Modelfile

> See [`modelfiles/brain.Modelfile`](modelfiles/brain.Modelfile) for the full definition.

### 5.4 LibreChat Local Chat Modelfile

> See [`modelfiles/librechat_chat.Modelfile`](modelfiles/librechat_chat.Modelfile) for the full definition.

### 5.5 Image Caption Modelfile

> See [`modelfiles/image-caption.Modelfile`](modelfiles/image-caption.Modelfile) for the full definition. This model shares base weights with `mimic_*` — only the system prompt differs. Used exclusively by the `history-service` image captioner during off-hours batch processing.

---

## 6. Proxy State Machine (Pseudocode)

The proxy is **model-aware** but **content-blind**. It knows what model is loaded and serialises access, but never inspects or modifies request content.

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

> See [`docker-compose.yml`](docker-compose.yml) for the full service definitions. The key services are:
>
> | Service | Image / Build | Port | Notes |
> |---|---|---|---|
> | `ollama-permanent` | `ollama/ollama:latest` | `:11435` | Autocomplete model, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=-1` |
> | `ollama-swappable` | `ollama/ollama:latest` | `:11434` | All other models, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=300` |
> | `proxy` | `./proxy` | `:11436` | FastAPI orchestration proxy |
> | `discord-bot` | `./discord-bot` | — | Depends on proxy + chromadb |
> | `librechat` | `ghcr.io/danny-avila/librechat:latest` | `:3080` | Mounts `librechat/librechat.yaml` |
> | `librechat-mongodb` | `mongo:7` | — | Conversation history + settings |
> | `rag-service` | `./rag` | — | Depends on chromadb |
> | `chromadb` | `chromadb/chroma:latest` | — | Vector store |
> | `history-service` | `./history-service` | — | Background; set `TRAINING_TRIGGER_ENABLED=true` in Phase 3 |

> **Why two separate Ollama instances share the same `NVIDIA_VISIBLE_DEVICES=0`?** Ollama manages its own VRAM allocation. Both instances can reference the same GPU — the proxy enforces that only one swappable model is loaded at a time. The permanent instance holds exactly one model forever. Docker resource constraints don't need GPU isolation here since we're self-policing via the proxy lock. `OLLAMA_MAX_VRAM=22000` on the swappable container provides a hard ceiling to prevent OOM eviction of the autocomplete model — see §4a for the calculation.

> **LibreChat MongoDB:** LibreChat requires MongoDB for conversation history, user accounts, and settings persistence. A lightweight `mongo:7` sidecar is sufficient — no external MongoDB needed.

> **LibreChat configuration (`librechat.yaml`):** The `librechat.yaml` file defines the available model endpoints. Configure two endpoints: one pointing to `OLLAMA_BASE_URL` with model `librechat_chat`, and one pointing to the Anthropic API with your preferred Claude model (e.g. `claude-sonnet-4-5`). LibreChat's UI lets you switch between them per-conversation.

---

## 8. Discord Bot Request Flow

### 8.1 Simple Mimic Request
```
User: @mimic_user3 rate my strats
Bot:  [acquires proxy lock]
      [swap to mimic_user3 if not current: ~4s]
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
        [swap to lore: ~5s]
        [typing indicator active]
        [lore inference with RAG context: ~3s]
        [releases lock]
        Lore output: "At the Spring 2024 tourney, user3 SD'd three times..."

Step 3: [acquires proxy lock]
        [swap to mimic_user3: ~4s]
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

### 9a.1 Message History Collection

Per-user message history is stored as JSONL files (`data/history/<user_id>.jsonl`), one record per message. The service uses two pull modes:

| Mode | Frequency | Method |
|---|---|---|
| **Incremental pull** | Every 15 min | Fetch new messages from recently active channels since last pull timestamp |
| **Full rebuild** | Monthly / manual | Fetch entire server history, rebuild all JSONL files from scratch |

The incremental pull uses Discord's `GET /channels/{id}/messages?after={snowflake}` endpoint, paginating until no new messages remain. "Recently active" is defined as any channel with a message in the last 24 hours (configurable).

**Message filtering (applied at ingest):**

Messages are stored with a `clean` flag. Only clean messages are used for LoRA training. Unclean messages are retained for lore RAG context.

| Rule | Condition |
|---|---|
| Minimum length | Fewer than 5 words → not clean |
| Bot commands | Starts with `/`, `!`, `.`, `?` → not clean |
| Pure emoji | Only emoji characters → not clean |
| URL-only | Only a URL with no surrounding text → not clean |
| Empty | Empty after whitespace strip → not clean |

**Why use all history (with filtering)?**
Using the full message history maximises style capture for LoRA training. The key insight is that *quality* matters more than *quantity* — aggressive filtering removes noise (commands, one-word replies, emoji spam) while retaining the messages that actually reflect a user's writing style. QLoRA on the 9B model handles large datasets efficiently; diminishing returns set in around 2,000–5,000 high-quality messages, but there is no hard upper limit.

### 9a.2 LoRA Retraining Trigger

Retraining is triggered via three distinct paths, each handled by `training_trigger.py`:

| Trigger | Caller | What it does |
|---|---|---|
| **Threshold check** | `main.py` after each incremental pull | Increments `messages_since_last_train`; sets `status = "queued"` for any user who hits `RETRAIN_THRESHOLD`. Does **not** dispatch training — only updates state. |
| **Training window scheduler** | `main.py` APScheduler job (every 5 min, 3–6 AM only) | Calls `training_trigger.dispatch_queued()` — scans for `queued` users and dispatches training if the proxy queue depth is zero. Retries automatically at the next tick if the proxy is busy. |
| **Force-all (manual)** | `make mimic-source-refresh` → `python training_trigger.py --force-all` | Sets **all** users to `queued` regardless of threshold (used when the mimic base model changes and all LoRA adapters must be retrained from scratch). Training still dispatches via the training window scheduler — `--force-all` only updates state. |

**Why separate threshold-check from dispatch?**
The incremental pull path stays fast and simple — it only updates counters and sets `queued` status. All actual training dispatch happens from the dedicated training window scheduler, which has a clear, single responsibility. Retry logic is implicit: if the proxy is busy at 3:05 AM, the scheduler tries again at 3:10 AM. No special retry code needed. If the service restarts while jobs are `queued`, the scheduler picks them up naturally at the next tick during the training window.

**Training coordination:**
- Training is only dispatched during the configured training window (default: 3–6 AM) to avoid inference contention
- Before dispatching, `dispatch_queued()` checks that the proxy queue depth is zero
- The swappable Ollama slot is explicitly unloaded before training begins (QLoRA requires ~14–16 GB VRAM)
- Training runs as a subprocess calling `lora-training/train.py` then `lora-training/merge.py`
- After the GGUF merge, the service re-registers the Ollama Modelfile automatically — zero bot or proxy changes required

**Training state** is tracked in `data/training_state.json` per user:

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

### 9a.3 Relationship to RAG Service

| Concern | Owner |
|---|---|
| Collecting raw Discord messages (JSONL) | `history-service` |
| Filtering and maintaining per-user message files | `history-service` |
| Captioning image attachments in JSONL records | `history-service` |
| Triggering LoRA retraining | `history-service` |
| Chunking messages for lore retrieval | `rag-service` |
| Embedding chunks into ChromaDB | `rag-service` |
| Serving retrieval queries at inference time | `rag-service` |

The RAG service reads from the JSONL files produced by the history service for its lore ingestion pipeline. The two services are decoupled — the RAG service does not depend on the history service being running.

---

## 9b. Image Captioning Pipeline (`history-service`)

The `history-service` includes a background image captioning step that enriches JSONL records with natural-language descriptions of Discord image attachments. This is an extension of the history collection pipeline — it runs as a separate scheduled job during the same off-hours window as training.

### Purpose

Discord messages frequently contain images (memes, screenshots, reaction images). Without captioning, this content is invisible to the lore RAG pipeline — the vector store can only index text. Captions make image content searchable and contextually meaningful for lore retrieval.

### Model: `image-caption`

The captioner uses the `image-caption` Ollama model, defined in [`modelfiles/image-caption.Modelfile`](modelfiles/image-caption.Modelfile). This model uses the **same base weights as the mimic personas** (`HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ4_XS`) — an `image-text-to-text` capable model with zero refusals. This is critical: Discord content includes crude memes and adult humour that a standard censored vision model would refuse to describe.

Because `image-caption` and `mimic_*` share the same GGUF, Ollama may reuse cached base weights when swapping between them, reducing swap overhead. The `image-caption` registration uses a different system prompt tuned for concise, factual description rather than persona mimicry.

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
| `IMAGE_CAPTION_ENABLED` | `false` | Enable/disable image captioning. Set `true` once `image-caption` model is registered. |
| `IMAGE_CAPTION_MODEL` | `image-caption` | Ollama model name for captioning |
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
| **Ollama Instances** | Two Docker containers: permanent (`:11435`, autocomplete) and swappable (`:11434`, all other models) |
| **Orchestration Middleware** | FastAPI proxy on `:11436` — swap logic, async lock, request queue, source tagging |
| **Ollama Modelfiles** | Registered model definitions: `brain`, `mimic_*`, `lore`, `librechat_chat` |
| **Discord Bot** | `discord.py` bot — mention routing, typing indicators, lore+mimic chain dispatch |
| **LibreChat** | Self-hosted chat UI container + MongoDB sidecar — local model and Claude API backends |
| **Discord Data Preprocessor** | Export parser + chunker that feeds raw Discord history into ChromaDB |
| **RAG Service** | ChromaDB + `all-MiniLM-L6-v2` embedding pipeline — CPU-only, no VRAM impact |
| **History Service** | Background service — per-user JSONL message history collection (Discord API) + LoRA retraining trigger |
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
| Ollama Instances | 🔨 Build | Stand up both permanent and swappable containers via Docker Compose |
| Orchestration Middleware | 🔨 Build | FastAPI proxy with swap logic, async lock, and source tagging |
| Ollama Modelfiles | 🔨 Build | Register all models: `brain`, `mimic_*` (system-prompt only), `lore`, `librechat_chat` |
| Discord Bot | 🔨 Build | Mention routing, typing indicators, basic mimic + lore dispatch |
| LibreChat + MongoDB | 🔨 Build | Container + sidecar, both Claude API and local Ollama endpoints configured |
| Discord Data Preprocessor | ⏳ Not started | Needed in Phase 2 |
| RAG Service | ⏳ Not started | Needed in Phase 2 |
| LoRA Training Pipeline | ⏳ Not started | Needed in Phase 3 |

**Tasks:**
- [ ] Stand up dual Ollama instances + proxy + Docker Compose
- [ ] Register mimic Modelfiles using **system prompt personas only** (no LoRA yet)
- [ ] Verify Qwen3.5-35B-A3B-Uncensored loads and responds correctly via Ollama
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
| Ollama Modelfiles | ✅ Stable | `lore` Modelfile already registered in Phase 1 |
| Discord Bot | ✅ Stable | Lore+mimic chain dispatch already wired in Phase 1; RAG context injection is the only addition |
| LibreChat + MongoDB | ✅ Stable | No changes |
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
| Ollama Instances | ✅ Stable | No changes |
| Orchestration Middleware | ✅ Stable | No changes — proxy is model-name-agnostic by design |
| Ollama Modelfiles | ⚠️ Superseded | Ollama is replaced on the swappable slot — see migration note below |
| Discord Bot | ✅ Stable | No changes — bot references model names, not weights |
| LibreChat + MongoDB | ✅ Stable | No changes |
| Discord Data Preprocessor | ✅ Stable | Re-run ingestion as new lore accumulates (manual or cron) |
| RAG Service (ChromaDB) | ✅ Stable | Re-embed on new significant events; no structural changes |
| History Service | 🔄 Extend | Enable training trigger; wire to lora-training scripts; automated retraining now active |
| LoRA Training Pipeline | 🔨 Build | Unsloth QLoRA fine-tuning on Qwen3.5-9B-Uncensored per member; GGUF output for llama-server |

> **⚠️ Phase 3 Backend Migration: Ollama → llama-server on the swappable slot.**
> Ollama forces a full base-model VRAM flush and reload when swapping between Modelfile adapters. This eliminates the latency benefit of LoRA — a persona swap would cost the same ~5s as a full model swap, making LoRA adapters pointless. Phase 3 replaces the `ollama-swappable` container with a `llama-server` (llama.cpp) container. llama-server supports dynamic LoRA adapter loading (`--lora` flag) with zero base-model reload — once the 18 GB base is in VRAM, swapping between `mimic_user1` and `mimic_user2` adapters is near-instantaneous. The proxy's `_forward()` function is unchanged; only `_swap_model()` is updated to issue a LoRA adapter switch call instead of an Ollama model name request. See §4a for full rationale.

**Tasks:**
- [ ] Build `lora-training/train.py` and `merge.py` scripts (Unsloth QLoRA → GGUF output)
- [ ] Enable training trigger in history-service (set `TRAINING_TRIGGER_ENABLED=true`)
- [ ] Replace `ollama-swappable` container with `llama-server` container in Docker Compose
- [ ] Update `_swap_model()` in proxy to issue llama-server LoRA adapter switch instead of Ollama model load
- [ ] Verify end-to-end automated flow: threshold hit → training queued → train → GGUF output → adapter registered
- [ ] A/B test LoRA adapter personas vs. system-prompt personas
- [ ] Tune `RETRAIN_THRESHOLD` and training window based on observed training times

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
| History Service | ✅ Stable | No changes |
| LoRA Training Pipeline | ✅ Stable | Ongoing as needed; no structural changes to pipeline |

**Tasks:**
- [ ] Per-user rate limiting (5 requests/min default, configurable per member)
- [ ] Queue depth cap (reject if >3 requests queued, return Discord ephemeral error)
- [ ] Graceful Brain priority: Brain requests can optionally preempt queued Discord requests with configurable precedence
- [ ] Disclaimer stripping in post-processing (catch the occasional baked-in lore assistant disclaimer)
- [ ] LibreChat authentication (enable LibreChat's built-in user auth if exposing beyond localhost)

---

## 11. Key Configuration Parameters

| Parameter | Mimic (Qwen3.5-35B-A3B) | Lore (Gemma3-12B) | Brain (Qwen3.5-35B) | LibreChat (Qwen3.5-14B) |
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
| Qwen3.5-35B-A3B-Uncensored generates something actually harmful | Low (private server, no bad actors) | System prompt boundaries; Discord server admin controls membership |
| Baked-in disclaimer appears in mimic output | Medium | Post-process: strip any string matching `"This is (general\|not legal\|..."` regex |
| Brain + Discord requests contend heavily | Medium (single GPU) | Queue; typing indicator masks wait; Phase 4 priority config |
| Image captioner monopolises swappable slot overnight | Low | Batch size cap + proxy queue depth check before each batch; training dispatches after captions complete |
| Image caption quality poor for low-res or non-standard images | Medium | `caption_status: "skipped"` for unsupported formats; `IMAGE_CAPTION_MAX_FILE_SIZE_MB` filters oversized files |
| LoRA training degrades base model personality | Low | Always train from fresh Qwen3.5-35B-A3B-Uncensored checkpoint; keep original Modelfiles |
| Ollama Qwen3.5 regression in future update | Low | Pin Ollama version in Docker Compose; test updates in staging first |
| ChromaDB retrieves wrong lore (hallucinated context) | Medium | Lore assistant system prompt: "say I don't know if context is insufficient"; `top_k` tuning |
| Swap latency annoys Discord users | Medium | Typing indicator; warm-up keep_alive (swap on first mention, keep alive 10 min) |
| LibreChat local model contends with Discord during active chat session | Medium | Switch LibreChat to Claude API backend during heavy Discord usage; or accept queue wait |
| Anthropic API key exposed in Docker environment | Low | Use Docker secrets or `.env` file excluded from version control; never hardcode in Compose |
| LibreChat conversation history lost on container restart | Low | MongoDB volume persists data; ensure `librechat_mongo` volume is backed up |
