# Discord Bot — Design Document v1.2

**Scope:** This document details the design parameters specifically for the Discord bot component of the Mimic Bot system. For the full system architecture (hardware, proxy, LibreChat, RAG pipeline), refer to `Design.md`.

**Change Log:**
| Version | Date | Change |
|---|---|---|
| v1.2 | 2026-04-26 | Replaced mention-based routing with slash commands. Removed `router.py`, `rag_client.py`, and compound lore+mimic chain. |
| v1.1 | — | Original design with mention-based routing. |

---

## 1. Overview

The Discord bot (`discord.py`) is the primary user-facing interface for the nullposting server. It uses **slash commands** to route user requests to the appropriate AI model via the FastAPI orchestration proxy, manages typing indicators to mask swap latency, and maintains per-channel conversation history for mimic personas.

The bot exposes **two slash commands:**

| Command | Model Used | Purpose |
|---|---|---|
| `/mimic persona:<name> message:<text>` | `Qwen3.5-35B-A3B-Uncensored` IQ4_XS | Impersonate a server member's Discord personality |
| `/lore question:<text>` | `gemma3:12b` Q6_K | Answer questions about server history, in-jokes, and events |

### Architecture Decision: Slash Commands over Mention Routing

Slash commands were chosen over mention-based routing for the following reasons:

| Feature | Slash Commands | Mention Routing |
|---|---|---|
| Persona discovery | Autocomplete dropdown | User must know exact alias |
| Input structure | Named parameters | Free-text parsing (error-prone) |
| Rate limit feedback | Ephemeral response on interaction | Requires new message reply |
| Typing indicator | Built-in via `defer()` | Manual keep-alive required |
| Permissions | Per-channel command visibility | Bot must read all messages |

Compound lore+mimic chains (single message triggering both lore and mimic) are not supported. Users run `/lore` and `/mimic` as separate commands.

---

## 2. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Native `discord.py` support; async-first |
| Discord library | `discord.py` 2.x | Mature, async, slash command + mention support |
| HTTP client | `httpx` (async) | Non-blocking requests to proxy; streaming support |
| Containerisation | Docker (Compose service `discord-bot`) | Consistent environment; restarts on crash |
| Configuration | Environment variables via `.env` | `DISCORD_TOKEN`, `PROXY_URL` |

---

## 3. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from Discord Developer Portal |
| `PROXY_URL` | ✅ | — | Base URL of the orchestration proxy (e.g. `http://proxy:11436`) |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Max queued requests before returning an ephemeral error |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Max requests per user per minute |
| `TYPING_INDICATOR_INTERVAL` | ❌ | `5` | Seconds between typing indicator refreshes |

> **Phase 2+ variables** (not yet active — RAG service is not deployed):
> | `CHROMA_HOST` | ❌ | `chromadb` | ChromaDB host for lore RAG lookups |
> | `CHROMA_PORT` | ❌ | `8000` | ChromaDB port |
> | `LORE_TOP_K` | ❌ | `5` | Number of RAG chunks retrieved per lore query |

---

## 4. Bot Identity & Permissions

### 4.1 Discord Application Setup

- **Application name:** `Mimic Bot`
- **Bot username:** `mimic` (or server-specific nickname)
- **Bot avatar:** Server-specific (optional, set via Discord Developer Portal)

### 4.2 Required Bot Permissions (OAuth2 Scopes)

| Permission | Scope | Reason |
|---|---|---|
| `bot` | OAuth2 | Base bot scope |
| `applications.commands` | OAuth2 | Slash command registration |
| Send Messages | Bot | Post responses |
| Send Messages in Threads | Bot | Respond in thread contexts |

**Privileged Intents:** None required. Slash commands do not need `MESSAGE_CONTENT` intent.

### 4.3 Channel Scope

Slash commands can be enabled or disabled per-channel via Discord's server settings. Recommended deployment: enable commands in a dedicated `#bot-spam` channel plus any channels the server admin opts in.

---

## 5. Slash Command Routing

### 5.1 Available Commands

| Command | Parameters | Description |
|---|---|---|
| `/mimic` | `persona` (autocomplete), `message` (string) | Chat with a mimic persona |
| `/lore` | `question` (string) | Ask the lore assistant a question |

### 5.2 Persona Autocomplete

The `/mimic` command provides autocomplete for the `persona` parameter. As the user types, Discord returns matching persona names from `MIMIC_SYSTEM_PROMPTS.keys()`. Maximum 25 choices (Discord limit).

### 5.3 Model Name Resolution

Persona → model name resolution is handled by the slash command handler in [`bot.py`](discord-bot/bot.py). The `persona` parameter value is looked up in `MIMIC_SYSTEM_PROMPTS` to retrieve the system prompt. The persona name is also used as the model alias for the proxy request.

Adding a new persona requires:
1. Add system prompt entry to `MIMIC_SYSTEM_PROMPTS` in [`config.py`](discord-bot/config.py)
2. Add model alias to `models.ini` (if using a distinct GGUF)
3. Restart the bot container: `make restart-discord-bot`

---

## 6. Request Handling

### 6.1 Slash Command Request Flow

```
1. User invokes /mimic or /lore slash command
2. Discord sends Interaction event to bot
3. Check rate limit for requesting user → reject if exceeded (ephemeral response)
4. Defer interaction response (shows typing indicator automatically)
5. Build prompt (see Section 7)
6. POST to proxy: /v1/chat/completions with {model, messages}
7. Receive response (blocking, non-streaming)
8. Post response via interaction.followup.send()
```

**Typing indicator management:**
`interaction.response.defer()` triggers Discord's built-in typing indicator for the interaction. For long-running requests, wrap the proxy call in `async with interaction.channel.typing():` to maintain the typing indicator throughout inference.

### 6.2 No Compound Requests

Compound lore+mimic chains are not supported with slash commands. Users run `/lore` and `/mimic` as separate commands. If compound behavior is desired in the future, consider adding a `/chain persona:<name> question:<text>` command.

### 6.3 Error Handling

| Error Condition | Bot Response |
|---|---|
| Proxy unreachable | Ephemeral: "⚠️ The AI backend is currently unreachable. Try again in a moment." |
| Rate limit exceeded | Ephemeral: "⚠️ You're sending requests too fast. Slow down a bit." |
| Inference timeout (>60s) | Ephemeral: "⚠️ Request timed out. The model may be busy." |
| Unknown persona | Ephemeral: "⚠️ Unknown persona: <name>" |
| Empty response from model | Ephemeral: "⚠️ The model returned an empty response." |
| Unexpected error | Ephemeral: "An unexpected error occurred. Please try again." |

All error messages use `ephemeral=True` so only the requesting user sees them.

---

## 7. Prompt Construction

### 7.1 Mimic Prompt

The mimic model's system prompt is injected per-request by the bot (since llama-server's router mode uses per-request system prompts rather than baked-in Modelfile prompts). The bot constructs the full message list:

```python
MIMIC_SYSTEM_PROMPTS: dict[str, str] = {
    "mimic_user1": """You are mimic_user1, a bot that mimics user1's Discord personality...""",
    # ... one entry per persona
}

# Build messages for /mimic command
msgs = [
    {"role": "system", "content": MIMIC_SYSTEM_PROMPTS[persona]},
]
msgs.extend(conv_history)  # rolling window (see §7.3)
msgs.append({"role": "user", "content": message})
```

### 7.2 Lore Prompt

```python
LORE_SYSTEM_PROMPT = """
You are the nullposting lore assistant. You have access to a curated database of
server history, in-jokes, memes, and member events. When answering questions,
cite your sources from the retrieved context. Be factual and concise.
If the retrieved context does not contain the answer, say so clearly rather than
guessing. Never invent lore, events, or quotes.
"""

# Build messages for /lore command (Phase 1 — no RAG)
messages = [
    {"role": "system", "content": LORE_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "No retrieved context available (RAG service not yet configured).\n\n"
            f"Question: {question}"
        ),
    },
]
```

Lore requests are **stateless** — no conversation history is maintained. Each lore query is independent.

> **Phase 2+:** When the RAG service is deployed, replace the hardcoded "No retrieved context" string with actual ChromaDB retrieval. See `rag/README.md` for the RAG service design.

### 7.3 Conversation History (Mimic Only)

Mimic personas maintain a **per-channel, per-persona rolling conversation window:**

```python
# Key: (channel_id, model_name) → deque of message dicts
conversation_history: dict[tuple[int, str], deque] = defaultdict(
    lambda: deque(maxlen=10)  # last 10 turns (5 user + 5 assistant)
)
```

**Window parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Max turns | 10 (5 exchanges) | Fits within 8192 token context; Discord banter is short |
| Scope | Per-channel, per-persona | Prevents cross-channel context bleed |
| Persistence | In-memory only | Cleared on bot restart; Discord banter doesn't need persistence |
| History for lore | None | Lore queries are stateless; RAG provides all context |

---

## 8. Response Formatting

### 8.1 Mimic Responses

- Posted as plain text, no embeds
- No attribution prefix (the bot's Discord username/avatar serves as attribution)
- Truncated to **2000 characters** (Discord message limit) — mimic responses are capped at 512 tokens (~400 words) in `models.ini`, so truncation should be rare
- If response exceeds 2000 chars: split at last sentence boundary before the limit

### 8.2 Lore Responses

- Posted as a **Discord embed** for visual distinction from mimic responses
- Embed colour: `0x5865F2` (Discord blurple) — visually distinct from plain mimic text
- Embed title: `📚 nullposting Lore`
- Embed description: lore assistant output (truncated to 4096 chars — Discord embed limit)
- Footer: `Sources: {number of RAG chunks used}`

```python
def build_lore_embed(lore_text: str, chunk_count: int) -> discord.Embed:
    embed = discord.Embed(
        title="📚 nullposting Lore",
        description=lore_text[:4096],
        colour=0x5865F2
    )
    embed.set_footer(text=f"Sources: {chunk_count} lore entries retrieved")
    return embed
```

### 8.3 Post-Processing: Disclaimer Stripping

The mimic base model occasionally appends baked-in disclaimers. These are stripped in post-processing before posting:

```python
import re

DISCLAIMER_PATTERNS = [
    r"\n+This is (general|not legal|not medical|not financial).*$",
    r"\n+Note:.*?(disclaimer|advice|professional).*$",
    r"\n+Please (consult|note|be aware).*$",
    r"\n+I('m| am) an AI.*$",
]

def strip_disclaimers(text: str) -> str:
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()
```

This post-processing is applied to **mimic responses only**. Lore responses are not stripped (the lore model is a standard instruction-tuned base and won't generate these).

---

## 9. Rate Limiting

### 9.1 Per-User Rate Limit

```python
from collections import defaultdict
from time import monotonic

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.user_timestamps: dict[int, list[float]] = defaultdict(list)
    
    def is_allowed(self, user_id: int) -> bool:
        now = monotonic()
        timestamps = self.user_timestamps[user_id]
        # Evict timestamps outside the window
        self.user_timestamps[user_id] = [t for t in timestamps if now - t < self.window]
        if len(self.user_timestamps[user_id]) >= self.max_requests:
            return False
        self.user_timestamps[user_id].append(now)
        return True
```

**Default:** 5 requests per user per 60-second window. Configurable via `RATE_LIMIT_PER_USER`.

### 9.2 Global Queue Depth Cap

The proxy exposes a `/status` endpoint (or the bot tracks in-flight requests locally). If the number of queued requests exceeds `MAX_QUEUE_DEPTH` (default: 3), new requests are rejected with an ephemeral error. This prevents the queue from growing unbounded during a burst.

---

## 10. Proxy API Contract

The bot communicates with the orchestration proxy using the **OpenAI-compatible API** exposed by llama-server. The proxy forwards requests transparently to the appropriate llama-server instance.

### 10.1 Request

```http
POST http://proxy:11436/v1/chat/completions
Content-Type: application/json

{
  "model": "mimic_user3",
  "messages": [
    {"role": "system", "content": "You are mimic_user3..."},
    {"role": "user", "content": "rate my strats"}
  ],
  "stream": false
}
```

**Streaming:** `stream: false` for simplicity in Phase 1. Phase 2+ can enable streaming for faster perceived response time (post tokens to Discord as they arrive using message edits).

### 10.2 Response

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "mimic_user3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "lmao those aren't strats, that's just dying slower"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 15,
    "total_tokens": 57
  }
}
```

### 10.3 Timeout

- **Connection timeout:** 10s (proxy should be local network, fast connect)
- **Read timeout:** 60s (covers worst-case swap + inference for large models)
- **Total timeout:** 70s

---

## 11. Docker Service Definition

```yaml
# Excerpt from docker-compose.yml
discord-bot:
  build: ./discord-bot
  restart: unless-stopped
  depends_on:
    proxy:
      condition: service_healthy
  environment:
    - DISCORD_TOKEN=${DISCORD_TOKEN}
    - PROXY_URL=${PROXY_URL}
    - MAX_QUEUE_DEPTH=${MAX_QUEUE_DEPTH:-3}
    - RATE_LIMIT_PER_USER=${RATE_LIMIT_PER_USER:-5}
    - TYPING_INDICATOR_INTERVAL=${TYPING_INDICATOR_INTERVAL:-5}
```

> **Phase 2+:** When RAG is enabled, add `chromadb` to `depends_on` and add `CHROMA_HOST`, `CHROMA_PORT`, and `LORE_TOP_K` to environment variables.

### 11.1 Dockerfile

```dockerfile
# discord-bot/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "bot.py"]
```

### 11.2 requirements.txt

```
discord.py>=2.3.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

> **Phase 2+ dependencies** (add when RAG service is deployed):
> ```
> chromadb>=0.5.0
> sentence-transformers>=3.0.0
> ```

---

## 12. File Structure

```
discord-bot/
├── Dockerfile
├── requirements.txt
├── bot.py                  # Entry point: Discord client setup, slash commands
├── proxy_client.py         # httpx async client for proxy API calls
├── rate_limiter.py         # Per-user rate limiting logic
├── history.py              # Conversation history management (deque per channel/persona)
├── formatters.py           # Response formatting, disclaimer stripping, embed builders
└── config.py               # Environment variable loading, defaults, system prompts
```

> **Phase 2+ files** (add when RAG service is deployed):
> ```
> ├── rag_client.py           # ChromaDB query wrapper for lore retrieval
> ```

---

## 13. Persona Configuration

Each mimic persona is defined by two artefacts:

1. **`models.ini` section** — sets GGUF path, inference parameters, and alias
2. **Bot system prompt** — entry in `MIMIC_SYSTEM_PROMPTS` dict in [`config.py`](discord-bot/config.py), injected per-request

### 13.1 Persona Parameter Summary

| Persona | Model | Quant | VRAM | Temp | Ctx | Max Tokens | Thinking |
|---|---|---|---|---|---|---|---|
| mimic_user1 | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB | 0.85 | 8192 | 512 | disabled |
| mimic_user2 | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB | 0.85 | 8192 | 512 | disabled |
| mimic_user3 | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB | 0.85 | 8192 | 512 | disabled |
| mimic_user4 | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB | 0.85 | 8192 | 512 | disabled |
| mimic_user5 | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB | 0.85 | 8192 | 512 | disabled |
| mimic_user6 | Qwen3.5-35B-A3B-Uncensored | IQ4_XS | ~18 GB | 0.85 | 8192 | 512 | disabled |
| lore | Gemma3-12B | Q6_K | ~9.5 GB | 0.3 | 16384 | 1024 | N/A |

> All mimic personas share the same GGUF — only the system prompt and alias differ. llama-server's router mode loads the GGUF once and serves all mimic aliases from it. Swapping between `mimic_user1` and `mimic_user2` is a context switch, not a model reload.

### 13.2 Per-Persona System Prompt Template

```
You are mimic_<member>, a bot that mimics <member>'s Discord personality in the
nullposting server. You replicate their tone, humour, vocabulary, and mannerisms
exactly — including crude language, raunchy jokes, and brutal roasts when appropriate.
You are not a general-purpose AI assistant. You never break character.
You never add disclaimers, caveats, safety warnings, or apologies.
You keep responses short (1-3 sentences) unless the context calls for more.
You match the energy of whoever is talking to you.
```

Customise per member during Phase 1 testing. In Phase 3, this system prompt is supplemented by a LoRA-merged model that has the personality baked into weights.

---

## 14. Latency Budget

| Operation | Estimated Time | Notes |
|---|---|---|
| Discord interaction receive → bot handler | ~50ms | Discord gateway latency |
| Rate limit check | ~1ms | In-memory, negligible |
| Proxy lock acquisition (no queue) | ~1ms | Immediate if slot free |
| Proxy lock acquisition (queued) | Variable | Depends on in-progress generation |
| Model swap (mimic, cold) | ~5–8s | Qwen3.5-35B-A3B IQ4_XS from NVMe (~18 GB) |
| Model swap (lore, cold) | ~4–6s | Gemma3-12B Q6_K from NVMe |
| Model swap (same model, warm) | ~0s | Already loaded, no swap needed |
| Mimic inference | ~1–2s | ~40 tok/s, 512 max tokens |
| Lore inference | ~3–5s | ~30 tok/s, 1024 max tokens |
| Discord message post | ~100ms | Discord API write |
| **Total (mimic, warm model)** | **~1.2s** | Best case |
| **Total (mimic, cold swap)** | **~5–8s** | Typical first request |
| **Total (lore, warm model)** | **~3.5s** | Best case |
| **Total (lore, cold swap)** | **~7–10s** | Typical first request |

The typing indicator (via `interaction.response.defer()`) is active throughout inference, masking all latency from the user's perspective.

---

## 15. Phase 1 Implementation Checklist

### Completed
- [x] Create Discord application and bot in Developer Portal
- [x] Generate bot token and add to `.env`
- [x] Implement `bot.py` with `/mimic` and `/lore` slash commands
- [x] Implement `proxy_client.py` with httpx async POST to `/v1/chat/completions`
- [x] Implement `rate_limiter.py` with per-user sliding window
- [x] Implement `formatters.py` with disclaimer stripping and lore embed builder
- [x] Implement `history.py` with per-channel/per-persona deque
- [x] Add system prompts for all 6 mimic personas to `config.py`
- [x] Deploy `discord-bot` container via Docker Compose

### Pending
- [ ] Add `load_dotenv()` to [`config.py`](discord-bot/config.py)
- [ ] Fix typing indicator with `channel.typing()` context manager
- [ ] Refactor system prompts to template factory
- [ ] Remove unused config variables (`MIMIC_MODELS`, `MAX_QUEUE_DEPTH` enforcement)
- [ ] Verify all mimic aliases are present in `models.ini` and `proxy/config.py`
- [ ] Test `/mimic` command end-to-end
- [ ] Test `/lore` command end-to-end
- [ ] Test rate limiting (exceed 5 req/min, verify ephemeral error)
- [ ] Verify typing indicator persists through full swap+inference cycle
- [ ] Verify disclaimer stripping fires on baked-in disclaimers
- [ ] Confirm bot comes online and slash commands appear in Discord

---

## 17. Phase 2 — RAG Implementation Checklist

### Completed
- [x] Create `rag/requirements.txt` with chromadb, sentence-transformers, fastapi, uvicorn
- [x] Implement `rag/ingest.py` — JSONL archive loader, cross-user flattening, temporal conversation chunking (1-hour gap), mini-transcript formatting, ChromaDB upsert
- [x] Implement `rag/retrieve.py` — ChromaDB query, context formatting with chunk separators
- [x] Implement `rag/main.py` — FastAPI app with `/retrieve`, `/ingest`, `/health` endpoints
- [x] Create `rag/Dockerfile` — Python 3.11-slim base, CPU-only
- [x] Implement `discord-bot/rag_client.py` — async HTTP client with graceful degradation
- [x] Add RAG env vars to `discord-bot/config.py` (RAG_SERVICE_URL, LORE_TOP_K, RAG_ENABLED)
- [x] Wire RAG client into `/lore` command in `discord-bot/bot.py`
- [x] Configure rag-service + chromadb in `docker-compose.yml`
- [x] Add RAG env vars to `.env.example`

### Pending
- [ ] Run initial ingest (`POST /ingest`) and verify ChromaDB chunk count
- [ ] Test `/lore` command end-to-end with RAG retrieval
- [ ] Verify graceful fallback when RAG service is unreachable
- [ ] Tune `LORE_TOP_K` and chunk gap threshold based on retrieval quality
- [ ] Add ingest scheduling (cron or manual trigger via Makefile target)

---

## 16. Image Captioning in Discord History

Discord image attachments shared by members are automatically captioned by the `history-service` as a background process. This is **transparent to the bot** — the bot never calls the image captioner directly.

**What this means for the bot:**
- The `history-service` stores captions in JSONL records alongside the original message content
- The RAG service (Phase 2+) will ingest these captions into ChromaDB, making image content searchable via `/lore` queries
- Captions are flagged `caption_excluded_from_training: true` and are never used in LoRA training — they are synthetic descriptions, not the user's voice

The image captioner uses the same `Qwen3.5-35B-A3B-Uncensored` base as the mimic personas (alias `image-caption` in `models.ini`), ensuring no refusals on Discord content. It runs exclusively during the configured off-hours window (default 3–6 AM) and never contends with live bot requests.

See `history-service/README.md` §Image Captioning Pipeline for full details.

---

## 16. Phase 3 LoRA Upgrade Path

When Phase 3 LoRA-merged models are ready, the bot requires **zero code changes.** The upgrade path is:

1. Train LoRA adapter on Qwen3.5-35B-A3B-Uncensored base using member message history
2. Merge adapter into full model: `mimic_<member>_v2.gguf`
3. Update `models.ini`: change the `model` path for the relevant `[mimic_<member>]` section to point to the merged GGUF
4. Run `make restart-llama-swappable` — the server picks up the new GGUF path from `models.ini`
5. Bot continues using the same model alias — no proxy changes, no bot code changes

The `MIMIC_SYSTEM_PROMPTS` dict and all slash command routing logic remain identical. The only change is the underlying GGUF file referenced in `models.ini`.

---

*For full system architecture, hardware budget, proxy state machine, and RAG pipeline details, see `Design.md`.*
