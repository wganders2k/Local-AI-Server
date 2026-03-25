# Discord Bot — Design Document v1.0

**Scope:** This document details the design parameters specifically for the Discord bot component of the Mimic Bot system. For the full system architecture (hardware, proxy, LibreChat, RAG pipeline), refer to `Design.md`.

---

## 1. Overview

The Discord bot (`discord.py`) is the primary user-facing interface for the nullposting server. It routes member mentions to the appropriate AI model via the FastAPI orchestration proxy, manages typing indicators to mask swap latency, and handles the sequential lore+mimic chain for compound queries.

The bot has **two distinct functional modes:**

| Mode | Trigger | Model Used | Purpose |
|---|---|---|---|
| Mimic | `@mimic_<member>` | `Qwen3.5-9B-Uncensored` Q6_K | Impersonate a server member's Discord personality |
| Lore | `@lore` | `gemma3:12b` Q6_K | Answer questions about server history, in-jokes, and events |

These modes can be chained in a single message (see Section 6.2).

---

## 2. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Native `discord.py` support; async-first |
| Discord library | `discord.py` 2.x | Mature, async, slash command + mention support |
| HTTP client | `httpx` (async) | Non-blocking requests to proxy; streaming support |
| Containerisation | Docker (Compose service `discord-bot`) | Consistent environment; restarts on crash |
| Configuration | Environment variables via `.env` | `DISCORD_TOKEN`, `OLLAMA_PROXY` |

---

## 3. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from Discord Developer Portal |
| `OLLAMA_PROXY` | ✅ | — | Base URL of the orchestration proxy (e.g. `http://proxy:11436`) |
| `BOT_PREFIX` | ❌ | `mimic_` | Prefix used to identify bot mention targets |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Max queued requests before returning an ephemeral error |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Max requests per user per minute |
| `TYPING_INDICATOR_INTERVAL` | ❌ | `5` | Seconds between typing indicator refreshes |
| `LORE_TOP_K` | ❌ | `5` | Number of RAG chunks retrieved per lore query |

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
| Read Messages / View Channels | Bot | Read messages in channels where bot is active |
| Send Messages | Bot | Post responses |
| Send Messages in Threads | Bot | Respond in thread contexts |
| Read Message History | Bot | Context window for conversation history |
| Use External Emojis | Bot | Optional — for persona-flavoured reactions |
| Add Reactions | Bot | Optional — reaction-based acknowledgements |

**Privileged Intents required:**
- `MESSAGE_CONTENT` — required to read message text for mention parsing

> Enable `MESSAGE_CONTENT` intent in the Discord Developer Portal under Bot → Privileged Gateway Intents.

### 4.3 Channel Scope

The bot responds only in channels where it has been explicitly granted access. It does **not** respond in DMs by default (configurable). Recommended deployment: a dedicated `#bot-spam` channel plus any channels the server admin opts in.

---

## 5. Mention Routing

### 5.1 Trigger Pattern

The bot activates on `@mention` of any registered bot user. Each persona is a **separate Discord bot application** or a **single bot with alias routing** — see Section 5.3 for the recommended approach.

**Mention format:**
```
@mimic_user3 rate my strats
@lore what happened at the Spring 2024 tournament?
@mimic_user1 @lore what did user1 say about the meta last month?
```

### 5.2 Model Name Resolution

Mention → model name mapping is maintained in a config dict:

```python
MENTION_TO_MODEL: dict[str, str] = {
    "mimic_user1": "mimic_user1",
    "mimic_user2": "mimic_user2",
    "mimic_user3": "mimic_user3",
    "mimic_user4": "mimic_user4",
    "mimic_user5": "mimic_user5",
    "mimic_user6": "mimic_user6",
    "lore":        "lore",
}
```

This dict is the single source of truth. Adding a new persona requires only a new entry here + a new Ollama Modelfile registration.

### 5.3 Single Bot vs. Multi-Bot Architecture

**Recommended: Single bot application with alias routing.**

One Discord bot token, one `discord-bot` container. The bot parses the mention text to determine which persona was invoked. This avoids managing multiple bot tokens, multiple OAuth flows, and multiple container processes.

**Alternative: Separate bot per persona.** Each persona has its own Discord application, token, and container. Cleaner Discord UX (each persona shows as a distinct user with its own avatar/name), but operationally heavier. Recommended only if Phase 3 LoRA personas are deployed and distinct visual identity per persona is desired.

**Phase 1 recommendation:** Single bot. Revisit for Phase 3.

---

## 6. Request Handling

### 6.1 Simple Request Flow

```
1. Message received with @mention
2. Parse mention → resolve model name
3. Check rate limit for requesting user → reject if exceeded
4. Check proxy queue depth → reject if > MAX_QUEUE_DEPTH
5. Send typing indicator to channel
6. Build prompt (see Section 7)
7. POST to proxy: /api/chat with {model, messages}
8. Receive response (streaming or blocking)
9. Post response to channel
10. Stop typing indicator
```

**Typing indicator management:**
Discord's typing indicator expires after ~10 seconds. The bot refreshes it every `TYPING_INDICATOR_INTERVAL` seconds (default: 5s) while waiting for a response. This masks the full swap + inference latency without the user seeing the indicator drop.

```python
async def send_with_typing(channel, generate_coro):
    async with channel.typing():
        return await generate_coro()
```

`channel.typing()` as an async context manager handles the keep-alive automatically in `discord.py` 2.x.

### 6.2 Lore + Mimic Chain (Compound Request)

When a message mentions both `@lore` and a mimic persona, the bot executes a **sequential two-step chain:**

```
Step 1: RAG lookup (CPU, ~0.5s)
        → Retrieve top-K lore chunks from ChromaDB matching the query

Step 2: Lore inference
        → POST to proxy with model: lore
        → Inject RAG chunks into user message context
        → Receive lore summary

Step 3: Mimic inference
        → POST to proxy with model: mimic_<member>
        → Inject lore summary as additional context
        → Receive mimic reaction

Step 4: Post both outputs to Discord
        → Lore output as first message (or embed)
        → Mimic output as follow-up reply
```

**Typing indicator is active throughout all steps.** Total wall time: ~14s (see Design.md §8.2).

**Compound mention detection:**
```python
def parse_mentions(message: discord.Message) -> list[str]:
    """Returns list of model names mentioned, in order of appearance."""
    mentioned_ids = [m.id for m in message.mentions]
    return [MENTION_TO_MODEL[bot_id_to_name[mid]] 
            for mid in mentioned_ids 
            if bot_id_to_name.get(mid) in MENTION_TO_MODEL]
```

If multiple mimic personas are mentioned (e.g. `@mimic_user1 @mimic_user3`), each is called sequentially and their responses are posted in order.

### 6.3 Error Handling

| Error Condition | Bot Response |
|---|---|
| Proxy unreachable | Ephemeral: "⚠️ The AI backend is currently unavailable. Try again in a moment." |
| Queue depth exceeded | Ephemeral: "⏳ Too many requests queued. Please wait a moment." |
| Rate limit exceeded | Ephemeral: "🚦 You're sending requests too fast. Slow down a bit." |
| Inference timeout (>60s) | Ephemeral: "⏱️ Request timed out. The model may be busy." |
| Unknown mention | Silent ignore (bot does not respond to unrecognised mentions) |
| Empty response from model | Ephemeral: "🤔 The model returned an empty response. Try rephrasing." |

All ephemeral messages are visible only to the requesting user (`ephemeral=True` in interaction response, or DM fallback for prefix-command style).

---

## 7. Prompt Construction

### 7.1 Mimic Prompt

The mimic model's system prompt is baked into the Ollama Modelfile (see Design.md §5.1). The bot constructs the user message as:

```python
def build_mimic_messages(
    user_message: str,
    conversation_history: list[dict],
    lore_context: str | None = None,
) -> list[dict]:
    messages = list(conversation_history)  # rolling window (see §7.3)
    
    user_content = user_message
    if lore_context:
        user_content = f"[Lore context: {lore_context}]\n\n{user_message}"
    
    messages.append({"role": "user", "content": user_content})
    return messages
```

The system prompt is **not** re-sent by the bot — it is embedded in the Modelfile and applied by Ollama automatically.

### 7.2 Lore Prompt

```python
def build_lore_messages(
    user_message: str,
    rag_chunks: list[str],
) -> list[dict]:
    context_block = "\n\n".join(rag_chunks) if rag_chunks else "No relevant lore found."
    
    return [
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n{context_block}\n\n"
                f"Question: {user_message}"
            )
        }
    ]
```

Lore requests are **stateless** — no conversation history is maintained. Each lore query is independent.

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
- Truncated to **2000 characters** (Discord message limit) — mimic responses are capped at 512 tokens (~400 words) in the Modelfile, so truncation should be rare
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

The bot communicates with the orchestration proxy using the **Ollama `/api/chat` format:**

### 10.1 Request

```http
POST http://proxy:11436/api/chat
Content-Type: application/json

{
  "model": "mimic_user3",
  "messages": [
    {"role": "user", "content": "rate my strats"}
  ],
  "stream": false,
  "options": {}
}
```

**Streaming:** `stream: false` for simplicity in Phase 1. Phase 2+ can enable streaming for faster perceived response time (post tokens to Discord as they arrive using message edits).

### 10.2 Response

```json
{
  "model": "mimic_user3",
  "message": {
    "role": "assistant",
    "content": "lmao those aren't strats, that's just dying slower"
  },
  "done": true
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
    - proxy
    - chromadb          # RAG lookups for lore chain
  environment:
    - DISCORD_TOKEN=${DISCORD_TOKEN}
    - OLLAMA_PROXY=http://proxy:11436
    - CHROMA_HOST=chromadb
    - CHROMA_PORT=8000
    - MAX_QUEUE_DEPTH=3
    - RATE_LIMIT_PER_USER=5
    - LORE_TOP_K=5
  volumes:
    - ./discord-bot:/app   # dev: live reload; remove for prod
```

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
chromadb-client>=0.5.0
sentence-transformers>=3.0.0
python-dotenv>=1.0.0
```

---

## 12. File Structure

```
discord-bot/
├── Dockerfile
├── requirements.txt
├── bot.py                  # Entry point: Discord client setup, event loop
├── router.py               # Mention parsing, model name resolution
├── proxy_client.py         # httpx async client for proxy API calls
├── rag_client.py           # ChromaDB query wrapper for lore retrieval
├── rate_limiter.py         # Per-user rate limiting logic
├── history.py              # Conversation history management (deque per channel/persona)
├── formatters.py           # Response formatting, disclaimer stripping, embed builders
├── config.py               # Environment variable loading and defaults
└── modelfiles/             # Ollama Modelfile templates (reference copies)
    ├── mimic_user1.Modelfile
    ├── mimic_user2.Modelfile
    ├── mimic_user3.Modelfile
    ├── mimic_user4.Modelfile
    ├── mimic_user5.Modelfile
    ├── mimic_user6.Modelfile
    └── lore.Modelfile
```

---

## 13. Persona Configuration

Each mimic persona is defined by two artefacts:

1. **Ollama Modelfile** — sets base model, quantisation, inference parameters, and system prompt
2. **Bot config entry** — maps Discord mention → model name

### 13.1 Persona Parameter Summary

| Persona | Model | Quant | VRAM | Temp | Ctx | Max Tokens | Thinking |
|---|---|---|---|---|---|---|---|
| mimic_user1 | Qwen3.5-9B-Uncensored | Q6_K | ~7.4 GB | 0.85 | 8192 | 512 | false |
| mimic_user2 | Qwen3.5-9B-Uncensored | Q6_K | ~7.4 GB | 0.85 | 8192 | 512 | false |
| mimic_user3 | Qwen3.5-9B-Uncensored | Q6_K | ~7.4 GB | 0.85 | 8192 | 512 | false |
| mimic_user4 | Qwen3.5-9B-Uncensored | Q6_K | ~7.4 GB | 0.85 | 8192 | 512 | false |
| mimic_user5 | Qwen3.5-9B-Uncensored | Q6_K | ~7.4 GB | 0.85 | 8192 | 512 | false |
| mimic_user6 | Qwen3.5-9B-Uncensored | Q6_K | ~7.4 GB | 0.85 | 8192 | 512 | false |
| lore | Gemma3-12B | Q6_K | ~9.5 GB | 0.3 | 16384 | 1024 | N/A |

> All mimic personas share the same base model and quant — only the system prompt differs. This means swapping between any two mimic personas is a **system prompt swap only**, not a model reload. The proxy's `current_model` key distinguishes them by name, but Ollama may cache the base weights and only swap the KV context. Verify this behaviour during Phase 1 testing.

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

Customise the italicised description per member during Phase 1 testing. In Phase 3, this system prompt is replaced by a LoRA-merged model that has the personality baked into weights.

---

## 14. Latency Budget

| Operation | Estimated Time | Notes |
|---|---|---|
| Discord message receive → bot handler | ~50ms | Discord gateway latency |
| Mention parse + rate limit check | ~1ms | In-memory, negligible |
| RAG lookup (lore chain only) | ~500ms | ChromaDB CPU query |
| Proxy lock acquisition (no queue) | ~1ms | Immediate if slot free |
| Proxy lock acquisition (queued) | Variable | Depends on in-progress generation |
| Model swap (mimic, cold) | ~3–5s | Qwen3.5-9B Q6_K from NVMe |
| Model swap (lore, cold) | ~4–6s | Gemma3-12B Q6_K from NVMe |
| Model swap (same model, warm) | ~0s | Already loaded, no swap needed |
| Mimic inference | ~1–2s | ~40 tok/s, 512 max tokens |
| Lore inference | ~3–5s | ~30 tok/s, 1024 max tokens, RAG context |
| Discord message post | ~100ms | Discord API write |
| **Total (mimic, warm model)** | **~1.2s** | Best case |
| **Total (mimic, cold swap)** | **~5–8s** | Typical first request |
| **Total (lore+mimic chain)** | **~12–16s** | Two swaps + two inferences |

The typing indicator is active from step 1 through Discord post, masking all latency from the user's perspective.

---

## 15. Phase 1 Implementation Checklist

- [ ] Create Discord application and bot in Developer Portal
- [ ] Enable `MESSAGE_CONTENT` privileged intent
- [ ] Generate bot token and add to `.env`
- [ ] Implement `bot.py` with `on_message` handler and mention detection
- [ ] Implement `router.py` with `MENTION_TO_MODEL` dict
- [ ] Implement `proxy_client.py` with httpx async POST to `/api/chat`
- [ ] Wire typing indicator keep-alive in request handler
- [ ] Implement `rate_limiter.py` with per-user sliding window
- [ ] Implement `formatters.py` with disclaimer stripping and lore embed builder
- [ ] Implement `history.py` with per-channel/per-persona deque
- [ ] Register all 6 mimic Modelfiles in Ollama swappable instance
- [ ] Register lore Modelfile in Ollama swappable instance
- [ ] Test single mimic request end-to-end
- [ ] Test lore request end-to-end
- [ ] Test lore+mimic compound chain
- [ ] Test rate limiting (exceed 5 req/min, verify ephemeral error)
- [ ] Test queue depth cap (simulate 4 concurrent requests)
- [ ] Verify typing indicator persists through full swap+inference cycle
- [ ] Verify disclaimer stripping fires on any baked-in disclaimers
- [ ] Deploy `discord-bot` container via Docker Compose
- [ ] Confirm bot comes online in Discord server

---

## 16. Phase 3 LoRA Upgrade Path

When Phase 3 LoRA-merged models are ready, the bot requires **zero code changes.** The upgrade path is:

1. Train LoRA adapter on Qwen3.5-9B-Uncensored base using member message history
2. Merge adapter into full model: `mimic_<member>_v2.gguf`
3. Update Modelfile: change `FROM` line to point to merged GGUF
4. `ollama rm mimic_<member>` + `ollama create mimic_<member> -f mimic_<member>_v2.Modelfile`
5. Bot continues using the same model name — no proxy changes, no bot code changes

The `MENTION_TO_MODEL` dict and all routing logic remain identical. The only change is the underlying weights.

---

*For full system architecture, hardware budget, proxy state machine, and RAG pipeline details, see `Design.md`.*
