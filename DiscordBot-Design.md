# Discord Bot — Design Document v2.0

**Scope:** design parameters for the Discord bot component. For system architecture (hardware, GPU arbitration, proxy, RAG pipeline), see [`Design.md`](Design.md).

**Change Log:**

| Version | Date | Change |
|---|---|---|
| v2.0 | 2026-08-07 | `/chat` thread-based conversation added. `/lore` rebuilt as an agentic tool-calling loop against the RAG service — it no longer does a single retrieval and no longer uses the `lore` model. All responses now stream. Privileged intents are required. Phase checklists replaced with current state. |
| v1.2 | 2026-04-26 | Replaced mention-based routing with slash commands. Removed `router.py` and the compound lore+mimic chain. |
| v1.1 | — | Original design with mention-based routing. |

---

## 1. Overview

The bot is the primary user-facing interface for the nullposting server. It routes requests to the orchestration proxy, masks swap latency with typing indicators, and owns all prompt construction — the proxy is content-blind, so anything the model sees was assembled here.

Three slash commands:

| Command | Model | Purpose |
|---|---|---|
| `/mimic persona message` | the persona alias itself (`mimic_user*`) | Impersonate a member's Discord personality |
| `/chat model` | any alias llama-server knows | Create a thread bound to a model; chat in it without further commands |
| `/lore question [rounds]` | `brain-dense-heretic` (`AGENT_MODEL`) | Agentic search over server history |

### 1.1 Slash commands over mention routing

| Concern | Slash commands | Mention routing |
|---|---|---|
| Persona discovery | Autocomplete dropdown | User must know the exact alias |
| Input structure | Named parameters | Free-text parsing, error-prone |
| Rate limit feedback | Ephemeral interaction response | Requires a new message reply |
| Typing indicator | Built in via `defer()` | Manual keep-alive |

Compound lore+mimic chains are not supported; the commands are run separately.

### 1.2 `/chat` is the exception that reintroduced message handling

`/chat` is a slash command only at creation. Afterwards the thread is the interface, and every message in it is handled by `on_message`. This deliberately reverses part of the v1.2 rationale — the ergonomics of a conversation are worth it, but the cost is real and stated in §4.2: the bot now needs a privileged intent it previously did without.

---

## 2. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 (`python:3.11-slim`) | Native `discord.py` support; async-first |
| Discord library | `discord.py` ≥2.3 | Mature, async, slash commands + app command trees |
| HTTP client | `httpx` (async) | Non-blocking calls to proxy and RAG service; streaming support |
| Configuration | Environment variables via `.env` | `python-dotenv` |

```
discord.py>=2.3.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

**No vector-store dependencies.** An earlier revision planned to add `chromadb` and `sentence-transformers` here for lore retrieval. Instead the bot calls the RAG service over HTTP (`rag_client.py`). That keeps the embedding model in exactly one process rather than loading a second copy into the bot, and keeps the bot image small.

---

## 3. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from the Developer Portal |
| `PROXY_URL` | ❌ | `http://proxy:11436` | Orchestration proxy base URL |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Reject with an ephemeral error above this proxy queue depth |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Requests per user per 60s |
| `TYPING_INDICATOR_INTERVAL` | ❌ | `5` | Seconds between typing refreshes |
| `RAG_SERVICE_URL` | ❌ | `http://rag-service:8001` | RAG service base URL |
| `LORE_TOP_K` | ❌ | `10` | Default chunks per retrieval |
| `RAG_ENABLED` | ❌ | `true` | Set false to disable retrieval for maintenance |
| `THREAD_REGISTRY_PATH` | ❌ | `data/threads.json` | Persisted `/chat` thread bindings |
| `LORE_CONTEXT_PATH` | ❌ | `prompts/lore_context.md` | Gitignored server background for the `/lore` agent |

`CHROMA_HOST` / `CHROMA_PORT` are **not** used by the bot — it never talks to ChromaDB directly.

Non-environment constants in [`config.py`](discord-bot/config.py): `AGENT_MODEL` (`brain-dense-heretic`), `AGENT_MAX_ROUNDS` (10), `AGENT_MAX_ROUNDS_HARD_CAP` (25), `AGENT_TEMPERATURE` (0.1), `AGENT_TOP_K` (10), `HISTORY_MAX_TURNS` (20), and the proxy timeouts in §10.3.

---

## 4. Bot Identity & Permissions

### 4.1 Discord application

- **Application name:** `Mimic Bot`
- **Bot avatar / nickname:** server-specific, set in the Developer Portal

### 4.2 Required permissions and intents

| Permission | Scope | Reason |
|---|---|---|
| `bot` | OAuth2 | Base scope |
| `applications.commands` | OAuth2 | Slash command registration |
| Send Messages | Bot | Post responses |
| Create Public Threads | Bot | `/chat` |
| Send Messages in Threads | Bot | Thread conversation |
| Read Message History | Bot | Thread restore on startup |

**Privileged intents are required** — this changed in v2.0:

| Intent | Why |
|---|---|
| `MESSAGE_CONTENT` | `on_message` reads the text of messages inside `/chat` threads. Slash commands alone would not need it |
| `SERVER_MEMBERS` | `intents.members = True` — display-name resolution for message labelling |

Both must be enabled in the Developer Portal or the bot fails to start.

### 4.3 Channel scope

Slash commands can be enabled or disabled per-channel in server settings. Note that `/chat` threads work wherever a thread can be created, and `on_message` responds in any registered thread regardless of where the parent channel sits.

---

## 5. Command Routing

### 5.1 Persona autocomplete (`/mimic`)

Suggestions come from `MIMIC_SYSTEM_PROMPTS.keys()`, capped at Discord's 25-choice limit.

**These are advertised whether or not they work.** The keys come from `MIMIC_PERSONAS` in the bot's config, but the backing presets are commented out in [`models.ini`](models.ini) — so every suggestion currently resolves to a model llama-server does not have. The autocomplete has no visibility into the preset file; making it accurate would mean sourcing it from `/v1/models` the way `/chat` does.

### 5.2 Model autocomplete (`/chat`)

Suggestions come from the proxy's `/v1/models`, i.e. from llama-server itself. Any alias actually loaded in the preset file is selectable, including ones absent from `SWAPPABLE_MODELS` in `proxy/config.py`. If the proxy is unreachable, the list is empty rather than an error.

### 5.3 Adding a persona

1. Uncomment or add the `[mimic_<member>]` section in `models.ini`
2. Add the member to `MIMIC_PERSONAS` in [`config.py`](discord-bot/config.py) — the prompt is generated from `MIMIC_PROMPT_TEMPLATE`
3. `make restart-llama-swappable && make restart-bot`

---

## 6. Request Handling

### 6.1 `/mimic` flow

```
1. Rate limit check → ephemeral rejection if exceeded
2. Validate persona against MIMIC_PERSONAS → ephemeral "Unknown persona"
3. Check proxy queue depth → ephemeral rejection at MAX_QUEUE_DEPTH
   (a ProxyError here is logged and ignored; the chat call reports it properly)
4. defer() the interaction
5. Build [system prompt] + rolling history + labelled user message
6. Stream from proxy under `async with interaction.channel.typing()`
7. Flush 2000-char chunks at sentence boundaries as they fill
8. Strip disclaimers from the final buffer, send the remainder
9. Record the turn in history
```

### 6.2 `/chat` flow

```
create:   rate limit → verify the channel supports threads (not a DM)
          → create_thread(auto_archive_duration=60)
          → bind thread_id → model in memory and in threads.json
          → greet in-thread, confirm ephemerally

message:  on_message → ignore self → thread must be registered → rate limit
          → build history + "[DisplayName]: text"
          → stream to the bound model, chunked at 2000 chars
          → record the turn
```

Threads are restored at startup from `threads.json`: each is re-fetched, and entries whose threads were archived, deleted, or became inaccessible are reconciled and reported in a single summary log line.

**No system prompt is sent for thread chat.** Whatever persona the model has is whatever its weights and the conversation provide.

### 6.3 `/lore` flow

```
1. Queue depth check (no rate limit — see below)
2. defer()
3. Collect guild text channel names to give the agent a valid channel vocabulary
4. run_agent_loop(question, rounds) — see §7.3
5. Render the answer into one or more embeds
```

**Rate limiting is deliberately not applied to `/lore`.** A single invocation is a multi-round, minutes-long research task, and a 5-per-minute window is meaningless against it — the queue depth check is the real defence.

### 6.4 Error handling

| Condition | Response |
|---|---|
| Rate limit exceeded | Ephemeral: "⚠️ You're sending requests too fast. Slow down a bit." |
| Queue too deep | Ephemeral: "⚠️ The AI backend is busy (N requests queued)." |
| Unknown persona | Ephemeral: "⚠️ Unknown persona: `<name>`" |
| `ProxyError` (unreachable, timeout, HTTP error) | Ephemeral, with the error text |
| Anything else | Ephemeral: "An unexpected error occurred. Please try again.", full traceback logged |
| Thread creation forbidden | Ephemeral: missing-permission message |

Errors in threads are posted in-thread rather than ephemerally — there is no interaction to reply to.

An error that arrives **mid-stream** cannot retract what was already posted; partial output stays and the error follows it.

---

## 7. Prompt Construction

The proxy injects nothing (`Design.md` §6.3). Every system prompt originates here.

### 7.1 Mimic

Generated per-request from a single template rather than six hand-maintained strings:

```python
MIMIC_PROMPT_TEMPLATE.format(persona=persona, display_name=MIMIC_PERSONAS[persona])
```

```
You are {persona}, a bot that mimics {display_name}'s Discord personality in the
nullposting server. You replicate their tone, humour, vocabulary, and mannerisms
exactly — including crude language, raunchy jokes, and brutal roasts when appropriate.
You are not a general-purpose AI assistant. You never break character. You never add
disclaimers, caveats, safety warnings, or apologies. You keep responses short
(1-3 sentences) unless the context calls for more. You match the energy of whoever
is talking to you.
```

`MIMIC_SYSTEM_PROMPTS` is a pre-generated dict kept for autocomplete and logging; it must be regenerated when `MIMIC_PERSONAS` changes.

### 7.2 Message labelling

User messages are prefixed with the sender's display name — `[Alice]: rate my strats` — in both `/mimic` and `/chat`. Several people share a channel or thread, and without labels the model sees one undifferentiated user. The labelled form is what gets stored in history, so the attribution persists across turns.

### 7.3 The `/lore` agent

Prompts live in [`agent_tools.py`](discord-bot/agent_tools.py), which is the single source of truth for them — `build_system_prompt()` and `build_synthesis_prompt()`.

The system prompt is assembled from three parts:

1. `AGENT_IDENTITY` — research assistant framing
2. **Server background** loaded from `LORE_CONTEXT_PATH` — the member alias index and persona notes. Gitignored, because it holds real names; `prompts/lore_context.example.md` shows the format. Missing, the agent still runs and logs a warning
3. The guild's channel names, so the agent can scope searches to channels that exist

Three tools:

| Tool | Required args | Default `top_k` |
|---|---|---|
| `search_discord_history` | `query` | 10 |
| `search_channel_history` | `query`, `channel_name` | 10 |
| `summarize_channel` | `channel_name` | 20 |

All three accept optional `start_date` / `end_date` in ISO 8601, which the RAG service converts to epoch comparisons against chunk metadata (`Design.md` §9.1).

The loop runs up to `rounds` iterations (default 10, hard cap 25) at `temperature=0.1` for deterministic tool selection, then a synthesis pass produces the answer. `AgentMetrics` records rounds, tool calls, tools used, and per-stage latency, logging a summary per run.

### 7.4 Conversation history

```python
# (channel_id | thread_id, model_name) → deque(maxlen=20)
```

| Parameter | Value | Rationale |
|---|---|---|
| Max entries | 20 messages ≈ 10 exchanges | Each turn appends a user and an assistant message |
| Scope | Per-channel/thread, per-model | Prevents cross-channel and cross-persona bleed |
| Persistence | In-memory only | Cleared on restart. Note that `/chat` **thread bindings** persist but their history does not — a restarted bot answers in the same thread with no memory of it |
| `/lore` | none | Each query is independent; the agent retrieves its own context |

---

## 8. Response Formatting

### 8.1 Streaming

All model responses stream. Tokens accumulate in a buffer; whenever it reaches Discord's 2000-character limit, `find_split_boundary()` picks a sentence boundary and that chunk is posted. The remainder is flushed when the stream ends.

Streaming was chosen over a single blocking post because swap plus inference on a 27B–35B model routinely exceeds Discord's 15-minute interaction window in the worst case, and partial output is better feedback than a spinner.

### 8.2 Mimic responses

Plain text, no embeds — the bot's own username and avatar are the attribution. Disclaimer patterns are stripped from the final buffer.

### 8.3 Lore responses

Discord embeds, blurple `0x5865F2`, titled `📚 nullposting Lore`. Answers longer than the 4096-character embed limit are split across several embeds at sentence boundaries; an empty answer produces a single `(No results)` embed.

There is no source-count footer. An earlier design specified one, but with an agentic loop "how many chunks were used" is not a single number — the agent may run a dozen retrievals across several rounds and discard most of them.

### 8.4 Disclaimer stripping

`DISCLAIMER_PATTERNS` in [`config.py`](discord-bot/config.py) — around a dozen anchored regexes covering "This is general…", "Please consult…", "I'm an AI…", and similar trailing additions. Applied to the tail of the response.

Applied to `/mimic` and `/chat` output. Not applied to `/lore`, whose model is not the uncensored base and whose output is a research answer where a caveat may be legitimate.

---

## 9. Rate Limiting

Per-user sliding window in [`rate_limiter.py`](discord-bot/rate_limiter.py): timestamps outside the window are evicted on each check, then the request is admitted or refused. Default 5 per 60s via `RATE_LIMIT_PER_USER`.

Applied to `/mimic`, `/chat` creation, and each message in a `/chat` thread. Not applied to `/lore` (§6.3).

**Queue depth cap:** before dispatching, the bot reads the proxy's `/status` and refuses at `MAX_QUEUE_DEPTH` (default 3) with an ephemeral error, which stops the queue growing unbounded during a burst.

---

## 10. Proxy API Contract

OpenAI-compatible, forwarded transparently by the proxy.

### 10.1 Request

```http
POST http://proxy:11436/v1/chat/completions
Content-Type: application/json

{
  "model": "mimic_user3",
  "messages": [
    {"role": "system", "content": "You are mimic_user3..."},
    {"role": "user", "content": "[Alice]: rate my strats"}
  ],
  "stream": true
}
```

`chat_with_tools()` additionally sends `tools` and `tool_choice` for the agent loop; the proxy passes them through to llama-server, which is started with `--jinja` so the chat template can render tool calls.

### 10.2 Client methods

[`proxy_client.py`](discord-bot/proxy_client.py) exposes `chat()`, `chat_stream()`, `chat_with_tools()`, `get_queue_depth()`, and `list_models()`.

### 10.3 Timeouts

| Timeout | Value | Covers |
|---|---|---|
| Connection | 10s | Local network connect |
| Read | 120s | Worst-case model swap plus first token |
| Total | 320s | Full generation |

These are considerably longer than the v1.2 values (10/60/70s). The swappable slot may need to evict an 18 GB model and load another before the first token, and a request can additionally wait on the arbiter while a junior GPU tenant is preempted.

The RAG client has its own 30s timeout and degrades to empty context rather than raising, so a slow retrieval costs the agent one tool result instead of failing the command.

---

## 11. Container Definition

```yaml
discord-bot:
  build: ./discord-bot
  restart: unless-stopped
  depends_on:
    proxy:
      condition: service_healthy
    rag-service:
      condition: service_started
  environment:
    - DISCORD_TOKEN=${DISCORD_TOKEN}
    - PROXY_URL=${PROXY_URL}
    - MAX_QUEUE_DEPTH=${MAX_QUEUE_DEPTH:-3}
    - RATE_LIMIT_PER_USER=${RATE_LIMIT_PER_USER:-5}
    - TYPING_INDICATOR_INTERVAL=${TYPING_INDICATOR_INTERVAL:-5}
    - RAG_SERVICE_URL=http://rag-service:8001
    - LORE_TOP_K=${LORE_TOP_K:-5}
    - RAG_ENABLED=${RAG_ENABLED:-true}
    - LORE_CONTEXT_PATH=${LORE_CONTEXT_PATH:-prompts/lore_context.md}
  volumes:
    - ./discord-bot/data:/app/data      # thread registry, read-write
    - ./discord-bot/prompts:/app/prompts:ro
```

`prompts/` is mounted read-only, so editing `lore_context.md` takes effect on restart with no image rebuild. `data/` must stay writable — the thread registry is saved there via atomic replace.

Note that compose passes `LORE_TOP_K` with a default of 5 while `config.py` defaults to 10; the environment wins in the deployed stack.

---

## 12. File Structure

```
discord-bot/
├── Dockerfile
├── requirements.txt
├── bot.py              # Entry point: client setup, the three slash commands, on_message
├── agent_tools.py      # /lore agent: tool schemas, prompt builders, executor, loop, metrics
├── proxy_client.py     # httpx client: chat, stream, tools, queue depth, model list
├── rag_client.py       # httpx client for rag-service /retrieve; degrades to empty context
├── thread_registry.py  # JSON-backed thread_id → model bindings, atomic writes
├── history.py          # (channel_id, model) → deque conversation windows
├── rate_limiter.py     # Per-user sliding window
├── formatters.py       # Streaming split boundaries, disclaimer strip, lore embeds
├── config.py           # Env loading, persona registry, prompt template, agent constants
├── data/threads.json   # Persisted thread registry (bind-mounted)
└── prompts/
    ├── lore_context.example.md
    └── lore_context.md        # gitignored — real names
```

---

## 13. Latency Budget

| Operation | Estimate | Notes |
|---|---|---|
| Interaction receive → handler | ~50ms | Gateway latency |
| Rate limit + queue depth check | ~1ms + one HTTP round trip | |
| Arbiter acquire (nothing to preempt) | ~ms | Lease grant only |
| Arbiter acquire (preempting the trainer) | **0.9–2.0s** | Measured: notice, SIGKILL, process exit, driver reclaim |
| Model swap, cold | ~5–8s | 17–18 GB from NVMe |
| Model swap, warm | ~0s | Same model already resident |
| First token → user | streaming | Chunks post as they fill |
| `/lore` full run | **tens of seconds to minutes** | Multi-round; each round is an inference plus retrieval |

The typing indicator is held for the whole of `/mimic` and `/chat`; `/lore` uses `defer()` and posts embeds when the loop finishes.

---

## 14. Current State

**Working:** all three commands, streaming, thread persistence and restore, agentic RAG with three tools, rate limiting, queue depth rejection, disclaimer stripping.

**Blocked:** `/mimic` — the persona presets are commented out in `models.ini`, so every persona resolves to a model llama-server does not have. Autocomplete still offers them (§5.1).

**Disabled:** `/admin-clear-history`, which had no authorization check — any user could clear anyone's history. Re-enable only behind role verification.

**Known rough edges:**

- Persona autocomplete advertises models that may not exist; `/chat`'s live model list is the pattern to copy.
- `/chat` thread bindings persist across restarts but conversation history does not.
- `LORE_TOP_K` defaults disagree between `config.py` (10) and `docker-compose.yml` (5).
- No tests. The bot is the only service in the repo without a `tests/` directory; the agent loop and the streaming split logic are the parts most worth covering.

---

## 15. LoRA Upgrade Path

When a merged persona GGUF exists, the bot needs **no code changes**:

1. Train an adapter on the uncensored base from per-member history
2. Merge and export `mimic_<member>_v2.gguf`
3. Point the `[mimic_<member>]` section in `models.ini` at it
4. `make restart-llama-swappable`

The bot references aliases, not weights. Note that the training half of this is not built, and training on per-member history has not been agreed as a decision — see [`lora-training/README.md`](lora-training/README.md).

---

*For system architecture, GPU arbitration, and the RAG pipeline, see [`Design.md`](Design.md).*
