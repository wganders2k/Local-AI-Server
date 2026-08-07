# Discord Bot

`discord.py` bot exposing three slash commands routed through the orchestration proxy. It owns all prompt construction — the proxy injects nothing.

## Commands

| Command | Parameters | Description |
|---|---|---|
| `/mimic` | `persona` (autocomplete), `message` | One-shot reply from a mimic persona |
| `/chat` | `model` (autocomplete) | Creates a thread bound to a model; every message in it is streamed to that model |
| `/lore` | `question`, `rounds` (optional, 1–25) | Agentic search over server history |

`/mimic` is blocked today: the `mimic_user*` presets are commented out in [`models.ini`](../models.ini), so every persona the autocomplete offers resolves to a model llama-server does not have. Uncomment them to restore it.

### Disabled

| Command | Reason |
|---|---|
| `/admin-clear-history` | No authorization check — any user could clear history. Re-enable only behind role verification |

## How each command works

**`/mimic`** — rate limit, persona validation, queue depth check, then a streamed request carrying `[system prompt] + rolling history + labelled user message`. Disclaimers are stripped from the tail before the final chunk is posted.

**`/chat`** — creates a public thread, binds `thread_id → model` in memory and in `data/threads.json`, and greets in-thread. From then on `on_message` handles it: any message in a registered thread is streamed to the bound model with no system prompt. Bindings are restored at startup, with archived and deleted threads reconciled.

The model list comes from the proxy's `/v1/models`, so anything llama-server actually has is selectable — including aliases missing from `proxy/config.py`. Persona autocomplete, by contrast, reads a hardcoded dict and cannot tell whether a preset exists.

**`/lore`** — an agentic loop, not a single retrieval. The agent (`brain-dense-heretic`) is given three tools and decides for itself what to search and when it has enough:

| Tool | Required | Default `top_k` |
|---|---|---|
| `search_discord_history` | `query` | 10 |
| `search_channel_history` | `query`, `channel_name` | 10 |
| `summarize_channel` | `channel_name` | 20 |

All accept optional ISO 8601 `start_date` / `end_date`. The loop runs up to `rounds` iterations (default 10, hard cap 25) at `temperature=0.1`, then synthesises an answer into Discord embeds. Rate limiting is deliberately not applied — a lore query is a minutes-long multi-round task, and a per-minute window means nothing against it. The queue depth check is the real defence.

## Setup: lore context file

The `/lore` agent injects server-specific background (member alias index, persona notes) into its system prompt. That content is **not committed** — it holds real names:

```bash
cp discord-bot/prompts/lore_context.example.md discord-bot/prompts/lore_context.md
# then edit with your server's real members
```

`prompts/` is bind-mounted read-only, so edits take effect on restart with no rebuild. If the file is missing the bot still starts and `/lore` still works — it logs a warning and answers without the alias index.

## Privileged intents

Both must be enabled in the Discord Developer Portal or the bot will not start:

| Intent | Why |
|---|---|
| `MESSAGE_CONTENT` | `on_message` reads message text inside `/chat` threads |
| `SERVER_MEMBERS` | Display-name resolution for message labelling |

Slash commands alone would need neither. `/chat` is what reintroduced the requirement.

## Message labelling

User messages are prefixed with the sender's display name — `[Alice]: rate my strats` — in both `/mimic` and `/chat`. Channels and threads have several participants, and without labels the model sees one undifferentiated user. The labelled form is what is stored in history, so attribution survives across turns.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token |
| `PROXY_URL` | ❌ | `http://proxy:11436` | Proxy base URL |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Reject above this proxy queue depth |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Requests per user per minute |
| `TYPING_INDICATOR_INTERVAL` | ❌ | `5` | Seconds between typing refreshes |
| `RAG_SERVICE_URL` | ❌ | `http://rag-service:8001` | RAG service |
| `LORE_TOP_K` | ❌ | `10` | Chunks per retrieval (compose passes `5`, which wins) |
| `RAG_ENABLED` | ❌ | `true` | False disables retrieval for maintenance |
| `THREAD_REGISTRY_PATH` | ❌ | `data/threads.json` | Persisted thread bindings |
| `LORE_CONTEXT_PATH` | ❌ | `prompts/lore_context.md` | Gitignored server background |

`CHROMA_HOST` / `CHROMA_PORT` are not used — the bot talks to the RAG service over HTTP and never to ChromaDB directly, which keeps the embedding model in one process instead of two.

Agent tuning lives in [`config.py`](config.py) rather than the environment: `AGENT_MODEL`, `AGENT_MAX_ROUNDS`, `AGENT_MAX_ROUNDS_HARD_CAP`, `AGENT_TEMPERATURE`, `AGENT_TOP_K`.

## Timeouts

Connect 10s, read 120s, total 320s. Long because a request may wait on an 18 GB model swap *and* on the arbiter preempting a junior GPU tenant before the first token arrives.

The RAG client has its own 30s timeout and degrades to empty context rather than raising, so a slow retrieval costs the agent one tool result instead of failing the command.

## File Structure

```
discord-bot/
├── Dockerfile
├── requirements.txt      # discord.py, httpx, python-dotenv — no vector-store deps
├── bot.py                # Client setup, the three commands, on_message, thread restore
├── agent_tools.py        # /lore: tool schemas, prompt builders, executor, loop, metrics
├── proxy_client.py       # chat, chat_stream, chat_with_tools, get_queue_depth, list_models
├── rag_client.py         # httpx client for rag-service /retrieve; degrades gracefully
├── thread_registry.py    # JSON-backed thread_id → model bindings, atomic writes
├── history.py            # (channel_id, model) → deque(maxlen=20)
├── rate_limiter.py       # Per-user sliding window
├── formatters.py         # Streaming split boundaries, disclaimer strip, lore embeds
├── config.py             # Env loading, persona registry, prompt template, agent constants
├── data/threads.json     # Persisted registry (bind-mounted read-write)
└── prompts/
    ├── lore_context.example.md
    └── lore_context.md   # gitignored
```

`data/` must stay writable — the registry is saved by atomic replace. `prompts/` is mounted read-only.

No tests. This is the only service in the repo without a `tests/` directory; the agent loop and the streaming split logic are the parts most worth covering.

## Design Reference

[`DiscordBot-Design.md`](../DiscordBot-Design.md) for full detail; [`Design.md`](../Design.md) §8 for request flows.
