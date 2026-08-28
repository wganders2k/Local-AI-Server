# Discord Bot

`discord.py` bot exposing `/mimic`, `/chat` and `/lore`, routed to models through
the orchestration proxy. Handles typing indicators, rate limiting, conversation
history, disclaimer stripping, and the agentic RAG loop behind `/lore`.

## Commands

| Command | Parameters | Description |
|---|---|---|
| `/mimic` | `persona` (autocomplete), `message` | One-shot reply in the current channel, in a persona's voice |
| `/chat` | `model` (autocomplete) | Creates a thread bound to one model; the bot answers every message posted in it |
| `/lore` | `question`, `rounds` (1–25, default 10) | Agentic RAG over the Discord archive, then an optional follow-up thread |

### Disabled Commands

| Command | Status | Reason |
|---|---|---|
| `/admin-clear-history` | **Removed** | No authorization checks — any user could clear history. Re-add only with role/permission verification. |

## How `/lore` works

1. The agent model (`AGENT_MODEL`, currently `brain-dense-heretic`) is given the
   five tools in [`lore/tools.py`](lore/tools.py) and loops: call a tool, read the
   result, decide whether to search again — up to `rounds` times.
2. If it answers before the budget runs out, that is the answer. If it doesn't,
   the research is flattened into plain text and a separate synthesis call writes
   one. (Flattened, because leaving tool-call markup in context teaches the model
   to reply with tool-call markup.)
3. The answer posts as embeds, followed by an offer message. React with ❓ and the
   bot opens a thread seeded with the research it already did — so follow-ups
   usually cost no searching at all.
4. Threads are swept after `LORE_THREAD_TTL_SECONDS` of silence, which is what
   actually reclaims the disk: a session stores every excerpt verbatim.

Two searching styles are exposed on purpose, because they fail in opposite ways:
semantic search finds paraphrases but only returns top-k and cannot count or
order; literal search matches exact text across every message in time order and
reports totals. The prompt in [`prompts/lore_agent.md`](prompts/lore_agent.md)
spends most of its length teaching the model which to reach for.

## File structure

```
discord-bot/
├── bot.py               # Entry point: client, setup_hook, extension loading
├── services.py          # Shared state (clients, stores, thread router)
├── config.py            # Environment reads and tuning constants — nothing else
├── prompt_loader.py     # Loads and renders prompts/*.md
├── discord_io.py        # Streaming, message splitting, posting
├── formatters.py        # Disclaimer stripping, split boundaries, embeds
├── history.py           # Rolling per-channel/per-model conversation window
├── rate_limiter.py      # Per-user sliding window
├── thread_registry.py   # thread_id -> model, persisted so restarts survive
├── proxy_client.py      # httpx client for the proxy (/v1/chat/completions)
├── rag_client.py        # httpx client for the RAG service
├── lore/                # The /lore agent. Imports no discord — see below.
│   ├── prompts.py       #   assembles prompts from prompts/*.md
│   ├── tools.py         #   tool schemas + the executor behind them
│   ├── agent.py         #   the tool-calling loop
│   ├── compaction.py    #   condensing a thread's oldest research
│   ├── research.py      #   rendering tool results as plain text, dedup
│   ├── session.py       #   LoreSession + its JSON store
│   ├── metrics.py       #   timing and token accounting
│   └── progress.py      #   ProgressReporter protocol
├── cogs/                # Discord presentation layer
│   ├── mimic.py  chat.py  lore.py
│   └── lore_status.py   #   the /lore progress message
├── prompts/             # Prompt text, committed, baked into the image
└── tests/
```

**`lore/` must not import discord.** That boundary is what makes the agent loop
testable without a gateway (see `tests/test_agent.py`, which drives a full run
against a fake backend). Progress reporting crosses it through
`lore.progress.ProgressReporter`, which `cogs.lore_status.LoreStatus` implements.

This is the only service in the repo with subpackages; see
[`CONVENTIONS.md`](../CONVENTIONS.md). At ~4,400 lines a flat layout stopped
distinguishing "the agent" from "the Discord client", which is exactly the split
these directories restore.

## Prompts

All prompt text lives in `prompts/*.md` and is rendered by `prompt_loader.render()`.
`prompts/` is **committed code baked into the image** — editing a prompt is a
rebuild, which is correct for text that is part of the program's behaviour. Do not
bind-mount over it: the container would then see the working tree instead of the
image, and a stale checkout would silently change the model's instructions.

### Setup: lore context file

The `/lore` agent injects server-specific background (member alias index, persona
notes) into its system prompt. That content holds real names, so it is **not
committed** and does **not** live in `prompts/`. It sits on the `bot_context`
volume beside the session store:

```bash
cp discord-bot/prompts/lore_context.example.md \
   /mnt/storage/array/DiscordArchive/bot_context/lore/context.md
# then edit it with your server's real members
```

If the file is missing the bot still starts and `/lore` still works; it logs a
warning and answers without the alias index.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from Discord Developer Portal |
| `PROXY_URL` | ✅ | `http://proxy:11436` | Proxy base URL |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Max queued backend requests before the command refuses |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Max requests per user per minute |
| `RAG_SERVICE_URL` | ❌ | `http://rag-service:8001` | RAG service base URL |
| `RAG_ENABLED` | ❌ | `true` | Set false to run without `/lore` retrieval |
| `THREAD_REGISTRY_PATH` | ❌ | `data/threads.json` | Thread → model registry |
| `LORE_SESSION_PATH` | ❌ | `bot_context/lore/sessions.json` | Lore thread transcripts |
| `LORE_CONTEXT_PATH` | ❌ | `bot_context/lore/context.md` | Host-local server background for the `/lore` prompt |

Tuning that is not environment-driven — round caps, context thresholds, TTLs —
lives in [`config.py`](config.py) with the reasoning attached.

> `AGENT_CTX_LIMIT` in `config.py` must mirror `ctx-size` for `AGENT_MODEL` in
> [`models.ini`](../models.ini). Nothing enforces the pairing; if you change one,
> change the other or the budget maths is silently wrong.

## Tests

```bash
cd discord-bot
python -m venv .venv-test && .venv-test/bin/pip install -r requirements.txt pytest pytest-asyncio
.venv-test/bin/python -m pytest
```

## Design Reference

See [`docs/design/discord-bot.md`](../docs/design/discord-bot.md) for full detail,
and [`docs/design/system.md`](../docs/design/system.md) §8 for request flow examples.
