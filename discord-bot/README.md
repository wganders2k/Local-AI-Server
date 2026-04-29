# Discord Bot

`discord.py` bot that exposes `/mimic` and `/lore` slash commands routed to the appropriate AI model via the orchestration proxy. Handles typing indicators, rate limiting, conversation history, and disclaimer stripping.

## Commands

| Command | Parameters | Description |
|---|---|---|
| `/mimic` | `persona` (autocomplete), `message` (string) | Chat with a mimic persona |
| `/lore` | `question` (string) | Ask the lore assistant a question |

### Disabled Commands

| Command | Status | Reason |
|---|---|---|
| `/admin-clear-history` | **Disabled** | No authorization checks — any user could clear history. Re-enable only after adding role/permission verification. |

## Responsibilities

- Expose `/mimic` and `/lore` slash commands with autocomplete for persona selection
- Route requests to the proxy (`/v1/chat/completions`) with the correct model name and system prompt
- Maintain typing indicator throughout swap + inference latency
- Enforce per-user rate limiting (default: 5 req/min)
- Maintain per-channel, per-persona conversation history (rolling 10-turn window)
- Strip baked-in disclaimers from mimic responses
- Format lore responses as Discord embeds

## Design Reference

See [`DiscordBot-Design.md`](../DiscordBot-Design.md) for full detail. See [`Design.md`](../Design.md) §8 for request flow examples.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from Discord Developer Portal |
| `PROXY_URL` | ✅ | — | Proxy base URL (e.g. `http://proxy:11436`) |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Max queued requests before ephemeral error |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Max requests per user per minute |
| `TYPING_INDICATOR_INTERVAL` | ❌ | `5` | Seconds between typing indicator refreshes |

### Phase 2+ Variables (RAG service not yet deployed)

| Variable | Required | Default | Description |
|---|---|---|---|
| `CHROMA_HOST` | ❌ | `chromadb` | ChromaDB host for lore RAG lookups |
| `CHROMA_PORT` | ❌ | `8000` | ChromaDB port |
| `LORE_TOP_K` | ❌ | `5` | RAG chunks retrieved per lore query |

## File Structure

```
discord-bot/
├── Dockerfile
├── requirements.txt
├── bot.py              # Entry point: Discord client setup, slash commands
├── proxy_client.py     # httpx async client for proxy API calls (/v1/chat/completions)
├── rate_limiter.py     # Per-user rate limiting logic
├── history.py          # Conversation history management (deque per channel/persona)
├── formatters.py       # Response formatting, disclaimer stripping, embed builders
└── config.py           # Environment variable loading, defaults, system prompts
```

### Phase 2+ Files (add when RAG service is deployed)

```
├── rag_client.py       # ChromaDB query wrapper for lore retrieval
```
