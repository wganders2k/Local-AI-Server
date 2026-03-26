# Discord Bot

`discord.py` bot that routes member mentions to the appropriate AI model via the orchestration proxy. Handles typing indicators, rate limiting, conversation history, and the sequential lore+mimic chain.

## Responsibilities

- Parse `@mimic_<member>` and `@lore` mentions
- Route requests to the proxy (`/v1/chat/completions`) with the correct model name and system prompt
- Maintain typing indicator throughout swap + inference latency
- Enforce per-user rate limiting (default: 5 req/min)
- Maintain per-channel, per-persona conversation history (rolling 10-turn window)
- Execute the lore+mimic sequential chain for compound mentions
- Strip baked-in disclaimers from mimic responses
- Format lore responses as Discord embeds

## Design Reference

See `DiscordBot-Design.md` for full detail. See `Design.md` §8 for request flow examples.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from Discord Developer Portal |
| `PROXY_URL` | ✅ | — | Proxy base URL (e.g. `http://proxy:11436`) |
| `CHROMA_HOST` | ✅ | — | ChromaDB hostname |
| `CHROMA_PORT` | ❌ | `8000` | ChromaDB port |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Max queued requests before ephemeral error |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Max requests per user per minute |
| `TYPING_INDICATOR_INTERVAL` | ❌ | `5` | Seconds between typing indicator refreshes |
| `LORE_TOP_K` | ❌ | `5` | RAG chunks retrieved per lore query |

## Planned file structure

```
discord-bot/
├── Dockerfile
├── requirements.txt
├── bot.py              # Entry point: Discord client setup, event loop
├── router.py           # Mention parsing, model name resolution
├── proxy_client.py     # httpx async client for proxy API calls (/v1/chat/completions)
├── rag_client.py       # ChromaDB query wrapper for lore retrieval
├── rate_limiter.py     # Per-user rate limiting logic
├── history.py          # Conversation history management (deque per channel/persona)
├── formatters.py       # Response formatting, disclaimer stripping, embed builders
└── config.py           # Environment variable loading, defaults, system prompts
```
