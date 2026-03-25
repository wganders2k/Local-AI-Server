# LibreChat Configuration

Configuration files for the LibreChat container. The LibreChat image itself is pulled from Docker Hub (`ghcr.io/danny-avila/librechat:latest`) — no custom Dockerfile needed.

## Files

- `librechat.yaml` — defines available model endpoints (local Ollama via proxy + Claude API)

## Design Reference

See `Design.md` §3a (LibreChat Model Selection) and §7 (Docker Compose).

## librechat.yaml overview

Two endpoints are configured:

1. **Local Ollama** — routes to `http://proxy:11436` with model `librechat_chat` (Qwen3.5-14B UD-IQ4_XS). Competes for the swappable slot under the proxy lock.
2. **Claude API** — routes directly to `api.anthropic.com` using `ANTHROPIC_API_KEY`. Bypasses the proxy entirely — zero VRAM impact. Leave `ANTHROPIC_API_KEY` blank in `.env` to disable this endpoint.

## Modelfile reference

The `librechat_chat` Modelfile is defined in `Design.md` §5.4. Register it in the swappable Ollama instance:

```bash
# From the server, after `make up`:
docker compose exec ollama-swappable ollama create librechat_chat -f /path/to/librechat_chat.Modelfile
```

## Notes

- LibreChat requires MongoDB for conversation history. The `librechat-mongodb` sidecar handles this automatically.
- Conversation history persists in the `librechat_mongo` Docker volume — survives container restarts.
- To enable user authentication (Phase 4), set `ALLOW_REGISTRATION=false` and configure credentials in `librechat.yaml`.
