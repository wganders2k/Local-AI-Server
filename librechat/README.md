# LibreChat Configuration

Configuration files for the LibreChat container. The LibreChat image itself is pulled from Docker Hub (`ghcr.io/danny-avila/librechat:latest`) — no custom Dockerfile needed.

## Files

- `librechat.yaml` — defines available model endpoints (local llama-server via proxy + Claude API)

## Design Reference

See `Design.md` §3a (LibreChat Model Selection) and §7 (Docker Compose).

## librechat.yaml overview

Two endpoints are configured:

1. **Local model** — routes to `http://proxy:11436/v1` (OpenAI-compatible) with model `librechat_chat` (Qwen3.5-35B-A3B UD-IQ4_NL, ~17.8 GB VRAM). Uses the **same GGUF as the Brain coding assistant** — no additional download required. Competes for the swappable slot under the proxy lock.
2. **Claude API** — routes directly to `api.anthropic.com` using `ANTHROPIC_API_KEY`. Bypasses the proxy entirely — zero VRAM impact. Leave `ANTHROPIC_API_KEY` blank in `.env` to disable this endpoint.

## Model configuration

The `librechat_chat` model is defined as `[librechat_chat]` in `models.ini`. It points to the same GGUF as `[brain]` (`unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf`) with different sampling parameters tuned for casual conversation:

| Parameter | Brain (coding) | LibreChat (casual chat) |
|---|---|---|
| temperature | 0.2 | 0.75 |
| top_k | 10 | 40 |
| top_p | 0.9 | 0.92 |
| repeat_penalty | — | 1.1 |
| n_ctx | 40960 | 16384 |

The GGUF is downloaded via `make models-download` (shared with brain — no extra disk cost). No manual registration step is needed — llama-server's router mode loads it on first request.

If Brain was the last loaded model when a LibreChat request arrives, the swap is near-zero (same file, no eviction). Otherwise expect ~5–8s cold load from NVMe.

## Notes

- LibreChat requires MongoDB for conversation history. The `librechat-mongodb` sidecar handles this automatically.
- Conversation history persists in the `librechat_mongo` Docker volume — survives container restarts.
- To enable user authentication (Phase 4), set `ALLOW_REGISTRATION=false` and configure credentials in `librechat.yaml`.
