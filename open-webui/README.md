# Open WebUI Configuration

Open WebUI is the self-hosted chat UI for this stack. The image is pulled directly from GitHub Container Registry — no custom Dockerfile needed.

## Enabling Open WebUI

Uncomment the `open-webui` service block in `docker-compose.yml`, then:

```bash
make up
```

Open WebUI will be available at **http://localhost:3000**. The first user to register becomes the admin.

## Endpoints

Two model backends are available:

1. **Local model (`chat`)** — routes to `http://proxy:11436/v1` (OpenAI-compatible). Uses `Qwen3.5-35B-A3B UD-IQ4_NL` (~17.8 GB VRAM), the same GGUF as the Brain coding assistant. Competes for the swappable slot under the proxy lock.

2. **Claude API** — set `ANTHROPIC_API_KEY` in `.env` to enable. Requests go directly to `api.anthropic.com` — bypasses the proxy entirely, zero VRAM impact.

## Model configuration

The `chat` model is defined as `[chat]` in `models.ini`. It points to the same GGUF as `[brain]` with different sampling parameters tuned for casual conversation:

| Parameter | Brain (coding) | Chat (casual) |
|---|---|---|
| temperature | 0.2 | 0.75 |
| top_k | 10 | 40 |
| top_p | 0.9 | 0.92 |
| repeat_penalty | — | 1.1 |
| ctx-size | 32768 | 16384 |

The GGUF is downloaded via `make models-download` (shared with brain — no extra disk cost). llama-server's router mode loads it on first request.

If Brain was the last loaded model when an Open WebUI request arrives, the swap is near-zero (same file, no eviction). Otherwise expect ~5–8s cold load from NVMe.

## Configuring backends in Open WebUI

Open WebUI discovers the local model automatically via `OPENAI_API_BASE_URL`. To add Claude:

1. Go to **Settings → Connections → OpenAI API**
2. The local proxy is already configured
3. For Claude, go to **Settings → Connections → Anthropic** and enter your API key (or set `ANTHROPIC_API_KEY` in `.env` — Open WebUI picks it up automatically)

## Notes

- All data (conversations, settings, uploads) persists in the `open_webui_data` Docker volume — survives container restarts
- No MongoDB sidecar needed — Open WebUI uses SQLite internally
- To enable user authentication beyond the first-user-is-admin default, configure it in Open WebUI's admin panel under **Settings → Users**
