# Proxy — FastAPI Orchestration Middleware

FastAPI service on `:11436` that routes AI requests to the correct Ollama instance and serialises access to the swappable model slot.

## Responsibilities

- Route autocomplete requests (permanent models) directly to `:11435` — no lock, no swap
- Serialise all swappable model requests via `asyncio.Lock` — one at a time
- Swap the loaded model on `:11434` when the requested model differs from `current_model`
- Expose a `/health` endpoint for Docker health checks
- Expose a `/status` endpoint so the Discord bot can check queue depth

## Design Reference

See `Design.md` §4 (Architecture), §6 (Proxy State Machine), and §7 (Docker Compose).

## Build context

Built by Docker Compose from this directory. Add `Dockerfile` and `requirements.txt` here when implementing.

## Planned file structure

```
proxy/
├── Dockerfile
├── requirements.txt
├── main.py          # FastAPI app, route handlers
└── state.py         # OrchestratorState, swap logic
```
