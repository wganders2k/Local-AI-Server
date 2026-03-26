# Proxy — FastAPI Orchestration Middleware

FastAPI service on `:11436` that routes AI requests to the correct llama-server instance and serialises access to the swappable model slot.

## Responsibilities

- Route autocomplete requests (permanent models) directly to `:11435` — no lock, no swap
- Serialise all swappable model requests via `asyncio.Lock` — one at a time
- Track which model is active in the swappable slot for `/status` reporting
- llama-server's router mode handles the actual model load/eviction automatically
- Expose a `/health` endpoint for Docker health checks
- Expose a `/status` endpoint so the Discord bot can check queue depth

## Design Reference

See `Design.md` §4a (Inference Backend), §6 (Proxy State Machine), and §7 (Docker Compose).

## File structure

```
proxy/
├── Dockerfile
├── requirements.txt
├── main.py          # FastAPI app, route handlers
├── config.py        # LLAMA_PERMANENT / LLAMA_SWAPPABLE URLs, model sets
└── state.py         # OrchestratorState — lock, queue depth, current_model
```
