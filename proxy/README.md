# Proxy — FastAPI Orchestration Middleware

FastAPI service on `:11436` that routes AI requests to the correct llama-server instance and serialises access to the swappable model slot.

## Responsibilities

- Route autocomplete requests (permanent models) directly to `:11435` — no lock, no swap
- Serialise all swappable model requests via `asyncio.Lock` — one at a time
- Track which model is active in the swappable slot for `/status` reporting
- **Inject system prompts from `system_prompts.ini` before forwarding requests**
- llama-server's router mode handles the actual model load/eviction automatically
- Expose a `/health` endpoint for Docker health checks
- Expose a `/status` endpoint so the Discord bot can check queue depth

## System Prompts

System prompts are loaded from `system_prompts.ini` and injected into chat completion requests. Each model alias defined in `config.py` can have a corresponding `[system_prompt:<alias>]` section.

Example:
```ini
[system_prompt:brain]
prompt = """
You are Brain, a deep coding assistant...
"""
```

The proxy will prepend the system prompt to the request's `messages` array (OpenAI format) or `system` field (llama-server format).

## Design Reference

See `../docs/design/system.md` §4a (Inference Backend), §6 (Proxy State Machine), and §7 (Docker Compose).

## File structure

## File structure

```
proxy/
├── Dockerfile
├── requirements.txt
├── main.py          # FastAPI app, route handlers
├── config.py        # LLAMA_PERMANENT / LLAMA_SWAPPABLE URLs, model sets
├── state.py         # OrchestratorState — lock, queue depth, current_model
├── system_prompts.py # System prompt loader from INI file
└── system_prompts.ini # System prompt definitions (optional)
```
