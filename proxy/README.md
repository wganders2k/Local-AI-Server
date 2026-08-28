# Proxy — FastAPI Orchestration Middleware

FastAPI service on `:11436`. It routes AI requests to the correct llama-server instance, serialises access to the swappable slot, and is the LLM's representative to the [GPU arbiter](../arbiter/README.md).

It has no GPU access. Giving it any would be a second answer to a question the arbiter already owns.

## What it does

- **Routes autocomplete requests to the permanent slot** (`:11435`) — no lock, no arbiter, no swap. If `AUTOCOMPLETE_ENABLED` is false, these get a `503` instead.
- **Serialises every other request** through an `asyncio.Lock`, one at a time.
- **Acquires the GPU lease before touching the backend.** The router auto-loads on first request, so the card has to be clear before anything is forwarded. One call to the arbiter stops whatever is running, waits for the driver, and answers. A refusal becomes a `503` with `Retry-After: 30`.
- **Releases the lease only when idle** — when in-flight count and queue depth both hit zero, not after each request, so a burst of Discord traffic does not hand the card away mid-conversation.
- **Evicts the resident model after an idle period** so background jobs get a real VRAM window (see below).
- **Exposes Prometheus metrics**, health, and proxy state.

## What it deliberately does not do

**It does not inject system prompts.** [`system_prompts.py`](system_prompts.py) and the bind-mounted `system_prompts.ini` exist, but `main.py` never imports the loader — the code is inert and editing the INI has no effect. Every prompt in use is built client-side: mimic prompts in the bot's `config.py`, the `/lore` agent's in `agent_tools.py`, Open WebUI's in its own UI. `/chat` threads send none at all.

This is unfinished business, not a design stance: either wire the loader up or delete it, because a config file that silently does nothing is worse than no config file.

**It does not report GPU state.** `/status` returns proxy state only. What is on the card and why is `/gpu/status` on the arbiter — asking two services the same question is how they end up disagreeing.

**It does not gate on the model list.** `SWAPPABLE_MODELS` is advisory. Any request carrying a model name that isn't an autocomplete alias goes to the swappable slot and llama-server decides whether it knows it; a request with no model in the body is forwarded unlocked (this is what makes `GET /v1/models` work). The set only labels swap logging, so a preset missing from it still works — and today it disagrees with `models.ini` in both directions.

## Idle eviction

llama-server's router keeps a model resident once loaded and never releases VRAM between requests. Without intervention, an external job would never get a window on a busy day — and an empty lease table is not headroom, because ~18 GB can still be sitting on the card with nothing in flight.

After `IDLE_EVICT_SECONDS` (default 600) with no LLM request, the evictor asks the router to unload and then **verifies** by re-reading `/v1/models` until nothing is resident. Verification rather than trust is the whole point: the guarantee being offered downstream is free memory.

Two details worth knowing:

- The router requires the model name to unload. An empty body is rejected with `400 "model is not found"` and nothing happens — hence reading the resident list first.
- This lives here rather than in the arbiter because nothing else may unload llama-server's models. Moving it would give two services authority over one container's state.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, used by the compose healthcheck |
| `GET` | `/status` | `current_model`, `queue_depth`, `llm_inflight`, `model_loaded_seconds_ago` |
| `GET` | `/metrics` | Prometheus |
| `GET` | `/history`, `/history/summary` | **Stubs returning `[]`.** Kept for Homepage compatibility; the data moved to Prometheus |
| `*` | `/{path:path}` | Everything else, forwarded to the appropriate llama-server |

## Metrics

`proxy_requests_total{model,status}`, `proxy_active_requests{model}`, `proxy_tokens_total{model,token_type}`, `proxy_queue_depth`, `proxy_request_duration_seconds{model}`, `proxy_current_model_info{model}`, `proxy_model_age_seconds`, `proxy_llm_requests_inflight`, `proxy_llm_idle_seconds`.

GPU metrics are the arbiter's — it is the only component that reads the card.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLAMA_PERMANENT` | `http://localhost:11435` | Permanent slot |
| `LLAMA_SWAPPABLE` | `http://localhost:11434` | Swappable slot |
| `ARBITER_URL` | `http://arbiter:11438` | The proxy's whole interface to the GPU |
| `ARBITER_JOB_NAME` | `llm` | Which entry in `arbiter/jobs.yaml` this proxy is. It sends this and nothing else — priority and VRAM are decided there. An unknown name falls back to priority 0 and is logged loudly |
| `AUTOCOMPLETE_ENABLED` | `true` | False makes the `autocomplete` alias return 503 |
| `IDLE_EVICT_ENABLED` | `true` | |
| `IDLE_EVICT_SECONDS` | `600` | Idle time before unloading |
| `IDLE_EVICT_POLL_SECONDS` | `30` | How often the evictor checks |
| `IDLE_EVICT_UNLOAD_TIMEOUT` | `60` | How long to wait for an unload to take effect |
| `SYSTEM_PROMPTS_PATH` | `../system_prompts.ini` | Read by the inert loader only |

Changing how the LLM ranks against the trainer is an edit to [`arbiter/jobs.yaml`](../arbiter/jobs.yaml), not to anything here.

## File structure

```
proxy/
├── Dockerfile
├── requirements.txt
├── main.py            # FastAPI app, routing, idle evictor, metrics, forwarding
├── config.py          # URLs, model sets, arbiter identity, eviction tuning
├── state.py           # OrchestratorState — lock, queue depth, in-flight, current model
├── arbiter.py         # ArbiterClient — acquire / release
├── system_prompts.py  # INI loader — currently unused by main.py
└── tests/
    ├── test_gpu_gate.py      # the lease is acquired before forwarding, released only when idle
    └── test_idle_evictor.py  # unload is verified, not assumed
```

`arbiter.py` is a deliberate copy of the two calls in `lora-training/arbiter.py` rather than a shared library. The previous design vendored a client into three repos and every protocol change became a three-repo change.

## Tests

```bash
.venv-test/bin/python -m pytest tests -q
```

## Design reference

[`docs/design/system.md`](../docs/design/system.md) §4b (GPU arbitration), §6 (proxy behaviour), §7 (container layout).
