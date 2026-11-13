# Proxy — FastAPI Orchestration Middleware

FastAPI service on `:11436`. It routes AI requests to the correct llama-server instance, serialises access to the swappable slot, and is the LLM's representative to the [GPU arbiter](../arbiter/README.md).

It has no GPU access. Giving it any would be a second answer to a question the arbiter already owns.

## What it does

- **Serialises every model request** through an `asyncio.Lock`, one at a time.
- **Acquires the GPU lease before touching the backend.** The router auto-loads on first request, so the card has to be clear before anything is forwarded. One call to the arbiter stops whatever is running, waits for the driver, and answers. A refusal becomes a `503` with `Retry-After: 30`.
- **Releases the lease only when idle** — when in-flight count and queue depth both hit zero, not after each request, so a burst of Discord traffic does not hand the card away mid-conversation.
- **Evicts the resident model after an idle period** so background jobs get a real VRAM window (see below).
- **Exposes Prometheus metrics**, health, and proxy state.

## What it deliberately does not do

**It does not inject system prompts.** Every prompt in use is built client-side: mimic prompts in the bot's `config.py`, the `/lore` agent's in `agent_tools.py`, Open WebUI's in its own UI. `/chat` threads send none at all.

**It does not report GPU state.** `/status` returns proxy state only. What is on the card and why is `/gpu/status` on the arbiter — asking two services the same question is how they end up disagreeing.

**It does not gate on the model list.** Any request carrying a model name goes to the swappable slot and llama-server decides whether it knows it; a request with no model in the body is forwarded unlocked (this is what makes `GET /v1/models` work). The model list is whatever `models.ini` defines — the proxy doesn't parse it, `GET /v1/models` just forwards to `llama-swappable` and surfaces its own parse.

## Idle eviction

llama-server's router keeps a model resident once loaded and never releases VRAM between requests. Without intervention, an external job would never get a window on a busy day — and an empty lease table is not headroom, because ~18 GB can still be sitting on the card with nothing in flight.

After `IDLE_EVICT_SECONDS` (default 600) with no LLM request, the evictor asks the router to unload and then **verifies** by re-reading `/v1/models` until nothing is resident. Verification rather than trust is the whole point: the guarantee being offered downstream is free memory.

Two details worth knowing:

- The router requires the model name to unload. An empty body is rejected with `400 "model is not found"` and nothing happens — hence reading the resident list first.
- This lives here rather than in the arbiter because nothing else may unload llama-server's models. Moving it would give two services authority over one container's state.

## Image generation

`POST /v1/images/generations` serves text-to-image from the `imagegen` compose service (stable-diffusion.cpp `sd-server`, Z-Image-Turbo). It is a **second backend behind the proxy's own lock and `llm` lease, not an arbiter tenant**. An image is interactive work a person waits on, the same as a chat message. See `imagegen.py` and [`docs/plans/image-gen.md`](../docs/plans/image-gen.md).

- **Text prompts only** (user decision, 2026-11-12: no seed images that could be used to depict server members). The proxy builds a fresh body from `prompt` and a `size` from an allowlist, with `n=1`; nothing else the caller sends is forwarded. `/v1/images/edits`, `/variations` and anything else under `/v1/images/` get a 403.
- **No VRAM held while idle.** The container is started on the first request (~8 s, then ~5 s to load weights) and stopped through `docker-socket-proxy` (`DOCKER_URL`):
  - by the idle evictor, together with the chat model;
  - before any chat model outside `IMAGEGEN_COEXIST_MODELS` loads;
  - when the proxy starts.
- **Coexistence.** It holds ~11.2 GB, so the 12B mimic base (~9.3 GB) can stay resident beside it. Anything bigger is unloaded before an image is generated. A failed stop refuses the chat request rather than risk an OOM.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, used by the compose healthcheck |
| `GET` | `/status` | `current_model`, `queue_depth`, `llm_inflight`, `model_loaded_seconds_ago` |
| `GET` | `/metrics` | Prometheus |
| `GET` | `/history`, `/history/summary` | **Stubs returning `[]`.** Kept for Homepage compatibility; the data moved to Prometheus |
| `POST` | `/v1/images/generations` | Text-to-image (see Image generation). 503 while `IMAGEGEN_ENABLED` is off |
| `*` | `/v1/images/{rest}` | 403: image inputs are not accepted |
| `*` | `/{path:path}` | Everything else, forwarded to the appropriate llama-server |

## Metrics

`proxy_requests_total{model,status}`, `proxy_active_requests{model}`, `proxy_tokens_total{model,token_type}`, `proxy_queue_depth`, `proxy_request_duration_seconds{model}`, `proxy_current_model_info{model}`, `proxy_model_age_seconds`, `proxy_llm_requests_inflight`, `proxy_llm_idle_seconds`.

GPU metrics are the arbiter's — it is the only component that reads the card.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SWAPPABLE` | `http://localhost:11434` | Swappable slot |
| `ARBITER_URL` | `http://arbiter:11438` | The proxy's whole interface to the GPU |
| `ARBITER_JOB_NAME` | `llm` | Which entry in `arbiter/jobs.yaml` this proxy is. It sends this and nothing else — priority and VRAM are decided there. An unknown name falls back to priority 0 and is logged loudly |
| `IDLE_EVICT_ENABLED` | `true` | |
| `IDLE_EVICT_SECONDS` | `600` | Idle time before unloading |
| `IDLE_EVICT_POLL_SECONDS` | `30` | How often the evictor checks |
| `IDLE_EVICT_UNLOAD_TIMEOUT` | `60` | How long to wait for an unload to take effect |
| `BATCH_MODELS` | `judge,caption` | Models ranked as batch work (`llm-batch`) |
| `IMAGEGEN_ENABLED` | `false` | Serve `/v1/images/generations` |
| `IMAGEGEN_URL` | `http://imagegen:8080` | The image server |
| `IMAGEGEN_CONTAINER` | `imagegen` | Container started/stopped; must match `container_name` in compose |
| `DOCKER_URL` | `http://docker-socket-proxy:2375` | Restricted Docker API used for that |
| `IMAGEGEN_START_TIMEOUT` / `IMAGEGEN_REQUEST_TIMEOUT` | `90` / `300` | Seconds |

Changing how the LLM ranks against the trainer is an edit to [`arbiter/jobs.yaml`](../arbiter/jobs.yaml), not to anything here.

## File structure

```
proxy/
├── Dockerfile
├── requirements.txt
├── main.py            # FastAPI app, routing, idle evictor, metrics, forwarding
├── config.py          # URLs, arbiter identity, eviction tuning
├── state.py           # OrchestratorState — lock, queue depth, in-flight, current model
├── arbiter.py         # ArbiterClient — acquire / release
├── imagegen.py        # image server lifecycle + the text-only request filter
└── tests/
    ├── test_gpu_gate.py      # the lease is acquired before forwarding, released only when idle
    ├── test_idle_evictor.py  # unload is verified, not assumed
    └── test_imagegen.py      # text only, same gate as chat, never shares the card with a big model
```

`arbiter.py` is a deliberate copy of the two calls in `lora-training/arbiter.py` rather than a shared library. The previous design vendored a client into three repos and every protocol change became a three-repo change.

## Tests

```bash
.venv-test/bin/python -m pytest tests -q
```

## Design reference

[`docs/design/system.md`](../docs/design/system.md) §4b (GPU arbitration), §6 (proxy behaviour), §7 (container layout).
