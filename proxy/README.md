# Proxy — FastAPI Orchestration Middleware

FastAPI service on `:11436`. It routes AI requests to the correct llama-server instance, schedules them one at a time with people ahead of batch work, and is the LLM's representative to the [GPU arbiter](../arbiter/README.md).

It has no GPU access. Giving it any would be a second answer to a question the arbiter already owns.

## What it does

- **Schedules every model request** (`dispatcher.py`): one on the backend at a time, interactive requests first in arrival order. A batch request (`BATCH_MODELS`) runs only when no interactive request is waiting or running and none has finished in the last `BATCH_YIELD_SECONDS`, so a person waits for at most the one batch call already running. A request whose client has gone away is dropped while it waits.
- **Is one arbiter tenant with one lease**, named for the most important work it has: `llm` while any interactive request is queued or running, `llm-batch` while it has only batch work. The lease is acquired before the backend is touched, because the router auto-loads on first request. An interactive refusal becomes a `503` with `Retry-After: 30`; a refused batch request stays queued and asks again after `BATCH_RETRY_SECONDS`, since its client is a background job.
- **Gives the batch lease back cooperatively.** `llm-batch` is a `cooperative` arbiter job: while holding it the proxy waits on the arbiter's reclaim notice, and when something ranked above batch work wants the card it stops dispatching batch requests, lets the running call finish, unloads its models and the image server, and releases. Moving up to `llm` when a person arrives is the same notice, answered at once with nothing unloaded.
- **Releases the lease only when idle** — when nothing is running and nothing is ready to run, not after each request, so a burst of Discord traffic does not hand the card away mid-conversation.
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

- **Allowlisted requests only.** The proxy builds a fresh body from `prompt`, plus a `size`, an `n` (up to 3), a `steps` (8 or 16) and an optional `cfg` (guidance, 1–7; unset is the server's 1), each from an allowlist. An optional `image` (base64 or a data: URL, up to 15 MB and 50 MP) makes it image-to-image, with `strength` (0.1–0.95, default 0.65): the proxy decodes the picture, crops it to `size` and re-encodes it as a fresh PNG, then runs it as a job on sd-server's native API. An optional `model` picks the image server (`alt`; anything else gets the default). Nothing else the caller sends is forwarded. `/v1/images/edits`, `/variations` and anything else under `/v1/images/` get a 403.
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
| `*` | `/v1/images/{rest}` | 403: edits and variations are not offered (image-to-image goes through `/generations`) |
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
| `BATCH_MODELS` | `qwen3.6-27b-judge-u,qwen3.6-27b-caption-u` | Models ranked as batch work (`llm-batch`) |
| `BATCH_YIELD_SECONDS` | `30` | Quiet time after the last interactive request before batch work runs |
| `BATCH_RETRY_SECONDS` | `5` | How long a refused batch request waits before asking again |
| `IMAGEGEN_ENABLED` | `false` | Serve `/v1/images/generations` |
| `IMAGEGEN_URL` | `http://imagegen:8080` | The image server |
| `IMAGEGEN_CONTAINER` | `imagegen` | Container started/stopped; must match `container_name` in compose |
| `IMAGEGEN_ALT_ENABLED` | `false` | Also serve the `imagegen-alt` container to requests with `"model": "alt"`; only one image server runs at a time |
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
├── dispatcher.py      # which request runs next, and the one arbiter lease it runs under
├── state.py           # current model, swap and idle clocks
├── arbiter.py         # ArbiterClient — acquire / release / reclaim notice
├── imagegen.py        # image server lifecycle + the request filter and image-to-image input
└── tests/
    ├── fakes.py              # a fake arbiter with the real admission rules, a stub backend
    ├── test_gpu_gate.py      # the lease before the backend; people before batch; handovers and reclaims
    ├── test_idle_evictor.py  # unload is verified, not assumed
    └── test_imagegen.py      # allowlisted requests, same gate as chat, never shares the card with a big model
```

`arbiter.py` is a deliberate copy of the two calls in `lora-training/arbiter.py` rather than a shared library. The previous design vendored a client into three repos and every protocol change became a three-repo change.

## Tests

```bash
.venv-test/bin/python -m pytest tests -q
```

## Design reference

[`docs/design/system.md`](../docs/design/system.md) §4b (GPU arbitration), §6 (proxy behaviour), §7 (container layout).
