import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from config import (
    ARBITER_BATCH_JOB_NAME,
    ARBITER_JOB_NAME,
    ARBITER_URL,
    BATCH_MODELS,
    BATCH_RETRY_SECONDS,
    BATCH_YIELD_SECONDS,
    IDLE_EVICT_ENABLED,
    IDLE_EVICT_POLL_SECONDS,
    IDLE_EVICT_SECONDS,
    IDLE_EVICT_UNLOAD_TIMEOUT,
    DOCKER_URL,
    IMAGEGEN_CONTAINER,
    IMAGEGEN_COEXIST_MODELS,
    IMAGEGEN_DEFAULT_SIZE,
    IMAGEGEN_ENABLED,
    IMAGEGEN_MAX_PROMPT_CHARS,
    IMAGEGEN_REQUEST_TIMEOUT,
    IMAGEGEN_SIZES,
    IMAGEGEN_START_TIMEOUT,
    IMAGEGEN_URL,
    LLAMA_SWAPPABLE,
)
from arbiter import ArbiterClient
from dispatcher import BATCH, INTERACTIVE, Abandoned, Dispatcher, Lease, Refused
from imagegen import ImageServer, RequestError, clean_request
from state import state

image_server = ImageServer(
    IMAGEGEN_URL, DOCKER_URL, IMAGEGEN_CONTAINER, IMAGEGEN_START_TIMEOUT, IMAGEGEN_REQUEST_TIMEOUT
)


def build_dispatcher(llm_arbiter, batch_arbiter) -> Dispatcher:
    """
    The proxy is one arbiter tenant: `llm` while it has interactive work,
    `llm-batch` (cooperative) while it has only batch work. See dispatcher.py.
    """
    return Dispatcher(
        Lease(llm_arbiter, batch_arbiter),
        BATCH_YIELD_SECONDS,
        batch_retry_seconds=BATCH_RETRY_SECONDS,
        # Looked up at call time: defined further down, and tests replace it.
        free_gpu=lambda: _free_gpu(),
    )


dispatcher = build_dispatcher(
    ArbiterClient(ARBITER_URL, ARBITER_JOB_NAME),
    # Asked under its own, lower-ranked name while only batch work is queued.
    ArbiterClient(ARBITER_URL, ARBITER_BATCH_JOB_NAME),
)

# HTTP client — shared across all requests, reuses connections
_http_client: httpx.AsyncClient | None = None

# How many bytes of the first response chunk to log as a preview.
_RESPONSE_PREVIEW_BYTES = 120

# Prometheus Metrics
PROXY_REQUESTS = Counter("proxy_requests_total", "Total LLM requests", ["model", "status"])
PROXY_ACTIVE_REQUESTS = Gauge("proxy_active_requests", "Currently processing requests", ["model"])
PROXY_TOKENS = Counter("proxy_tokens_total", "Total tokens processed", ["model", "token_type"])
PROXY_QUEUE_DEPTH = Gauge("proxy_queue_depth", "Interactive requests waiting for their turn")
PROXY_BATCH_QUEUE_DEPTH = Gauge("proxy_batch_queue_depth", "Batch requests waiting for their turn")
PROXY_REQUEST_DURATION = Histogram("proxy_request_duration_seconds", "Total request time", ["model"])
PROXY_CURRENT_MODEL_INFO = Gauge(
    "proxy_current_model_info",
    "Currently loaded swappable model, one series per model (value is always 1)",
    ["model"],
)
PROXY_MODEL_AGE_SECONDS = Gauge(
    "proxy_model_age_seconds", "Seconds since the last model swap in the swappable slot"
)
PROXY_LLM_INFLIGHT = Gauge("proxy_llm_requests_inflight", "In-flight LLM requests")
PROXY_LLM_IDLE_SECONDS = Gauge("proxy_llm_idle_seconds", "Seconds since the last LLM request")

# GPU metrics are the arbiter's — it is the only component that reads the card.


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    logger.info(f"HTTP client initialised — arbiter at {ARBITER_URL}, asking as {ARBITER_JOB_NAME!r}")
    # A lease from before a restart would outrank every job forever (`llm` is
    # kind: none and never reaped); this process holds nothing yet.
    await dispatcher.release_lease_if_idle()
    if IMAGEGEN_ENABLED:
        # `compose up` may have started it; the proxy cannot know what it holds.
        image_server.running = True
        await image_server.stop(_http_client)
    tasks = []
    if IDLE_EVICT_ENABLED:
        tasks.append(asyncio.create_task(_idle_evictor()))
    yield
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
    await dispatcher.lease.aclose()
    await _http_client.aclose()
    await dispatcher.lease.llm.aclose()
    await dispatcher.lease.batch.aclose()
    logger.info("HTTP client closed")


app = FastAPI(title="Orchestration Proxy", lifespan=lifespan)


# Idle eviction — hand the GPU to external jobs when the LLMs are quiet

async def _resident_models() -> list[str]:
    """Model ids the swappable router currently has loaded."""
    resp = await _http_client.get(f"{LLAMA_SWAPPABLE}/v1/models", timeout=10.0)
    resp.raise_for_status()
    return [
        m.get("id", "?")
        for m in resp.json().get("data", [])
        if (m.get("status") or {}).get("value", "unloaded") != "unloaded"
    ]


async def _unload_resident_models(resident: list[str] | None = None) -> bool:
    """
    Ask the router to unload, then verify it actually happened.

    Returns True once nothing is resident. We verify rather than trust the
    response because the whole point is to guarantee free VRAM before telling
    an external job it may run.

    The router requires the model name — an empty body is rejected with
    400 "model is not found" and nothing is unloaded.
    """
    if resident is None:
        try:
            resident = await _resident_models()
        except Exception as exc:
            logger.warning(f"Idle evictor: could not read model status: {exc}")
            return False

    for model in resident:
        try:
            resp = await _http_client.post(
                f"{LLAMA_SWAPPABLE}/models/unload",
                json={"model": model},
                timeout=30.0,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"Idle evictor: unload of '{model}' returned "
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return False
        except Exception as exc:
            logger.warning(f"Idle evictor: unload of '{model}' failed: {exc}")
            return False

    deadline = time.monotonic() + IDLE_EVICT_UNLOAD_TIMEOUT
    while time.monotonic() < deadline:
        try:
            resident = await _resident_models()
        except Exception as exc:
            logger.warning(f"Idle evictor: could not read model status: {exc}")
            return False
        if not resident:
            state.current_model = None
            return True
        await asyncio.sleep(2.0)

    logger.warning(f"Idle evictor: models still resident after {IDLE_EVICT_UNLOAD_TIMEOUT}s")
    return False


async def _free_gpu() -> bool:
    """
    Unload every resident model and stop the image server, verified. True once
    the card really holds nothing of the proxy's. Used by the idle evictor and
    when the arbiter reclaims the batch lease.
    """
    resident = await _resident_models()
    if resident:
        logger.info(f"Unloading {resident}")
        if not await _unload_resident_models(resident):
            return False
    if image_server.running:
        logger.info("Stopping the image server")
        if not await image_server.stop(_http_client):
            return False
    return True


async def _idle_evictor() -> None:
    """
    Unload the swappable slot once the backend has been idle long enough.

    The router keeps a model resident indefinitely once loaded, so without this
    a background job would be permanently starved on any day with LLM traffic —
    it would be told it may run while 18 GB of model sat on the card.

    This is the proxy's job rather than the arbiter's precisely because
    llama-server is privileged: nothing else is allowed to unload its models. The
    arbiter is told afterwards, and decides for itself whether the freed memory
    is enough for whatever is waiting.

    It runs as a turn of its own (dispatcher.maintenance): nothing can start on
    the backend while it is unloading, so it can never stop the image server or
    release the lease under a request that began meanwhile.
    """
    logger.info(f"Idle evictor active: unload after {IDLE_EVICT_SECONDS:.0f}s idle")
    while True:
        await asyncio.sleep(IDLE_EVICT_POLL_SECONDS)
        try:
            idle = state.idle_seconds
            if not dispatcher.idle or idle is None or idle < IDLE_EVICT_SECONDS:
                continue
            async with dispatcher.maintenance() as ok:
                if not ok:
                    continue
                resident = await _resident_models()
                if not resident and not image_server.running:
                    continue
                logger.info(f"Backend idle {idle:.0f}s — freeing the GPU")
                if not await _free_gpu():
                    continue
                # Only now is there really room. Telling the arbiter before the
                # unload lands would invite it to start a job under a resident
                # model, which is the OOM this whole arrangement exists to avoid.
                await dispatcher.release_lease_if_idle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Idle evictor error: {exc}")


# Internal helpers

def _extract_model(body_bytes: bytes) -> str | None:
    """
    Extract the model name from a JSON request body strictly for routing.
    Does NOT modify the body.
    """
    if not body_bytes:
        return None
    try:
        body_json = json.loads(body_bytes)
        return body_json.get("model")
    except (json.JSONDecodeError, AttributeError):
        return None


def _sanitise_preview(raw: bytes, max_bytes: int = _RESPONSE_PREVIEW_BYTES) -> str:
    """
    Cleans up the log output preview (does NOT affect the actual stream data).
    """
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    if text.startswith("data:"):
        try:
            json_part = text[5:].strip()
            if json_part and json_part != "[DONE]":
                chunk_json = json.loads(json_part)
                choices = chunk_json.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content") or \
                              choices[0].get("text", "")
                    if content:
                        return repr(content[:80])
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    return repr(text.replace("\n", "\\n").replace("\r", ""))


async def _forward(
    target_base_url: str,
    request: Request,
    model: str | None = None,
    on_complete=None,
) -> Response:
    """
    Transparently forward the raw request to the target_base_url.

    ``on_complete`` runs once the response stream is fully consumed (or the
    client disconnects) — it hands the request's turn back to the dispatcher.
    """
    try:
        body = await request.body()
        
        # Log the outgoing request
        if "chat/completions" in request.url.path or "completions" in request.url.path:
            logger.info(f"\n{'='*60}\nINTERCEPTED OUTGOING REQUEST TO: {target_base_url}{request.url.path}\n{'='*60}")
            try:
                body_json = json.loads(body.decode("utf-8"))
                logger.info(json.dumps(body_json, indent=2))
            except Exception:
                logger.info(f"Could not parse body as JSON. Raw body: {body}")
            logger.info(f"{'='*60}\n")

        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        req = _http_client.build_request(
            method=request.method,
            url=f"{target_base_url}{request.url.path}",
            headers=headers,
            content=body,
            params=request.query_params,
        )

        label = model or "unknown_model"
        t_start = time.monotonic()

        # Track request start in Prometheus
        if model:
            PROXY_ACTIVE_REQUESTS.labels(model=label).inc()

        try:
            response = await _http_client.send(req, stream=True)
        except Exception as exc:
            if model:
                PROXY_ACTIVE_REQUESTS.labels(model=label).dec()
                PROXY_REQUESTS.labels(model=label, status="error").inc()
            logger.error(f"[{label}] request failed: {exc}")
            raise
        
    except BaseException:
        if on_complete:
            on_complete()
        raise

    t_first_byte = time.monotonic()
    logger.info(f"[{label}] HTTP {response.status_code} (first byte in {t_first_byte - t_start:.2f}s)")

    async def _stream_with_timing():
        full_content = []  # Buffer to store the text response
        input_tokens = 0    # Track prompt/input tokens from usage field
        output_tokens = 0   # Track completion tokens from usage field
        try:
            first = True
            async for chunk in response.aiter_bytes():
                if first and chunk:
                    preview = _sanitise_preview(chunk)
                    logger.info(f"[{label}] response preview: {preview}")
                    first = False
                
                # --- RESPONSE CAPTURE LOGIC ---
                try:
                    # Decode chunk and split by lines (SSE format)
                    lines = chunk.decode("utf-8", errors="replace").splitlines()
                    for line in lines:
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                continue

                            data_json = json.loads(data_str)
                            # Extract content from 'chat/completions' or 'completions' formats
                            choices = data_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content") or choices[0].get("text", "")
                                if content:
                                    full_content.append(content)
                            # Extract token usage if present (usually in final chunk)
                            usage = data_json.get("usage")
                            if usage:
                                if "completion_tokens" in usage:
                                    output_tokens = usage["completion_tokens"]
                                if "prompt_tokens" in usage:
                                    input_tokens = usage["prompt_tokens"]
                except Exception:
                    # If it's not JSON/SSE (like a direct error message), skip parsing
                    pass
                # ------------------------------

                yield chunk

            t_done = time.monotonic()

            # Print the final combined response to logs
            if full_content:
                combined_text = "".join(full_content)
                logger.info(f"\n{'*'*60}\nFULL LLM RESPONSE ({label}):\n{combined_text}\n{'*'*60}")

            logger.info(f"[{label}] completed in {t_done - t_start:.2f}s total")

        finally:
            # Always record Prometheus metrics, even if client disconnects mid-stream
            if model:
                # Fallback: estimate tokens from character count if usage not available (~4 chars/token)
                if output_tokens == 0 and full_content:
                    output_tokens = max(1, len("".join(full_content)) // 4)

                PROXY_TOKENS.labels(model=label, token_type="input").inc(input_tokens)
                PROXY_TOKENS.labels(model=label, token_type="output").inc(output_tokens)
                PROXY_REQUESTS.labels(model=label, status=str(response.status_code)).inc()
                PROXY_REQUEST_DURATION.labels(model=label).observe(time.monotonic() - t_start)
                PROXY_ACTIVE_REQUESTS.labels(model=label).dec()

            if on_complete:
                on_complete()
            await response.aclose()

    return StreamingResponse(
        content=_stream_with_timing(),
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )


# Routes

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    """
    Proxy state only. What is on the GPU and why is `/gpu/status` on the arbiter —
    asking two services the same question is how they end up disagreeing.
    """
    age = state.time_since_swap
    running = dispatcher.running
    return {
        "current_model": state.current_model,
        # Interactive requests waiting; the bot refuses commands past its
        # MAX_QUEUE_DEPTH on this, so batch work is counted separately.
        "queue_depth": dispatcher.queue_depth,
        "batch_queue_depth": dispatcher.batch_waiting,
        "llm_inflight": 1 if running is not None else 0,
        "running": running.klass if running is not None else None,
        "lease": dispatcher.lease.held,
        "model_loaded_seconds_ago": round(age, 1) if age is not None else None,
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    PROXY_QUEUE_DEPTH.set(dispatcher.queue_depth)
    PROXY_BATCH_QUEUE_DEPTH.set(dispatcher.batch_waiting)
    PROXY_LLM_INFLIGHT.set(1 if dispatcher.running is not None else 0)

    PROXY_CURRENT_MODEL_INFO.clear()
    if state.current_model:
        PROXY_CURRENT_MODEL_INFO.labels(model=state.current_model).set(1)

    idle = state.idle_seconds
    if idle is not None:
        PROXY_LLM_IDLE_SECONDS.set(idle)

    age = state.time_since_swap
    if age is not None:
        PROXY_MODEL_AGE_SECONDS.set(age)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/history")
async def history() -> list[dict]:
    """
    Stub endpoint for Homepage compatibility.

    Previously returned job history as Homepage-compatible JSON.
    Now returns empty list — data migrated to Prometheus metrics.
    Re-implement later by querying Prometheus API if needed.
    """
    return []


@app.get("/history/summary")
async def history_summary() -> list[dict]:
    """
    Stub endpoint for Homepage compatibility.

    Previously returned a minimal summary of the last 10 jobs.
    Now returns empty list — data migrated to Prometheus metrics.
    Re-implement later by querying Prometheus API if needed.
    """
    return []


@app.post("/v1/images/generations")
async def images_generations(request: Request) -> Response:
    """
    Text-to-image. Interactive work: a turn on the dispatcher and the `llm`
    lease, like a chat message — see imagegen.py for why it is not a tenant.
    """
    if not IMAGEGEN_ENABLED:
        return _json_error(503, "imagegen_disabled", "Image generation is not enabled on this server.")
    try:
        body = clean_request(await request.json(), IMAGEGEN_SIZES, IMAGEGEN_MAX_PROMPT_CHARS, IMAGEGEN_DEFAULT_SIZE)
    except (RequestError, ValueError) as exc:
        return _json_error(400, "invalid_request", str(exc))

    ticket, refusal = await _take_turn(INTERACTIVE, request, "imagegen")
    if refusal is not None:
        return refusal
    try:
        resident = await _resident_models()
        if any(m not in IMAGEGEN_COEXIST_MODELS for m in resident):
            logger.info(f"Image request: unloading {resident} to make room")
            if not await _unload_resident_models(resident):
                return _json_error(503, "vram_unavailable", "Could not unload the chat model. Retry shortly.")
        t0 = time.monotonic()
        resp = await image_server.generate(_http_client, body)
        logger.info(f"[imagegen] HTTP {resp.status_code} in {time.monotonic() - t0:.1f}s ({body['size']})")
        PROXY_REQUESTS.labels(model="imagegen", status=str(resp.status_code)).inc()
        PROXY_REQUEST_DURATION.labels(model="imagegen").observe(time.monotonic() - t0)
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
    except Exception as exc:
        logger.error(f"[imagegen] failed: {exc}")
        PROXY_REQUESTS.labels(model="imagegen", status="error").inc()
        return _json_error(502, "imagegen_failed", "Image generation failed.")
    finally:
        _finisher(ticket)()


@app.api_route("/v1/images/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def images_other(rest: str) -> Response:
    """
    Edits and variations take a source image, which is exactly what this
    server does not accept (imagegen.py) — refused, never forwarded.
    """
    return _json_error(403, "text_prompts_only", "Only text-to-image (/v1/images/generations) is available.")


def _json_error(status: int, code: str, message: str, detail: str | None = None) -> Response:
    body = {"error": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return Response(
        content=json.dumps(body),
        status_code=status,
        headers={"Retry-After": "30"} if status == 503 else None,
        media_type="application/json",
    )


async def _take_turn(klass: str, request: Request, label: str):
    """(ticket, None) once it is this request's turn, or (None, the response to send instead)."""
    state.record_request()
    try:
        return await dispatcher.turn(klass, request, label), None
    except Refused as exc:
        return None, _json_error(
            503, "vram_unavailable", "The GPU could not be freed for this request. Retry shortly.", exc.detail
        )
    except Abandoned:
        # The client is gone; nobody reads this. 499 is nginx's "client closed request".
        return None, Response(status_code=499)


def _finisher(ticket):
    """Hands a request's turn back exactly once, from whichever exit path gets there first."""
    done = False

    def finish():
        nonlocal done
        if done:
            return
        done = True
        state.record_request()
        dispatcher.finish(ticket)

    return finish


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    body_bytes = await request.body()
    model = _extract_model(body_bytes)

    if model is not None:
        klass = BATCH if model in BATCH_MODELS else INTERACTIVE
        ticket, refusal = await _take_turn(klass, request, model)
        if refusal is not None:
            return refusal
        finish = _finisher(ticket)
        try:
            if image_server.running and model not in IMAGEGEN_COEXIST_MODELS:
                # Too big to share the card with it: stop it before the
                # router loads anything, or the load OOMs.
                if not await image_server.stop(_http_client):
                    raise RuntimeError("the image server could not be stopped to make room")
            if state.current_model != model:
                logger.info(f"Model switch: {state.current_model} → {model} (swappable slot evict + load)")
                state.record_swap(model)
            return await _forward(LLAMA_SWAPPABLE, request, model=model, on_complete=finish)
        except BaseException:
            finish()
            raise

    # No model in the body (GET /v1/models and the like): nothing is loaded, so no turn.
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(LLAMA_SWAPPABLE, request)
