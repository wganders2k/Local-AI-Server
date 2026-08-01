import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from config import (
    ARBITER_JOB_NAME,
    ARBITER_URL,
    AUTOCOMPLETE_ENABLED,
    AUTOCOMPLETE_MODELS,
    IDLE_EVICT_ENABLED,
    IDLE_EVICT_POLL_SECONDS,
    IDLE_EVICT_SECONDS,
    IDLE_EVICT_UNLOAD_TIMEOUT,
    LLAMA_PERMANENT,
    LLAMA_SWAPPABLE,
    SWAPPABLE_MODELS,
)
from arbiter import ArbiterClient
from state import state

arbiter = ArbiterClient(ARBITER_URL, ARBITER_JOB_NAME)

# HTTP client — shared across all requests, reuses connections
_http_client: httpx.AsyncClient | None = None

# How many bytes of the first response chunk to log as a preview.
_RESPONSE_PREVIEW_BYTES = 120

# Prometheus Metrics
PROXY_REQUESTS = Counter("proxy_requests_total", "Total LLM requests", ["model", "status"])
PROXY_ACTIVE_REQUESTS = Gauge("proxy_active_requests", "Currently processing requests", ["model"])
PROXY_TOKENS = Counter("proxy_tokens_total", "Total tokens processed", ["model", "token_type"])
PROXY_QUEUE_DEPTH = Gauge("proxy_queue_depth", "Current requests waiting for a model swap")
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
    await _http_client.aclose()
    await arbiter.aclose()
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


async def _idle_evictor() -> None:
    """
    Unload the swappable slot once the LLMs have been idle long enough.

    The router keeps a model resident indefinitely once loaded, so without this
    a background job would be permanently starved on any day with LLM traffic —
    it would be told it may run while 18 GB of model sat on the card.

    This is the proxy's job rather than the arbiter's precisely because
    llama-server is privileged: nothing else is allowed to unload its models. The
    arbiter is told afterwards, and decides for itself whether the freed memory
    is enough for whatever is waiting.

    Deliberately unconditional on whether a job is waiting. The proxy no longer
    knows what jobs exist, and evicting an idle model is the right thing anyway —
    the cost is one cold load on the next request after ten quiet minutes.
    """
    logger.info(f"Idle evictor active: unload after {IDLE_EVICT_SECONDS:.0f}s idle")
    while True:
        await asyncio.sleep(IDLE_EVICT_POLL_SECONDS)
        try:
            if state.llm_inflight > 0 or state.queue_depth > 0:
                continue
            idle = state.idle_seconds
            if idle is None or idle < IDLE_EVICT_SECONDS:
                continue

            resident = await _resident_models()
            if resident:
                logger.info(f"LLMs idle {idle:.0f}s — unloading {resident}")
                if not await _unload_resident_models(resident):
                    continue
                # Only now is there really room. Telling the arbiter before the
                # unload lands would invite it to start a job under a resident
                # model, which is the OOM this whole arrangement exists to avoid.
                await arbiter.release()
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
    lock: asyncio.Lock | None = None,
    on_complete=None,
) -> Response:
    """
    Transparently forward the raw request to the target_base_url.

    ``on_complete`` runs once the response stream is fully consumed (or the
    client disconnects), alongside the lock release — used to drop the
    in-flight LLM refcount that gates external-job yielding.
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
        if lock:
            lock.release()
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

            if lock:
                lock.release()
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
    return {
        "current_model": state.current_model,
        "queue_depth": state.queue_depth,
        "llm_inflight": state.llm_inflight,
        "model_loaded_seconds_ago": round(age, 1) if age is not None else None,
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    PROXY_QUEUE_DEPTH.set(state.queue_depth)
    PROXY_LLM_INFLIGHT.set(state.llm_inflight)

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


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    body_bytes = await request.body()
    model = _extract_model(body_bytes)

    # Fast path: autocomplete models bypass the lock entirely
    if model in AUTOCOMPLETE_MODELS:
        if not AUTOCOMPLETE_ENABLED:
            logger.warning(f"Autocomplete model '{model}' requested but autocomplete is disabled")
            return Response(
                content=json.dumps({
                    "error": "autocomplete_disabled",
                    "message": "The autocomplete model is currently disabled. Enable it with 'make enable-autocomplete'."
                }),
                status_code=503,
                media_type="application/json",
            )
        logger.debug(f"Fast path: {model} → permanent slot")
        return await _forward(LLAMA_PERMANENT, request, model=model)

    # Swappable path: serialise via lock
    if model in SWAPPABLE_MODELS or model is not None:
        state.increment_queue()
        state.record_request()
        state.llm_request_started()
        released = False

        def _finish():
            """Drop the refcount; tell the arbiter once nothing is in flight."""
            nonlocal released
            if released:
                return
            released = True
            state.llm_request_finished()
            if state.llm_inflight == 0 and state.queue_depth == 0:
                asyncio.create_task(arbiter.release())

        try:
            # Clear the GPU before touching it. The router auto-loads on first
            # request, so this must complete before we forward anything.
            #
            # One call: the arbiter stops whatever is running, waits for the
            # driver to give the memory back, and answers. The proxy no longer
            # knows what a job is, how many there are, or how one is stopped.
            granted, detail = await arbiter.acquire()
            if not granted:
                logger.error(f"Arbiter would not hand over the GPU: {detail}")
                state.decrement_queue()
                _finish()
                return Response(
                    content=json.dumps({
                        "error": "vram_unavailable",
                        "message": "The GPU could not be freed for this request. Retry shortly.",
                        "detail": detail,
                    }),
                    status_code=503,
                    headers={"Retry-After": "30"},
                    media_type="application/json",
                )

            try:
                await state.lock.acquire()
            except BaseException:
                state.decrement_queue()
                raise

            state.decrement_queue()

            try:
                if model in SWAPPABLE_MODELS:
                    if state.current_model != model:
                        logger.info(
                            f"Model switch: {state.current_model} → {model} "
                            f"(swappable slot evict + load)"
                        )
                        state.record_swap(model)
            except BaseException:
                state.lock.release()
                raise

            return await _forward(
                LLAMA_SWAPPABLE, request, model=model, lock=state.lock, on_complete=_finish
            )
        except BaseException:
            _finish()
            raise

    # Fallback: no model in body
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(LLAMA_SWAPPABLE, request)