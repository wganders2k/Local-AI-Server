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
    AUTOCOMPLETE_ENABLED,
    AUTOCOMPLETE_MODELS,
    EXTERNAL_JOB_YIELD_ON_TIMEOUT,
    EXTERNAL_JOB_YIELD_TIMEOUT,
    GLANCES_URL,
    IDLE_EVICT_ENABLED,
    IDLE_EVICT_POLL_SECONDS,
    IDLE_EVICT_SECONDS,
    IDLE_EVICT_UNLOAD_TIMEOUT,
    LLAMA_PERMANENT,
    LLAMA_SWAPPABLE,
    SWAPPABLE_MODELS,
)
from state import state, external_jobs

# HTTP client — shared across all requests, reuses connections
_http_client: httpx.AsyncClient | None = None

# How many bytes of the first response chunk to log as a preview.
_RESPONSE_PREVIEW_BYTES = 120

# Prometheus Metrics
PROXY_REQUESTS = Counter("proxy_requests_total", "Total LLM requests", ["model", "status"])
PROXY_ACTIVE_REQUESTS = Gauge("proxy_active_requests", "Currently processing requests", ["model"])
PROXY_TOKENS = Counter("proxy_tokens_total", "Total tokens processed", ["model", "token_type"])
PROXY_QUEUE_DEPTH = Gauge("proxy_queue_depth", "Current requests waiting for a model swap")
PROXY_EXTERNAL_JOBS = Gauge("proxy_external_jobs_active", "Number of active external VRAM jobs")
GPU_VRAM_PERCENT = Gauge("gpu_vram_percent", "GPU VRAM usage percentage")
PROXY_REQUEST_DURATION = Histogram("proxy_request_duration_seconds", "Total request time", ["model"])

# Initialize VRAM gauge to 0 so it appears in /metrics before first Glances read
GPU_VRAM_PERCENT.set(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    logger.info("HTTP client initialised")
    # Background tasks
    tasks = [asyncio.create_task(_vram_monitor())]
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
    logger.info("HTTP client closed")


app = FastAPI(title="Orchestration Proxy", lifespan=lifespan)


# VRAM monitor — runs in background, logs GPU memory usage every 30s

async def _vram_monitor(interval: int = 30) -> None:
    """Poll Glances API for GPU VRAM usage and expose via Prometheus gauges."""
    if not GLANCES_URL:
        logger.info("GLANCES_URL not set — VRAM monitoring disabled")
        return

    gpu_api = f"{GLANCES_URL.rstrip('/')}/api/4/gpu"
    # NB: use _http_client directly — an `async with` here would close the
    # shared client for the whole proxy when this task exits.
    while True:
        try:
            resp = await _http_client.get(gpu_api, timeout=5.0)
            if resp.status_code == 200:
                gpus = resp.json()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    mem_percent = float(gpu.get("mem", 0))
                    GPU_VRAM_PERCENT.set(mem_percent)

                    model_label = state.current_model or "none"
                    age = state.time_since_swap
                    age_str = f"{age:.0f}s ago" if age is not None else "never swapped"
                    logger.info(
                        f"VRAM: {mem_percent:.1f}% | "
                        f"swappable slot: {model_label} (loaded {age_str})"
                    )
                else:
                    logger.warning("VRAM monitor: Glances returned empty GPU list")
            else:
                logger.warning(f"VRAM monitor: Glances API returned HTTP {resp.status_code}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"VRAM monitor error: {exc}")
        await asyncio.sleep(interval)


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
    Free the swappable slot once the LLMs have been idle long enough, so a
    registered external job gets a real window.

    The router keeps a model resident indefinitely once loaded, so without this
    an external job would be permanently starved on any day with LLM traffic.
    """
    logger.info(
        f"Idle evictor active: unload after {IDLE_EVICT_SECONDS:.0f}s idle "
        f"when an external job is waiting"
    )
    while True:
        await asyncio.sleep(IDLE_EVICT_POLL_SECONDS)
        try:
            if not external_jobs.external_job_running:
                continue
            if external_jobs.llm_inflight > 0 or state.queue_depth > 0:
                continue
            idle = state.idle_seconds
            if idle is not None and idle < IDLE_EVICT_SECONDS:
                continue

            resident = await _resident_models()
            if resident:
                logger.info(
                    f"LLMs idle {idle if idle is not None else float('inf'):.0f}s and external "
                    f"job(s) waiting — unloading {resident}"
                )
                if not await _unload_resident_models(resident):
                    continue

            if external_jobs.yield_requested:
                external_jobs.release_yield()
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
    age = state.time_since_swap
    return {
        "current_model": state.current_model,
        "queue_depth": state.queue_depth,
        "model_loaded_seconds_ago": round(age, 1) if age is not None else None,
        "external_job_running": external_jobs.external_job_running,
        "active_external_jobs": external_jobs.active_job_ids,
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    PROXY_QUEUE_DEPTH.set(state.queue_depth)
    PROXY_EXTERNAL_JOBS.set(len(external_jobs.active_job_ids))
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


# === External Job VRAM Coordination Endpoints ===

@app.post("/external-job/register")
async def external_job_register(body: dict) -> dict:
    """Register an external job (batch processing, ML inference, etc.)."""
    job_id = body.get("job_id", "")
    if not job_id:
        return {"error": "job_id required"}
    return external_jobs.register(job_id)


@app.post("/external-job/unregister")
async def external_job_unregister(body: dict) -> dict:
    """Unregister an external job."""
    job_id = body.get("job_id", "")
    if not job_id:
        return {"error": "job_id required"}
    return external_jobs.unregister(job_id)


@app.get("/external-job/status")
async def external_job_status(job_id: str = "") -> dict:
    """Check status of an external job (including yield state)."""
    if not job_id:
        return {
            "external_job_running": external_jobs.external_job_running,
            "active_jobs": external_jobs.active_job_ids,
            "yield_requested": external_jobs.yield_requested,
            "may_run": not external_jobs.yield_requested,
        }
    return external_jobs.get_status(job_id)


@app.post("/external-job/yield")
async def external_job_yield(body: dict) -> dict:
    """
    Handle yield/resume signals from external jobs.

    ``action: "yield"`` means "my VRAM is released" — the job is expected to
    have verified that before calling, since this is what unblocks the proxy
    from loading a model.
    """
    job_id = body.get("job_id", "")
    action = body.get("action", "")
    if not job_id:
        return {"error": "job_id required"}
    if action == "yield":
        return external_jobs.confirm_yield(job_id)
    elif action == "resume":
        return external_jobs.resume(job_id)
    else:
        return {"error": f"Unknown action: {action}"}


@app.get("/external-job/debug")
async def external_job_debug() -> dict:
    """Full coordination state — for validating the handover protocol."""
    return external_jobs.snapshot() | {
        "llm_idle_seconds": state.idle_seconds,
        "current_model": state.current_model,
        "queue_depth": state.queue_depth,
    }


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
        external_jobs.llm_request_started()
        released = False

        def _finish():
            """Drop the refcount; hand VRAM back once nothing is in flight."""
            nonlocal released
            if released:
                return
            released = True
            external_jobs.llm_request_finished()
            if external_jobs.llm_inflight == 0 and state.queue_depth == 0:
                external_jobs.release_yield()

        try:
            # Reclaim VRAM from external jobs before touching the GPU. The
            # router auto-loads on first request, so this must complete before
            # we forward anything — not after.
            if external_jobs.external_job_running:
                logger.info(
                    f"LLM model '{model}' requested — {len(external_jobs.active_job_ids)} "
                    f"external job(s) registered, requesting VRAM release: "
                    f"{external_jobs.active_job_ids}"
                )
                external_jobs.request_yield_all()
                try:
                    await asyncio.wait_for(
                        external_jobs.wait_for_yield(),
                        timeout=EXTERNAL_JOB_YIELD_TIMEOUT,
                    )
                    logger.info("External job(s) released VRAM — proceeding with model load")
                except asyncio.TimeoutError:
                    if EXTERNAL_JOB_YIELD_ON_TIMEOUT == "wait":
                        logger.warning(
                            f"External job(s) still holding VRAM after "
                            f"{EXTERNAL_JOB_YIELD_TIMEOUT:.0f}s — continuing to wait"
                        )
                        await external_jobs.wait_for_yield()
                    else:
                        # Refusing is the safe failure: loading a model on top
                        # of a job still holding several GB OOMs both workloads.
                        logger.error(
                            f"External job(s) did not release VRAM within "
                            f"{EXTERNAL_JOB_YIELD_TIMEOUT:.0f}s — refusing request "
                            f"rather than risking an OOM"
                        )
                        state.decrement_queue()
                        _finish()
                        return Response(
                            content=json.dumps({
                                "error": "vram_unavailable",
                                "message": (
                                    "An external GPU job has not released VRAM yet. "
                                    "Retry shortly."
                                ),
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