import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from config import AUTOCOMPLETE_MODELS, LLAMA_PERMANENT, LLAMA_SWAPPABLE, SWAPPABLE_MODELS
from state import state, external_jobs
from job_history import job_history

# HTTP client — shared across all requests, reuses connections
_http_client: httpx.AsyncClient | None = None

# How many bytes of the first response chunk to log as a preview.
_RESPONSE_PREVIEW_BYTES = 120


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    logger.info("HTTP client initialised")
    # Start background VRAM monitor
    vram_task = asyncio.create_task(_vram_monitor())
    # Start background job history heartbeat
    heartbeat_task = asyncio.create_task(_job_history_heartbeat())
    yield
    vram_task.cancel()
    heartbeat_task.cancel()
    try:
        await vram_task
    except asyncio.CancelledError:
        pass
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    await _http_client.aclose()
    logger.info("HTTP client closed")


app = FastAPI(title="Orchestration Proxy", lifespan=lifespan)


# VRAM monitor — runs in background, logs GPU memory usage every 30s

async def _vram_monitor(interval: int = 30) -> None:
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                line = stdout.decode().strip().splitlines()[0]
                used, total = (x.strip() for x in line.split(","))
                model_label = state.current_model or "none"
                age = state.time_since_swap
                age_str = f"{age:.0f}s ago" if age is not None else "never swapped"
                logger.info(
                    f"VRAM: {used} / {total} MiB | "
                    f"swappable slot: {model_label} (loaded {age_str})"
                )
        except FileNotFoundError:
            logger.warning("nvidia-smi not found — VRAM monitoring disabled")
            return
        except Exception as exc:
            logger.warning(f"VRAM monitor error: {exc}")
        await asyncio.sleep(interval)


async def _job_history_heartbeat(interval: int = 60) -> None:
    """
    Periodically pulse the job history to force-finalize any stale tasks
    that were never properly closed (e.g. client disconnected mid-stream).
    """
    while True:
        await asyncio.sleep(interval)
        try:
            stale_count = job_history.heartbeat()
            if stale_count > 0:
                logger.info(f"Job history heartbeat: force-finalized {stale_count} stale task(s)")
        except Exception as exc:
            logger.warning(f"Job history heartbeat error: {exc}")


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
) -> Response:
    """
    Transparently forward the raw request to the target_base_url.
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

        label = model or request.url.path
        t_start = time.monotonic()

        # Track request start in job history
        if model:
            job_history.request_start(model)

        response = await _http_client.send(req, stream=True)
        
    except BaseException:
        if lock:
            lock.release()
        raise

    t_first_byte = time.monotonic()
    logger.info(f"[{label}] HTTP {response.status_code} (first byte in {t_first_byte - t_start:.2f}s)")

    async def _stream_with_timing():
        full_content = []  # Buffer to store the text response
        input_tokens = 0    # Track prompt/input tokens from usage field
        output_tokens = 0   # Track completion tokens from usage field
        request_end_called = False  # Track to avoid double-calling
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
            # Always call request_end, even if client disconnects mid-stream
            if model and not request_end_called:
                t_done = time.monotonic()
                generation_time = t_done - t_first_byte
                # Fallback: estimate tokens from character count if usage not available (~4 chars/token)
                if output_tokens == 0 and full_content:
                    output_tokens = max(1, len("".join(full_content)) // 4)
                job_history.request_end(model, input_tokens, output_tokens, generation_time)
                request_end_called = True

            if lock:
                lock.release()
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


@app.get("/history")
async def history() -> list[dict]:
    """
    Return job history as Homepage-compatible JSON.

    Groups recent requests to the same model into tasks and returns
    metrics (request count, avg tokens/sec, total tokens, active status).
    """
    return job_history.get_homepage_json()


@app.get("/history/summary")
async def history_summary() -> list[dict]:
    """
    Return a minimal summary of the last 10 jobs.

    Each entry contains only two fields:
    - name: "model-name (X,XXX tk)"
    - description: "HH:MM:SS (active|completed)"
    """
    return job_history.get_summary_json()


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
        }
    return external_jobs.get_status(job_id)


@app.post("/external-job/yield")
async def external_job_yield(body: dict) -> dict:
    """Handle yield/resume actions from external jobs."""
    job_id = body.get("job_id", "")
    action = body.get("action", "")
    if not job_id:
        return {"error": "job_id required"}
    if action == "yield":
        return external_jobs.clear_yield(job_id)
    elif action == "resume":
        return external_jobs.resume(job_id)
    else:
        return {"error": f"Unknown action: {action}"}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    body_bytes = await request.body()
    model = _extract_model(body_bytes)

    # Fast path: autocomplete models bypass the lock entirely
    if model in AUTOCOMPLETE_MODELS:
        logger.debug(f"Fast path: {model} → permanent slot")
        return await _forward(LLAMA_PERMANENT, request, model=model)

    # Swappable path: serialise via lock
    if model in SWAPPABLE_MODELS or model is not None:
        state.increment_queue()
        
        # Yield to external jobs if we need to load/switch a model
        if model in SWAPPABLE_MODELS and external_jobs.external_job_running:
            active_jobs = external_jobs.active_job_ids
            if active_jobs:
                logger.info(
                    f"LLM model '{model}' requested — {len(active_jobs)} external job(s) running, "
                    f"requesting yield: {active_jobs}"
                )
                # Signal all external jobs to yield
                for job_id in active_jobs:
                    external_jobs.request_yield(job_id)
                # Wait for external jobs to acknowledge yield
                try:
                    await asyncio.wait_for(external_jobs.wait_for_yield(), timeout=120.0)
                    logger.info("External job(s) yielded — proceeding with model load")
                except asyncio.TimeoutError:
                    logger.warning("External job(s) did not yield in time — proceeding anyway")

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

        return await _forward(LLAMA_SWAPPABLE, request, model=model, lock=state.lock)

    # Fallback: no model in body
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(LLAMA_SWAPPABLE, request)