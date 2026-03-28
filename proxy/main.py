import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from config import AUTOCOMPLETE_MODELS, LLAMA_PERMANENT, LLAMA_SWAPPABLE, SWAPPABLE_MODELS
from state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HTTP client — shared across all requests, reuses connections
# ──────────────────────────────────────────────────────────────────────────────
# Timeout is generous: model load + prefill on a large context can take several minutes.
# 600s covers: ~20k token prefill (~13s), long generation (n-predict=-1), and model swap (~8s).
# The client is created at startup and closed at shutdown via lifespan.
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
    yield
    vram_task.cancel()
    try:
        await vram_task
    except asyncio.CancelledError:
        pass
    await _http_client.aclose()
    logger.info("HTTP client closed")


app = FastAPI(title="Orchestration Proxy", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────────────
# VRAM monitor — runs in background, logs GPU memory usage every 30s
# ──────────────────────────────────────────────────────────────────────────────

async def _vram_monitor(interval: int = 30) -> None:
    """
    Periodically query nvidia-smi for GPU memory usage and log it alongside
    the currently loaded swappable model. Helps correlate VRAM consumption
    with model load/evict events.
    """
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
            return  # Don't keep retrying if nvidia-smi isn't available
        except Exception as exc:
            logger.warning(f"VRAM monitor error: {exc}")
        await asyncio.sleep(interval)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_model(body_bytes: bytes) -> str | None:
    """
    Extract the model name from a JSON request body.
    Handles both OpenAI format ({"model": "..."}) and any other JSON body
    that carries a top-level "model" field.
    Returns None if the body is empty, not JSON, or has no model field.
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
    Decode the first `max_bytes` of a response chunk into a loggable string.
    Strips newlines and control characters for readability.
    For SSE streams (data: {...}), extracts the content delta if present.
    """
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    # Try to extract the actual text content from an SSE chunk
    # SSE format: "data: {\"choices\":[{\"delta\":{\"content\":\"...\"},...}]}"
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
    # Fallback: return raw text with whitespace collapsed
    return repr(text.replace("\n", "\\n").replace("\r", ""))


async def _peek_and_stream(response: httpx.Response, label: str):
    """
    Async generator that peeks at the first chunk of a streaming response,
    logs a preview of its content, then yields all chunks unchanged.
    """
    first = True
    async for chunk in response.aiter_bytes():
        if first and chunk:
            preview = _sanitise_preview(chunk)
            logger.info(f"Response preview [{label}]: {preview}")
            first = False
        yield chunk


async def _forward(
    target_base_url: str,
    request: Request,
    model: str | None = None,
) -> Response:
    """
    Forward the incoming request to `target_base_url`, preserving method,
    headers, and body. Streams the response back to the caller.

    Strips the Host header — httpx sets it correctly for the target.
    The proxy is a transparent forwarder: it does not inspect or modify
    request/response content. Both sides speak OpenAI-compatible API.

    Logs:
    - Request: model name, endpoint path
    - Response: first ~120 bytes of generated content, total elapsed time
    """
    body = await request.body()
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

    response = await _http_client.send(req, stream=True)

    t_first_byte = time.monotonic()
    logger.info(
        f"[{label}] {request.method} {request.url.path} → "
        f"HTTP {response.status_code} "
        f"(first byte in {t_first_byte - t_start:.2f}s)"
    )

    async def _stream_with_timing():
        first = True
        async for chunk in response.aiter_bytes():
            if first and chunk:
                preview = _sanitise_preview(chunk)
                logger.info(f"[{label}] response preview: {preview}")
                first = False
            yield chunk
        t_done = time.monotonic()
        logger.info(
            f"[{label}] completed in {t_done - t_start:.2f}s total "
            f"({t_done - t_first_byte:.2f}s streaming)"
        )

    return StreamingResponse(
        content=_stream_with_timing(),
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

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
    }


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    """
    Main routing handler. All llama-server API calls pass through here.

    The proxy speaks OpenAI-compatible API on both the client-facing side
    and the backend side. llama-server's router mode handles model loading
    automatically — no explicit warm-up or swap call is needed.

    Routing logic:
      1. Parse the model name from the request body (if present).
      2. If it's an autocomplete model → forward to permanent slot, no lock.
      3. If it's a swappable model → acquire lock, update state, forward.
         llama-server's router loads/evicts the model automatically.
      4. If the model is unknown or absent → forward to swappable slot as a
         fallback (handles management calls like /v1/models, /health).
    """
    body_bytes = await request.body()
    model = _extract_model(body_bytes)

    # ── Fast path: autocomplete models bypass the lock entirely ──────────────
    if model in AUTOCOMPLETE_MODELS:
        logger.debug(f"Fast path: {model} → permanent slot")
        return await _forward(LLAMA_PERMANENT, request, model=model)

    # ── Swappable path: serialise via lock ───────────────────────────────────
    # llama-server's router mode handles the actual model load/eviction.
    # The proxy lock ensures only one request at a time reaches the swappable
    # slot, preventing concurrent requests from triggering simultaneous swaps.
    if model in SWAPPABLE_MODELS or model is not None:
        state.increment_queue()
        try:
            async with state.lock:
                # Decrement once we hold the lock — we're no longer waiting
                state.decrement_queue()
                if model in SWAPPABLE_MODELS:
                    # Track which model is active for /status reporting.
                    # The actual load/eviction is handled by llama-server's router.
                    if state.current_model != model:
                        logger.info(
                            f"Model switch: {state.current_model} → {model} "
                            f"(swappable slot evict + load)"
                        )
                        state.record_swap(model)
                return await _forward(LLAMA_SWAPPABLE, request, model=model)
        except Exception:
            raise

    # ── Fallback: no model in body (e.g. /v1/models, /health) ────────────────
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(LLAMA_SWAPPABLE, request)
