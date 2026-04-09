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
from state import state
from system_prompts import system_prompts


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
    yield
    vram_task.cancel()
    try:
        await vram_task
    except asyncio.CancelledError:
        pass
    await _http_client.aclose()
    logger.info("HTTP client closed")


app = FastAPI(title="Orchestration Proxy", lifespan=lifespan)


# VRAM monitor — runs in background, logs GPU memory usage every 30s

async def _vram_monitor(interval: int = 30) -> None:
    """
    Periodically query nvidia-smi for GPU memory usage and log it alongside
    the currently loaded swappable model.
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


# Internal helpers

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

async def _inject_system_prompt(
    body_bytes: bytes,
    model: str | None,
    headers: dict
) -> bytes:
    """
    Inject system prompt into the request body if one exists for the model.
    Handles both OpenAI chat format and llama-server format.
    """
    user_agent = headers.get("user-agent", "").lower()
    if "continue" in user_agent or "vscode" in user_agent:
        # Skip injection so we don't break Continue's strict formatting
        return body_bytes
    
    if not model or not system_prompts.has(model):
        return body_bytes

    try:
        body_json = json.loads(body_bytes)
    except (json.JSONDecodeError, AttributeError):
        return body_bytes

    # OpenAI chat format: {"messages": [{"role": "system", ...}, ...]}
    if "messages" in body_json:
        messages = body_json["messages"]
        # Check if system prompt already exists
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            system_prompt = system_prompts.get(model)
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
                logger.info(f"Injected system prompt for model: {model}")
        return json.dumps(body_json, ensure_ascii=False).encode("utf-8")

    # llama-server format: {"prompt": "...", "system": "..."}
    if "system" in body_json:
        existing_system = body_json.get("system", "")
        new_system = system_prompts.get(model)
        if new_system:
            # Prepend to existing system prompt
            body_json["system"] = f"{new_system}\n{existing_system}".strip()
            logger.info(f"Injected system prompt for model: {model}")
        return json.dumps(body_json, ensure_ascii=False).encode("utf-8")

    return body_bytes

async def _forward(
    target_base_url: str,
    request: Request,
    model: str | None = None,
    lock: asyncio.Lock | None = None,
) -> Response:
    """
    Forward the incoming request to target_base_url and stream the response.
    Safely releases the lock when the stream is completely finished or if an error occurs.
    """
    # 1. Wrap ALL setup in a try block to guarantee lock release on early disconnects
    try:
        body = await request.body()
        headers_dict = await request.headers
        body = await _inject_system_prompt(body, model, headers_dict)
        # Log the system prompt that was injected into the request body
        if model and system_prompts.has(model):
            logger.info(f"System prompt injected for model '{model}': {system_prompts.get(model)}")
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
        
    except BaseException:
        # BaseException catches asyncio.CancelledError if the client aborts early
        if lock:
            lock.release()
        raise

    t_first_byte = time.monotonic()
    logger.info(f"[{label}] HTTP {response.status_code} (first byte in {t_first_byte - t_start:.2f}s)")

    async def _stream_with_timing():
        try:
            first = True
            async for chunk in response.aiter_bytes():
                if first and chunk:
                    preview = _sanitise_preview(chunk)
                    logger.info(f"[{label}] response preview: {preview}")
                    first = False
                yield chunk
            t_done = time.monotonic()
            logger.info(f"[{label}] completed in {t_done - t_start:.2f}s total")
        finally:
            # THIS IS CRITICAL: Release the lock when the stream naturally ends
            # or if the client disconnects mid-stream.
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
        logger.debug(f"Fast path: {model} → permanent slot")
        return await _forward(LLAMA_PERMANENT, request, model=model)

    # Swappable path: serialise via lock
    if model in SWAPPABLE_MODELS or model is not None:
        state.increment_queue()
        
        try:
            await state.lock.acquire()
        except BaseException:
            # Client disconnected while waiting their turn in the queue
            state.decrement_queue()
            raise

        # We now have the lock. 
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
            # If our internal state logic fails for any reason, don't leak the lock
            state.lock.release()
            raise

        # 2. Hand off control to _forward, passing the lock!
        return await _forward(LLAMA_SWAPPABLE, request, model=model, lock=state.lock)

    # Fallback: no model in body (e.g. /v1/models, /health)
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(LLAMA_SWAPPABLE, request)