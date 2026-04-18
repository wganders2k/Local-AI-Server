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

        response = await _http_client.send(req, stream=True)
        
    except BaseException:
        if lock:
            lock.release()
        raise

    t_first_byte = time.monotonic()
    logger.info(f"[{label}] HTTP {response.status_code} (first byte in {t_first_byte - t_start:.2f}s)")

    async def _stream_with_timing():
        full_content = []  # Buffer to store the text response
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