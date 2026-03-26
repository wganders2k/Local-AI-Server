import json
import logging
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
# Timeout is generous: model load + inference on a large model can take 60–90s.
# The client is created at startup and closed at shutdown via lifespan.
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    logger.info("HTTP client initialised")
    yield
    await _http_client.aclose()
    logger.info("HTTP client closed")


app = FastAPI(title="Orchestration Proxy", lifespan=lifespan)


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


async def _forward(target_base_url: str, request: Request) -> Response:
    """
    Forward the incoming request to `target_base_url`, preserving method,
    headers, and body. Streams the response back to the caller.

    Strips the Host header — httpx sets it correctly for the target.
    The proxy is a transparent forwarder: it does not inspect or modify
    request/response content. Both sides speak OpenAI-compatible API.
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

    response = await _http_client.send(req, stream=True)

    return StreamingResponse(
        content=response.aiter_bytes(),
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
    return {
        "current_model": state.current_model,
        "queue_depth": state.queue_depth,
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
        return await _forward(LLAMA_PERMANENT, request)

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
                        logger.info(f"Model switch: {state.current_model} → {model}")
                        state.current_model = model
                return await _forward(LLAMA_SWAPPABLE, request)
        except Exception:
            raise

    # ── Fallback: no model in body (e.g. /v1/models, /health) ────────────────
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(LLAMA_SWAPPABLE, request)
