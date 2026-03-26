import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from config import AUTOCOMPLETE_MODELS, OLLAMA_PERMANENT, OLLAMA_SWAPPABLE, SWAPPABLE_MODELS
from state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HTTP client — shared across all requests, reuses connections
# ──────────────────────────────────────────────────────────────────────────────
# Timeout is generous: swap + inference on a large model can take 60–90s.
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

async def _swap_model(model: str) -> None:
    """
    Ask the swappable Ollama instance to load `model` by sending a
    zero-message generate request. Ollama loads the model into VRAM
    on the first request that names it — there is no explicit load API.

    The actual user request follows immediately after this returns.
    """
    logger.info(f"Swapping to model: {model} (was: {state.current_model})")
    try:
        # Sending keep_alive: -1 ensures the model stays loaded after this
        # warm-up request. The real keep_alive from the Modelfile takes over
        # once the model is resident.
        await _http_client.post(
            f"{OLLAMA_SWAPPABLE}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": -1},
        )
        state.current_model = model
        logger.info(f"Swap complete: {model} is now loaded")
    except httpx.RequestError as e:
        logger.error(f"Swap failed for {model}: {e}")
        raise


async def _forward(target_base_url: str, request: Request) -> Response:
    """
    Forward the incoming request to `target_base_url`, preserving method,
    headers, and body. Streams the response back to the caller.

    Strips the Host header — httpx sets it correctly for the target.
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
    Main routing handler. All Ollama API calls pass through here.

    Routing logic:
      1. Parse the model name from the request body (if present).
      2. If it's an autocomplete model → forward to permanent slot, no lock.
      3. If it's a swappable model → acquire lock, swap if needed, forward.
      4. If the model is unknown or absent → forward to swappable slot as a
         fallback (handles Ollama management calls like /api/tags, /api/pull).
    """
    body_bytes = await request.body()
    model: str | None = None

    # Try to extract the model name from the JSON body.
    # Not all Ollama endpoints carry a model field (e.g. /api/tags) — fine.
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
            model = body_json.get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Fast path: autocomplete models bypass the lock entirely ──────────────
    if model in AUTOCOMPLETE_MODELS:
        logger.debug(f"Fast path: {model} → permanent slot")
        return await _forward(OLLAMA_PERMANENT, request)

    # ── Swappable path: serialise via lock ───────────────────────────────────
    if model in SWAPPABLE_MODELS or model is not None:
        state.increment_queue()
        try:
            async with state.lock:
                # Decrement once we hold the lock — we're no longer waiting
                state.decrement_queue()
                if model in SWAPPABLE_MODELS and state.current_model != model:
                    await _swap_model(model)
                return await _forward(OLLAMA_SWAPPABLE, request)
        except Exception:
            # If we error before acquiring the lock, the increment was never
            # balanced by the decrement inside the lock block — fix it here.
            # (If we're inside the lock block, decrement already happened.)
            raise

    # ── Fallback: no model in body (e.g. /api/tags) → swappable slot ─────────
    logger.debug(f"No model in request body, forwarding to swappable: /{path}")
    return await _forward(OLLAMA_SWAPPABLE, request)
