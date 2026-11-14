"""
Image generation: a second GPU backend behind the proxy's own lock and lease.

The image server (stable-diffusion.cpp's sd-server, compose service
`imagegen`) is not an arbiter tenant. Generating an image is interactive work
a person is waiting on, exactly like a chat message, so it runs under the
proxy's `llm` lease and the same swappable lock. A separate tenant would need
its own reclaim protocol, and would compete with the proxy for a card the
proxy already holds whenever anyone is chatting.

The container is stopped whenever it is not needed, so it holds no VRAM at
all — not even the ~300 MB of an idle CUDA context, which matters next to a
21 GB lore agent on a 24 GB card. It is started on the first request (1–8 s
to listening, then several seconds of weights on the first image) and stopped:
- before a chat model that does not fit beside it is loaded;
- by the idle evictor, with everything else;
- when the proxy starts, so a container `compose up` started is not left
  holding memory the proxy does not know about.
Start and stop go through docker-socket-proxy — the restricted Docker API
history-service and the arbiter already use — never the real socket.

Text prompts only, by the user's decision (2026-11-12): their concern is
server members being depicted negatively, and a seed image is how that would
happen. `clean_request` builds a fresh body from the prompt and a size from an
allowlist; nothing else the caller sends reaches the server. The edit and
variation endpoints are refused outright by main.py.

Measured 2026-11-13 on the sm_86 build (docs/plans/image-gen.md): 6–7 s per
warm 1024² image. Peak memory depends on the model .env selects: ~14 GB for
Z-Image-Turbo Q8, ~20 GB for a 12B int8 DiT.
"""

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


class RequestError(ValueError):
    """The caller's request cannot be served as asked; the message says why."""


# sd-server lifts `<sd_cpp_extra_args>{json}</sd_cpp_extra_args>` out of the
# prompt and applies it as generation parameters, reference images included.
# Allowing it would bypass both the size allowlist and the text-only rule.
_EXTRA_ARGS_TAG = "<sd_cpp_extra_args"


def clean_request(body: object, sizes: set[str], max_prompt_chars: int, default_size: str,
                  max_images: int = 1) -> dict:
    """
    The only body ever sent to the image server: prompt, size, how many images,
    base64. Raises RequestError for anything the caller must fix.
    """
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("'prompt' is required")
    if len(prompt) > max_prompt_chars:
        raise RequestError(f"'prompt' is longer than {max_prompt_chars} characters")
    if _EXTRA_ARGS_TAG in prompt.lower():
        raise RequestError("'prompt' may not carry sd_cpp_extra_args")
    size = body.get("size") or default_size
    if not isinstance(size, str) or size not in sizes:
        raise RequestError(f"'size' must be one of {sorted(sizes)}")
    n = body.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= max_images:
        raise RequestError(f"'n' must be an integer from 1 to {max_images}")
    return {"prompt": prompt.strip(), "size": size, "n": n, "response_format": "b64_json"}


class ImageServer:
    """Starts, stops and calls the image server container."""

    def __init__(self, url: str, docker_url: str, container: str, start_timeout: float, request_timeout: float):
        self.url = url.rstrip("/")
        self.docker_url = docker_url.rstrip("/")
        self.container = container
        self.start_timeout = start_timeout
        self.request_timeout = request_timeout
        # Whether the container may be holding VRAM. Set on start, cleared only
        # once a stop has been confirmed — when unsure, assume it is running.
        self.running = False
        self.last_used: float | None = None

    async def _docker(self, client: httpx.AsyncClient, action: str) -> None:
        # t=0: sd-server ignores SIGTERM (measured: `docker stop` always ran
        # out its 10 s grace), and it holds nothing worth flushing.
        params = {"t": "0"} if action == "stop" else None
        resp = await client.post(
            f"{self.docker_url}/containers/{self.container}/{action}", params=params, timeout=60.0
        )
        # 204 done, 304 already in that state.
        if resp.status_code not in (204, 304):
            raise RuntimeError(f"docker {action} {self.container}: HTTP {resp.status_code} {resp.text[:200]}")

    async def start(self, client: httpx.AsyncClient) -> None:
        self.running = True
        await self._docker(client, "start")
        deadline = time.monotonic() + self.start_timeout
        while True:
            try:
                await client.get(f"{self.url}/", timeout=5.0)
                return
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"image server not listening after {self.start_timeout:.0f}s")
                await asyncio.sleep(1.0)

    async def stop(self, client: httpx.AsyncClient) -> bool:
        """Stop the container. True once confirmed; a failure leaves `running` set."""
        try:
            await self._docker(client, "stop")
        except Exception as exc:
            logger.warning(f"Could not stop the image server: {exc}")
            return False
        self.running = False
        logger.info("Image server stopped — its VRAM is free")
        return True

    async def generate(self, client: httpx.AsyncClient, body: dict) -> httpx.Response:
        for attempt in (1, 2):
            if not self.running:
                logger.info("Starting the image server")
                await self.start(client)
            try:
                resp = await client.post(
                    f"{self.url}/v1/images/generations", json=body, timeout=self.request_timeout
                )
            except httpx.ConnectError:
                # Nothing listening: sd-server is the container's only process,
                # so it has exited (a crash, an OOM) and its memory is free.
                # Without this `running` stays set, every request goes to a dead
                # server, and each failure resets the idle clock that would
                # otherwise stop it. A read timeout is different — the server
                # may still be generating — so it keeps `running` set.
                if attempt == 2:
                    raise
                logger.warning("Image server not answering — it has exited; starting it again")
                self.running = False
                continue
            self.last_used = time.monotonic()
            return resp
        raise AssertionError("unreachable")
