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

`clean_request` builds a fresh body from the prompt and allowlisted values;
nothing else the caller sends reaches the server. Text-to-image goes to
sd-server's OpenAI route. Image-to-image (an optional `image`, added
2026-11-15) goes to its native async API instead, because the OpenAI route
only takes a starting image inside the prompt tag that is refused below. The
input is decoded, cropped to the requested size and re-encoded as a fresh PNG
here, so sd-server only ever sees an image this module made. Reference-image
editing (the edit and variation endpoints) is still refused by main.py.

Measured 2026-11-13 on the sm_86 build (docs/plans/image-gen.md): 6–7 s per
warm 1024² image. Peak memory depends on the model .env selects: ~14 GB for
Z-Image-Turbo Q8, ~20 GB for a 12B int8 DiT.
"""

import asyncio
import base64
import binascii
import io
import json
import logging
import time

import httpx
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class RequestError(ValueError):
    """The caller's request cannot be served as asked; the message says why."""


# sd-server lifts `<sd_cpp_extra_args>{json}</sd_cpp_extra_args>` out of the
# prompt and applies it as generation parameters, reference images included.
# Allowing it would bypass every allowlist here, the size and steps included.
_EXTRA_ARGS_TAG = "<sd_cpp_extra_args"


def clean_request(body: object, sizes: set[str], max_prompt_chars: int, default_size: str,
                  max_images: int = 1, steps: tuple[int, ...] = (8,), max_input_bytes: int = 0,
                  max_input_pixels: int = 0, strength_range: tuple[float, float] = (0.1, 0.95),
                  cfg_range: tuple[float, float] = (1.0, 1.0)) -> dict:
    """
    The only body ever sent to the image server: prompt, size, how many images,
    the step count (the first of `steps` unless the caller picks another of
    them) and, when the caller sets one, the guidance (`cfg`). With an `image`, the native image-to-image body instead (see
    `init_image`), which `ImageServer.generate` sends as a job. Raises
    RequestError for anything the caller must fix.
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
    n_steps = body.get("steps", steps[0])
    if isinstance(n_steps, bool) or n_steps not in steps:
        raise RequestError(f"'steps' must be one of {list(steps)}")
    sample_params: dict = {"sample_steps": n_steps}
    if body.get("cfg") is not None:
        sample_params["guidance"] = {"txt_cfg": _number(body, "cfg", cfg_range)}
    if body.get("image") is not None:
        strength = _number(body, "strength", strength_range, default=0.65)
        width, height = (int(v) for v in size.split("x"))
        return {
            "prompt": prompt.strip(),
            "init_image": init_image(body["image"], width, height, max_input_bytes, max_input_pixels),
            "strength": strength,
            "width": width,
            "height": height,
            "batch_count": n,
            "sample_params": sample_params,
            "output_format": "png",
        }
    # The OpenAI route has no steps or guidance fields: sd-server reads them
    # from this tag, built here from checked numbers after the caller's own
    # tag was refused.
    args = json.dumps({"sample_params": sample_params}, separators=(",", ":"))
    tag = f"<sd_cpp_extra_args>{args}</sd_cpp_extra_args>"
    return {"prompt": prompt.strip() + tag, "size": size, "n": n, "response_format": "b64_json"}


def _number(body: dict, key: str, allowed: tuple[float, float], default: float | None = None) -> float:
    """body[key] as a float inside `allowed`, inclusive. Raises RequestError otherwise."""
    value = body.get(key, default)
    low, high = allowed
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise RequestError(f"'{key}' must be a number from {low} to {high}")
    return float(value)


def init_image(data: object, width: int, height: int, max_bytes: int, max_pixels: int) -> str:
    """
    The caller's image (base64, or a data: URL) as a base64 PNG of exactly
    width × height: scaled to cover that size, then centre-cropped, so the
    result keeps the input's framing without stretching it. A GIF gives its
    first frame; alpha and metadata are dropped with the re-encode.
    """
    if not isinstance(data, str) or not data:
        raise RequestError("'image' must be a base64 string or a data: URL")
    if data.startswith("data:"):
        data = data.partition(",")[2]
    # The base64 is a third larger than the file; refuse before decoding it.
    if len(data) > max_bytes * 4 // 3 + 4:
        raise RequestError(f"'image' is larger than {max_bytes // (1024 * 1024)} MB")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError("'image' is not valid base64") from None
    try:
        with Image.open(io.BytesIO(raw)) as im:
            # The header's size, read before any pixels are decoded.
            if im.width * im.height > max_pixels:
                raise RequestError(f"'image' has more than {max_pixels:,} pixels")
            im.seek(0)
            im = im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise RequestError("'image' is not a readable PNG, JPEG, WebP or GIF") from None
    scale = max(width / im.width, height / im.height)
    resized = im.resize((max(width, round(im.width * scale)), max(height, round(im.height * scale))),
                        Image.Resampling.LANCZOS)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    out = io.BytesIO()
    resized.crop((left, top, left + width, top + height)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


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
        # How often a native (image-to-image) job is polled; a warm image takes 3–12 s.
        self.poll_interval = 0.5

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
        """
        The images for a clean_request body, as an OpenAI-shaped response
        either way: text-to-image straight from the OpenAI route, image-to-
        image from a native job.
        """
        if "init_image" not in body:
            return await self._post(client, "/v1/images/generations", body)
        submitted = await self._post(client, "/sdcpp/v1/img_gen", body)
        if submitted.status_code != 202:
            return submitted
        resp = await self._wait_for_job(client, submitted.json()["id"])
        self.last_used = time.monotonic()
        return resp

    async def _post(self, client: httpx.AsyncClient, path: str, body: dict) -> httpx.Response:
        for attempt in (1, 2):
            if not self.running:
                logger.info("Starting the image server")
                await self.start(client)
            try:
                resp = await client.post(f"{self.url}{path}", json=body, timeout=self.request_timeout)
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

    async def _wait_for_job(self, client: httpx.AsyncClient, job_id: str) -> httpx.Response:
        """Poll a native job to its end. On a timeout it is cancelled, so it frees the card."""
        deadline = time.monotonic() + self.request_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            try:
                job = (await client.get(f"{self.url}/sdcpp/v1/jobs/{job_id}", timeout=10.0)).json()
            except httpx.ConnectError:
                self.running = False  # exited mid-job, as in _post
                raise
            if job["status"] == "completed":
                images = [{"b64_json": image["b64_json"]} for image in job["result"]["images"]]
                return httpx.Response(200, json={"data": images})
            if job["status"] in ("failed", "cancelled"):
                return httpx.Response(500, json={"error": job.get("error") or job["status"]})
        await client.post(f"{self.url}/sdcpp/v1/jobs/{job_id}/cancel", timeout=10.0)
        raise httpx.ReadTimeout(f"image job {job_id} still running after {self.request_timeout:.0f}s")
