# Image generation — spike results and design

Status: **live since 2026-11-12; runtime rebuilt and model made configurable
2026-11-13** (see [Runtime and model update](#runtime-and-model-update-2026-11-13)).
It launched text-to-image only. Image-to-image (a starting picture plus
`strength`) was added 2026-11-15; reference-image editing is not offered.
Implementation:
`proxy/imagegen.py`, `discord-bot/cogs/imagine.py`, the `imagegen` service in
docker-compose.yml.

## Runtime and model update (2026-11-13)

The service is used by a small private group of friends, so the model is
picked on image quality and prompt adherence for that group rather than kept
fixed. Two changes:

1. **sd.cpp is built from source for sm_86** (pinned commit, compose `build:`
   from the upstream repo with `CUDA_ARCHITECTURES=86`). The prebuilt image
   ships no sm_86 kernels, which is what made flash attention segfault, and it
   also crashes on the first step of any int8 DiT checkpoint. Built, flash
   attention works and every model got faster.
2. **The model is chosen in `.env`** (`IMAGEGEN_DIFFUSION_MODEL`,
   `IMAGEGEN_TEXT_ENCODER`, `IMAGEGEN_VAE`; the defaults are Z-Image-Turbo).
   The live deployment runs a 12B int8 DiT checkpoint that won a side-by-side
   on 14 server prompts (sharper detail, better text rendering, closer prompt
   adherence).

Measured on the 3090, 1024², 8 steps, seed 42, median of 10–28 images:

| setup | per image | first image | peak VRAM |
|---|---|---|---|
| Z-Image-Turbo Q8, prebuilt image (before) | 11.3 s | 16.7 s | 17.8 GB |
| Z-Image-Turbo Q8, sm_86 build | 6.2 s | 10.7 s | 14.0 GB |
| 12B int8 DiT, sm_86 build | 6.9 s | 13.8 s | 20.2 GB |
| 12B int8 DiT, sm_86 build, `--offload-to-cpu` | 9.5 s | 22.3 s | 14.4 GB |
| 12B int8 DiT, ComfyUI 0.45, torch cu128 | 21.4 s | 26.4 s | 19.5 GB |
| 12B int8 DiT, ComfyUI 0.45, torch cu130 | 7.3 s | 16.0 s | 19.0 GB |

**ComfyUI was evaluated as a replacement backend and not adopted.** On torch
cu128 its comfy_kitchen int8 kernels are disabled and it falls back to eager
dequantisation (3× slower); on cu130 (which also needs a C compiler for
triton) it matches sd.cpp. With equal speed and output, sd.cpp keeps the
existing proxy integration, starts in 1 s instead of 9, and is a 2.7 GB image
rather than ~8–10 GB.

**Consequence for chat:** at ~20 GB the 12B model leaves no room for the mimic
base (~9.3 GB), so `.env` sets `IMAGEGEN_COEXIST_MODELS` empty and the proxy
unloads any chat model before starting the image server. /imagine's prompt
rewriter (which runs on the mimic base) therefore costs a model swap, ~10 s,
and is off by default. Both low-memory modes peak around 14.4 GB, which with
the mimic base would total ~23.7 GB of 24.1 GB: too tight to rely on.

The sections below are the original 2026-11-12 spike and design, kept as
written.

## What was measured

Hardware: RTX 3090 24 GB. The spike ran beside the 2.2 GB always-resident
autocomplete server (`llama-permanent`), which was removed later the same day;
totals below are without it.

Stack: `ghcr.io/leejet/stable-diffusion.cpp:master-cuda`
(`sha256:07285a1c…`, built 2026-11-10) running `sd-server`, which serves an
OpenAI-style `POST /v1/images/generations`.

Model: Z-Image-Turbo (6B DiT, Apache-2.0, 8 steps, cfg 1.0), three files under
`/srv/models`:

| file | size |
|---|---|
| `leejet/Z-Image-Turbo-GGUF/z_image_turbo-Q8_0.gguf` | 6.6 GB |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf` (text encoder) | 4.3 GB |
| `Comfy-Org/z_image_turbo/ae.safetensors` (FLUX VAE) | 0.3 GB |

| | result |
|---|---|
| server start to listening | 8 s (weights load lazily) |
| first 1024² image (includes weight load) | 15.5 s |
| warm 1024² image | 10.9 s, 10.8 s |
| warm 768² image | 5.0 s |
| VRAM while loaded | ~11.2 GB |
| `[mimic-multi]` cold reload after the image server stops | 5.3 s to first answer |

Quality: good. Prompt adherence is strong and it renders legible text: a
"LORE NIGHT" sign came out exactly right. Samples from the run were not kept.

### Two problems found

1. **The prebuilt image crashes with flash attention on.** sd.cpp's auto
   setting picks the CUDA MMA flash-attention kernel. That kernel was not built
   for sm_86 ("cp.async instructions not enabled"), so the image segfaults on
   the first sample. It works with `--flash-attn off --conv-direct off`, which
   is how every number above was measured. Building the image ourselves from
   `docker/Dockerfile.cuda` with `CUDA_ARCHITECTURES=86` should bring flash
   attention back and make generation faster; not measured.
2. **sd-server ignores SIGTERM.** `docker stop` takes the full 10 s grace
   period every time. Use `stop_signal: SIGKILL` or `docker kill`, which is
   safe because it holds no state worth flushing.

## How it fits

**sd-server is a second backend behind the proxy, under the proxy's existing
`llm` lease. It is not a new arbiter tenant.**

- The proxy already serialises the GPU between models and owns unloading
  llama-server's models. Image generation is one more thing that holds the
  card while a person waits, so it belongs under the same lock and the same
  priority.
- A separate arbiter job would need its own reclaim protocol. It would also
  compete with the proxy for a card the proxy already holds whenever anyone is
  chatting: the proxy would release, the image job would acquire, a chat
  message would arrive, and the card would change hands again.

Request flow for `POST /v1/images/generations` at the proxy:

1. Take the swappable lock and acquire the `llm` lease, the same as chat.
2. **Fit check:** the image server needs ~11.2 GB.
   - `[mimic-multi]` (~9.3 GB) can stay resident beside it: 11.2 + 9.3 ≈
     20.5 GB of 24.
   - `brain-dense-heretic` (21.25 GB) cannot. Unload it first with
     `_unload_resident_models`.
3. Forward the request to `imagegen:8080`.
4. Afterwards, idle-evict the image server like any model: kill its weights
   after `IDLE_EVICT_SECONDS`, or recreate the container. sd-server has no
   unload endpoint, so the plan is to stop the container and let the next
   request start it (8 s plus a 5 s load).

**Who stops the container (decided).** The proxy starts and stops it
through `docker-socket-proxy`, the same restricted Docker API history-service
uses. The proxy still never touches the real socket or the card itself.

What chat pays:
- **From `[mimic-multi]`:** nothing, because the two models coexist.
- **From the lore agent:** one unload, plus a ~6–11 s reload on the next lore
  turn, which is the same as any preset change today.

## `/imagine` design (if go)

- **Command:** `/imagine prompt:<text> [size: square|portrait|landscape]`.
  Answered with a "generating…" message that is edited when the image is done.
  The image is attached as a PNG, with the prompt in the embed.
- **Queue:** depth 1 per user plus the bot's existing `MAX_QUEUE_DEPTH`. One
  image at a time on the card.
- **Rate limit:** the shared `RateLimiter`, plus a per-user image cap
  (e.g. 10/hour), because images are the costliest request the bot serves.
- **Sizes:** 1024² by default (~11 s warm). 768² is ~5 s if latency matters
  more than detail.
- **Access:** the bot serves one private server; `/imagine` is gated by
  `IMAGINE_ENABLED` and the bot's shared per-user rate limit. (A separate
  hourly image cap was removed on 2026-11-13.)
- **Later:** persona-driven images (a mimic persona "posts a meme") reuse the
  same proxy route. Wait until the plain command has been used for a while.

## Files a build would touch

- `docker-compose.yml`: an `imagegen` service (`profiles: ["managed"]` style).
  Started and stopped by the proxy, never by `compose up`, with
  `stop_signal: SIGKILL`.
- `proxy/main.py`, `proxy/config.py`: the `/v1/images/*` route and the fit
  check.
- `discord-bot/cogs/imagine.py`, `proxy_client.py` (an `images()` call), and
  `config.py`.
- `scripts/download_models.py`: list the three files. They are not in
  `models.ini`, so add an explicit list (`IMAGEGEN_MODELS`).
