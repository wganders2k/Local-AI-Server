import os

# The llama-server router — injected by Docker Compose
LLAMA_SWAPPABLE = os.environ.get("LLAMA_SWAPPABLE", "http://localhost:11434")

# The GPU arbiter. The proxy asks it for the card and tells it when idle; it owns
# every other decision about who runs, so this is the proxy's whole interface to
# the GPU.
ARBITER_URL = os.environ.get("ARBITER_URL", "http://arbiter:11438")

# Which entry in the arbiter's jobs.yaml the proxy is. It sends this and nothing
# else — priority, VRAM, and what gets stopped are all decided there. Changing
# how the LLM ranks against the trainer is an edit to that file, not to this one.
ARBITER_JOB_NAME = os.environ.get("ARBITER_JOB_NAME", "llm")

# Batch traffic: models whose requests are bulk work — labelling a dataset,
# judging an eval — with nobody waiting on any one answer (dispatcher.py). While
# the proxy has only batch work it holds the arbiter lease under its own name,
# ranked below the LLM in jobs.yaml, and the dispatcher runs a batch request only
# once no interactive request has been queued or running for
# BATCH_YIELD_SECONDS. So a person on /mimic waits for at most the one batch call
# already running, and the slot is not swapped back and forth between the two
# models on every message of a busy chat.
# `caption` is history-service's archive backfill: ~43k images at a few seconds
# each, i.e. a day or more of GPU time that nobody is waiting on.
BATCH_MODELS: set[str] = {m.strip() for m in os.environ.get("BATCH_MODELS", "judge,caption").split(",") if m.strip()}
ARBITER_BATCH_JOB_NAME = os.environ.get("ARBITER_BATCH_JOB_NAME", "llm-batch")
BATCH_YIELD_SECONDS = float(os.environ.get("BATCH_YIELD_SECONDS", "30"))
# How long a batch request refused by the arbiter waits before asking again. It
# stays queued meanwhile: its client is a background job, not a person.
BATCH_RETRY_SECONDS = float(os.environ.get("BATCH_RETRY_SECONDS", "5"))

# Idle eviction — llama-server's router keeps a model resident once loaded, so
# without this an external job would never get a VRAM window on a busy day.
# After this many seconds with no LLM request, unload the resident model and let
# registered external jobs run.
IDLE_EVICT_ENABLED = os.environ.get("IDLE_EVICT_ENABLED", "true").lower() not in ("false", "0", "no")
IDLE_EVICT_SECONDS = float(os.environ.get("IDLE_EVICT_SECONDS", "600"))

# How often the idle evictor checks, and how long it waits for the unload to
# actually take effect.
IDLE_EVICT_POLL_SECONDS = float(os.environ.get("IDLE_EVICT_POLL_SECONDS", "30"))
IDLE_EVICT_UNLOAD_TIMEOUT = float(os.environ.get("IDLE_EVICT_UNLOAD_TIMEOUT", "60"))


# Image generation (imagegen.py) — off until the imagegen service is deployed.
IMAGEGEN_ENABLED = os.environ.get("IMAGEGEN_ENABLED", "false").lower() not in ("false", "0", "no")
IMAGEGEN_URL = os.environ.get("IMAGEGEN_URL", "http://imagegen:8080")
# The container the proxy starts and stops; must match container_name in
# docker-compose.yml. Reached through docker-socket-proxy, never the real socket.
IMAGEGEN_CONTAINER = os.environ.get("IMAGEGEN_CONTAINER", "imagegen")
DOCKER_URL = os.environ.get("DOCKER_URL", "http://docker-socket-proxy:2375")
# Start to listening measured at 8 s; the rest is headroom for a cold disk.
IMAGEGEN_START_TIMEOUT = float(os.environ.get("IMAGEGEN_START_TIMEOUT", "90"))
# A warm 1024² image took 10.9 s, the first one 15.5 s with its weight load.
IMAGEGEN_REQUEST_TIMEOUT = float(os.environ.get("IMAGEGEN_REQUEST_TIMEOUT", "300"))
# Z-Image-Turbo works in multiples of 64; these are ~1 megapixel each, so all
# cost about the same 11 s. Anything else is refused rather than guessed at.
IMAGEGEN_SIZES: set[str] = {"1024x1024", "832x1216", "1216x832", "768x768"}
IMAGEGEN_DEFAULT_SIZE = "1024x1024"
IMAGEGEN_MAX_PROMPT_CHARS = 2000
# Chat models small enough to stay loaded beside the image server: ~11.2 GB +
# the 12B mimic base's ~9.3 GB ≈ 20.5 GB of 24. Anything
# else (the 27B lore agent alone holds 21.25 GB) is unloaded first, and the
# image server is stopped before any other model loads.
IMAGEGEN_COEXIST_MODELS: set[str] = {"mimic-multi", "mimic-base", "mimic-candidate"}
