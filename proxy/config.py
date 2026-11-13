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
# judging an eval — with nobody waiting on any one answer. They go to the arbiter
# under their own name, ranked below the LLM in jobs.yaml, and the proxy holds
# each one back until no interactive request has been queued or in flight for
# BATCH_YIELD_SECONDS. So a person on /mimic waits for at most the one batch call
# already running, and the slot is not swapped back and forth between the two
# models on every message of a busy chat.
BATCH_MODELS: set[str] = {m.strip() for m in os.environ.get("BATCH_MODELS", "judge").split(",") if m.strip()}
ARBITER_BATCH_JOB_NAME = os.environ.get("ARBITER_BATCH_JOB_NAME", "llm-batch")
BATCH_YIELD_SECONDS = float(os.environ.get("BATCH_YIELD_SECONDS", "30"))

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
