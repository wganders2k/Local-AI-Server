import os

# llama-server instance URLs — injected by Docker Compose
LLAMA_PERMANENT = os.environ.get("LLAMA_PERMANENT", "http://localhost:11435")
LLAMA_SWAPPABLE = os.environ.get("LLAMA_SWAPPABLE", "http://localhost:11434")

# System prompts config path
SYSTEM_PROMPTS_PATH = os.environ.get("SYSTEM_PROMPTS_PATH", "../system_prompts.ini")

# The GPU arbiter. The proxy asks it for the card and tells it when idle; it owns
# every other decision about who runs, so this is the proxy's whole interface to
# the GPU.
ARBITER_URL = os.environ.get("ARBITER_URL", "http://arbiter:11438")

# Which entry in the arbiter's jobs.yaml the proxy is. It sends this and nothing
# else — priority, VRAM, and what gets stopped are all decided there. Changing
# how the LLM ranks against the trainer is an edit to that file, not to this one.
ARBITER_JOB_NAME = os.environ.get("ARBITER_JOB_NAME", "llm")

# Autocomplete model toggle — set to "false" to disable the permanent autocomplete
# model and free ~1.2 GB VRAM. When disabled, requests for the "autocomplete"
# model will receive a 503 error. Controlled by AUTOCOMPLETE_ENABLED env var.
AUTOCOMPLETE_ENABLED = os.environ.get("AUTOCOMPLETE_ENABLED", "true").lower() not in ("false", "0", "no")

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


# Models that always route to the permanent slot — never swapped, never locked.
# The autocomplete model is loaded at llama-server startup and stays resident.
# Using a stable alias ("autocomplete") decouples the proxy from the underlying
# model — future model upgrades only require updating docker-compose.yml and
# re-downloading the GGUF, not the proxy.
AUTOCOMPLETE_MODELS: set[str] = {
    "autocomplete",
}

# Models that compete for the swappable slot.
# These are the aliases defined in models.ini.
# The router loads a model on first request and evicts it when a different
# model is requested. The proxy serialises access via asyncio.Lock.
SWAPPABLE_MODELS: set[str] = {
    "brain",
    "brain-dense",  # Qwen3.6-27B — agentic RAG tool-calling model
    "brain-dense-heretic",
    "agent",
    "mimic_user1",
    "mimic_user2",
    "mimic_user3",
    "mimic_user4",
    "mimic_user5",
    "mimic_user6",
    "lore",
    "chat",
    "image-caption",
}

