import os

# llama-server instance URLs — injected by Docker Compose
LLAMA_PERMANENT = os.environ.get("LLAMA_PERMANENT", "http://localhost:11435")
LLAMA_SWAPPABLE = os.environ.get("LLAMA_SWAPPABLE", "http://localhost:11434")

# System prompts config path
SYSTEM_PROMPTS_PATH = os.environ.get("SYSTEM_PROMPTS_PATH", "../system_prompts.ini")

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

