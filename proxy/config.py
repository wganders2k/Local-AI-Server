import os

# Ollama instance URLs — injected by Docker Compose
OLLAMA_PERMANENT = os.environ.get("OLLAMA_PERMANENT", "http://localhost:11435")
OLLAMA_SWAPPABLE = os.environ.get("OLLAMA_SWAPPABLE", "http://localhost:11434")

# Models that always route to the permanent slot — never swapped, never locked
# Registered via: ollama create autocomplete -f modelfiles/autocomplete.Modelfile
# Using a stable name ("autocomplete") decouples the proxy from the underlying model —
# future model upgrades only require updating the Modelfile, not the proxy.
AUTOCOMPLETE_MODELS: set[str] = {
    "autocomplete",
}

# Models that compete for the swappable slot
SWAPPABLE_MODELS: set[str] = {
    "brain",
    "mimic_user1",
    "mimic_user2",
    "mimic_user3",
    "mimic_user4",
    "mimic_user5",
    "mimic_user6",
    "lore",
    "librechat_chat",
    "image-caption",
}
