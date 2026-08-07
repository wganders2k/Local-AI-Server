
### Chores
* Consolidate all model config/names/etc to one source of truth. Currently all this info is split across a few too many files (models.ini, proxy/config.py, discord-bot/config.py, system_prompts.ini, docker-compose.yml, scripts/download_models.py). This causes adding new models or tweaking configs to be a headache.
  * These have now actually drifted, in both directions: `proxy/config.py` lists `mimic_user1..6`, `agent`, `chat`, and `image-caption`, none of which exist as presets; `models.ini` defines `gemma-brain-dense`, `chat-liberal`, `chat-chinese`, and `bomb-you`, none of which are in `SWAPPABLE_MODELS`. It degrades rather than breaks — the proxy forwards unknown names to the swappable slot anyway — but it means neither file can be trusted as the list.

### OpenWebUI Search
* Set up SearXNG + redis or similar search provider to improve chatbot utility

### Discord Bot
* Re-enable `/mimic`: the `mimic_user*` presets are commented out in `models.ini`, so every persona the autocomplete offers resolves to a model llama-server doesn't have.
* Persona autocomplete reads a hardcoded dict and can advertise models that don't exist. `/chat` already does this properly by reading the proxy's `/v1/models` — copy that.
* No tests. Only service in the repo without a `tests/` directory; the agent loop and the streaming split logic are the parts worth covering.
* `/admin-clear-history` stays disabled until there's a role check.

### Dead code / half-wired things
* `proxy/system_prompts.py` is never imported by `proxy/main.py`. `system_prompts.ini` is bind-mounted and edits to it do nothing. Either wire the loader up or delete both — a config file that silently has no effect is worse than no config file.
* `history-service/main.py` calls `_notify_lora_training()` after every merge, which POSTs to `http://lora-training:11438/notify` — a service that doesn't exist, on a port belonging to the arbiter. Fails harmlessly and logs a warning. Delete it or point it somewhere real.
* `EXCLUDED_CHANNELS` is read by `history-service/config.py` but never passed into the container by `docker-compose.yml`, so setting it has no effect. One line in the service definition fixes it.
* `LORE_TOP_K` defaults disagree: 10 in `discord-bot/config.py`, 5 in `docker-compose.yml`.

### Training
* Nothing consumes the JSONL archive for training — Tier 3 (the filtered training dataset) doesn't exist.
* `merge.py` doesn't exist: no path from a trained adapter to a GGUF that `models.ini` can point at.
* Whether to train on per-member message history at all hasn't actually been decided — it's inherited from the original design sketch, not chosen.

### Hardening
* `ghcr.io/ggml-org/llama.cpp:server-cuda` is unpinned. A bad upstream build lands on the next `make pull`.
* The arbiter is root-equivalent via `docker-socket-proxy` (`CONTAINERS`+`POST` admits `/containers/create`). Since `kind: container` currently has no user, dropping the Docker dependency entirely would remove the exposure outright.
* `docker-compose.yml` bind-mounts configs by absolute `/home/peacow/local-ai-server` paths, so the repo only runs from that location.

### Done
* ~~Break ground on the Discord bot~~ — `/mimic`, `/chat` (thread-based), and `/lore` (agentic RAG) all shipped.
* ~~Proxy + Homepage integration: job history records and an endpoint returning homepage-compatible JSON~~ — built, then superseded by Prometheus + Grafana. `/history` and `/history/summary` remain as `[]` stubs for compatibility; re-implement against the Prometheus API if the panels are ever wanted back.
