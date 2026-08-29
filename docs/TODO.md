
### OpenWebUI Search
* Set up SearXNG + redis or similar search provider to improve chatbot utility

### Discord Bot
* Re-enable `/mimic`: the `mimic_user*` presets are commented out in `models.ini`, so every persona the autocomplete offers resolves to a model llama-server doesn't have.
* Persona autocomplete reads a hardcoded dict and can advertise models that don't exist. `/chat` already does this properly by reading the proxy's `/v1/models` — copy that.
* `/admin-clear-history` stays disabled until there's a role check.

### Dead code / half-wired things
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
* ~~Consolidate all model config to one source of truth~~ — model identity now lives in
  exactly two places: `models.ini` (swappable slot) and the `llama-permanent` command:
  in `docker-compose.yml` (permanent slot). `proxy/config.py`'s `SWAPPABLE_MODELS` set is
  gone (it was advisory only); `scripts/download_models.py` derives its swappable download
  list from `models.ini` directly.
* ~~`proxy/system_prompts.py` / `system_prompts.ini` are dead code~~ — both deleted; no
  loader, no bind mount.
* ~~`history-service` notifies a service that doesn't exist~~ — `_notify_lora_training()`
  deleted.
* ~~`EXCLUDED_CHANNELS` never reaches its container~~ — passed through in
  `docker-compose.yml`, and the filter now matches by channel name or id.
* ~~Discord bot has no tests~~ — `discord-bot/tests/` now has 6 test files covering the agent loop, formatters, prompts, research, session, and tools.
* ~~Break ground on the Discord bot~~ — `/mimic`, `/chat` (thread-based), and `/lore` (agentic RAG) all shipped.
* ~~Proxy + Homepage integration: job history records and an endpoint returning homepage-compatible JSON~~ — built, then superseded by Prometheus + Grafana. `/history` and `/history/summary` remain as `[]` stubs for compatibility; re-implement against the Prometheus API if the panels are ever wanted back.
