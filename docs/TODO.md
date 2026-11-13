
### OpenWebUI Search
* Set up SearXNG + redis or similar search provider to improve chatbot utility

### Discord Bot
* `/admin-clear-history` stays disabled until there's a role check.

### Dead code / half-wired things
* `LORE_TOP_K` defaults disagree: 10 in `discord-bot/config.py`, 5 in `docker-compose.yml`.

### Training
* Replace the estimated `required_mb: 14000` in `arbiter/jobs.yaml` with the measured peak.
* Tuning experiments against the live adapters (lora-training/README.md §Judging a run; the cut pipeline and topic eval are the current round).

### Hardening
* The arbiter is root-equivalent via `docker-socket-proxy` (`CONTAINERS`+`POST` admits `/containers/create`). Since `kind: container` currently has no user, dropping the Docker dependency entirely would remove the exposure outright.
* `docker-compose.yml` bind-mounts configs by absolute `/home/peacow/local-ai-server` paths, so the repo only runs from that location.

### Done
* ~~Re-enable `/mimic`~~ — all seven personas are trained and served from `[mimic-multi]`, one base with every adapter applied per request.
* ~~Persona autocomplete reads a hardcoded dict~~ — `personas.PersonaCatalog` offers only personas on the roster that also have an adapter id from the proxy's `/v1/models`.
* ~~First real persona run~~ — done for every persona; `make train-datasets` / `train-queue` / `train-export-latest` retrain the roster.
* ~~Confirm llama-server loads a Gemma 4 GGUF LoRA via `lora =`~~ — it does; the adapters are live.
* ~~`ghcr.io/ggml-org/llama.cpp:server-cuda` is unpinned~~ — `llama-swappable` is pinned by digest to `server-cuda12-b11977`.
* ~~Nothing consumes the JSONL archive for training~~: `lora-training/build_dataset.py` builds Tier 3 per persona.
* ~~`merge.py` doesn't exist~~: replaced by `export_adapter.py`, which writes a GGUF LoRA over a shared base rather than a merged model.
* ~~Whether to train on per-member history at all~~: decided 2026-11-04, with the members' agreement.
* ~~Consolidate all model config to one source of truth~~ — model identity now lives in
  `models.ini` (and the `imagegen` command: for the image model; the permanent
  autocomplete slot was removed 2026-11-12). `proxy/config.py`'s `SWAPPABLE_MODELS` set is
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
