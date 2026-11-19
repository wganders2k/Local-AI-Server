# TODO

### Open WebUI search
* Set up SearXNG (plus Redis) or a similar search provider so the chat models can search the web.

### Discord bot
* `/admin-clear-history` was removed because it had no authorization check. Re-add it only behind a role check.

### Discord archive
* Clarify how the Discord archive is backed up, and write it down. Today:
  * Nothing under `/mnt/storage_cold/array/DiscordArchive` is in the restic backup, which covers `/opt` only. The archive's only protection is SnapRAID parity, and parity is not a backup.
  * `raw/` (~650 MB of DCE exports) is the one tier that can't be rebuilt: a message deleted on Discord since its export exists only here. `archive/`, `training/` and the ChromaDB index are all derived from it.
  * `media/` (~25 GB) holds local copies of attachments, which Discord's CDN links don't keep reachable forever.
  * `bot_context/` (the hand-written lore context, gitignored for its real names) sits on the hot pool, so it can stay on the SSD cache without parity until the mover runs.
  * Before a risky reparse, the archive gets copied by hand into a dated sibling directory (`archive-backup-20261104-pre-reparse`, `archive-backup-20261112-pre-attachments`, `archive.pre-mention-handles-20261118`). Those copies sit on the same disks, follow no naming scheme, and are never pruned.
  * To decide: which tiers go into restic (or another copy), what happens to the pre-change copies, and where the policy is written down (`history-service/README.md` and `server-ops`).
* Decide whether to store and caption video attachments. Today:
  * The archive has 2,750 videos (~15 GiB: 2,186 `.mp4`, 388 `.mov`, 175 `.webm`, 1 `.mkv`). All are `caption_status: skipped`, and none is downloaded, because the media store only fetches images. Discord's CDN links don't stay valid forever.
  * The live bot ignores them too (`discord-bot/vision.py` accepts `image/*` only), so in `/lore` and RAG a video counts only through its message text and filename.
  * Options: a single keyframe through the existing JPEG captioner (cheap, GIF-quality), or 4–6 frames as one grid with a sequence-aware caption prompt (about the same GPU time, captures what happens). Either needs ffmpeg in the history-service image. Neither covers audio, which would need a separate speech-to-text model.
  * Storing them adds ~15 GiB to `media/`, which ties into the backup decision above.
* Decide whether to caption GIFs sent as Tenor/Giphy links. Today:
  * The archive's message text holds ~2,000 of them (1,989 from `tenor.com`), roughly seven times the 280 uploaded GIFs, which are all captioned (first frame).
  * Discord renders these as embeds, not attachments, and `history-service/dce_parser.py` drops embeds, so the media store and captioner never see them. Only the URL survives in `content`; a Tenor slug (`tenor.com/view/cat-dancing-gif-123`) is the only hint `/lore` and RAG get.
  * The live bot misses them too: `discord-bot/vision.py` looks at attachments only.
  * The change: keep GIF embeds in the parser with their preview/thumbnail URL, let the media store fetch them like attachments, and caption the stored copy. They'd be captioned from the first frame like uploaded GIFs; the frame-grid option for video would cover motion here as well.

### Training
* Tuning experiments against the live adapters (lora-training/README.md §Judging a run; the cut pipeline and topic eval are the current round).

### Hardening
* `docker-socket-proxy` admits `/containers/create` (`CONTAINERS` + `POST`), so anything that can reach it is root-equivalent. The arbiter, the proxy (to start and stop the image servers) and history-service (to run DCE) all reach it. The arbiter could drop Docker entirely, since no job in `jobs.yaml` uses `kind: container`, but the exposure stays as long as the other two need to create and start containers. The image is also unpinned (`:latest`).
* `docker-compose.yml` bind-mounts host paths absolutely (configs under `/home/peacow/local-ai-server`, plus `/home/peacow/lora-outputs` and `/home/peacow/.cache/huggingface`), so the repo only runs from that location.

### Done
* ~~`LORE_TOP_K` defaults disagree between `discord-bot/config.py` and `docker-compose.yml`~~ — the setting no longer exists in either.
* ~~Replace the estimated `required_mb` for `lora-trainer` with the measured peak~~ — measured at 14303 MiB; `jobs.yaml` sets 16000 to leave headroom for the longest batches and eval passes.
* ~~Re-enable `/mimic`~~ — all seven personas are trained and served from `[gemma4-12b-mimic-u]`, one base with every adapter applied per request.
* ~~Persona autocomplete reads a hardcoded dict~~ — `personas.PersonaCatalog` offers only personas on the roster that also have an adapter id from the proxy's `/v1/models`.
* ~~First real persona run~~ — done for every persona; `make train-datasets` / `train-queue` / `train-export-latest` retrain the roster.
* ~~Confirm llama-server loads a Gemma 4 GGUF LoRA via `lora =`~~ — it does; the adapters are live.
* ~~`ghcr.io/ggml-org/llama.cpp:server-cuda` is unpinned~~ — `llama-swappable` is pinned by digest to `server-cuda12-b11977`.
* ~~Nothing consumes the JSONL archive for training~~ — `lora-training/build_dataset.py` builds Tier 3 per persona.
* ~~`merge.py` doesn't exist~~ — replaced by `export_adapter.py`, which writes a GGUF LoRA over a shared base rather than a merged model.
* ~~Whether to train on per-member history at all~~ — decided 2026-11-04, with the members' agreement.
* ~~Consolidate all model config to one source of truth~~ — chat models live in `models.ini`, and the image models in `.env` (`IMAGEGEN_*`). `proxy/config.py`'s advisory `SWAPPABLE_MODELS` set is gone, and `scripts/download_models.py` derives its download list from `models.ini`. The permanent autocomplete slot was removed 2026-11-12.
* ~~`proxy/system_prompts.py` / `system_prompts.ini` are dead code~~ — both deleted; no loader, no bind mount.
* ~~`history-service` notifies a service that doesn't exist~~ — `_notify_lora_training()` deleted.
* ~~`EXCLUDED_CHANNELS` never reaches its container~~ — passed through in `docker-compose.yml`, and the filter now matches by channel name or id.
* ~~Discord bot has no tests~~ — `discord-bot/tests/` has 18 test files (294 tests), covering the lore agent, tools, citations, mimic chat, personas, `/imagine`, vision and threads.
* ~~Break ground on the Discord bot~~ — `/mimic`, `/mimic-chat`, `/chat`, `/lore` and `/imagine` all shipped.
* ~~Proxy + Homepage integration: job history records and an endpoint returning homepage-compatible JSON~~ — built, then superseded by Prometheus + Grafana. `/history` and `/history/summary` remain as `[]` stubs for compatibility; re-implement against the Prometheus API if the panels are ever wanted back.
