# History Service

Background service that orchestrates Discord message history collection via DiscordChatExporter (DCE), merges raw exports into per-user JSONL archives, and triggers LoRA retraining when users accumulate enough new messages. Runs on a schedule — not in the inference hot path.

Runs on port `:11437` (internal only, not exposed externally).

## Responsibilities

1. **Channel evaluation** — Monthly cron pings the service to evaluate all channels in the guild. The service checks `channel_state.json` for last export timestamps and determines which channels need targeted exports.

2. **DCE orchestration** — Invokes `discord-chat-exporter` as a one-shot Docker container via `docker compose run`. Raw exports land on cold storage at `/mnt/storage_cold/array/DiscordArchive/raw/`.

3. **Archive merging** — Parses raw DCE output and merges into per-user JSONL archive files at `/mnt/storage_cold/array/DiscordArchive/archive/<user_id>.jsonl`. Messages are deduplicated by `message_id`. **No filtering is applied** — all messages are retained unfiltered.

4. **LoRA retraining trigger** — After each merge, check if any user has accumulated ≥ `RETRAIN_THRESHOLD` new messages since their last training run. If so, queue a training job.

5. **Training coordination** — Training uses the same RTX 3090 as inference. The service checks proxy idle status before starting and respects a configurable training window (default: 3–6 AM) to avoid contention. After training and GGUF merge complete, the service updates `models.ini` and restarts `llama-swappable` — zero bot or proxy changes required.

## Design Reference

See `Design.md` §9a (History & Training Pipeline) and §10 Phase 3 (LoRA Persona Refinement).

---

## Three-Tier Data Architecture

The history pipeline uses a **three-tier data architecture** that separates raw exports from processed archives and filtered training datasets. This design allows iterating on filtering methods without re-pulling data from Discord.

| Tier | Location | Format | Owner | Description |
|---|---|---|---|---|
| **Tier 1: Raw DCE Exports** | `/mnt/storage_cold/array/DiscordArchive/raw/` | DCE native JSON | `discord-chat-exporter` | Unmodified output from DiscordChatExporter. One directory per export run. |
| **Tier 2: Per-User JSONL Archive** | `/mnt/storage_cold/array/DiscordArchive/archive/` | JSONL (one line per message) | `history-service` | Normalized, per-user message archive. **Unfiltered** — all messages retained. |
| **Tier 3: Filtered Training Dataset** | Separate service (not `history-service`) | JSONL chat format | Training pipeline | Filtered subset of Tier 2 for LoRA training. Filtering is **NOT** a responsibility of history-service. |

**Key design principle:** Raw data is preserved in the archive. Filtering logic is applied at training time, not at ingest. This allows experimenting with different filtering methods without re-pulling data from Discord.

---

## File Structure

```
history-service/
├── Dockerfile
├── requirements.txt
├── main.py                  # Entry point: registers APScheduler jobs —
│                            #   (1) channel evaluation endpoint (triggered by host cron)
│                            #   (2) training window dispatch job (every 5 min, 3–6 AM only)
│                            # Also exposes internal HTTP endpoints on :11437.
├── dce_orchestrator.py      # Invokes DCE via `docker compose run` subprocess
│                            # Handles channel evaluation, date-range calculation,
│                            # and post-export merge into per-user JSONL archive
├── dce_parser.py            # Parses DCE JSON output, maps fields to internal archive schema
├── jsonl_store.py           # JSONL read/write, per-user file management
│                            # Deduplication by message_id, append-only writes
├── channel_state.py         # Tracks last_export_at per channel
│                            # State persisted to /archive/state/channel_state.json
├── training_state.py        # Tracks last_trained_at, message_count per user
├── training_trigger.py      # Three entry points:
│                            #   check_thresholds() — called after each merge; increments
│                            #     messages_since_last_train and sets status="queued" for
│                            #     users who hit RETRAIN_THRESHOLD. Does NOT dispatch.
│                            #   dispatch_queued() — called by training window scheduler;
│                            #     scans for queued users and dispatches training if proxy
│                            #     queue depth is zero. Retries at next tick if busy.
│                            #   CLI --force-all — invoked by `make mimic-source-refresh`;
│                            #     sets ALL users to queued regardless of threshold (used
│                            #     when the mimic base model changes and all adapters must
│                            #     be retrained from scratch).
├── llama_registrar.py       # Post-training: updates models.ini + restarts llama-swappable
├── image_captioner.py       # Batch processor: scans pending attachments, downloads images,
│                            # calls proxy with image-caption model, writes captions to JSONL
├── config.py                # Environment variable loading and defaults
└── data/
    └── (no longer used for history data; archive lives on cold storage)
```


> **Cold storage paths** (bind-mounted, survive `make nuke`):
> - `/archive/raw/` → `/mnt/storage_cold/array/DiscordArchive/raw/`
> - `/archive/archive/` → `/mnt/storage_cold/array/DiscordArchive/archive/`
> - `/archive/state/` → `/mnt/storage_cold/array/DiscordArchive/state/`

---

## Data Flow

```
Host cron (monthly, e.g. "0 3 1 * *")
    ↓
curl POST http://localhost:11437/evaluate
    ↓
history-service reads channel_state.json
    ↓
For each channel with activity since last export:
    history-service invokes: docker compose run --rm discord-chat-exporter \
        export --channel <id> --after <date> --before <date> --format Json
    ↓
DCE writes raw JSON → /mnt/storage_cold/array/DiscordArchive/raw/<timestamp>/
    ↓
history-service parses raw exports → merges into /archive/<user_id>.jsonl
    ↓
history-service updates channel_state.json with new last_export_at
    ↓
history-service runs training threshold check
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token (same token as discord-bot service) |
| `DISCORD_GUILD_ID` | ✅ | — | Target server (guild) ID |
| `PROXY_URL` | ✅ | — | Orchestration proxy URL (e.g. `http://proxy:11436`) |
| `LLAMA_SWAPPABLE` | ✅ | — | Swappable llama-server URL (e.g. `http://llama-swappable:8080`) |
| `RETRAIN_THRESHOLD` | ❌ | `200` | New messages per user before triggering retraining |
| `TRAINING_WINDOW_START` | ❌ | `3` | Hour (0–23) when training jobs may start |
| `TRAINING_WINDOW_END` | ❌ | `6` | Hour (0–23) when training window closes |
| `LORA_OUTPUTS_DIR` | ❌ | `/app/lora-outputs` | Directory for merged GGUF outputs |
| `MODELS_INI_PATH` | ❌ | `/models.ini` | Path to the llama-server preset config (updated after training) |

---

## Data Schemas

### JSONL Record (one line per message, per user file)

```json
{
  "message_id": "123456789012345678",
  "user_id": "987654321098765432",
  "username": "user3",
  "channel_id": "111222333444555666",
  "channel_name": "general",
  "timestamp": "2025-03-15T14:23:01Z",
  "content": "lmao those aren't strats, that's just dying slower",
  "attachments": []
}
```

**Important:** The archive is **unfiltered**. All messages are retained regardless of length, content type, or quality. Filtering into a training dataset is the responsibility of a separate service, not the history-service.

### Channel State (`/archive/state/channel_state.json`)

```json
{
  "111222333444555666": {
    "channel_id": "111222333444555666",
    "channel_name": "general",
    "last_export_at": "2026-03-01T03:15:00Z",
    "last_message_at": "2026-03-28T22:45:00Z",
    "total_messages_exported": 15234
  }
}
```

### Training State (`/archive/state/training_state.json`)

```json
{
  "user3": {
    "user_id": "987654321098765432",
    "username": "user3",
    "total_messages": 1247,
    "messages_since_last_train": 43,
    "last_trained_at": "2026-03-01T03:00:00Z",
    "last_pull_at": "2026-03-25T18:00:00Z",
    "model_version": 2,
    "training_status": "idle"
  }
}
```

**`training_status` values:**

| Status | Meaning |
|---|---|
| `idle` | No training in progress or queued |
| `queued` | Threshold met, waiting for training window / proxy idle |
| `training` | Unsloth QLoRA fine-tuning in progress |
| `merging` | Adapter merge + GGUF export in progress |
| `registering` | Updating `models.ini` + restarting `llama-swappable` |
| `done` | Training cycle complete — resets to `idle` |
| `error` | Training failed — check logs; manual intervention required |

---

## Pull Strategy

### Monthly Evaluation (host cron → history-service)

1. Host cron sends `POST /evaluate` to history-service on `:11437`
2. Service reads `channel_state.json` for last export timestamp per channel
3. For each channel:
   - If no record exists (new channel), schedule full channel export
   - If `last_message_at` > `last_export_at` (recent activity), schedule targeted date-range export
4. For each scheduled export, invoke DCE via `docker compose run`
5. After DCE completes, parse raw output and merge into per-user JSONL archive
6. Update `channel_state.json` with new `last_export_at`
7. Run training threshold check for all users with new messages

### Makefile Targets

| Target | Description |
|---|---|
| `make dce-export-full` | Full guild export via DCE. Useful for initial setup or disaster recovery. |
| `make dce-export-channel` | Export a specific channel. Set `CHANNEL_ID`, `DATE_FROM`, `DATE_TO` in environment. |
| `make dce-evaluate` | Trigger history-service to evaluate channels and run targeted exports (`POST /evaluate`). |
| `make history-refresh` | Alias for `dce-evaluate` — single-command pipeline trigger. |

### DiscordChatExporter CLI Reference

DCE is invoked via `docker compose run` as a one-shot CLI container. Key subcommands:

| Command | Description |
|---|---|
| `export` | Export one or more specific channels |
| `exportguild` | Export all channels within a guild |
| `exportdm` | Export all direct message channels |

Common options:

| Option | Description |
|---|---|
| `--channel <id>` | Channel ID(s) for `export` |
| `--guild <id>` | Guild ID for `exportguild` |
| `--output <path>` | Output directory (default: `/out/` → cold storage bind mount) |
| `--after <date>` | Only messages after this date (inclusive) |
| `--before <date>` | Only messages before this date (inclusive) |
| `--format Json` | JSON output format (used for all history-service exports) |
| `--include-threads` | Include threads: `None`, `Active`, `All` |
| `--parallel <n>` | Parallel channel exports (default: `1`) |

Example invocations:

```bash
# Single channel, date range (incremental pull)
docker compose run --rm discord-chat-exporter \
  export --channel 111222333444555666 \
  --format Json --output /out/ \
  --after 2026-03-29 --before 2026-04-29

# Full guild export
docker compose run --rm discord-chat-exporter \
  exportguild --guild 123456789012345678 \
  --format Json --output /out/
```

DCE JSON output contains rich message objects with `Id`, `Author`, `Content`, `Timestamp`, `Attachments`, `Embeds`, `Reactions`, `ReferencedMessage`, and `EditedTimestamp`. The `dce_parser.py` module maps these fields into the internal JSONL archive schema.

---

## Training Trigger Flow

Retraining is triggered via three distinct paths, all routed through `training_trigger.py`:

### Path 1 — Threshold check (after each DCE merge)

`main.py` calls `training_trigger.check_thresholds()` after every merge:

```
for each user with new messages:
  messages_since_last_train += new_count
  if messages_since_last_train >= RETRAIN_THRESHOLD:
    set status = "queued"   ← that's it; no dispatch here
```

This path only updates state. It never dispatches training directly.

### Path 2 — Training window scheduler (every 5 min, 3–6 AM only)

`main.py` registers an APScheduler job that runs every 5 minutes but only executes within the training window. It calls `training_trigger.dispatch_queued()`:

```
for each user with status == "queued":
  if proxy queue depth == 0:
    dispatch_training_job(user)
  # else: do nothing — scheduler retries at next tick
```

Retry logic is implicit: if the proxy is busy at 3:05 AM, the job runs again at 3:10 AM. No special retry code needed. If the service restarts while jobs are `queued`, the scheduler picks them up naturally at the next tick during the training window.

### Path 3 — Force-all (manual, via Makefile)

`make mimic-source-refresh` calls `python training_trigger.py --force-all` directly inside the container. This sets **all** users to `queued` regardless of their `messages_since_last_train` count:

```
for each user in training_state.json:
  set status = "queued"
```

Used when the mimic base model changes (new base or quant in `models.ini`) — all existing LoRA adapters are trained against the old base and must be retrained from scratch. The `--force-all` flag only updates state; actual training still dispatches via the training window scheduler (Path 2).

### Training job (dispatched by Path 2, one user at a time)

```
  1. Export user's archive JSONL → formatted training dataset (JSONL chat format)
     → Filtering applied here, NOT at ingest time
  2. Stop llama-swappable to free VRAM for training
  3. Run lora-training/train.py (Unsloth QLoRA, 1–2 epochs)
     → checkpoint saved to LORA_OUTPUTS_DIR/<user_id>/checkpoint/
  4. Run lora-training/merge.py → mimic_<username>_v{n+1}.gguf
     → saved to LORA_OUTPUTS_DIR/<user_id>/mimic_<username>_v{n+1}.gguf
  5. Update models.ini: set model path for [mimic_<username>] to new GGUF
  6. Restart llama-swappable (picks up new models.ini)
  7. Update training_state.json:
     - messages_since_last_train = 0
     - model_version += 1
     - last_trained_at = now()
     - training_status = "idle"
```

**GPU contention note:** Training is only dispatched when the proxy reports zero queued requests. The training window (3–6 AM by default) further reduces the chance of contention. If a Discord request arrives during training, it will queue at the proxy as normal — training does not hold the proxy lock, it uses the GPU directly via Unsloth. The only contention is raw VRAM: training the 35B-A3B model with QLoRA requires ~14–16 GB VRAM, which means the swappable llama-server slot must be stopped before training begins.

---

## HTTP Endpoints (internal only, not exposed externally)

| Method | Path | Description |
|---|---|---|
| `GET /health` | — | Health check |
| `GET /status` | — | Training state for all users (JSON) |
| `POST /evaluate` | — | Trigger channel evaluation + targeted DCE exports (called by host cron) |
| `POST /train/{username}` | — | Manually trigger training for a specific user (sets that user to `queued` and dispatches immediately if proxy is idle, regardless of training window) |
| `GET /archive/{user_id}/count` | — | Message count for a user in the archive |

> **Force-all vs per-user manual trigger:** `POST /train/{username}` is for ad-hoc single-user retraining (e.g. testing a new persona or recovering from an `error` state). `make mimic-source-refresh` → `training_trigger.py --force-all` is for bulk retraining after a base model change — it queues all users and lets the training window scheduler dispatch them in sequence overnight.

---

## Image Captioning Pipeline

The history service includes a background image captioning step that enriches JSONL records with natural-language descriptions of Discord image attachments. This runs as a separate scheduled job — it is **not** in the export/merge hot path.

### Purpose

Discord messages frequently contain images (memes, screenshots, reaction images). Without captioning, this content is invisible to:
- **Lore RAG** — the vector store can't retrieve image content it can't read
- **LoRA training context** — the model has no signal about what images a user shared

Captions make image content searchable and contextually meaningful, while the `caption_excluded_from_training` flag ensures synthetic descriptions never pollute the LoRA training corpus.

### Model: `image-caption`

The captioner uses the `image-caption` alias defined in `models.ini`. This alias points to the **same GGUF as the mimic personas** (`HauhauCS/Qwen3.5-35B-A3B-Uncensored` IQ4_XS) — an `image-text-to-text` capable model with zero refusals. This is critical: Discord content includes crude memes and adult humour that a standard censored vision model would refuse to describe.

Because `image-caption` and `mimic_*` share the same GGUF, llama-server's router loads the file once and serves all aliases from it. Swapping from a mimic persona to `image-caption` is a context switch, not a model reload.

**VRAM:** ~18 GB (same as mimic slot). The captioner only runs during the configured caption window when no inference is active.

### Scheduling

Image captioning runs as a dedicated APScheduler job in `main.py`. By default it shares the training window (3–6 AM) and runs **before** training dispatch to avoid VRAM contention:

```
Caption window (3–6 AM):
  1. image_captioner.process_pending_batch()   ← runs first
  2. training_trigger.dispatch_queued()        ← runs after captions complete
```

The captioner checks proxy queue depth before starting each batch. If the proxy is busy (e.g. a late-night Discord request), it defers to the next scheduler tick (every 5 minutes).

### Batch Processing

Images are processed in configurable batches (`IMAGE_CAPTION_BATCH_SIZE`, default 10) to avoid monopolising the swappable slot for extended periods. Uncaptioned images are tracked via `caption_status` in the JSONL record and are picked up on subsequent runs.

**Per-image flow:**
```
1. Scan JSONL records for attachments with caption_status == "pending"
2. Filter: skip if content_type is not image/* or file size > IMAGE_CAPTION_MAX_FILE_SIZE_MB
3. Download image to a temporary file
4. Check proxy queue depth — defer if busy
5. POST to proxy: model=image-caption, with image + caption prompt
6. Store caption text in the attachment record
7. Set caption_status = "done" (or "skipped" on error/unsupported format)
8. Set caption_excluded_from_training = true
9. Write updated record back to JSONL
10. Delete temporary image file
```

### JSONL Schema Changes

The existing JSONL record gains an optional `attachments` array. Messages without image attachments have no `attachments` field (fully backward compatible).

```json
{
  "message_id": "123456789012345678",
  "user_id": "987654321098765432",
  "username": "user3",
  "channel_id": "111222333444555666",
  "channel_name": "general",
  "timestamp": "2025-03-15T14:23:01Z",
  "content": "check this out",
  "attachments": [
    {
      "url": "https://cdn.discordapp.com/attachments/.../meme.png",
      "content_type": "image/png",
      "filename": "meme.png",
      "file_size_bytes": 204800,
      "caption": "A man in a suit pointing at a whiteboard that reads 'dying slower'.",
      "caption_status": "done",
      "caption_excluded_from_training": true
    }
  ]
}
```

**`caption_status` values:**

| Status | Meaning |
|---|---|
| `pending` | Attachment detected; caption not yet generated |
| `done` | Caption successfully generated and stored |
| `skipped` | Skipped — unsupported format, file too large, or download failed |
| `error` | Caption request failed (model error); will retry on next run |

### Training Exclusion

The `caption_excluded_from_training: true` flag on each attachment signals the LoRA dataset exporter in `training_trigger.py` to treat caption text as **read-only context**, not training data. Specifically:

- **RAG ingestion:** The caption is appended to the message's effective content string when chunking for ChromaDB: `"check this out [image: A man in a suit pointing at a whiteboard...]"`. This makes image content searchable via lore retrieval.
- **LoRA training dataset:** The caption text is **never** included in training samples. Only the original `content` field (the user's actual words) is used.

### New Environment Variables

| Variable | Default | Description |
|---|---|---|
| `IMAGE_CAPTION_ENABLED` | `false` | Enable/disable image captioning. Set `true` once GGUF is downloaded. |
| `IMAGE_CAPTION_MODEL` | `image-caption` | Model alias for captioning (must match alias in `models.ini`) |
| `IMAGE_CAPTION_BATCH_SIZE` | `10` | Images processed per captioning run |
| `IMAGE_CAPTION_WINDOW_START` | `3` | Hour (0–23) when captioning may run |
| `IMAGE_CAPTION_WINDOW_END` | `6` | Hour (0–23) when captioning window closes |
| `IMAGE_CAPTION_MAX_FILE_SIZE_MB` | `10` | Skip images larger than this (avoids downloading huge files) |

### Phase Timeline

Image captioning is introduced in **Phase 2** alongside history collection. It is disabled by default (`IMAGE_CAPTION_ENABLED=false`) until the GGUF is downloaded via `make models-download`.

| Phase | Image Captioning |
|---|---|
| Phase 1 | ⏳ Not started |
| Phase 2 | 🔨 Build + enable (set `IMAGE_CAPTION_ENABLED=true` after `make models-download`) |
| Phase 3 | ✅ Running |
| Phase 4 | ✅ Stable |

---

## Why Not Merge Into the RAG Service?

The RAG service is a **live inference dependency** — the Discord bot calls it synchronously during every lore request. It must be fast, lightweight, and always available.

The history service is a **background maintenance process** — it runs on a schedule, invokes external Docker containers (DCE), performs heavy I/O, and occasionally kicks off multi-hour training jobs. Mixing these responsibilities would make the RAG service heavier, harder to reason about, and potentially unstable during training runs.

Keeping them separate means:
- RAG service can be restarted independently without interrupting history collection
- Training failures don't affect lore retrieval
- Each service has a clear, single responsibility

---

## Relationship to `lora-training/`

The `lora-training/` directory contains the training scripts (`train.py`, `merge.py`) that are invoked by this service. Those scripts are designed to be run standalone (manually) or called as subprocesses by `training_trigger.py`.

```
history-service/training_trigger.py
    └── subprocess: lora-training/train.py --user <user_id> --data <jsonl_path>
    └── subprocess: lora-training/merge.py --user <user_id> --checkpoint <path>
```

The training scripts remain independently runnable for manual Phase 3 workflows. The history service adds the automated scheduling layer on top.

---

## Docker Compose Service

```yaml
history-service:
  build: ./history-service
  restart: unless-stopped
  ports:
    - "11437:11437"    # Internal HTTP API (not exposed externally)
  depends_on:
    proxy:
      condition: service_healthy
  environment:
    - DISCORD_TOKEN=${DISCORD_TOKEN}
    - DISCORD_GUILD_ID=${DISCORD_GUILD_ID}
    - PROXY_URL=http://proxy:11436
    - LLAMA_SWAPPABLE=http://llama-swappable:8080
    - RETRAIN_THRESHOLD=${RETRAIN_THRESHOLD:-200}
    - TRAINING_WINDOW_START=${TRAINING_WINDOW_START:-3}
    - TRAINING_WINDOW_END=${TRAINING_WINDOW_END:-6}
  volumes:
    - /mnt/storage_cold/array/DiscordArchive:/archive    # Cold storage bind mount
    - lora_outputs:/app/lora-outputs
    - /models.ini:/models.ini         # Preset config updated after training
    - ./lora-training:/lora-training:ro  # Training scripts (read-only mount)
    - /var/run/docker.sock:/var/run/docker.sock  # For invoking DCE via docker compose run
```

**Volume notes:**
- `/archive` is a **bind mount** (not a named volume) so data persists across `make nuke`
- `/var/run/docker.sock` is mounted so history-service can invoke `docker compose run` for DCE exports

---

## Phase Timeline

This service is introduced in **Phase 2** (alongside the RAG service) for history collection only. The training trigger is activated in **Phase 3** once the LoRA training scripts are ready.

| Phase | History Collection | Training Trigger |
|---|---|---|
| Phase 1 | ⏳ Not started | ⏳ Not started |
| Phase 2 | 🔨 Build | ⏳ Disabled (training scripts not ready) |
| Phase 3 | ✅ Running | 🔨 Enable + wire to lora-training scripts |
| Phase 4 | ✅ Stable | ✅ Stable |
