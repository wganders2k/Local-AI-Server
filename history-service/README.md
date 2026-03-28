# History Service

Background service that manages per-user Discord message history (JSONL) and automatically triggers LoRA retraining when a user accumulates enough new clean messages. Runs on a schedule — not in the inference hot path.

## Responsibilities

1. **Message history collection** — two pull modes:
   - **Incremental pull** (every 15 min): Fetch only new messages from recently active channels since the last pull timestamp. Low API cost, keeps history current.
   - **Full rebuild** (monthly / manual): Fetch entire server history via Discord API, rebuild all JSONL files from scratch. Ensures no gaps from missed incremental pulls.

2. **JSONL management** — one file per user: `data/history/<user_id>.jsonl`. Each line is a cleaned, filtered message record. Messages are filtered on ingest (minimum word count, no bot commands, no pure emoji/URL-only messages).

3. **LoRA retraining trigger** — after each incremental pull, check if any user has accumulated ≥ `RETRAIN_THRESHOLD` new clean messages since their last training run. If so, queue a training job.

4. **Training coordination** — training uses the same RTX 3090 as inference. The service checks proxy idle status before starting and respects a configurable training window (default: 3–6 AM) to avoid contention. After training and GGUF merge complete, the service updates `models.ini` and restarts `llama-swappable` — zero bot or proxy changes required.

## Design Reference

See `Design.md` §9a (History & Training Pipeline) and §10 Phase 3 (LoRA Persona Refinement).

---

## File Structure

```
history-service/
├── Dockerfile
├── requirements.txt
├── main.py                  # Entry point: registers two APScheduler jobs —
│                            #   (1) incremental pull job (every 15 min)
│                            #   (2) training window dispatch job (every 5 min, 3–6 AM only)
│                            # Also exposes internal HTTP endpoints.
├── discord_fetcher.py       # Discord API client — full pull + incremental pull
├── message_filter.py        # Filtering logic (min words, command detection, etc.)
├── jsonl_store.py           # JSONL read/write, per-user file management
├── training_state.py        # Tracks last_trained_at, clean_msg_count per user
├── training_trigger.py      # Three entry points:
│                            #   check_thresholds() — called after each pull; increments
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
    ├── history/             # <user_id>.jsonl files (gitignored — can be large)
    └── training_state.json  # Per-user training metadata (persisted)
```

> `data/history/` and any LoRA outputs are excluded from git. Store on the server filesystem via the `history_data` Docker volume.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token (same token as discord-bot service) |
| `DISCORD_GUILD_ID` | ✅ | — | Target server (guild) ID |
| `PROXY_URL` | ✅ | — | Orchestration proxy URL (e.g. `http://proxy:11436`) |
| `LLAMA_SWAPPABLE` | ✅ | — | Swappable llama-server URL (e.g. `http://llama-swappable:8080`) |
| `RETRAIN_THRESHOLD` | ❌ | `200` | New clean messages per user before triggering retraining |
| `INCREMENTAL_PULL_INTERVAL` | ❌ | `15` | Minutes between incremental pulls |
| `FULL_PULL_CRON` | ❌ | `0 3 1 * *` | Cron expression for full rebuild (default: monthly at 3 AM) |
| `TRAINING_WINDOW_START` | ❌ | `3` | Hour (0–23) when training jobs may start |
| `TRAINING_WINDOW_END` | ❌ | `6` | Hour (0–23) when training window closes |
| `MIN_WORD_COUNT` | ❌ | `5` | Minimum words for a message to be considered clean |
| `ACTIVE_CHANNEL_LOOKBACK_HOURS` | ❌ | `24` | Hours to look back when identifying recently active channels |
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
  "word_count": 9,
  "clean": true
}
```

**Filtering rules (applied at ingest — `clean: false` messages are stored but excluded from training):**

| Rule | Condition |
|---|---|
| Minimum length | `word_count < MIN_WORD_COUNT` → not clean |
| Bot commands | Content starts with `/`, `!`, `.`, `?` → not clean |
| Pure emoji | Content contains only emoji characters → not clean |
| URL-only | Content is only a URL (no surrounding text) → not clean |
| Bot mentions | Content is only a `@mention` with no other text → not clean |
| Empty after strip | Content is empty after stripping whitespace → not clean |

Clean messages are the training corpus. Unclean messages are retained in JSONL for completeness (useful for lore RAG context) but excluded from LoRA training datasets.

### Training State (`data/training_state.json`)

```json
{
  "user3": {
    "user_id": "987654321098765432",
    "username": "user3",
    "total_clean_messages": 1247,
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

### Incremental Pull (every 15 minutes)

1. Query Discord API for channels in the guild
2. For each channel, check if any message was posted in the last `ACTIVE_CHANNEL_LOOKBACK_HOURS` hours
3. For recently active channels, fetch messages after the stored `last_message_id` for that channel
4. Filter and append new messages to the appropriate `<user_id>.jsonl` file
5. Update `last_message_id` per channel and `last_pull_at` in training state
6. Run threshold check for all users with new messages

**Discord API endpoint used:**
```
GET /channels/{channel_id}/messages?after={snowflake}&limit=100
```
Paginate with `after` until no more results. Respects Discord rate limits (50 req/s for bot tokens).

### Full Rebuild (monthly / manual trigger)

1. For each channel in the guild, fetch **all** message history (paginate from the beginning)
2. Rebuild all `<user_id>.jsonl` files from scratch (overwrite existing)
3. Recompute `total_clean_messages` and `messages_since_last_train` for all users
4. Does **not** reset `last_trained_at` or `model_version` — training state is preserved

**Manual trigger:** `POST /full-pull` on the service's internal HTTP endpoint (not exposed externally).

---

## Training Trigger Flow

Retraining is triggered via three distinct paths, all routed through `training_trigger.py`:

### Path 1 — Threshold check (after each incremental pull)

`main.py` calls `training_trigger.check_thresholds()` after every pull:

```
for each user with new clean messages:
  messages_since_last_train += new_clean_count
  if messages_since_last_train >= RETRAIN_THRESHOLD:
    set status = "queued"   ← that's it; no dispatch here
```

This path only updates state. It never dispatches training directly.

### Path 2 — Training window scheduler (every 5 min, 3–6 AM only)

`main.py` registers a second APScheduler job that runs every 5 minutes but only executes within the training window. It calls `training_trigger.dispatch_queued()`:

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
  1. Export user's clean JSONL messages → formatted training dataset (JSONL chat format)
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
| `POST /full-pull` | — | Trigger a full history rebuild immediately |
| `POST /train/{username}` | — | Manually trigger training for a specific user (sets that user to `queued` and dispatches immediately if proxy is idle, regardless of training window) |
| `GET /history/{user_id}/count` | — | Clean message count for a user |

> **Force-all vs per-user manual trigger:** `POST /train/{username}` is for ad-hoc single-user retraining (e.g. testing a new persona or recovering from an `error` state). `make mimic-source-refresh` → `training_trigger.py --force-all` is for bulk retraining after a base model change — it queues all users and lets the training window scheduler dispatch them in sequence overnight.

---

## Image Captioning Pipeline

The history service includes a background image captioning step that enriches JSONL records with natural-language descriptions of Discord image attachments. This runs as a separate scheduled job — it is **not** in the incremental pull hot path.

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
  "word_count": 3,
  "clean": true,
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

The history service is a **background maintenance process** — it runs on a schedule, performs heavy I/O (Discord API pagination), and occasionally kicks off multi-hour training jobs. Mixing these responsibilities would make the RAG service heavier, harder to reason about, and potentially unstable during training runs.

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
  depends_on:
    proxy:
      condition: service_healthy
  environment:
    - DISCORD_TOKEN=${DISCORD_TOKEN}
    - DISCORD_GUILD_ID=${DISCORD_GUILD_ID}
    - PROXY_URL=http://proxy:11436
    - LLAMA_SWAPPABLE=http://llama-swappable:8080
    - RETRAIN_THRESHOLD=${RETRAIN_THRESHOLD:-200}
    - INCREMENTAL_PULL_INTERVAL=${INCREMENTAL_PULL_INTERVAL:-15}
    - FULL_PULL_CRON=${FULL_PULL_CRON:-"0 3 1 * *"}
    - TRAINING_WINDOW_START=${TRAINING_WINDOW_START:-3}
    - TRAINING_WINDOW_END=${TRAINING_WINDOW_END:-6}
  volumes:
    - history_data:/app/data
    - lora_outputs:/app/lora-outputs
    - /models.ini:/models.ini         # Preset config updated after training
    - ./lora-training:/lora-training:ro  # Training scripts (read-only mount)
```

---

## Phase Timeline

This service is introduced in **Phase 2** (alongside the RAG service) for history collection only. The training trigger is activated in **Phase 3** once the LoRA training scripts are ready.

| Phase | History Collection | Training Trigger |
|---|---|---|
| Phase 1 | ⏳ Not started | ⏳ Not started |
| Phase 2 | 🔨 Build | ⏳ Disabled (training scripts not ready) |
| Phase 3 | ✅ Running | 🔨 Enable + wire to lora-training scripts |
| Phase 4 | ✅ Stable | ✅ Stable |
