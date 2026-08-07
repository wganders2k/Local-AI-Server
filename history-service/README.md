# History Service

Background service that orchestrates Discord message collection via DiscordChatExporter (DCE) and merges raw exports into per-user JSONL archives. Runs on `:11437` (bound to `127.0.0.1`), on a schedule — never in the inference hot path.

## What it does

1. **Channel evaluation** — on `POST /evaluate`, fetches the guild's channels from the Discord API, compares each channel's `last_message_at` against the `last_export_at` recorded in `channel_state.json`, and decides which need exporting.
2. **DCE orchestration** — invokes `discord-chat-exporter` as a one-shot container per channel. Raw exports land on cold storage.
3. **Archive merging** — parses raw DCE output and merges it into per-user JSONL files, deduplicated by `message_id`. **No filtering is applied** — every message is retained.
4. **Image captioning** — an optional scheduled job that describes image attachments using the GPU (see below).

## What it does not do

**It does not trigger LoRA training.** Earlier revisions of this document described a three-path training trigger backed by `training_trigger.py`, `training_state.py`, and `llama_registrar.py`, a `training_state.json`, a 3–6 AM training window, and an automatic `models.ini` update after each merge. **None of that exists.** There is no training state, no window, no dispatch, and no registrar.

What remains is a vestige: after each merge, `main.py` calls `_notify_lora_training()`, which POSTs to `http://lora-training:11438/notify` — a service that does not exist, on a port that belongs to the arbiter. It fails, logs a warning, and is otherwise inert. Either delete it or point it somewhere real.

A training run is started by a human running `make train-submit` and by nothing else. See [`lora-training/README.md`](../lora-training/README.md).

**It does not filter.** Turning the archive into a training dataset is Tier 3's job, and Tier 3 does not exist yet.

---

## Three-Tier Data Architecture

| Tier | Location | Format | Owner | Description |
|---|---|---|---|---|
| **1: Raw DCE exports** | `/mnt/storage_cold/array/DiscordArchive/raw/` | DCE native JSON | `discord-chat-exporter` | Unmodified output, one timestamped directory per export run |
| **2: Per-user JSONL archive** | `…/DiscordArchive/archive/` | JSONL, one line per message | `history-service` | Normalised, **unfiltered** |
| **3: Filtered training dataset** | not built | JSONL chat format | training pipeline | Filtered subset of Tier 2 |

**Key design principle:** raw data is preserved and filtering is applied at training time, not at ingest. This means filtering rules can be revised without re-pulling from Discord — which is also what makes `POST /reparse` useful: it rebuilds Tier 2 from existing Tier 1 without touching the Discord API.

Cold storage paths are bind mounts, so they survive `make nuke`:

```
/archive/raw/     → /mnt/storage_cold/array/DiscordArchive/raw/
/archive/archive/ → /mnt/storage_cold/array/DiscordArchive/archive/
/archive/state/   → /mnt/storage_cold/array/DiscordArchive/state/
```

---

## Data Flow

```
make dce-evaluate  →  POST /evaluate
    ↓
fetch guild channels from the Discord API
    ↓
load channel_state.json; compare last_message_at vs last_export_at
    ↓
drop anything in EXCLUDED_CHANNELS
    ↓
for each channel needing export:
    first time  → full channel export (no date range)
    seen before → targeted export --after <last_export_at> --before <today>
    ↓
DCE writes raw JSON → /archive/raw/<timestamp>/
    ↓
parse and merge into /archive/archive/<user_id>.jsonl, deduplicated by message_id
    ↓
update channel_state.json (only if the export actually succeeded)
```

A failed export leaves the channel's state untouched, so the next run retries the same range rather than silently skipping a gap.

### How DCE is invoked

Not through the main `docker-compose.yml`. The service runs `docker compose -f /etc/dce-compose.yml run` against a dedicated one-service compose file, with the token supplied from a separate `/etc/dce.env` — both bind-mounted read-only. Docker access goes through `docker-socket-proxy` (`DOCKER_HOST=tcp://docker-socket-proxy:2375`), not the real socket.

`--user` is passed through to `docker compose run` so DCE writes files owned by the invoking user rather than root; without it, `POST /clear` hits permission errors trying to remove them.

Subprocess output is streamed to the log line by line as it happens, rather than being captured and dumped at the end — a full-guild export takes long enough that a silent subprocess is indistinguishable from a hung one.

---

## HTTP Endpoints

Bound to `127.0.0.1:11437`. No authentication.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/status` | Users in the archive and per-user message counts |
| `POST` | `/evaluate` | Channel evaluation + targeted DCE exports (the main entry point) |
| `POST` | `/reparse` | Rebuild Tier 2 from every existing Tier 1 export. No Discord API calls, no DCE. Use it to iterate on parser changes |
| `POST` | `/clear` | ⚠️ Deletes all per-user archives, all raw exports, and resets channel state |
| `GET` | `/archive/{user_id}/count` | Message count for one user |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Same token as the bot |
| `DISCORD_GUILD_ID` | ✅ | — | Target guild |
| `PROXY_URL` | ❌ | `http://proxy:11436` | Used by the captioner |
| `EXCLUDED_CHANNELS` | ❌ | *(empty)* | Comma-separated channel IDs never exported |
| `IMAGE_CAPTION_ENABLED` | ❌ | `false` | Enables the captioning job |
| `IMAGE_CAPTION_MODEL` | ❌ | `image-caption` | Must match an alias in `models.ini` |
| `IMAGE_CAPTION_BATCH_SIZE` | ❌ | `10` | Images per run |
| `IMAGE_CAPTION_WINDOW_START` | ❌ | `3` | Hour (UTC) the window opens |
| `IMAGE_CAPTION_WINDOW_END` | ❌ | `6` | Hour (UTC) it closes |
| `IMAGE_CAPTION_MAX_FILE_SIZE_MB` | ❌ | `10` | Skip larger images |
| `HOST` / `PORT` | ❌ | `0.0.0.0` / `11437` | Bind address |

Window hours are **UTC**, not local time.

If you are working from an older `.env`, note that `RETRAIN_THRESHOLD`, `TRAINING_WINDOW_START`, `TRAINING_WINDOW_END`, `TRAINING_TRIGGER_ENABLED`, `INCREMENTAL_PULL_INTERVAL`, `FULL_PULL_CRON`, `MIN_WORD_COUNT`, and `ACTIVE_CHANNEL_LOOKBACK_HOURS` are **not read by this service**. They belonged to the training trigger described above and have been removed from `.env.example`.

---

## Data Schemas

### JSONL record (one line per message, per user file)

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

The archive is **unfiltered**: every message is retained regardless of length, content type, or quality.

### Channel state (`/archive/state/channel_state.json`)

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

`total_messages_exported` is cumulative across runs.

---

## Image Captioning

Discord messages are full of memes and screenshots. Without captions that content is invisible to lore retrieval, which can only index text.

**Model.** Uses the `image-caption` alias — the same GGUF as the mimic personas (`Qwen3.5-35B-A3B-Uncensored` IQ4_XS), chosen because Discord content includes crude memes that a standard vision model would refuse to describe. Sharing the GGUF means switching between a mimic persona and the captioner is a context switch, not a reload.

**This is a GPU consumer**, ~18 GB in the swappable slot, competing under the same lock and the same arbiter lease as live inference. That is why it is windowed.

**Scheduling.** An APScheduler job every 300s that returns immediately unless `IMAGE_CAPTION_ENABLED` is true *and* the current UTC hour is inside the window.

**Per-image flow:**

```
1. Scan archives for attachments with caption_status == "pending"
2. Skip non-images and anything over IMAGE_CAPTION_MAX_FILE_SIZE_MB
3. Download to a temp file
4. Check proxy queue depth — defer to the next tick if busy
5. POST to the proxy: model=image-caption, image base64-inlined as a data: URI
6. Write the caption back into the record; set caption_status
7. Delete the temp file
```

**`caption_status`:** `pending` → `done`, or `skipped` (unsupported format, oversized, download failed) / `error` (model failure — retried next run).

**Training exclusion.** Every caption carries `caption_excluded_from_training: true`. The distinction it encodes: a caption is *read-only context*. RAG ingestion appends it to the message's effective text so image content becomes searchable; a training exporter must never treat it as training data, because a synthetic description is not the user's voice.

```json
{
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

**Status:** disabled by default, and the `image-caption` preset is currently commented out in [`models.ini`](../models.ini) — so setting the flag alone is not enough. Uncomment the preset first.

---

## Makefile Targets

| Target | Description |
|---|---|
| `make dce-evaluate` | `POST /evaluate` — the normal pipeline trigger |
| `make history-refresh` | Alias for `dce-evaluate` |
| `make dce-export-full` | Full guild export, for initial setup or disaster recovery |
| `make dce-export-guild` | Guild export with `AFTER=` / `BEFORE=` |
| `make dce-export-channel` | Single channel; `CHANNEL_ID=`, optional `AFTER=` / `BEFORE=` |
| `make dce-help` | DCE CLI help |
| `make logs-history` | Tail the logs |

---

## File Structure

```
history-service/
├── Dockerfile
├── requirements.txt
├── main.py              # FastAPI app, APScheduler registration, endpoints
├── dce_orchestrator.py  # Channel evaluation, date ranges, DCE invocation, merge
├── dce_parser.py        # DCE JSON → internal archive schema
├── jsonl_store.py       # Per-user JSONL read/write, message_id dedup, counts
├── channel_state.py     # last_export_at / last_message_at tracking
├── image_captioner.py   # Batch captioner
├── config.py            # Environment loading and defaults
└── tests/
    ├── test_channel_state.py
    ├── test_dce_parser.py
    └── test_jsonl_store.py
```

```bash
python -m pytest tests -q
```

---

## Why Not Merge Into the RAG Service?

The RAG service is a **live inference dependency** — the bot calls it synchronously during every `/lore` round, so it must stay fast and available. This service is a **background process** that shells out to Docker, performs heavy I/O, and occasionally takes the GPU for captioning.

Keeping them separate means the RAG service can restart without interrupting collection, and an export failure cannot affect lore retrieval.

The coupling that does exist is one-directional and loose: the RAG service reads Tier 2 read-only and does not care whether this service is running.

---

## Design Reference

[`Design.md`](../Design.md) §9a (history pipeline), §9b (image captioning).
