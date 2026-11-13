# History Service

Background service that orchestrates Discord message collection via DiscordChatExporter (DCE) and merges raw exports into per-user JSONL archives. Runs on `:11437` (bound to `127.0.0.1`), on a schedule — never in the inference hot path.

## What it does

1. **Channel evaluation** — on `POST /evaluate`, fetches the guild's channels from the Discord API, compares each channel's `last_message_at` against the `last_export_at` recorded in `channel_state.json`, and decides which need exporting.
2. **DCE orchestration** — invokes `discord-chat-exporter` as a one-shot container per channel. Raw exports land on cold storage.
3. **Archive merging** — parses raw DCE output and merges it into per-user JSONL files, deduplicated by `message_id`. **No filtering is applied** — every message is retained.
4. **Archive media** — an optional scheduled job that keeps a local copy of every image attachment under `/archive/media` (see Image Captioning).
5. **Image captioning** — an optional scheduled job that describes those images using the GPU, as proxy batch work (see below).

## What it does not do

**It does not trigger LoRA training.** Earlier revisions of this document described a three-path training trigger backed by `training_trigger.py`, `training_state.py`, and `llama_registrar.py`, a `training_state.json`, a 3–6 AM training window, and an automatic `models.ini` update after each merge. **None of that exists.** There is no training state, no window, no dispatch, and no registrar.

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

Not through the main `docker-compose.yml`. The service runs `docker compose -f /etc/dce-compose.yml run` against a dedicated one-service compose file, bind-mounted read-only. `DISCORD_TOKEN` isn't passed separately — the subprocess inherits it from history-service's own process environment, which Compose's variable substitution prefers over any `--env-file`. Docker access goes through `docker-socket-proxy` (`DOCKER_HOST=tcp://docker-socket-proxy:2375`), not the real socket.

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
| `POST` | `/reparse` | Rebuild Tier 2 from every existing Tier 1 export: append new messages, refresh parser-owned fields on archived ones. No Discord API calls, no DCE. Use it after any parser change |
| `POST` | `/clear` | ⚠️ Deletes all per-user archives, all raw exports, and resets channel state |
| `GET` | `/media/status` | Attachment counts by `media_status` and `caption_status` (backfill progress) |
| `GET` | `/archive/{user_id}/count` | Message count for one user |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Same token as the bot |
| `DISCORD_GUILD_ID` | ✅ | — | Target guild |
| `PROXY_URL` | ❌ | `http://proxy:11436` | Used by the captioner |
| `EXCLUDED_CHANNELS` | ❌ | *(empty)* | Comma-separated channel IDs or names never exported |
| `MEDIA_FETCH_ENABLED` | ❌ | `false` | Enables the media fetch job (network + disk, no GPU) |
| `MEDIA_FETCH_BATCH_SIZE` | ❌ | `200` | Images per run (every 120s) |
| `MEDIA_FETCH_WORKERS` | ❌ | `4` | Parallel downloads |
| `MEDIA_FETCH_MAX_FILE_SIZE_MB` | ❌ | `50` | Larger images are marked `too_large` |
| `MEDIA_FETCH_MAX_ATTEMPTS` | ❌ | `3` | Transient failures retried this many runs |
| `IMAGE_CAPTION_ENABLED` | ❌ | `false` | Enables the captioning job (every 60s) |
| `IMAGE_CAPTION_MODEL` | ❌ | `caption` | Must be an alias in `models.ini` **and** in the proxy's `BATCH_MODELS` |
| `IMAGE_CAPTION_BATCH_SIZE` | ❌ | `20` | Images per run; results written once per user file per run |
| `IMAGE_CAPTION_MAX_ATTEMPTS` | ❌ | `3` | Proxy failures retried this many runs, then `error` |
| `IMAGE_CAPTION_MAX_SIDE` | ❌ | `1536` | Longest side sent to the model |
| `HOST` / `PORT` | ❌ | `0.0.0.0` / `11437` | Bind address |

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
  "content": "@Sir Lags-a-Lot (AFK) lmao those aren't strats, that's just dying slower",
  "is_bot": false,
  "type": "Reply",
  "reference": {"message_id": "123456789012345600", "channel_id": "111222333444555666", "type": "Default"},
  "is_pinned": true,
  "mentions": [{"user_id": "222333444555666777", "username": "user1", "display": "Sir Lags-a-Lot (AFK)"}],
  "attachments": []
}
```

`content` is DCE's text as exported, unaltered — mentions included, rendered as
"@" + nickname. The optional fields appear only when they say something:

| field | meaning |
|---|---|
| `type` | DCE's message type: a name where DCE has one (`Default`, `Reply`, `ChannelPinnedMessage`, `ThreadCreated`), else Discord's number as a string (`"46"`). Always present |
| `reference` | the message this one points at: what a reply answers, a forward's source (`type: "Forward"`), what a pin event pinned |
| `is_pinned` | the message is pinned in its channel |
| `mentions` | who each "@Name" in `content` is. `display` is the exact text after the "@". Nicknames hold spaces and punctuation, so this is the only reliable way to resolve one |
| `attachments` | see Image Captioning |

The archive is **unfiltered**: every human message is retained regardless of
length, type or quality — system messages included (a pin event, "Pinned a
message.", is kept with its `type` and a `reference` to the pinned message).
Consumers filter by `type`. The one exception is bot-authored messages, which
the parser drops.

After a parser change, `POST /reparse` brings existing records up to date: it
appends what is new and refreshes `type`, `reference`, `is_pinned` and
`mentions` on what is already archived. Attachments get their parser-owned
fields (`attachment_id`, `url`, `content_type`, `filename`, `file_size_bytes`)
refreshed the same way, and an image the parser now recognises moves from
`skipped` to `pending`. Captions and media fields are never touched.

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

Two jobs, both off by default, both running inside this service:

**1. Media fetch (`media_store.py`, `MEDIA_FETCH_ENABLED`).** The archive holds signed Discord CDN links that expire about a day after export. This job keeps a copy of every image attachment at `/archive/media/<channel_id>/<attachment_id><ext>`. Before downloading, it re-signs links in batches of 50 through Discord's `POST /attachments/refresh-urls` using the bot token. An attachment that gets no fresh link was deleted on Discord and is marked `gone`; that image cannot be recovered. If the refresh service fails as a whole (bad token, Discord 5xx, rate limited out, network down), the run records nothing and later runs back off, 2 min doubling to an hour. If Discord rejects a call because of one link, the batch is halved until that link is alone, and only it counts toward `MEDIA_FETCH_MAX_ATTEMPTS`. Network and disk only. Measured 2026-11-12: 41.8k images, ~24 GiB, ~200 images per 30 s run.

**2. Captioning (`image_captioner.py`, `IMAGE_CAPTION_ENABLED`).** Sends each stored, uncaptioned image to the proxy's `caption` preset (models.ini: the Qwen3.6-27B heretic finetune plus its vision projector, thinking off). Every image is re-encoded as JPEG with a longest side of 1536, because llama.cpp's loader reads no WebP and a GIF is captioned by its first frame. The prompt is `prompts/caption.md`; the bot's live captions use an identical copy. Measured: 2–4 s per image warm. A request waits as long as the proxy holds it (`IMAGE_CAPTION_TIMEOUT_S`, default 6 h): a client timeout would only abandon a request the GPU still runs later. Only an image's own failures (an empty answer, a 4xx) count toward `IMAGE_CAPTION_MAX_ATTEMPTS`; when the proxy is unreachable, refusing or failing, the batch stops there with no attempt recorded and the next run carries on.

**GPU scheduling is the proxy's job, not this service's.** `caption` is in the proxy's `BATCH_MODELS`, so the proxy queues every request behind interactive traffic and runs it, one at a time, once people have been quiet for `BATCH_YIELD_SECONDS`, under arbiter job `llm-batch` (priority 25: below any person and the profiler, above the video job). A request the arbiter refuses stays queued in the proxy rather than failing. That replaced the old 3–6 AM window and queue-depth check. Note that 25 also **outranks the LoRA trainer** (-10), so a running backfill keeps training off the card. Don't enable captioning while a training queue needs to finish.

**Attachment fields:**

| field | written by | values |
|---|---|---|
| `attachment_id`, `url`, `content_type`, `filename`, `file_size_bytes` | parser (refreshed by `/reparse`) | `content_type` is inferred from the extension for png/jpg/jpeg/webp/gif, because DCE writes none |
| `media_status`, `media_path`, `media_attempts` | media fetch | `stored` (path relative to `/archive`), `gone`, `too_large`, `error` (retried) |
| `caption_status` | parser, then captioner | `skipped` (not an image), `pending`, `done`, `error` (gave up after max attempts), `unreadable` |
| `caption`, `caption_model`, `caption_prompt_version`, `caption_attempts` | captioner | the version is a hash of the prompt, so a re-caption can find stale entries |

`GET /media/status` counts attachments by `media_status` and `caption_status` across the archive, which shows backfill progress.

**Training exclusion.** Every caption carries `caption_excluded_from_training: true`. The distinction it encodes: a caption is *read-only context*. RAG ingestion appends it to the message's effective text so image content becomes searchable; a training exporter must never treat it as training data, because a synthetic description is not the user's voice.

```json
{
  "attachments": [
    {
      "attachment_id": "653318014426546186",
      "url": "https://cdn.discordapp.com/attachments/.../meme.png?ex=...",
      "content_type": "image/png",
      "filename": "meme.png",
      "file_size_bytes": 204800,
      "media_status": "stored",
      "media_path": "media/488918223203336204/653318014426546186.png",
      "caption": "A meme of Master Oogway in a forest. Top text: \"THIS WILL BE GRAPHICS\", bottom: \"IN 2013\".",
      "caption_status": "done",
      "caption_model": "caption",
      "caption_prompt_version": "3f1c9a0b2d4e",
      "caption_excluded_from_training": true
    }
  ]
}
```

History: until 2026-11-12 captioning had never worked. The parser keyed on a `content_type` DCE never writes, so every attachment was `skipped`; the links had expired; the preset was commented out; and the write-back clobbered concurrent writes.

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
├── media_store.py       # Local copies of image attachments (re-signs expired links)
├── image_captioner.py   # Batch captioner (proxy batch work)
├── prompts/caption.md   # Caption prompt (identical copy in discord-bot/prompts/)
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

The RAG service is a **live inference dependency** — the bot calls it synchronously during every `/lore` round, so it must stay fast and available. This service is a **background process** that shells out to Docker, performs heavy I/O, and takes the GPU for captioning as low-priority proxy batch work.

Keeping them separate means the RAG service can restart without interrupting collection, and an export failure cannot affect lore retrieval.

The coupling that does exist is one-directional and loose: the RAG service reads Tier 2 read-only and does not care whether this service is running.

---

## Design Reference

[`docs/design/system.md`](../docs/design/system.md) §9a (history pipeline), §9b (image captioning).
