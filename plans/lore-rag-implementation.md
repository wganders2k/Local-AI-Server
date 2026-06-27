# Lore RAG Service — Implementation Plan

**Phase:** 2 (RAG + Lore)
**Goal:** Wire the `/lore` slash command to an actual retrieval-augmented generation pipeline so Gemma3-12B can answer questions grounded in real server history.

---

## 1. Current State

The `/lore` command in [`discord-bot/bot.py`](discord-bot/bot.py:471) currently sends a placeholder message to the model:

```python
"No retrieved context available (RAG service not yet configured).\n\nQuestion: {question}"
```

The [`rag/`](rag/) directory exists with a design README but no code. ChromaDB and the `rag-service` are commented out in [`docker-compose.yml`](docker-compose.yml:157).

The JSONL archive at `/mnt/storage_cold/array/DiscordArchive/archive/<user_id>.jsonl` is being populated by the history-service. This is the data source for lore ingestion.

---

## 2. Data Transformation Strategy

### 2.1 Source Data Shape

The history-service writes per-user JSONL files. Each line is one message record:

```json
{
  "message_id": "123456789",
  "user_id": "987654321",
  "username": "user3",
  "channel_id": "111222333",
  "channel_name": "general",
  "timestamp": "2024-03-15T21:34:02+00:00",
  "content": "check this out",
  "attachments": [
    {
      "url": "...",
      "content_type": "image/png",
      "filename": "meme.png",
      "caption": "A man pointing at a whiteboard that reads dying slower.",
      "caption_status": "done",
      "caption_excluded_from_training": true
    }
  ]
}
```

**Problem:** Files are split per-user. Conversations involve multiple users in the same channel. The ingest step must reconstruct the full conversational thread across all users.

### 2.2 Ingest Pipeline: Cross-User Flattening

```
Step 1: Load all <user_id>.jsonl files from archive dir
Step 2: Flatten into a single list of message dicts
Step 3: Sort by (channel_id, timestamp) ascending
Step 4: Group into conversation chunks using 1-hour temporal gaps
Step 5: Format each chunk as a mini-transcript
Step 6: Embed with ibm-granite/granite-embedding-311m-multilingual-r2
Step 7: Upsert into ChromaDB collection "lore" (dedup by chunk_id)
```

### 2.3 Conversation Chunking Rules

- **Gap threshold:** 1 hour — if the next message in the same channel arrives more than 60 minutes after the previous, start a new chunk
- **Channel boundary:** Always start a new chunk when the channel changes (even within 1 hour)
- **No minimum size:** Keep all messages including short reactions ("lol", "💀") — they add tonal/contextual signal
- **Token cap:** Soft limit at 512 tokens per chunk. If a chunk exceeds this, split at the earliest paragraph/sentence boundary after the 400-token mark (prevents embedding truncation mid-sentence)

### 2.4 Chunk Text Format

Each chunk is serialised as a mini-transcript before embedding:

```
[#general | 2024-03-15 21:34]
user3: check this out [image: A man pointing at a whiteboard that reads dying slower.]
user1: lmao
user2: 💀
user3: i had one job. ONE JOB.
user4: classic user3 speedrun strats
```

Format rules:
- Header line: `[#channel_name | YYYY-MM-DD HH:MM]` (timestamp of first message in chunk)
- Each message line: `username: content`
- Image captions appended inline when `caption_status == "done"`: `[image: <caption>]`
- Messages with no content and no caption are omitted from the transcript (embed-only with no text or caption)

### 2.5 ChromaDB Chunk Metadata Schema

```python
{
    "chunk_id": str,          # sha256 of (channel_id + first_message_id)
    "channel_id": str,
    "channel_name": str,
    "timestamp_start": str,   # ISO 8601 of first message
    "timestamp_end": str,     # ISO 8601 of last message
    "authors": str,           # comma-separated list of unique usernames in chunk
    "message_count": int,
    "message_ids": str,       # comma-separated list of message_ids
}
```

The `chunk_id` is used for idempotent upserts — re-running ingest does not duplicate chunks.

### 2.6 Re-ingestion Strategy

The ingest script supports incremental re-runs:
- Pass `--since <ISO-date>` to only process messages newer than a given date
- Default (no flag): full ingest (safe due to upsert dedup by `chunk_id`)
- Manual trigger via `POST /ingest` on the RAG service HTTP API

---

## 3. Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│  RAG Service (rag-service container, CPU-only)           │
│                                                          │
│  rag/ingest.py     — JSONL loader, chunker, embedder,   │
│                       ChromaDB upserter                  │
│  rag/retrieve.py   — query ChromaDB, format context     │
│  rag/main.py       — FastAPI: /retrieve, /ingest,       │
│                       /health                            │
└─────────────────────────────────────────────────────────┘
         │ reads                      │ queries
         ▼                            ▼
  /archive/*.jsonl             chromadb (chroma_data vol)
  (history-service output)

discord-bot/rag_client.py  ←→  rag-service /retrieve
discord-bot/bot.py lore_command:
  1. POST /retrieve {query, top_k}
  2. Build lore context string from results
  3. Inject into user message before proxy call
```

---

## 4. File-by-File Implementation Plan

### 4.1 `rag/ingest.py`

Responsibilities:
- `load_archive(archive_dir) -> list[dict]` — read all JSONL files, flatten to sorted message list
- `chunk_by_conversation(messages, gap_minutes=60, max_tokens=512) -> list[Chunk]` — temporal chunking
- `format_chunk_text(chunk) -> str` — render mini-transcript
- `embed_and_upsert(chunks, chroma_client, collection_name="lore")` — batch embed + upsert
- `run_ingest(archive_dir, chroma_host, chroma_port, since=None)` — orchestrates full pipeline

### 4.2 `rag/retrieve.py`

Responsibilities:
- `build_lore_context(query, chroma_client, top_k=5) -> tuple[str, int]` — returns (context_string, chunk_count)
- `format_context_block(results) -> str` — renders retrieved chunks into a block for the lore prompt

The context block injected into the lore prompt:

```
Retrieved context (from server history):

--- [#general | 2024-03-15 21:34] ---
user3: check this out [image: A man pointing...]
user1: lmao
...

--- [#gaming | 2024-09-02 18:10] ---
user4: the tournament incident last year was legendary
...

Question: {user_question}
```

### 4.3 `rag/main.py`

FastAPI app with three endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}` |
| `/retrieve` | POST | `{query: str, top_k: int = 5}` → `{context: str, chunk_count: int}` |
| `/ingest` | POST | `{since: str = null}` → triggers background ingest task |

### 4.4 `rag/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 4.5 `rag/requirements.txt`

```
chromadb>=0.5.0
sentence-transformers>=3.0.0
fastapi>=0.110.0
uvicorn>=0.29.0
```

### 4.6 `discord-bot/rag_client.py`

Async httpx client wrapping the RAG service `/retrieve` endpoint:

```python
class RAGClient:
    async def retrieve(self, query: str, top_k: int = 5) -> tuple[str, int]:
        # POST /retrieve → returns (context_str, chunk_count)
        # Returns ("", 0) if RAG service unreachable (graceful fallback)
```

Graceful degradation: if the RAG service is unreachable, the lore command falls back to the current placeholder message rather than failing hard.

### 4.7 `discord-bot/config.py` additions

```python
RAG_SERVICE_URL: str = os.environ.get("RAG_SERVICE_URL", "http://rag-service:8001")
LORE_TOP_K: int = int(os.environ.get("LORE_TOP_K", "5"))
RAG_ENABLED: bool = os.environ.get("RAG_ENABLED", "true").lower() == "true"
```

### 4.8 `discord-bot/bot.py` — `lore_command` update

Replace the placeholder message block:

```python
# Before (Phase 1 placeholder):
messages = [
    {"role": "system", "content": LORE_SYSTEM_PROMPT},
    {"role": "user", "content": "No retrieved context available...\n\nQuestion: {question}"},
]

# After (Phase 2 RAG):
context, chunk_count = await rag_client.retrieve(question, top_k=LORE_TOP_K)
context_block = (
    f"Retrieved context:\n\n{context}\n\nQuestion: {question}"
    if context else
    f"No retrieved context available.\n\nQuestion: {question}"
)
messages = [
    {"role": "system", "content": LORE_SYSTEM_PROMPT},
    {"role": "user", "content": context_block},
]
```

The `chunk_count` flows through to `build_lore_embed_discord(lore_text, chunk_count=chunk_count)` so the embed footer shows the correct source count.

### 4.9 `docker-compose.yml` additions

Uncomment and configure:

```yaml
rag-service:
  build: ./rag
  restart: unless-stopped
  environment:
    - CHROMA_HOST=chromadb
    - CHROMA_PORT=8000
    - ARCHIVE_DIR=/archive
  depends_on:
    - chromadb
  volumes:
    - /mnt/storage_cold/array/DiscordArchive/archive:/archive:ro
    - chroma_data:/chroma/chroma  # shared with chromadb

chromadb:
  image: chromadb/chroma:latest
  restart: unless-stopped
  volumes:
    - chroma_data:/chroma/chroma
```

Update `discord-bot` service to add:
- `depends_on: rag-service` (condition: service_healthy)
- `CHROMA_HOST`, `CHROMA_PORT`, `LORE_TOP_K`, `RAG_SERVICE_URL`, `RAG_ENABLED` env vars

---

## 5. Data Flow Diagram

```
User: /lore question:"what happened at the tournament"
         │
         ▼
discord-bot/bot.py  lore_command
         │
         ├─ POST rag-service:8001/retrieve
         │         {query: "what happened at the tournament", top_k: 5}
         │         │
         │         ▼
         │    retrieve.py: chromadb.query(query_texts=[query], n_results=5)
         │         │
         │         ▼
         │    format_context_block(results) → context_str, chunk_count=3
         │
         ├─ Build messages:
         │    system: LORE_SYSTEM_PROMPT
         │    user:   "Retrieved context:\n\n[#general | 2024-09-02]...\n\nQuestion: ..."
         │
         ├─ POST proxy:11436/v1/chat/completions {model: "lore", messages: [...]}
         │         │
         │         ▼
         │    gemma3-12b answers grounded in retrieved chunks
         │
         └─ build_lore_embed_discord(lore_text, chunk_count=3)
              → embed footer: "Sources: 3 lore entries retrieved"
```

---

## 6. Ingest Trigger Strategy

| Trigger | Method | Frequency |
|---|---|---|
| Initial load | `docker compose exec rag-service python -c "from ingest import run_ingest; run_ingest(...)"` or `POST /ingest` | Once on first deploy |
| After new DCE export | `POST rag-service:8001/ingest` with `since=<last_export_date>` from history-service | After each monthly DCE pull |
| Manual re-embed | `POST /ingest` (no since param = full re-ingest) | When chunking strategy changes |

The history-service can optionally POST to the RAG service after each successful DCE merge — this is a future enhancement, not required for Phase 2.

---

## 7. Environment Variable Summary

### New variables (add to `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `RAG_SERVICE_URL` | `http://rag-service:8001` | RAG service base URL |
| `LORE_TOP_K` | `5` | Number of chunks retrieved per lore query |
| `RAG_ENABLED` | `true` | Toggle RAG retrieval without changing bot code |
| `CHROMA_HOST` | `chromadb` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |

---

## 8. Graceful Degradation

If the RAG service is unreachable (container down, ChromaDB unresponsive), [`rag_client.py`](discord-bot/rag_client.py) catches the exception and returns `("", 0)`. The lore command falls back to the Phase 1 placeholder: `"No retrieved context available."` — the user still gets a response from Gemma3-12B, just without grounding. This prevents the `/lore` command from hard-failing if the RAG stack is down for maintenance.

---

## 9. Implementation Order

The tasks should be implemented in this sequence to keep things testable at each step:

1. **`rag/ingest.py`** — can be tested standalone against the JSONL archive
2. **`rag/retrieve.py`** — depends on ChromaDB having data from step 1
3. **`rag/main.py`** + **`rag/Dockerfile`** + **`rag/requirements.txt`** — wraps steps 1-2 in HTTP API
4. **`docker-compose.yml`** — uncomment chromadb + rag-service, run initial ingest
5. **`discord-bot/rag_client.py`** — client for the HTTP API from step 3
6. **`discord-bot/config.py`** — add new env vars
7. **`discord-bot/bot.py`** — wire RAG context into lore_command
8. **`discord-bot/formatters.py`** — chunk_count already flows through; verify footer text
9. **`.env.example`** + **`DiscordBot-Design.md`** — documentation updates
