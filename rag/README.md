# RAG Service

CPU-only FastAPI service on `:8001` (internal). Ingests the per-user Discord archive into ChromaDB and answers retrieval queries for the bot's `/lore` agent. Zero VRAM impact — which is why lore retrieval can run while the GPU is busy with something else.

Embeddings: `ibm-granite/granite-embedding-311m-multilingual-r2`, loaded lazily on first use and cached for the process lifetime.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/retrieve` | Query the `lore` collection. Supports channel and date-range filtering |
| `POST` | `/ingest` | Chunk, embed, and upsert the archive. Idempotent |

`make rag-ingest` triggers ingestion; `make rag-status` shows collection counts.

## Chunking is temporal, not per-message

A single Discord message is rarely a meaningful unit of retrieval — a joke and its punchline are usually two messages thirty seconds apart, and embedding them separately loses both. So [`ingest.py`](ingest.py) groups messages into conversations:

- A new chunk starts when the **channel changes**, regardless of timing.
- A new chunk starts after a **60-minute gap** between consecutive messages (`DEFAULT_GAP_MINUTES`).
- Chunks above a **512-token soft cap** (`DEFAULT_MAX_TOKENS`) are split at a message boundary rather than mid-conversation.

The archive is flattened across users first, so a chunk is a slice of the channel as it actually happened — several people talking — not one person's messages in isolation.

Image captions produced by the history service are folded into a message's effective text at this point, which is what makes image content searchable at all.

## Chunk metadata

```python
{
    "chunk_id": str,
    "channel_id": str,
    "channel_name": str,
    "timestamp_start": int,   # Unix epoch
    "timestamp_end": int,     # Unix epoch
    "authors": str,           # comma-joined
    "message_count": str,
    "message_ids": str,       # comma-joined
}
```

**Timestamps are epoch integers, and this is load-bearing.** ChromaDB's `$gte`/`$lte` operators only work on numeric fields, and ChromaDB type-infers a field from its first insert. A collection that has ever seen a string timestamp will raise `ValueError` on every numeric comparison from then on — so the agent's `start_date` / `end_date` filters would fail permanently. Fixing that means rebuilding the collection, not editing a query.

Lists are comma-joined strings because ChromaDB metadata values must be scalars.

## Retrieval

`build_lore_context()` in [`retrieve.py`](retrieve.py) embeds the query, applies an optional `where` clause built from channel name and date range, and returns formatted context blocks with chunk separators.

Callers are the `/lore` agent's three tools — whole-history search, per-channel search, and channel summarisation — which differ in the filters they pass rather than in the code path they take. See [`DiscordBot-Design.md`](../DiscordBot-Design.md) §7.3.

## Data source

Reads Tier 2 of the archive, bind-mounted read-only at `/archive` (`/mnt/storage_cold/array/DiscordArchive/archive` on the host). The RAG service does not produce that data and does not depend on the history service running — see [`history-service/README.md`](../history-service/README.md).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CHROMA_HOST` | `chromadb` | |
| `CHROMA_PORT` | `8000` | |
| `COLLECTION_NAME` | `lore` | |
| `ARCHIVE_DIR` | `/archive` | Tier 2 JSONL, read-only |

## File structure

```
rag/
├── Dockerfile
├── requirements.txt
├── main.py        # FastAPI app: /health, /retrieve, /ingest
├── ingest.py      # Archive loader, conversation chunker, embedder, ChromaDB upsert
└── retrieve.py    # Embedding model cache, where-clause builder, context formatting
```

No tests.

## Design reference

[`Design.md`](../Design.md) §9 (RAG pipeline).
