# RAG Service

CPU-only FastAPI service on `:8001` (internal). Ingests the per-user Discord archive into ChromaDB and answers retrieval queries for the bot's `/lore` agent. Zero VRAM impact — which is why lore retrieval can run while the GPU is busy with something else.

Embeddings: `ibm-granite/granite-embedding-311m-multilingual-r2`, loaded in **float32** lazily on first use and cached for the process lifetime. Its default bfloat16 runs about 5× slower on this CPU (110 s against 22 s per 256 chunks), so ingest and retrieval both force float32.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/retrieve` | Semantic search over conversation chunks. Supports channel and date-range filtering |
| `POST` | `/search_images` | Semantic search over image captions; returns the conversations the images were posted in |
| `POST` | `/surrounding` | The messages before/after one message in its channel, across chunk boundaries |
| `POST` | `/search_literal` | Exact-text / per-author search over every chunk, in time order |
| `POST` | `/aggregate` | Message counts grouped by author, channel or month |
| `POST` | `/ingest` | Chunk, embed, and upsert the archive. Idempotent, but see below |

`make rag-ingest` triggers ingestion; `make rag-status` shows collection counts.

**Full rebuilds** go into a fresh collection and are switched to afterwards, because upsert never deletes: when chunk boundaries move, the old chunks stay behind as near-duplicates.

```bash
docker compose exec rag-service python ingest.py lore_v2 --rebuild   # writes lore_v2 + lore_v2_images
# check it, then set RAG_COLLECTION=lore_v2 in .env and recreate rag-service
```

## Chunking is temporal, not per-message

A single Discord message is rarely a meaningful unit of retrieval — a joke and its punchline are usually two messages thirty seconds apart, and embedding them separately loses both. So [`ingest.py`](ingest.py) groups messages into conversations:

- A new chunk starts when the **channel changes**, regardless of timing.
- A new chunk starts after a **60-minute gap** between consecutive messages (`DEFAULT_GAP_MINUTES`).
- Chunks above a **512-token soft cap** (`DEFAULT_MAX_TOKENS`) are split at a message boundary rather than mid-conversation.

The archive is flattened across users first, so a chunk is a slice of the channel as it actually happened — several people talking — not one person's messages in isolation.

## Image captions are kept out of the conversation embeddings

The history service captions every archived image. Those captions are long, generic descriptions ("This is a screenshot of…"), and once the archive was captioned they were **59% of all text**. With captions inlined, more than half the chunks would have been mostly descriptions of pictures, and conversation search would rank on what images looked like instead of on what people said. So captions are kept separate:

- In a chunk's embedded text, an image is a bare `[image]` marker. The captions travel in the chunk's `captions` metadata.
- At retrieval, *after* ranking, `retrieve.render_chunk()` swaps the markers for `[image: <caption>]`, so the agent knows what was posted without the caption having influenced the search. Each caption is capped at `CAPTION_CONTEXT_MAX_CHARS`, and each chunk at `CAPTION_CONTEXT_CHUNK_CHARS` in total; past that, images stay bare markers, so one meme dump can't take over the context.
- Every caption is also embedded on its own (with the poster's message text) in `<collection>_images`, which backs `/search_images`.
- Literal search and counting match only what people typed. The markers are stripped before matching, and captions are never in the matched text.

## Citation tags

Every message line the service returns starts with `[@<channel_id>/<message_id>]`. The bot swaps those for short per-conversation refs (`[m12]`), the model cites with them, and the posted answer turns them into links to the message (see `discord-bot/lore/citations.py`). Splitting a chunk back into exact messages uses the `line_counts` metadata. A chunk without it, from a collection built before this change, is returned untagged.

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
    "line_counts": str,       # comma-joined, lines each message occupies in the document
    "captions": str,          # JSON {message_id: [caption | null, ...]} for messages with captions
}
```

**Timestamps are epoch integers, and this is load-bearing.** ChromaDB's `$gte`/`$lte` operators only work on numeric fields, and ChromaDB type-infers a field from its first insert. A collection that has ever seen a string timestamp will raise `ValueError` on every numeric comparison from then on — so the agent's `start_date` / `end_date` filters would fail permanently. Fixing that means rebuilding the collection, not editing a query.

Lists are comma-joined strings because ChromaDB metadata values must be scalars.

## Retrieval

**Semantic search is exact, not approximate.** ChromaDB's HNSW index missed results badly on this data: top-10 recall averaged about 0.8 against an exact search, and one query ("relationship drama") got 0 of its true top 10. Raising `ef_search` didn't help. The collections are small (23.5k chunk vectors and 41.8k image vectors, ~200 MB of float32), so `retrieve._ExactIndex` caches the vectors and the metadata the filters use, and ranks everything with one matrix product (about 4 ms a query). ChromaDB stays the store, and literal search, counting and surrounding-message reads use its metadata filters as before. The cache is warmed at startup and reloads when a collection's size changes. `/ingest` clears it, since a re-ingest can rewrite chunks without changing the count.

`build_lore_context()` in [`retrieve.py`](retrieve.py) embeds the query, applies an optional `where` clause built from channel name and date range, and returns formatted context blocks with chunk separators.

Callers are the `/lore` agent's three tools — whole-history search, per-channel search, and channel summarisation — which differ in the filters they pass rather than in the code path they take. See [`docs/design/discord-bot.md`](../docs/design/discord-bot.md) §7.3.

## Data source

Reads Tier 2 of the archive, bind-mounted read-only at `/archive` (`/mnt/storage_cold/array/DiscordArchive/archive` on the host). The RAG service does not produce that data and does not depend on the history service running — see [`history-service/README.md`](../history-service/README.md).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CHROMA_HOST` | `chromadb` | |
| `CHROMA_PORT` | `8000` | |
| `COLLECTION_NAME` | `lore` | Set from `RAG_COLLECTION` in `.env`. Images live in `<name>_images` |
| `ARCHIVE_DIR` | `/archive` | Tier 2 JSONL, read-only |
| `CAPTION_CONTEXT_MAX_CHARS` | `600` | Longest caption put back into retrieved text (0 = no cap) |
| `CAPTION_CONTEXT_CHUNK_CHARS` | `3000` | Caption characters put back per chunk (0 = no cap) |

## File structure

```
rag/
├── Dockerfile
├── requirements.txt
├── main.py        # FastAPI app
├── ingest.py      # Archive loader, chunker, embedder, ChromaDB upsert; CLI for rebuilds
└── retrieve.py    # Search, chunk rendering (citation tags, caption injection), image search
```

No tests.

## Design reference

[`docs/design/system.md`](../docs/design/system.md) §9 (RAG pipeline).
