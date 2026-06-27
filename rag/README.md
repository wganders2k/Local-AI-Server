# RAG Service

CPU-only service that handles Discord history ingestion and lore retrieval for the lore assistant. Uses ChromaDB as the vector store and `ibm-granite/granite-embedding-311m-multilingual-r2` for embeddings. Zero VRAM impact.

## Responsibilities

- Parse Discord history exports and chunk by conversation thread (max 512 tokens/chunk)
- Embed chunks using `ibm-granite/granite-embedding-311m-multilingual-r2` and store in ChromaDB collection `lore`
- Expose a retrieval function used by the Discord bot at inference time
- Support re-ingestion as new lore accumulates (manual trigger or cron)

## Design Reference

See `Design.md` §9 (RAG Pipeline).

## Chunk metadata schema

```python
{
    "author": str,
    "timestamp": str,   # ISO 8601
    "channel": str,
    "topic_tags": list[str]
}
```

## Planned file structure

```
rag/
├── Dockerfile
├── requirements.txt
└── ingest.py           # Discord export parser, chunker, embedder, ChromaDB loader
```
