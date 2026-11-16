"""
RAG Service — FastAPI Application

Exposes HTTP endpoints for lore retrieval and ingestion.
CPU-only — no VRAM impact.
"""

import asyncio
import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import run_ingest
from retrieve import (
    aggregate_stats,
    build_lore_context,
    clear_exact_indexes,
    search_images,
    search_literal,
    surrounding_messages,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "/archive")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "lore")

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Service",
    description="Lore retrieval and ingestion for the Mimic Bot system.",
    version="0.1.0",
)

# Track in-flight ingest task to prevent concurrent runs
_ingest_task: Optional[asyncio.Task] = None


# ──────────────────────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    channel_name: Optional[str] = None  # Filter to a specific channel
    start_date: Optional[str] = None    # ISO 8601 — only results after this date
    end_date: Optional[str] = None      # ISO 8601 — only results before this date


class RetrieveResponse(BaseModel):
    context: str
    chunk_count: int


class ImageSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    channel_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SurroundingRequest(BaseModel):
    """The messages either side of one message, across chunk boundaries."""
    channel_id: str
    message_id: str
    direction: str = "after"  # "before" | "after" | "both"
    count: int = 20           # per side; capped at SURROUNDING_MAX_COUNT


class LiteralSearchRequest(BaseModel):
    """Chronological/literal search — no embedding involved."""
    term: Optional[str] = None       # literal text, matched case-insensitively
    author: Optional[str] = None     # username whose messages to match
    channel_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    order: str = "earliest"          # "earliest" | "latest"
    limit: int = 20
    whole_word: bool = False         # True = don't match inside larger words


class AggregateRequest(BaseModel):
    """Counts grouped by author, channel or month."""
    term: Optional[str] = None
    author: Optional[str] = None
    channel_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    group_by: str = "author"         # "author" | "channel" | "month"
    top_n: int = 25
    whole_word: bool = False
    exclude_channels: Optional[List[str]] = None


class IngestRequest(BaseModel):
    since: Optional[str] = None
    # Defaults to the live collection. A full rebuild is better done into a
    # fresh name with `python ingest.py <name> --rebuild`, then switched to.
    collection: Optional[str] = None
    rebuild: bool = False


class IngestResponse(BaseModel):
    status: str  # "running" | "completed"
    messages_loaded: int = 0
    chunks_created: int = 0
    chunks_upserted: int = 0
    images_upserted: int = 0


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@app.on_event("startup")
async def warm_exact_index():
    """
    Load the exact-search vectors in the background at startup, so the first
    /lore search does not wait several seconds for them.
    """
    def _warm():
        try:
            from retrieve import exact_index, get_chroma_client, get_embedding_model

            client = get_chroma_client()
            for name in (COLLECTION_NAME, f"{COLLECTION_NAME}_images"):
                exact_index(client.get_or_create_collection(name=name))
            get_embedding_model()
        except Exception:
            logger.exception("Warming the exact-search index failed; it will load on first use")

    asyncio.get_running_loop().run_in_executor(None, _warm)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """
    Retrieve relevant lore chunks for a query.

    Returns formatted context string and the number of chunks retrieved.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    context, chunk_count = build_lore_context(
        query=req.query,
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT,
        collection_name=COLLECTION_NAME,
        top_k=req.top_k,
        channel_name=req.channel_name,
        start_date=req.start_date,
        end_date=req.end_date,
    )

    return RetrieveResponse(context=context, chunk_count=chunk_count)


@app.post("/search_images", response_model=RetrieveResponse)
async def search_images_endpoint(req: ImageSearchRequest):
    """
    Semantic search over image captions only. Returns the conversations the
    matching images were posted in; chunk_count is how many.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    context, count = search_images(
        query=req.query,
        collection_name=COLLECTION_NAME,
        top_k=req.top_k,
        channel_name=req.channel_name,
        start_date=req.start_date,
        end_date=req.end_date,
    )
    return RetrieveResponse(context=context, chunk_count=count)


@app.post("/surrounding", response_model=RetrieveResponse)
async def surrounding_endpoint(req: SurroundingRequest):
    """
    Read on from (or back from) one message. chunk_count is the number of
    messages returned besides the anchor; 0 with empty context means the
    anchor is not in the collection.
    """
    if req.direction not in ("before", "after", "both"):
        raise HTTPException(status_code=400, detail="direction must be 'before', 'after' or 'both'")
    if not (req.channel_id.isdigit() and req.message_id.isdigit()):
        raise HTTPException(status_code=400, detail="channel_id and message_id must be Discord ids")
    context, count = surrounding_messages(
        channel_id=req.channel_id,
        message_id=req.message_id,
        direction=req.direction,
        count=req.count,
        collection_name=COLLECTION_NAME,
    )
    return RetrieveResponse(context=context, chunk_count=count)


@app.post("/search_literal", response_model=RetrieveResponse)
async def search_literal_endpoint(req: LiteralSearchRequest):
    """
    Literal / chronological search over the whole collection.

    Unlike /retrieve this does not embed anything: it selects chunks by the
    text they contain and by who spoke in them, then orders by timestamp.
    That is what makes "who said X first" answerable — ranking by similarity
    and sorting the top-k only reorders a similarity-chosen subset.
    """
    if not req.term and not req.author:
        raise HTTPException(status_code=400, detail="Provide at least one of: term, author")
    if req.order not in ("earliest", "latest"):
        raise HTTPException(status_code=400, detail="order must be 'earliest' or 'latest'")

    context, total = search_literal(
        term=req.term,
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT,
        collection_name=COLLECTION_NAME,
        author=req.author,
        channel_name=req.channel_name,
        start_date=req.start_date,
        end_date=req.end_date,
        order=req.order,
        limit=req.limit,
        whole_word=req.whole_word,
    )
    return RetrieveResponse(context=context, chunk_count=total)


@app.post("/aggregate", response_model=RetrieveResponse)
async def aggregate_endpoint(req: AggregateRequest):
    """
    Counts of matching messages grouped by author, channel or month.

    Returns a small report rather than the chunks themselves, so questions
    like "who says X most" do not have to spend the whole context budget.
    """
    if req.group_by not in ("author", "channel", "month"):
        raise HTTPException(status_code=400, detail="group_by must be 'author', 'channel' or 'month'")

    report, total = aggregate_stats(
        term=req.term,
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT,
        collection_name=COLLECTION_NAME,
        author=req.author,
        channel_name=req.channel_name,
        start_date=req.start_date,
        end_date=req.end_date,
        group_by=req.group_by,
        top_n=req.top_n,
        whole_word=req.whole_word,
        exclude_channels=req.exclude_channels,
    )
    return RetrieveResponse(context=report, chunk_count=total)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    Trigger a lore ingestion run.

    If an ingest is already in progress, returns status "running".
    Runs asynchronously to avoid blocking the HTTP response for long runs.
    """
    global _ingest_task

    if _ingest_task is not None and not _ingest_task.done():
        return IngestResponse(status="running")

    def _run():
        global _ingest_task
        try:
            result = run_ingest(
                archive_dir=ARCHIVE_DIR,
                chroma_host=CHROMA_HOST,
                chroma_port=CHROMA_PORT,
                since=req.since,
                collection_name=req.collection or COLLECTION_NAME,
                rebuild=req.rebuild,
            )
            return result
        finally:
            # Re-ingested chunks keep their ids, so the cached vectors would
            # not notice a change by size alone.
            clear_exact_indexes()
            _ingest_task = None

    _ingest_task = asyncio.create_task(asyncio.to_thread(_run))
    result = await _ingest_task

    return IngestResponse(
        status="completed",
        messages_loaded=result.get("messages_loaded", 0),
        chunks_created=result.get("chunks_created", 0),
        chunks_upserted=result.get("chunks_upserted", 0),
        images_upserted=result.get("images_upserted", 0),
    )


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
