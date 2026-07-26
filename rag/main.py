"""
RAG Service — FastAPI Application

Exposes HTTP endpoints for lore retrieval and ingestion.
CPU-only — no VRAM impact.
"""

import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import run_ingest
from retrieve import build_lore_context

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


class IngestRequest(BaseModel):
    since: Optional[str] = None


class IngestResponse(BaseModel):
    status: str  # "running" | "completed"
    messages_loaded: int = 0
    chunks_created: int = 0
    chunks_upserted: int = 0


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


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
            )
            return result
        finally:
            _ingest_task = None

    _ingest_task = asyncio.create_task(asyncio.to_thread(_run))
    result = await _ingest_task

    return IngestResponse(
        status="completed",
        messages_loaded=result.get("messages_loaded", 0),
        chunks_created=result.get("chunks_created", 0),
        chunks_upserted=result.get("chunks_upserted", 0),
    )


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
