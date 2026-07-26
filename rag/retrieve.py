"""
RAG Retrieval Module

Queries ChromaDB for relevant lore chunks and formats them into a
context block for the lore assistant's prompt.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Shared Embedding Model (lazy-loaded singleton)
# ──────────────────────────────────────────────────────────────

_MODEL_NAME = "ibm-granite/granite-embedding-311m-multilingual-r2"
_embedding_model = None


def get_embedding_model():
    """Load the embedding model once and cache it for subsequent calls."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s ...", _MODEL_NAME)
        _embedding_model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedding_model


# ──────────────────────────────────────────────────────────────
# Shared ChromaDB Client (lazy-loaded singleton)
# ──────────────────────────────────────────────────────────────

_chroma_client = None
_CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
_CHROMA_PORT = os.environ.get("CHROMA_PORT", "8000")


def get_chroma_client():
    """Get or create a singleton ChromaDB HTTP client.

    Creating a new client for every query leaks file descriptors and
    eventually causes 'Too many open files' errors. This singleton
    ensures the underlying HTTP connection is reused.
    """
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        logger.info("Connecting to ChromaDB at %s:%s ...", _CHROMA_HOST, _CHROMA_PORT)
        try:
            _chroma_client = chromadb.HttpClient(host=_CHROMA_HOST, port=str(_CHROMA_PORT))
            logger.info("ChromaDB client connected.")
        except Exception:
            logger.exception("Failed to connect to ChromaDB")
            raise
    return _chroma_client


def _iso_to_epoch(iso_str: str) -> int:
    """Convert an ISO 8601 date string to a Unix epoch integer.

    ChromaDB stores timestamps as integers (epoch seconds). Incoming
    ISO strings from the agent must be converted before comparison.
    """
    ts = iso_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    return int(dt.timestamp())


def _build_where_clause(
    channel_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[dict]:
    """
    Build a ChromaDB where-clause dict from optional filter parameters.

    ChromaDB stores timestamps as Unix epoch integers. Incoming ISO 8601
    date strings are converted to epoch seconds before building the clause.

    IMPORTANT: ChromaDB only accepts a SINGLE top-level key in the where
    clause (or $and/$or logical operators). When multiple field constraints
    are needed, wrap them in $and.

    Args:
        channel_name: Exact channel name to filter (e.g. "general").
        start_date: ISO 8601 date string -- only include results on or after this date.
        end_date: ISO 8601 date string -- only include results on or before this date.

    Returns:
        ChromaDB where-clause dict, or None if no filters are set.
    """
    constraints = []
    if channel_name:
        constraints.append({"channel_name": {"$eq": channel_name}})
    if start_date:
        constraints.append({"timestamp_start": {"$gte": _iso_to_epoch(start_date)}})
    if end_date:
        constraints.append({"timestamp_end": {"$lte": _iso_to_epoch(end_date)}})

    if not constraints:
        return None
    if len(constraints) == 1:
        return constraints[0]
    # Multiple constraints -- must use $and (ChromaDB rejects multiple top-level keys)
    return {"$and": constraints}


def build_lore_context(
    query: str,
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    collection_name: str = "lore",
    top_k: int = 5,
    channel_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Retrieve relevant lore chunks for a query and format them into a
    context block.

    Uses the same sentence-transformer model as ingest to encode the query,
    then searches by embedding (the ChromaDB collection has no built-in
    embedding function because embeddings are computed externally).

    Args:
        query: The user's lore question.
        chroma_host: ChromaDB server hostname.
        chroma_port: ChromaDB server port.
        collection_name: Name of the ChromaDB collection.
        top_k: Number of chunks to retrieve.
        channel_name: Optional — only search within this channel.
        start_date: Optional ISO 8601 date — only include results after this date.
        end_date: Optional ISO 8601 date — only include results before this date.

    Returns:
        Tuple of (context_string, chunk_count).
        Returns ("", 0) if no results or ChromaDB is unreachable.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        logger.exception("Failed to connect to ChromaDB")
        return "", 0

    # Encode query with the shared (lazy-loaded) embedding model
    try:
        model = get_embedding_model()
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
    except Exception:
        logger.exception("Failed to encode query")
        return "", 0

    # Build optional where-clause for metadata filtering
    where = _build_where_clause(channel_name, start_date, end_date)

    try:
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas"],
        }
        if where:
            kwargs["where"] = where
            logger.info("RAG query filters: %s", where)

        results = collection.query(**kwargs)
    except Exception:
        logger.exception("ChromaDB query failed")
        return "", 0

    # Extract documents and metadata
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        logger.info(f"No lore chunks found for query: {query!r}")
        return "", 0

    context = format_context_block(documents, metadatas)
    return context, len(documents)


def format_context_block(
    documents: List[str],
    metadatas: Optional[List[dict]] = None,
) -> str:
    """
    Format retrieved chunks into a context block for the lore prompt.

    Each chunk is wrapped with a separator and its channel/timestamp header
    (which is already embedded in the document text from ingest).

    Args:
        documents: List of chunk text strings from ChromaDB.
        metadatas: Optional list of metadata dicts (for future filtering).

    Returns:
        Formatted context string ready to prepend to the user's question.
    """
    blocks = []
    for i, doc in enumerate(documents):
        # Each document already starts with the [#channel | timestamp] header
        # from format_chunk_text(). Add a separator between chunks.
        blocks.append(doc.strip())

    return "\n\n---\n\n".join(blocks)
