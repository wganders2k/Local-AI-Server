"""
RAG Retrieval Module

Queries ChromaDB for relevant lore chunks and formats them into a
context block for the lore assistant's prompt.
"""

import logging
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


def build_lore_context(
    query: str,
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    collection_name: str = "lore",
    top_k: int = 5,
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

    Returns:
        Tuple of (context_string, chunk_count).
        Returns ("", 0) if no results or ChromaDB is unreachable.
    """
    import chromadb

    try:
        client = chromadb.HttpClient(host=chroma_host, port=str(chroma_port))
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

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
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
