"""
RAG Retrieval Module

Queries ChromaDB for relevant lore chunks and formats them into a
context block for the lore assistant's prompt.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
    exclude_channels: Optional[List[str]] = None,
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
        exclude_channels: Channel names to drop from the scope entirely. Applied
            as a single $nin so it composes with the other constraints.

    Returns:
        ChromaDB where-clause dict, or None if no filters are set.
    """
    constraints = []
    if channel_name:
        constraints.append({"channel_name": {"$eq": channel_name}})
    if exclude_channels:
        # Callers write channels the way Discord renders them ('#lore'), but
        # metadata holds the bare name, so an unstripped '#' would exclude
        # nothing at all — silently, since $nin against no match is legal.
        cleaned = [c.lstrip("#").strip() for c in exclude_channels if c and c.strip()]
        if cleaned:
            constraints.append({"channel_name": {"$nin": cleaned}})
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


# ──────────────────────────────────────────────────────────────
# Literal / Chronological / Aggregate Retrieval
# ──────────────────────────────────────────────────────────────
#
# These bypass the embedding model entirely. Semantic search answers "what is
# relevant to X"; it cannot answer "who said X *first*", because ranking by
# cosine distance and then sorting the top-k only ever reorders a similarity-
# chosen subset. Selecting on the document text and ordering by timestamp
# metadata answers a different class of question, over the whole collection.
#
# ChromaDB applies `limit` before any ordering we do here, so selection has to
# pull the full matching set and sort it in-process. _MAX_SCAN caps that.

_MAX_SCAN = 5000

# Chunk documents are rendered by ingest.format_chunk_text() as a header line
# followed by one "username: message" line per message, so a speaker is matched
# by anchoring to the start of a line in multiline mode.
def _author_regex(author: str) -> str:
    """Regex matching any chunk in which `author` spoke."""
    return f"(?im)^{re.escape(author.lstrip('@'))}:"


def _term_regex(term: str, whole_word: bool = False) -> str:
    """
    Case-insensitive regex for `term`.

    Substring matching by default: "oink" matches "oink", "oinking", "oinked"
    and also "yoink". Setting whole_word requires the match to begin at a word
    boundary, which drops "yoink"/"sploinky" while still keeping "oinking" and
    "oinks" — useful when a short term collides with unrelated words.
    """
    core = re.escape(term)
    return f"(?i)\\b{core}" if whole_word else f"(?i){core}"


# "username: message text" — the line format written by ingest.format_chunk_text.
_LINE_RE = re.compile(r"^(?P<user>[^:\n]+): (?P<text>.*)$")


def _parse_chunk_messages(doc: str, meta: dict) -> List[List[str]]:
    """
    Split a chunk document back into its individual [speaker, text] messages.

    A line only starts a new message if its name is one of the chunk's known
    authors: message text often contains colons ("Result: ..."), which would
    otherwise be read as a speaker. Lines that are not a new message are joined
    onto the previous one, so a multi-line Discord message stays one message and
    is searched in full.
    """
    known = {
        a.strip().casefold()
        for a in str((meta or {}).get("authors", "")).split(",")
        if a.strip()
    }
    messages: List[List[str]] = []
    for line in doc.splitlines():
        m = _LINE_RE.match(line)
        if m and (not known or m.group("user").strip().casefold() in known):
            messages.append([m.group("user").strip(), m.group("text")])
        elif messages:
            messages[-1][1] += "\n" + line
        # else: the leading "[#channel | timestamp]" header
    return messages


def _build_document_filter(
    term: Optional[str] = None,
    author: Optional[str] = None,
    whole_word: bool = False,
) -> Optional[dict]:
    """
    Build a ChromaDB where_document clause from a term and/or an author.

    Both conditions are regexes over the document text; combining them with
    $and finds chunks where that person spoke *and* the term appears.

    Returns:
        where_document dict, or None when neither filter is set.
    """
    # Deliberately kept as two independent conditions even when both are set:
    # anchoring the term to the author's own line would miss multi-line
    # messages. This is a broad prefilter; _matching_messages() below does the
    # precise per-message attribution that the caller actually asked for.
    clauses = []
    if term:
        clauses.append({"$regex": _term_regex(term, whole_word)})
    if author:
        clauses.append({"$regex": _author_regex(author)})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_literal(
    term: Optional[str] = None,
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    collection_name: str = "lore",
    author: Optional[str] = None,
    channel_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    order: str = "earliest",
    limit: int = 20,
    whole_word: bool = False,
) -> Tuple[str, int]:
    """
    Find chunks containing a literal term and/or spoken by an author, returned
    in chronological order rather than by similarity.

    This is the tool for "who said X first", "everyone's first X", "what did
    <person> say about X" — questions semantic search structurally cannot
    answer because it never sees the whole matching set.

    Args:
        term: Literal text to match, case-insensitively. Optional if author set.
        author: Username whose messages to match. Optional if term set.
        channel_name: Restrict to one channel.
        start_date / end_date: ISO 8601 bounds.
        order: "earliest" (oldest first) or "latest" (newest first).
        limit: Maximum chunks to return after ordering.

    Returns:
        Tuple of (context_string, total_matches). total_matches is the full
        number of matching chunks, which may exceed the number returned.
    """
    if not term and not author:
        return "", 0

    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        logger.exception("Failed to connect to ChromaDB")
        return "", 0

    where = _build_where_clause(channel_name, start_date, end_date)
    where_document = _build_document_filter(term, author, whole_word)

    try:
        kwargs = {
            "include": ["documents", "metadatas"],
            "limit": _MAX_SCAN,
            "where_document": where_document,
        }
        if where:
            kwargs["where"] = where
        logger.info(
            "Literal search: term=%r author=%r order=%s filters=%s",
            term, author, order, where,
        )
        results = collection.get(**kwargs)
    except Exception:
        logger.exception("ChromaDB literal query failed")
        return "", 0

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    if not documents:
        logger.info("No chunks matched term=%r author=%r", term, author)
        return "", 0

    # The Chroma filter is a chunk-level prefilter: with both term and author it
    # matches "this person spoke in a conversation where the term appeared",
    # not "this person said it". Re-check per message so an author+term search
    # means what the caller thinks it means.
    term_re = re.compile(_term_regex(term, whole_word)) if term else None
    want_author = author.lstrip("@").casefold() if author else None

    kept = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        hits = []
        if term_re or want_author:
            for speaker, text in _parse_chunk_messages(doc, meta):
                if want_author and speaker.casefold() != want_author:
                    continue
                if term_re and not term_re.search(text):
                    continue
                hits.append(f"{speaker}: {text}")
            if not hits:
                continue  # prefilter matched the chunk, but nobody we asked for said it
        kept.append((doc, meta, hits))

    if not kept:
        logger.info(
            "No message actually matched term=%r by author=%r (%d chunk(s) prefiltered)",
            term, author, len(documents),
        )
        return "", 0

    kept.sort(key=lambda x: (x[1] or {}).get("timestamp_start") or 0,
              reverse=(order == "latest"))
    total = len(kept)
    kept = kept[:limit]

    logger.info(
        "Literal search: %d chunk(s) contain a real match, returning %d (%s first)",
        total, len(kept), order,
    )

    # Lead each chunk with the exact message(s) that matched, so "what was the
    # exact message" is answerable without re-reading the whole conversation.
    blocks = []
    for doc, meta, hits in kept:
        block = doc.strip()
        if hits:
            shown = "\n".join(f">>> MATCH — {h}" for h in hits[:5])
            if len(hits) > 5:
                shown += f"\n>>> ... and {len(hits) - 5} more matching message(s) in this chunk"
            block = shown + "\n--- full conversation ---\n" + block
        blocks.append(block)
    return "\n\n---\n\n".join(blocks), total


def aggregate_stats(
    term: Optional[str] = None,
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    collection_name: str = "lore",
    author: Optional[str] = None,
    channel_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: str = "author",
    top_n: int = 25,
    whole_word: bool = False,
    exclude_channels: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """
    Count matching messages grouped by author, channel, or month, without
    returning the underlying chunks.

    Answers "how many times", "who says X most", "when was X most active"
    without spending the context budget that dumping the chunks would.

    Counting is done per *message line*, not per chunk: a chunk holds a whole
    conversation, so attributing its hits to every speaker in it would credit
    the term to people who never said it.

    Args:
        term: Literal text to count, case-insensitively.
        author: Restrict counting to one speaker.
        channel_name / start_date / end_date: Scope filters.
        group_by: "author", "channel", or "month".
        top_n: Number of groups to report.
        exclude_channels: Channel names to omit. Bulk-posted catalogue or log
            channels are written by one person and can dominate an author
            ranking without reflecting how the term is actually used in
            conversation; excluding them makes the ranking answer the question
            people usually mean.

    Returns:
        Tuple of (formatted_report, total_matching_messages).
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        logger.exception("Failed to connect to ChromaDB")
        return "", 0

    where = _build_where_clause(channel_name, start_date, end_date, exclude_channels)
    where_document = _build_document_filter(term, author, whole_word)

    try:
        kwargs = {
            "include": ["documents", "metadatas"],
            "limit": _MAX_SCAN,
            "where_document": where_document,
        }
        if where:
            kwargs["where"] = where
        results = collection.get(**kwargs)
    except Exception:
        logger.exception("ChromaDB aggregate query failed")
        return "", 0

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    if not documents:
        return "", 0

    term_re = re.compile(_term_regex(term, whole_word)) if term else None
    want_author = author.lstrip("@").casefold() if author else None

    counts: Dict[str, int] = {}
    total = 0
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        for speaker, text in _parse_chunk_messages(doc, meta):
            if want_author and speaker.casefold() != want_author:
                continue
            if term_re and not term_re.search(text):
                continue

            if group_by == "channel":
                key = "#" + str(meta.get("channel_name", "unknown"))
            elif group_by == "month":
                ts = meta.get("timestamp_start")
                key = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
                    if ts else "unknown"
                )
            else:
                key = speaker

            counts[key] = counts.get(key, 0) + 1
            total += 1

    if not total:
        return "", 0

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if group_by == "month":
        ordered = sorted(counts.items())  # chronological reads better for time
    ordered = ordered[:top_n]

    scope = []
    if term:
        scope.append(f"messages containing {term!r}")
    else:
        scope.append("messages")
    if author:
        scope.append(f"by {author}")
    if channel_name:
        scope.append(f"in {channel_name}")
    if exclude_channels:
        scope.append(
            "excluding " + ", ".join("#" + c.lstrip("#").strip() for c in exclude_channels)
        )
    if start_date or end_date:
        scope.append(f"between {start_date or 'any'} and {end_date or 'any'}")

    header = f"Counts of {' '.join(scope)} — grouped by {group_by}, {total} total:"
    lines = [f"  {key}: {count}" for key, count in ordered]
    if len(counts) > len(ordered):
        lines.append(f"  ... and {len(counts) - len(ordered)} more {group_by}(s)")

    logger.info("Aggregate: %d matching message(s) across %d %s(s)", total, len(counts), group_by)
    return header + "\n" + "\n".join(lines), total
