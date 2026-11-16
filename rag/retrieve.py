"""
RAG Retrieval Module

Queries ChromaDB for relevant lore chunks and formats them into a
context block for the lore assistant's prompt.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Shared Embedding Model (lazy-loaded singleton)
# ──────────────────────────────────────────────────────────────

_embedding_model = None


def get_embedding_model():
    """Load the embedding model once (float32, as ingest does) and cache it."""
    global _embedding_model
    if _embedding_model is None:
        from ingest import EMBEDDING_MODEL, load_embedding_model

        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        _embedding_model = load_embedding_model()
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


# ──────────────────────────────────────────────────────────────
# Rendering a chunk for the agent
# ──────────────────────────────────────────────────────────────
#
# The stored document is what was embedded: messages with bare [image]
# markers. What the agent reads is richer, and is built here after ranking so
# none of it can move a search result:
#
#   * each message line starts with a citation tag, [@<channel_id>/<message_id>],
#     which the bot swaps for a short per-conversation ref and later turns into
#     a link to the message;
#   * each [image] marker is replaced by [image: <caption>] where one exists.
#
# Both need the chunk split back into exact messages, which the ingest-time
# ``line_counts`` metadata allows. Chunks without it (an older collection) are
# returned as stored.

from ingest import IMAGE_MARKER, message_text  # noqa: E402

# Per caption, and per chunk in total. A conversation that is mostly a meme
# dump can hold dozens of images; past the chunk budget the rest stay bare
# [image] markers so one such chunk cannot eat the agent's context.
CAPTION_CONTEXT_MAX_CHARS = int(os.environ.get("CAPTION_CONTEXT_MAX_CHARS", "600"))
CAPTION_CONTEXT_CHUNK_CHARS = int(os.environ.get("CAPTION_CONTEXT_CHUNK_CHARS", "3000"))


def split_messages(doc: str, meta: dict) -> Optional[List[dict]]:
    """
    The chunk's messages as dicts {id, channel_id, user, text}, or None when
    the chunk predates line_counts metadata.
    """
    meta = meta or {}
    counts_raw, ids_raw = meta.get("line_counts"), meta.get("message_ids")
    if not counts_raw or not ids_raw:
        return None
    counts = [int(n) for n in str(counts_raw).split(",")]
    ids = str(ids_raw).split(",")
    lines = doc.split("\n")
    if len(counts) != len(ids) or 1 + sum(counts) != len(lines):
        logger.warning("Chunk %s: line_counts disagree with its text", meta.get("chunk_id"))
        return None
    out, pos = [], 1  # line 0 is the [#channel | time] header
    for mid, n in zip(ids, counts):
        line = "\n".join(lines[pos:pos + n])
        pos += n
        user, _, text = line.partition(": ")
        out.append({"id": mid, "channel_id": meta.get("channel_id", ""), "user": user, "text": text})
    return out


def _clip(caption: str) -> str:
    if CAPTION_CONTEXT_MAX_CHARS and len(caption) > CAPTION_CONTEXT_MAX_CHARS:
        return caption[:CAPTION_CONTEXT_MAX_CHARS].rstrip() + "…"
    return caption


def _with_captions(
    text: str, captions: Optional[List[Optional[str]]], budget: Optional[List[int]] = None,
) -> str:
    """
    Replace a message's trailing [image] markers with its captions.

    ``budget`` is a one-element list of characters still available in this
    chunk, spent as captions are added; None means unlimited.
    """
    if not captions:
        return text
    markers = message_text("", len(captions))
    if not text.endswith(markers):
        return text
    content = text[: len(text) - len(markers)].rstrip()
    parts = []
    for c in captions:
        c = _clip(c) if c else None
        if c and (budget is None or budget[0] >= len(c)):
            if budget is not None:
                budget[0] -= len(c)
            parts.append(f"[image: {c}]")
        else:
            parts.append(IMAGE_MARKER)
    rendered = " ".join(parts)
    return f"{content} {rendered}" if content else rendered


def cite_tag(channel_id: str, message_id: str) -> str:
    return f"[@{channel_id}/{message_id}]"


def render_message(msg: dict, captions: Optional[dict] = None, budget: Optional[List[int]] = None) -> str:
    text = _with_captions(msg["text"], (captions or {}).get(msg["id"]), budget)
    return f"{cite_tag(msg['channel_id'], msg['id'])} {msg['user']}: {text}"


def _captions_of(meta: dict) -> dict:
    try:
        return json.loads((meta or {}).get("captions") or "{}")
    except ValueError:
        return {}


def render_chunk(doc: str, meta: dict) -> str:
    """The chunk as the agent reads it: tagged messages, captions restored."""
    msgs = split_messages(doc, meta)
    if msgs is None:
        return doc.strip()
    captions = _captions_of(meta)
    header = doc.split("\n", 1)[0]
    budget = [CAPTION_CONTEXT_CHUNK_CHARS] if CAPTION_CONTEXT_CHUNK_CHARS else None
    return "\n".join([header] + [render_message(m, captions, budget) for m in msgs])


def _searchable(text: str) -> str:
    """Message text for literal matching and counting: markers are not words anyone typed."""
    return text.replace(IMAGE_MARKER, "").rstrip()


# ──────────────────────────────────────────────────────────────
# Exact nearest-neighbour search
# ──────────────────────────────────────────────────────────────
#
# ChromaDB's HNSW index missed results badly on this data. Measured against an
# exact search, top-10 recall averaged 0.79-0.80 on both the old and rebuilt
# collections, and for "relationship drama" it returned 0 of the true top 10:
# the best-matching conversation came back only when 1,000 results were
# requested. Raising ef_search on the collection did not change it, and
# over-fetching 300 only reached 0.92.
#
# The collections are small enough not to need an approximate index: 23.5k
# chunk vectors and 41.8k image vectors (768 float32 each, ~200 MB together)
# are scored exactly with one matrix product in a few milliseconds. So the
# vectors and the metadata the filters use are cached here, filtered and
# ranked with numpy, and ChromaDB is asked only for the winners' documents.
# ChromaDB stays the store, and literal search, counting and surrounding
# messages still use its metadata filters directly.

class _ExactIndex:
    """One collection's vectors plus the metadata semantic filters need."""

    def __init__(self, collection) -> None:
        import numpy as np

        ids, embs, channels, starts, ends = [], [], [], [], []
        offset = 0
        while True:
            page = collection.get(include=["embeddings", "metadatas"], limit=5000, offset=offset)
            if not page["ids"]:
                break
            ids += page["ids"]
            embs.extend(page["embeddings"])
            for meta in page["metadatas"]:
                meta = meta or {}
                channels.append(meta.get("channel_name", ""))
                starts.append(meta.get("timestamp_start", 0))
                ends.append(meta.get("timestamp_end", 0))
            offset += 5000
        self.count = len(ids)
        self.ids = ids
        self.vectors = np.asarray(embs, dtype=np.float32).reshape(len(ids), -1)
        self.channels = np.asarray(channels, dtype=object)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)

    def nearest(
        self, query_vec, k: int,
        channel_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """Ids of the k most similar vectors that pass the filters, best first."""
        import numpy as np

        if not self.count:
            return []
        mask = np.ones(self.count, dtype=bool)
        # Same semantics as _build_where_clause, so both search paths agree.
        if channel_name:
            mask &= self.channels == channel_name
        if start_date:
            mask &= self.starts >= _iso_to_epoch(start_date)
        if end_date:
            mask &= self.ends <= _iso_to_epoch(end_date)
        allowed = int(mask.sum())
        if not allowed:
            return []
        # Score everything and mask afterwards: indexing the matrix by the
        # allowed rows first copies it, which cost ~90 ms a query.
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)
        scores[~mask] = -np.inf
        k = min(k, allowed)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.ids[i] for i in top]


_exact_indexes: Dict[str, _ExactIndex] = {}


def exact_index(collection) -> _ExactIndex:
    """
    The cached index for a collection, reloaded when its size changes (a
    rebuild, or an ingest that added chunks). An ingest that only rewrites
    existing ids keeps the size, so /ingest calls clear_exact_indexes().
    """
    idx = _exact_indexes.get(collection.name)
    if idx is None or idx.count != collection.count():
        logger.info("Loading vectors of '%s' for exact search ...", collection.name)
        idx = _ExactIndex(collection)
        _exact_indexes[collection.name] = idx
        logger.info("Loaded %d vectors of '%s'", idx.count, collection.name)
    return idx


def clear_exact_indexes() -> None:
    _exact_indexes.clear()


def _get_in_order(collection, ids: List[str]) -> Tuple[List[str], List[dict]]:
    """Documents and metadata for ids, in the order given."""
    if not ids:
        return [], []
    got = collection.get(ids=ids, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    pairs = [by_id[i] for i in ids if i in by_id]
    return [d for d, _ in pairs], [m for _, m in pairs]


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
    then ranks every chunk exactly (see "Exact nearest-neighbour search").

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

    try:
        ids = exact_index(collection).nearest(query_embedding, top_k, channel_name, start_date, end_date)
        documents, metadatas = _get_in_order(collection, ids)
    except Exception:
        logger.exception("Semantic search failed")
        return "", 0

    if not documents:
        logger.info(f"No lore chunks found for query: {query!r}")
        return "", 0

    context = format_context_block(documents, metadatas)
    return context, len(documents)


def search_images(
    query: str,
    collection_name: str = "lore",
    top_k: int = 5,
    channel_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Semantic search over image captions, kept apart from conversation search
    so captions never compete with what people said.

    Each hit is returned as the conversation the image was posted in, led by
    the matching image, so the agent sees why it was posted and how people
    reacted. Several hits in one conversation share one block.

    Returns:
        (context, number of conversations returned)
    """
    try:
        client = get_chroma_client()
        images = client.get_or_create_collection(name=f"{collection_name}_images")
        chunks = client.get_or_create_collection(name=collection_name)
        q = get_embedding_model().encode([query], normalize_embeddings=True)[0]
        # Over-fetch: several images often come from one conversation.
        image_ids = exact_index(images).nearest(q, top_k * 3, channel_name, start_date, end_date)
        _, image_metas = _get_in_order(images, image_ids)
    except Exception:
        logger.exception("Image search failed")
        return "", 0

    hits: Dict[str, List[dict]] = {}
    for meta in image_metas:
        cid = (meta or {}).get("chunk_id")
        if not cid:
            continue
        if cid not in hits and len(hits) >= top_k:
            continue
        hits.setdefault(cid, []).append(meta)
    if not hits:
        return "", 0

    got = chunks.get(ids=list(hits), include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}

    blocks = []
    for cid, metas in hits.items():  # dicts keep rank order
        if cid not in by_id:
            continue
        doc, meta = by_id[cid]
        msgs = {m["id"]: m for m in split_messages(doc, meta) or []}
        captions = _captions_of(meta)
        lead = []
        for im in metas:
            msg = msgs.get(im["message_id"])
            if msg and render_message(msg, captions) not in lead:
                lead.append(render_message(msg, captions))
        shown = "\n".join(f">>> IMAGE MATCH — {line}" for line in lead)
        blocks.append(f"{shown}\n--- conversation it was posted in ---\n{render_chunk(doc, meta)}")
    logger.info("Image search %r: %d conversation(s)", query, len(blocks))
    return "\n\n---\n\n".join(blocks), len(blocks)


# ──────────────────────────────────────────────────────────────
# Surrounding messages
# ──────────────────────────────────────────────────────────────
#
# A chunk can end mid-exchange: busy conversations are cut at the 512-token
# cap, wherever the nearest message boundary falls. This reads the messages
# either side of one message, ignoring chunk boundaries, so the agent can
# finish reading an exchange instead of guessing how it went.

_DISCORD_EPOCH_MS = 1420070400000
# Half-widths tried in turn until enough messages are found. Most anchors are
# satisfied by the first; the wider ones are for quiet channels.
_SURROUNDING_WINDOWS = (6 * 3600, 2 * 86400, 14 * 86400, 90 * 86400)
_GAP_NOTE_SECONDS = 3600  # matches ingest's DEFAULT_GAP_MINUTES
SURROUNDING_MAX_COUNT = 60


def snowflake_time(message_id: str) -> float:
    """Unix seconds at which a Discord message was sent, read from its id."""
    return ((int(message_id) >> 22) + _DISCORD_EPOCH_MS) / 1000


def _surrounding_where(channel_id: str, t: int, half: int, direction: str) -> dict:
    """
    Chunks in the channel overlapping the part of [t - half, t + half] the
    direction asks for. A chunk overlaps [lo, hi] when it starts no later than
    hi and ends no earlier than lo; the anchor's own chunk always qualifies.
    """
    lo = t if direction == "after" else t - half
    hi = t if direction == "before" else t + half
    return {"$and": [
        {"channel_id": {"$eq": channel_id}},
        {"timestamp_start": {"$lte": hi}},
        {"timestamp_end": {"$gte": lo}},
    ]}


def _gap(seconds: float) -> str:
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.0f} hour(s)" if hours >= 1.5 else f"{seconds / 60:.0f} minutes"
    return f"{hours / 24:.0f} day(s)"


def surrounding_messages(
    channel_id: str,
    message_id: str,
    direction: str = "after",
    count: int = 20,
    collection_name: str = "lore",
) -> Tuple[str, int]:
    """
    The ``count`` messages before and/or after one message in its channel,
    across chunk boundaries, rendered like any other result (citation tags,
    captions). The anchor is marked, and silences of an hour or more are noted
    so a cut conversation can be told from one that ended.

    Args:
        direction: "before", "after", or "both" (count on each side).

    Returns:
        (context, number of messages returned besides the anchor).
        ("", 0) when the anchor is not in the collection.
    """
    count = max(1, min(int(count), SURROUNDING_MAX_COUNT))
    t = int(snowflake_time(message_id))
    try:
        collection = get_chroma_client().get_or_create_collection(name=collection_name)
    except Exception:
        logger.exception("Failed to connect to ChromaDB")
        return "", 0

    msgs: Dict[str, dict] = {}
    for half in _SURROUNDING_WINDOWS:
        where = _surrounding_where(channel_id, t, half, direction)
        try:
            got = collection.get(where=where, include=["documents", "metadatas"], limit=_MAX_SCAN)
        except Exception:
            logger.exception("Surrounding-messages query failed")
            return "", 0
        for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
            captions = _captions_of(meta)
            for m in split_messages(doc, meta) or []:
                msgs.setdefault(m["id"], {
                    **m, "captions": captions.get(m["id"]),
                    "channel_name": (meta or {}).get("channel_name", ""),
                })
        # Snowflakes increase with time, so sorting by id is sorting by time.
        ordered = sorted(msgs.values(), key=lambda m: int(m["id"]))
        ids = [m["id"] for m in ordered]
        if message_id not in ids:
            continue  # the window may simply not reach a chunk holding it yet
        i = ids.index(message_id)
        enough_before = direction == "after" or i >= count
        enough_after = direction == "before" or len(ids) - 1 - i >= count
        if enough_before and enough_after:
            break

    if message_id not in msgs:
        logger.info("Surrounding: message %s/%s is not in '%s'", channel_id, message_id, collection_name)
        return "", 0

    start = i if direction == "after" else max(0, i - count)
    end = i + 1 if direction == "before" else min(len(ordered), i + 1 + count)
    window = ordered[start:end]

    # One caption budget per ordinary chunk's worth of messages.
    per = CAPTION_CONTEXT_CHUNK_CHARS * max(1, len(window) // 20)
    budget = [per] if CAPTION_CONTEXT_CHUNK_CHARS else None
    first = window[0]
    lines = [
        f"[#{first['channel_name']} | "
        f"{datetime.fromtimestamp(snowflake_time(first['id']), tz=timezone.utc):%Y-%m-%d %H:%M}]"
    ]
    prev_t = None
    for m in window:
        mt = snowflake_time(m["id"])
        if prev_t is not None and mt - prev_t >= _GAP_NOTE_SECONDS:
            lines.append(
                f"— {_gap(mt - prev_t)} later, "
                f"{datetime.fromtimestamp(mt, tz=timezone.utc):%Y-%m-%d %H:%M} —"
            )
        prev_t = mt
        line = render_message(m, {m["id"]: m["captions"]} if m["captions"] else None, budget)
        lines.append(f">>> ANCHOR — {line}" if m["id"] == message_id else line)

    returned = len(window) - 1
    # Short only after the widest window was searched, so say what that means.
    reach = _gap(_SURROUNDING_WINDOWS[-1])
    if direction != "before" and end == len(ordered) and len(ordered) - 1 - i < count:
        lines.append(f"(no later messages in this channel within {reach})")
    if direction != "after" and start == 0 and i < count:
        lines.insert(1, f"(no earlier messages in this channel within {reach})")
    logger.info(
        "Surrounding %s/%s %s x%d: %d message(s) returned",
        channel_id, message_id, direction, count, returned,
    )
    return "\n".join(lines), returned


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
    metadatas = metadatas or [{}] * len(documents)
    blocks = []
    for doc, meta in zip(documents, metadatas):
        # Each document already starts with the [#channel | timestamp] header
        # from format_chunk_text(). Add a separator between chunks.
        blocks.append(render_chunk(doc, meta))

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


def _messages_for_matching(doc: str, meta: dict) -> List[dict]:
    """
    The chunk's messages for literal matching and counting, as dicts with
    "user" and searchable "text" (image markers removed). Uses exact
    line_counts splitting when the chunk has it, the author heuristic otherwise.
    """
    exact = split_messages(doc, meta)
    if exact is not None:
        return [{**m, "text": _searchable(m["text"])} for m in exact]
    return [{"id": None, "user": u, "text": t} for u, t in _parse_chunk_messages(doc, meta)]


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
            captions = _captions_of(meta)
            for msg in _messages_for_matching(doc, meta):
                if want_author and msg["user"].casefold() != want_author:
                    continue
                if term_re and not term_re.search(msg["text"]):
                    continue
                hits.append(
                    render_message(msg, captions) if msg["id"] else f"{msg['user']}: {msg['text']}"
                )
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
        block = render_chunk(doc, meta)
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
        for msg in _messages_for_matching(doc, meta):
            speaker, text = msg["user"], msg["text"]
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
