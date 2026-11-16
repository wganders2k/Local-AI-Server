"""
RAG Ingest Pipeline

Loads per-user JSONL archives from the history-service, flattens across
users, chunks by conversation thread (1-hour temporal gaps), embeds with
ibm-granite/granite-embedding-311m-multilingual-r2, and upserts into ChromaDB.

Designed for idempotent re-runs — chunks are deduplicated by chunk_id
(sha256 of channel_id + first_message_id).

Image captions are deliberately NOT embedded in the conversation chunks. They
are long, generic descriptions ("This is a screenshot of...") and made up 59%
of all text once the archive was captioned, so inlining them turned half the
index into descriptions of pictures. Instead:

  * a chunk shows each image as a bare ``[image]`` marker, and carries the
    captions in its metadata, so retrieval can put them back into the text the
    agent reads after the search has already ranked the chunk;
  * every caption is also embedded on its own, in a separate ``<name>_images``
    collection, which backs a dedicated image search.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────


@dataclass
class Message:
    """A single Discord message from the JSONL archive."""
    message_id: str
    user_id: str
    username: str
    channel_id: str
    channel_name: str
    timestamp: datetime
    content: str
    # One entry per image attachment, in order: its caption, or None if it has
    # none (not yet captioned, or captioning failed). Rendered as [image] markers.
    images: List[Optional[str]] = field(default_factory=list)


@dataclass
class Chunk:
    """A conversation chunk ready for embedding."""
    messages: List[Message] = field(default_factory=list)

    @property
    def channel_id(self) -> str:
        return self.messages[0].channel_id if self.messages else ""

    @property
    def channel_name(self) -> str:
        return self.messages[0].channel_name if self.messages else ""

    @property
    def timestamp_start(self) -> datetime:
        return self.messages[0].timestamp if self.messages else datetime.now(timezone.utc)

    @property
    def timestamp_end(self) -> datetime:
        return self.messages[-1].timestamp if self.messages else datetime.now(timezone.utc)

    @property
    def chunk_id(self) -> str:
        """Deterministic ID for dedup — sha256(channel_id + first_message_id)."""
        raw = f"{self.channel_id}:{self.messages[0].message_id}" if self.messages else ""
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def authors(self) -> List[str]:
        return list(dict.fromkeys(m.username for m in self.messages))  # unique, ordered

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def message_ids(self) -> List[str]:
        return [m.message_id for m in self.messages]


# ──────────────────────────────────────────────────────────────
# JSONL Loading
# ──────────────────────────────────────────────────────────────


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle both Z suffix and +00:00 offset
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse timestamp: {ts_str!r}")
        return None


IMAGE_MARKER = "[image]"


def _is_image(att: dict) -> bool:
    return (att.get("content_type") or "").startswith("image/") or att.get("caption_status") == "done"


def _extract_images(record: dict) -> List[Optional[str]]:
    """
    Every image on the message, in order, with its caption or None. A message
    posting several images is common (a screenshot series), so all are kept.
    """
    return [
        (att.get("caption") or None) if att.get("caption_status") == "done" else None
        for att in record.get("attachments") or []
        if _is_image(att)
    ]


def message_text(content: str, image_count: int) -> str:
    """A message's text as embedded: its content, then one marker per image."""
    markers = " ".join([IMAGE_MARKER] * image_count)
    if not markers:
        return content
    return f"{content} {markers}" if content.strip() else markers


def _record_to_message(record: dict) -> Optional[Message]:
    """Convert a JSONL archive record to a Message dataclass."""
    ts = _parse_timestamp(record.get("timestamp", ""))
    if ts is None:
        return None

    content = record.get("content", "") or ""
    images = _extract_images(record)

    # Skip messages with no text and no image — they add no signal
    if not content.strip() and not images:
        return None

    return Message(
        message_id=str(record.get("message_id", "")),
        user_id=str(record.get("user_id", "")),
        username=record.get("username", ""),
        channel_id=str(record.get("channel_id", "")),
        channel_name=record.get("channel_name", ""),
        timestamp=ts,
        content=content,
        images=images,
    )


def load_archive(archive_dir: str, since: Optional[str] = None) -> List[Message]:
    """
    Load all JSONL files from the archive directory and flatten into a
    single sorted list of messages.

    Args:
        archive_dir: Path to the directory containing <user_id>.jsonl files.
        since: Optional ISO 8601 date string. Only include messages after this date.

    Returns:
        List of Message objects sorted by (channel_id, timestamp).
    """
    since_dt = _parse_timestamp(since) if since else None

    if not os.path.isdir(archive_dir):
        logger.error(f"Archive directory does not exist: {archive_dir}")
        return []

    messages: List[Message] = []
    skipped = 0
    filtered = 0

    for filename in sorted(os.listdir(archive_dir)):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(archive_dir, filename)
        logger.info(f"Loading {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    msg = _record_to_message(record)
                    if msg is None:
                        skipped += 1
                        continue

                    # Apply date filter
                    if since_dt and msg.timestamp < since_dt:
                        filtered += 1
                        continue

                    messages.append(msg)

        except IOError as e:
            logger.error(f"Failed to read {filepath}: {e}")

    # Sort by channel, then timestamp
    messages.sort(key=lambda m: (m.channel_id, m.timestamp))

    logger.info(
        f"Loaded {len(messages)} messages "
        f"({skipped} skipped, {filtered} filtered by date)"
    )
    return messages


# ──────────────────────────────────────────────────────────────
# Conversation Chunking
# ──────────────────────────────────────────────────────────────

DEFAULT_GAP_MINUTES = 60
DEFAULT_MAX_TOKENS = 512


def chunk_by_conversation(
    messages: List[Message],
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[Chunk]:
    """
    Group messages into conversation chunks using temporal gaps.

    Rules:
      - New chunk when channel changes (even within gap window)
      - New chunk when time gap between consecutive messages exceeds gap_minutes
      - Soft token cap at max_tokens — if exceeded, split at earliest sentence boundary

    Args:
        messages: Sorted list of Message objects (by channel_id, timestamp).
        gap_minutes: Minutes of silence before starting a new chunk.
        max_tokens: Soft token limit per chunk.

    Returns:
        List of Chunk objects.
    """
    if not messages:
        return []

    gap = timedelta(minutes=gap_minutes)
    chunks: List[Chunk] = [Chunk()]
    current = chunks[0]

    for msg in messages:
        # Always start new chunk on channel boundary
        if current.messages and current.channel_id != msg.channel_id:
            current = Chunk()
            chunks.append(current)

        # Start new chunk on temporal gap
        elif current.messages:
            last_ts = current.messages[-1].timestamp
            if (msg.timestamp - last_ts) > gap:
                current = Chunk()
                chunks.append(current)

        current.messages.append(msg)

    # Split oversized chunks by token count
    final_chunks: List[Chunk] = []
    for chunk in chunks:
        if _estimate_tokens(chunk) <= max_tokens:
            final_chunks.append(chunk)
        else:
            split_chunks = _split_chunk(chunk, max_tokens)
            final_chunks.extend(split_chunks)

    # Remove empty chunks
    final_chunks = [c for c in final_chunks if c.messages]

    logger.info(
        f"Created {len(final_chunks)} chunks from {len(messages)} messages "
        f"(gap={gap_minutes}min, max_tokens={max_tokens})"
    )
    return final_chunks


def _estimate_tokens(chunk: Chunk) -> int:
    """
    Rough token estimate for a chunk.

    Uses the rule-of-thumb that 1 token ≈ 4 characters for English text.
    This is approximate but sufficient for the soft cap check.
    """
    text = format_chunk_text(chunk)
    return len(text) // 4


def _split_chunk(chunk: Chunk, max_tokens: int) -> List[Chunk]:
    """
    Split an oversized chunk at the earliest sentence boundary after
    the token budget is exceeded.

    Uses a simple heuristic: walk messages until adding the next one
    would exceed max_tokens, then split there.
    """
    splits: List[Chunk] = []
    current = Chunk()

    for msg in chunk.messages:
        test_chunk = Chunk(messages=current.messages + [msg])
        if _estimate_tokens(test_chunk) > max_tokens and current.messages:
            splits.append(current)
            current = Chunk()
        current.messages.append(msg)

    if current.messages:
        splits.append(current)

    return splits


# ──────────────────────────────────────────────────────────────
# Chunk Text Formatting
# ──────────────────────────────────────────────────────────────


def format_chunk_text(chunk: Chunk) -> str:
    """
    Render a chunk as a mini-transcript for embedding.

    Format:
        [#channel_name | YYYY-MM-DD HH:MM]
        username: content [image]
        username: content
        ...

    Retrieval splits this back into messages using the ``line_counts``
    metadata, so the line layout here and in _build_metadata must agree.
    """
    if not chunk.messages:
        return ""

    lines = []

    # Header with channel and timestamp of first message
    ts_str = chunk.timestamp_start.strftime("%Y-%m-%d %H:%M")
    lines.append(f"[#{chunk.channel_name} | {ts_str}]")

    for msg in chunk.messages:
        lines.append(f"{msg.username}: {message_text(msg.content, len(msg.images))}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# ChromaDB Embedding & Upsert
# ──────────────────────────────────────────────────────────────


def _build_metadata(chunk: Chunk) -> dict:
    """Build ChromaDB metadata dict for a chunk.

    Timestamps are stored as Unix epoch integers so that ChromaDB's $gte/$lte
    numeric comparison operators work correctly for date-range filtering.
    ChromaDB type-infers field types on first insert — once a field is seen
    as a string, numeric operators will fail with ValueError.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "channel_id": chunk.channel_id,
        "channel_name": chunk.channel_name,
        "timestamp_start": int(chunk.timestamp_start.timestamp()),
        "timestamp_end": int(chunk.timestamp_end.timestamp()),
        "authors": ",".join(chunk.authors),
        "message_count": str(chunk.message_count),
        "message_ids": ",".join(chunk.message_ids),
        # Lines each message occupies in the document (Discord messages can be
        # multi-line). Lets retrieval split a chunk back into exact messages,
        # to tag each one for citation and to put captions back in.
        "line_counts": ",".join(
            str(message_text(m.content, len(m.images)).count("\n") + 1)
            for m in chunk.messages
        ),
        # {message_id: [caption or null, ...]} for messages with at least one
        # caption — kept out of the embedded text, injected at retrieval.
        "captions": json.dumps(
            {m.message_id: m.images for m in chunk.messages if any(m.images)},
            ensure_ascii=False,
        ),
    }


EMBEDDING_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
EMBED_BATCH = 512  # chunks encoded and upserted per round, so progress is logged and kept


def load_embedding_model():
    """
    The embedding model in float32. Loaded in its default bfloat16 it runs
    about 5x slower on this CPU (110s vs 22s per 256 chunks measured), which
    made a full ingest take over two hours. Retrieval loads it the same way so
    query and document vectors come from the same weights.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"dtype": torch.float32})


@dataclass
class ImageDoc:
    """One captioned image, embedded on its own for image search."""
    doc_id: str
    text: str
    metadata: dict


def image_docs(chunks: List[Chunk]) -> List[ImageDoc]:
    """
    Every captioned image as a document of its own: the caption, plus whatever
    the poster wrote alongside it. Each points back at its chunk, so a hit can
    be returned as the conversation the image was posted in.
    """
    docs: List[ImageDoc] = []
    for chunk in chunks:
        for msg in chunk.messages:
            for n, caption in enumerate(msg.images):
                if not caption:
                    continue
                text = caption
                if msg.content.strip():
                    text = f"{caption}\n{msg.username}: {msg.content}"
                docs.append(ImageDoc(
                    doc_id=f"{msg.message_id}:{n}",
                    text=text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "message_id": msg.message_id,
                        "channel_id": msg.channel_id,
                        "channel_name": msg.channel_name,
                        "author": msg.username,
                        # Same field names as chunks, so the date/channel where
                        # clause builder works on both collections.
                        "timestamp_start": int(msg.timestamp.timestamp()),
                        "timestamp_end": int(msg.timestamp.timestamp()),
                    },
                ))
    return docs


def _encode(model, texts: List[str]):
    """
    Encode a batch, re-encoding any row that came back non-finite on its own.

    A full rebuild once died on a NaN embedding (Chroma rejects them) that did
    not reproduce when the same batch was encoded again, so a bad row is
    retried alone before it is given up on. Returns (embeddings, bad_rows).
    """
    import numpy as np

    emb = model.encode(texts, normalize_embeddings=True)
    bad = [i for i in range(len(texts)) if not np.isfinite(emb[i]).all()]
    still_bad = []
    for i in bad:
        emb[i] = model.encode([texts[i]], normalize_embeddings=True)[0]
        if not np.isfinite(emb[i]).all():
            still_bad.append(i)
    if bad:
        logger.warning(f"{len(bad)} non-finite embedding(s) re-encoded, {len(still_bad)} still bad and skipped")
    return emb, set(still_bad)


def _upsert_batched(model, collection, ids: List[str], texts: List[str], metadatas: List[dict], what: str) -> int:
    done = 0
    for i in range(0, len(ids), EMBED_BATCH):
        end = min(i + EMBED_BATCH, len(ids))
        emb, bad = _encode(model, texts[i:end])
        keep = [j for j in range(end - i) if j not in bad]
        for j in bad:
            logger.error(f"Skipping {what} {ids[i + j]}: embedding is not finite")
        collection.upsert(
            ids=[ids[i + j] for j in keep],
            documents=[texts[i + j] for j in keep],
            metadatas=[metadatas[i + j] for j in keep],
            embeddings=[emb[j].tolist() for j in keep],
        )
        done += len(keep)
        logger.info(f"Upserted {what} {end}/{len(ids)}")
    return done


def _unstored(collection, ids: List[str], *rows: List) -> tuple:
    """Drop rows whose id the collection already holds (for --resume)."""
    have = set()
    for i in range(0, len(ids), 5000):
        have.update(collection.get(ids=ids[i:i + 5000], include=[])["ids"])
    keep = [k for k, x in enumerate(ids) if x not in have]
    logger.info(f"Resume: {len(have)} already stored, {len(keep)} to embed")
    return ([ids[k] for k in keep],) + tuple([r[k] for k in keep] for r in rows)


def embed_and_upsert(
    chunks: List[Chunk],
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    collection_name: str = "lore",
    rebuild: bool = False,
    resume: bool = False,
) -> Dict[str, int]:
    """
    Embed chunks and their captioned images, and upsert both into ChromaDB:
    chunks into ``collection_name``, images into ``<collection_name>_images``.

    Upsert is idempotent by id but never deletes, so when chunk boundaries move
    (as they do whenever the set of included messages changes) the old chunks
    stay behind as near-duplicates. ``rebuild=True`` drops both collections
    first; the safe way to use it is on a fresh collection name, then point
    COLLECTION_NAME at it once it has been checked. ``resume=True`` skips ids
    the collections already hold, to finish an interrupted rebuild.

    Returns:
        {"chunks_upserted": n, "images_upserted": m}
    """
    if not chunks:
        logger.info("No chunks to embed")
        return {"chunks_upserted": 0, "images_upserted": 0}

    import chromadb  # noqa: F811 — lazy import to avoid startup cost when not ingesting

    images_name = f"{collection_name}_images"
    client = chromadb.HttpClient(host=chroma_host, port=str(chroma_port))
    if rebuild:
        for name in (collection_name, images_name):
            try:
                client.delete_collection(name)
                logger.info(f"Dropped collection '{name}' for rebuild")
            except Exception:
                pass  # did not exist
    collection = client.get_or_create_collection(name=collection_name)
    image_collection = client.get_or_create_collection(name=images_name)

    model = load_embedding_model()
    rows = ([c.chunk_id for c in chunks], [format_chunk_text(c) for c in chunks], [_build_metadata(c) for c in chunks])
    if resume:
        rows = _unstored(collection, *rows)
    logger.info(f"Embedding {len(rows[0])} chunks into '{collection_name}' with {EMBEDDING_MODEL}")
    n_chunks = _upsert_batched(model, collection, *rows, "chunks")

    docs = image_docs(chunks)
    rows = ([d.doc_id for d in docs], [d.text for d in docs], [d.metadata for d in docs])
    if resume:
        rows = _unstored(image_collection, *rows)
    logger.info(f"Embedding {len(rows[0])} captioned images into '{images_name}'")
    n_images = _upsert_batched(model, image_collection, *rows, "images")
    return {"chunks_upserted": n_chunks, "images_upserted": n_images}


# ──────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────


def run_ingest(
    archive_dir: str = "/archive",
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    since: Optional[str] = None,
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    collection_name: str = "lore",
    rebuild: bool = False,
    resume: bool = False,
) -> Dict[str, int]:
    """
    Run the full ingest pipeline: load → chunk → embed → upsert.

    Args:
        archive_dir: Path to JSONL archive directory.
        chroma_host: ChromaDB hostname.
        chroma_port: ChromaDB port.
        since: Optional ISO date — only process messages after this date.
        gap_minutes: Temporal gap threshold for conversation chunking.
        max_tokens: Soft token cap per chunk.
        collection_name: Target collection (images go to ``<name>_images``).
        rebuild: Drop the target collections first. See embed_and_upsert().
        resume: Skip ids already stored, to finish an interrupted rebuild.

    Returns:
        Dict with counts: {messages_loaded, chunks_created, chunks_upserted, images_upserted}
    """
    logger.info(
        f"Starting ingest pipeline (archive={archive_dir}, since={since}, "
        f"collection={collection_name}, rebuild={rebuild})"
    )
    empty = {"messages_loaded": 0, "chunks_created": 0, "chunks_upserted": 0, "images_upserted": 0}

    # Step 1: Load messages
    messages = load_archive(archive_dir, since=since)
    if not messages:
        logger.warning("No messages loaded — aborting ingest")
        return empty

    # Step 2: Chunk by conversation
    chunks = chunk_by_conversation(messages, gap_minutes=gap_minutes, max_tokens=max_tokens)
    if not chunks:
        logger.warning("No chunks created — aborting ingest")
        return {**empty, "messages_loaded": len(messages)}

    # Step 3: Embed and upsert
    counts = embed_and_upsert(
        chunks, chroma_host=chroma_host, chroma_port=chroma_port,
        collection_name=collection_name, rebuild=rebuild, resume=resume,
    )

    result = {"messages_loaded": len(messages), "chunks_created": len(chunks), **counts}
    logger.info(f"Ingest complete: {result}")
    return result


if __name__ == "__main__":
    # Long full rebuilds run here rather than through POST /ingest, whose
    # caller would otherwise have to hold an HTTP request open for the whole run:
    #   docker compose exec rag-service python ingest.py lore_v2 --rebuild
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Ingest the archive into ChromaDB")
    ap.add_argument("collection", nargs="?", default=os.environ.get("COLLECTION_NAME", "lore"))
    ap.add_argument("--rebuild", action="store_true", help="drop the target collections first")
    ap.add_argument("--resume", action="store_true", help="skip ids already stored (finish an interrupted rebuild)")
    ap.add_argument("--since", default=None)
    a = ap.parse_args()
    if a.rebuild and a.resume:
        ap.error("--rebuild and --resume are opposites")
    run_ingest(
        archive_dir=os.environ.get("ARCHIVE_DIR", "/archive"),
        chroma_host=os.environ.get("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
        since=a.since, collection_name=a.collection, rebuild=a.rebuild, resume=a.resume,
    )
