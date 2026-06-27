"""
RAG Ingest Pipeline

Loads per-user JSONL archives from the history-service, flattens across
users, chunks by conversation thread (1-hour temporal gaps), embeds with
ibm-granite/granite-embedding-311m-multilingual-r2, and upserts into ChromaDB.

Designed for idempotent re-runs — chunks are deduplicated by chunk_id
(sha256 of channel_id + first_message_id).
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
    caption: Optional[str] = None  # Image caption (if any attachment has caption_status == "done")


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


def _extract_caption(record: dict) -> Optional[str]:
    """Extract caption text from the first attachment with caption_status == 'done'."""
    attachments = record.get("attachments", [])
    for att in attachments:
        if att.get("caption_status") == "done" and att.get("caption"):
            return att["caption"]
    return None


def _record_to_message(record: dict) -> Optional[Message]:
    """Convert a JSONL archive record to a Message dataclass."""
    ts = _parse_timestamp(record.get("timestamp", ""))
    if ts is None:
        return None

    content = record.get("content", "") or ""
    caption = _extract_caption(record)

    # Skip messages with no text and no caption — they add no signal
    if not content.strip() and not caption:
        return None

    return Message(
        message_id=str(record.get("message_id", "")),
        user_id=str(record.get("user_id", "")),
        username=record.get("username", ""),
        channel_id=str(record.get("channel_id", "")),
        channel_name=record.get("channel_name", ""),
        timestamp=ts,
        content=content,
        caption=caption,
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
        username: content [image: caption]
        username: content
        ...
    """
    if not chunk.messages:
        return ""

    lines = []

    # Header with channel and timestamp of first message
    ts_str = chunk.timestamp_start.strftime("%Y-%m-%d %H:%M")
    lines.append(f"[#{chunk.channel_name} | {ts_str}]")

    for msg in chunk.messages:
        text = msg.content
        if msg.caption:
            text = f"{text} [image: {msg.caption}]"
        lines.append(f"{msg.username}: {text}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# ChromaDB Embedding & Upsert
# ──────────────────────────────────────────────────────────────


def _build_metadata(chunk: Chunk) -> Dict[str, str]:
    """Build ChromaDB metadata dict for a chunk."""
    return {
        "chunk_id": chunk.chunk_id,
        "channel_id": chunk.channel_id,
        "channel_name": chunk.channel_name,
        "timestamp_start": chunk.timestamp_start.isoformat(),
        "timestamp_end": chunk.timestamp_end.isoformat(),
        "authors": ",".join(chunk.authors),
        "message_count": str(chunk.message_count),
        "message_ids": ",".join(chunk.message_ids),
    }


def embed_and_upsert(
    chunks: List[Chunk],
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
    collection_name: str = "lore",
) -> int:
    """
    Embed chunks using ibm-granite/granite-embedding-311m-multilingual-r2 and upsert into ChromaDB.

    Uses idempotent upsert by chunk_id — re-running does not duplicate data.

    Args:
        chunks: List of Chunk objects to embed.
        chroma_host: ChromaDB server hostname.
        chroma_port: ChromaDB server port.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Number of chunks upserted.
    """
    if not chunks:
        logger.info("No chunks to embed")
        return 0

    import chromadb  # noqa: F811 — lazy import to avoid startup cost when not ingesting

    from sentence_transformers import SentenceTransformer

    logger.info(
        f"Embedding {len(chunks)} chunks using ibm-granite/granite-embedding-311m-multilingual-r2 "
        f"(ChromaDB at {chroma_host}:{chroma_port})"
    )

    # Load embedding model (CPU-only, ~311 MB)
    model = SentenceTransformer("ibm-granite/granite-embedding-311m-multilingual-r2")

    # Connect to ChromaDB server
    client = chromadb.HttpClient(host=chroma_host, port=str(chroma_port))
    collection = client.get_or_create_collection(name=collection_name)

    # Prepare batch data
    texts = [format_chunk_text(c) for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [_build_metadata(c) for c in chunks]

    # Embed in batches (sentence-transformers handles batching internally)
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Upsert into ChromaDB in batches to respect max batch size limit
    CHROMA_MAX_BATCH_SIZE = 5000  # Stay under ChromaDB's dynamic limit (~5461 typical)
    upserted = 0
    for i in range(0, len(chunks), CHROMA_MAX_BATCH_SIZE):
        batch_end = min(i + CHROMA_MAX_BATCH_SIZE, len(chunks))
        batch_slice = slice(i, batch_end)

        collection.upsert(
            ids=ids[batch_slice],
            documents=texts[batch_slice],
            metadatas=metadatas[batch_slice],
            embeddings=embeddings[batch_slice].tolist(),
        )
        upserted += batch_end - i
        logger.info(
            f"Upserted batch {i // CHROMA_MAX_BATCH_SIZE + 1}: "
            f"records {i+1}-{batch_end}/{len(chunks)}"
        )

    logger.info(f"Upserted {upserted} chunks into ChromaDB collection '{collection_name}'")
    return upserted


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

    Returns:
        Dict with counts: {messages_loaded, chunks_created, chunks_upserted}
    """
    logger.info(f"Starting ingest pipeline (archive={archive_dir}, since={since})")

    # Step 1: Load messages
    messages = load_archive(archive_dir, since=since)
    if not messages:
        logger.warning("No messages loaded — aborting ingest")
        return {"messages_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}

    # Step 2: Chunk by conversation
    chunks = chunk_by_conversation(messages, gap_minutes=gap_minutes, max_tokens=max_tokens)
    if not chunks:
        logger.warning("No chunks created — aborting ingest")
        return {"messages_loaded": len(messages), "chunks_created": 0, "chunks_upserted": 0}

    # Step 3: Embed and upsert
    upserted = embed_and_upsert(chunks, chroma_host=chroma_host, chroma_port=chroma_port)

    result = {
        "messages_loaded": len(messages),
        "chunks_created": len(chunks),
        "chunks_upserted": upserted,
    }
    logger.info(f"Ingest complete: {result}")
    return result
