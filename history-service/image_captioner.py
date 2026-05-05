"""
Image captioning pipeline.

Batch processor: scans pending attachments, downloads images,
calls proxy with image-caption model, writes captions to JSONL.

Runs as a scheduled job during the caption window.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import httpx

from config import (
    IMAGE_CAPTION_BATCH_SIZE,
    IMAGE_CAPTION_MAX_FILE_SIZE_MB,
    IMAGE_CAPTION_MODEL,
    PROXY_URL,
)
from jsonl_store import get_pending_captions, load_all_records, rewrite_archive

logger = logging.getLogger(__name__)

# Caption prompt — tuned for the uncensored mimic base model
CAPTION_PROMPT = (
    "Describe this image in 1-2 sentences. Be direct and factual. "
    "Include key visual elements, text, and context. "
    "Do not add disclaimers or safety warnings."
)


def _get_proxy_queue_depth() -> int:
    """
    Check proxy queue depth via HTTP GET /status.
    
    Returns:
        Queue depth (0 means idle). Returns -1 on error.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{PROXY_URL}/status")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("queue_depth", 0)
    except Exception as e:
        logger.error(f"Failed to check proxy queue depth: {e}")
    
    return -1


def _download_image(url: str, max_size_mb: int) -> Optional[str]:
    """
    Download an image from URL to a temporary file.
    
    Args:
        url: Image URL.
        max_size_mb: Maximum file size in MB.
        
    Returns:
        Path to temporary file, or None on failure.
    """
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                logger.warning(f"Failed to download image: HTTP {resp.status_code}")
                return None
            
            size_bytes = len(resp.content)
            if size_bytes > max_size_mb * 1024 * 1024:
                logger.warning(
                    f"Image too large: {size_bytes / (1024*1024):.1f} MB > {max_size_mb} MB"
                )
                return None
            
            # Write to temp file
            fd, path = tempfile.mkstemp(suffix=".jpg")
            try:
                os.write(fd, resp.content)
            finally:
                os.close(fd)
            
            return path
    
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def _caption_image(image_path: str) -> Optional[str]:
    """
    Send an image to the proxy for captioning.
    
    Args:
        image_path: Path to the local image file.
        
    Returns:
        Caption text, or None on failure.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Base64 encode the image for the OpenAI-compatible API
        import base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Detect content type from file extension
        ext = os.path.splitext(image_path)[1].lower()
        content_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        content_type = content_type_map.get(ext, "image/jpeg")
        
        with httpx.Client(timeout=120.0) as client:
            payload = {
                "model": IMAGE_CAPTION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": CAPTION_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            }
            
            resp = client.post(
                f"{PROXY_URL}/v1/chat/completions",
                json=payload,
            )
            
            if resp.status_code != 200:
                logger.error(
                    f"Caption request failed: HTTP {resp.status_code} — {resp.text[:200]}"
                )
                return None
            
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.error("Caption request returned no choices")
                return None
            
            content = choices[0].get("message", {}).get("content", "")
            return content.strip() if content else None
    
    except Exception as e:
        logger.error(f"Caption request failed: {e}")
        return None


def process_pending_batch() -> int:
    """
    Process a batch of pending image captions.
    
    Scans all user archives for attachments with caption_status == "pending",
    downloads images, generates captions via the proxy, and updates the JSONL.
    
    Returns:
        Number of images processed in this batch.
    """
    # Check proxy queue depth — defer if busy
    queue_depth = _get_proxy_queue_depth()
    if queue_depth < 0:
        logger.warning("Cannot reach proxy — deferring caption batch")
        return 0
    if queue_depth > 0:
        logger.info(
            f"Proxy has {queue_depth} queued request(s) — deferring caption batch"
        )
        return 0
    
    # Get pending captions
    pending = get_pending_captions(limit=IMAGE_CAPTION_BATCH_SIZE)
    
    if not pending:
        logger.info("No pending captions to process")
        return 0
    
    logger.info(f"Processing {len(pending)} pending captions")
    
    processed = 0
    # Track which users need archive rewrites
    user_updates: Dict[str, List[dict]] = {}
    
    for user_id, record, att_idx, attachment in pending:
        url = attachment.get("url", "")
        filename = attachment.get("filename", "")
        file_size = attachment.get("file_size_bytes", 0)
        
        logger.info(f"Captioning: {filename} (user {user_id})")
        
        # Download image
        image_path = _download_image(url, IMAGE_CAPTION_MAX_FILE_SIZE_MB)
        if not image_path:
            # Mark as skipped
            attachment["caption_status"] = "skipped"
            logger.info(f"Skipped {filename}: download failed or too large")
            _record_update(user_updates, user_id, record, att_idx, attachment)
            processed += 1
            continue
        
        try:
            # Generate caption
            caption = _caption_image(image_path)
            
            if caption:
                attachment["caption"] = caption
                attachment["caption_status"] = "done"
                logger.info(f"Captioned {filename}: {caption[:80]}...")
            else:
                attachment["caption_status"] = "error"
                logger.warning(f"Caption generation failed for {filename}")
            
            _record_update(user_updates, user_id, record, att_idx, attachment)
            processed += 1
        
        finally:
            # Clean up temp file
            try:
                os.unlink(image_path)
            except OSError:
                pass
    
    # Rewrite archives with updated records
    for user_id, updated_records in user_updates.items():
        rewrite_archive(user_id, updated_records)
    
    logger.info(f"Caption batch complete: {processed} images processed")
    return processed


def _record_update(
    user_updates: Dict[str, List[dict]],
    user_id: str,
    record: dict,
    att_idx: int,
    updated_attachment: dict,
) -> None:
    """
    Record an attachment update for later archive rewrite.
    
    If this is the first update for this user, load all records
    so we can rewrite the complete archive.
    """
    if user_id not in user_updates:
        user_updates[user_id] = load_all_records(user_id)
    
    records = user_updates[user_id]
    
    # Find the matching record and update the attachment
    for r in records:
        if r.get("message_id") == record.get("message_id"):
            attachments = r.get("attachments", [])
            if att_idx < len(attachments):
                attachments[att_idx] = updated_attachment
            break
