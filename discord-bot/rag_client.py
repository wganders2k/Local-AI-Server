"""
RAG Service Client

Async HTTP client for the RAG service retrieval endpoint.
Provides graceful degradation — returns empty context if the
RAG service is unreachable rather than raising.
"""

import logging
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class RAGClient:
    """Async client for the RAG service /retrieve endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Args:
            base_url: Base URL of the RAG service (e.g. http://rag-service:8001).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialise the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        channel_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Retrieve relevant lore chunks for a query.

        Args:
            query: The user's lore question.
            top_k: Number of chunks to retrieve.
            channel_name: Optional — only search within this channel.
            start_date: Optional ISO 8601 date — only include results after this date.
            end_date: Optional ISO 8601 date — only include results before this date.

        Returns:
            Tuple of (context_string, chunk_count).
            Returns ("", 0) if the RAG service is unreachable.
        """
        client = await self._get_client()
        try:
            payload = {"query": query, "top_k": top_k}
            if channel_name:
                payload["channel_name"] = channel_name
            if start_date:
                payload["start_date"] = start_date
            if end_date:
                payload["end_date"] = end_date

            resp = await client.post(
                "/retrieve",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            context = data.get("context", "")
            chunk_count = data.get("chunk_count", 0)
            logger.info(
                f"RAG retrieval: {chunk_count} chunks for query {query!r}"
            )
            return context, chunk_count

        except httpx.ConnectError:
            logger.warning(
                "RAG service unreachable — returning empty context. "
                f"Is the rag-service container running? ({self.base_url})"
            )
            return "", 0
        except httpx.TimeoutException:
            logger.warning(
                f"RAG service timed out after {self.timeout}s — returning empty context."
            )
            return "", 0
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG service HTTP error: {e.response.status_code} — {e.response.text}")
            return "", 0
        except Exception:
            logger.exception("Unexpected error calling RAG service")
            return "", 0

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
