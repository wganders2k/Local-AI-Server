"""
RAG Service Client

Async HTTP client for the RAG service retrieval endpoint.
Provides graceful degradation — returns empty context if the
RAG service is unreachable rather than raising.
"""

import logging
from typing import List, Optional, Tuple

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

    async def _post(self, path: str, payload: dict, what: str) -> Tuple[str, int]:
        """
        POST to a RAG endpoint, degrading to empty rather than raising.

        Args:
            path: Endpoint path, e.g. "/search_literal".
            payload: JSON body, with None values stripped.
            what: Short description used in log messages.

        Returns:
            Tuple of (context_string, count). ("", 0) if the service is
            unreachable or returns an error.
        """
        client = await self._get_client()
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            resp = await client.post(path, json=payload)
            resp.raise_for_status()
            data = resp.json()
            context = data.get("context", "")
            count = data.get("chunk_count", 0)
            logger.info("RAG %s: %d result(s) for %s", what, count, payload)
            return context, count
        except httpx.ConnectError:
            logger.warning("RAG service unreachable (%s) — returning empty.", self.base_url)
            return "", 0
        except httpx.TimeoutException:
            logger.warning("RAG %s timed out after %ss — returning empty.", what, self.timeout)
            return "", 0
        except httpx.HTTPStatusError as e:
            logger.error("RAG %s HTTP %s — %s", what, e.response.status_code, e.response.text[:200])
            return "", 0
        except Exception:
            logger.exception("Unexpected error calling RAG %s", what)
            return "", 0

    async def search_literal(
        self,
        term: Optional[str] = None,
        author: Optional[str] = None,
        channel_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        order: str = "earliest",
        limit: int = 20,
        whole_word: bool = False,
    ) -> Tuple[str, int]:
        """
        Literal, chronologically-ordered search (no embeddings involved).

        Returns:
            Tuple of (context_string, total_matches). total_matches counts
            every match found, which may exceed the number returned.
        """
        return await self._post(
            "/search_literal",
            {
                "term": term, "author": author, "channel_name": channel_name,
                "start_date": start_date, "end_date": end_date,
                "order": order, "limit": limit, "whole_word": whole_word,
            },
            "search_literal",
        )

    async def aggregate(
        self,
        term: Optional[str] = None,
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
        Counts of matching messages grouped by author, channel or month.

        Returns:
            Tuple of (report_string, total_matching_messages).
        """
        return await self._post(
            "/aggregate",
            {
                "term": term, "author": author, "channel_name": channel_name,
                "start_date": start_date, "end_date": end_date,
                "group_by": group_by, "top_n": top_n, "whole_word": whole_word,
                "exclude_channels": exclude_channels,
            },
            "aggregate",
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
