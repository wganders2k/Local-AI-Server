"""
Async HTTP client for the orchestration proxy.

Wraps httpx to send chat completion requests to the proxy's
OpenAI-compatible API endpoint (/v1/chat/completions).
"""

import json
from typing import AsyncGenerator

import httpx
from config import (
    PROXY_URL,
    PROXY_CONNECTION_TIMEOUT,
    PROXY_READ_TIMEOUT,
    PROXY_TOTAL_TIMEOUT,
)


class ProxyError(Exception):
    """Raised when the proxy returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ProxyClient:
    """
    Async HTTP client for the FastAPI orchestration proxy.

    Uses httpx with configurable timeouts. Connection timeout is short
    (proxy is local), read timeout covers worst-case swap + inference.
    """

    def __init__(self, proxy_url: str = PROXY_URL):
        self.proxy_url = proxy_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.proxy_url,
                timeout=httpx.Timeout(
                    connect=PROXY_CONNECTION_TIMEOUT,
                    read=PROXY_READ_TIMEOUT,
                    timeout=PROXY_TOTAL_TIMEOUT,
                ),
            )
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
    ) -> str:
        """
        Send a chat completion request to the proxy.

        Args:
            model: Model alias (e.g. "mimic_user3", "lore").
            messages: List of message dicts with "role" and "content".
            stream: Enable streaming response (Phase 2+).

        Returns:
            The assistant's response text.

        Raises:
            ProxyError: When the proxy is unreachable or returns an error.
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        try:
            response = await client.post(
                "/v1/chat/completions",
                json=payload,
            )
        except httpx.ConnectError:
            raise ProxyError(
                "The AI backend is currently unreachable. Try again in a moment."
            )
        except httpx.TimeoutException:
            raise ProxyError(
                "Request timed out. The model may be busy."
            )

        if response.status_code != 200:
            raise ProxyError(
                f"Proxy returned error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            raise ProxyError("The model returned an empty response.")

        content = choices[0].get("message", {}).get("content", "").strip()

        if not content:
            raise ProxyError("The model returned an empty response.")

        return content

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from the proxy using Server-Sent Events.

        Args:
            model: Model alias (e.g. "mimic_user3", "lore").
            messages: List of message dicts with "role" and "content".

        Yields:
            Non-empty content strings from each SSE delta chunk.

        Raises:
            ProxyError: When the proxy is unreachable or returns an error.
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        try:
            response = await client.post(
                "/v1/chat/completions",
                json=payload,
            )
        except httpx.ConnectError:
            raise ProxyError(
                "The AI backend is currently unreachable. Try again in a moment."
            )
        except httpx.TimeoutException:
            raise ProxyError(
                "Request timed out. The model may be busy."
            )

        if response.status_code != 200:
            raise ProxyError(
                f"Proxy returned error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    async def get_queue_depth(self) -> int:
        """
        Query the proxy /status endpoint for current queue depth.

        Returns:
            Current queue depth (number of pending requests).

        Raises:
            ProxyError: When the proxy is unreachable.
        """
        client = await self._get_client()
        try:
            response = await client.get("/status")
        except httpx.ConnectError:
            raise ProxyError("The AI backend is currently unreachable.")
        except httpx.TimeoutException:
            raise ProxyError("Request to backend timed out.")

        if response.status_code != 200:
            raise ProxyError(
                f"Proxy /status returned error {response.status_code}",
                status_code=response.status_code,
            )

        data = response.json()
        return data.get("queue_depth", 0)

    async def list_models(self) -> list[str]:
        """
        Query the proxy /v1/models endpoint for available model IDs.

        Returns:
            List of model name strings. Returns an empty list on failure
            so that autocomplete degrades gracefully.
        """
        client = await self._get_client()
        try:
            response = await client.get("/v1/models")
        except (httpx.ConnectError, httpx.TimeoutException):
            return []

        if response.status_code != 200:
            return []

        try:
            data = response.json()
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except (ValueError, KeyError):
            return []

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
