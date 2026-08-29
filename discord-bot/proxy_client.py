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

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.1,
    ) -> dict:
        """
        Send a chat completion request with tool definitions attached.

        Returns the full assistant message dict, which may contain either
        `content` (text response) or `tool_calls` (function invocation requests).

        Args:
            model: Model alias (e.g. "brain-dense").
            messages: Conversation history including system, user, assistant, and tool messages.
            tools: List of tool schema dicts (OpenAI function-calling format).
            temperature: Sampling temperature (low for deterministic tool calling).

        Returns:
            Dict with keys: "content" (str|None), "tool_calls" (list|None).

        Raises:
            ProxyError: When the proxy is unreachable or returns an error.
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "stream": False,
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

        message = choices[0].get("message", {})
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            # prompt_tokens is how full the model's context actually was for
            # this call — the only trustworthy number for context accounting,
            # since estimating from characters drifts badly on tool-result text.
            "usage": data.get("usage"),
            # Thinking models return their chain of thought separately. The agent
            # loop feeds it back so the model can see why it made its previous
            # choices; without it, it re-derives its plan from an almost
            # unchanged context each round and reissues identical queries.
            "reasoning_content": message.get("reasoning_content"),
        }

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        usage_sink: dict | None = None,
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from the proxy using Server-Sent Events.

        Args:
            model: Model alias (e.g. "mimic_user3", "lore").
            messages: List of message dicts with "role" and "content".
            usage_sink: Optional dict updated in place with the token counts
                from the final SSE chunk (prompt_tokens, completion_tokens,
                and prompt_tokens_details.cached_tokens). A sink rather than a
                return value so this stays a plain AsyncGenerator[str] for
                callers that only want the text.
            tools: Optional tool schemas. Passing them does not invite a tool
                call so much as keep the prompt SHAPE constant: the chat
                template renders tool schemas into the prompt, so a no-tools
                request shares almost no prefix with a tools request. In a
                conversation that alternates between the two, every call is a
                cold prefill — measured at sim_best 0.161 versus 0.995 when the
                shape holds. Pass the same tools you passed the surrounding
                calls, and instruct the model in-prompt not to use them.
            enable_thinking: False suppresses the model's reasoning block via
                the chat template. The agent model is a hybrid reasoner and
                deliberates by default even on mechanical work — a research
                digest measured 6,230 completion tokens for a ~1,400-token
                answer, 184s of it reasoning. Turning it off cut an equivalent
                call from 52.2s/724 tokens to 1.0s/27 with the same output.
                Leave it on for anything requiring judgement; turn it off for
                summarising and compression. Note that reasoning_effort is only
                partially honoured by this build (14.1s at "none"), so this is
                the control that actually works.

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
        if tools:
            payload["tools"] = tools
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if usage_sink is not None:
            # Without this the streamed response carries no token counts at all.
            payload["stream_options"] = {"include_usage": True}

        # client.stream() keeps the body unread so chunks surface as the model
        # emits them. client.post() would buffer the whole response first,
        # which makes PROXY_READ_TIMEOUT a deadline on the entire generation
        # and leaves the backend generating into a dropped socket on timeout.
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProxyError(
                        f"Proxy returned error {response.status_code}: {body[:200]}",
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
                    if usage_sink is not None and chunk.get("usage"):
                        usage_sink.update(chunk["usage"])
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except httpx.ConnectError:
            raise ProxyError(
                "The AI backend is currently unreachable. Try again in a moment."
            )
        except httpx.TimeoutException:
            raise ProxyError(
                "The model stopped sending output. It may be overloaded."
            )

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

    async def _models_payload(self) -> list[dict]:
        """
        The raw ``data`` array from the proxy /v1/models endpoint.

        Returns:
            One dict per model, or an empty list on any failure — callers
            degrade rather than raise.

        The request forwards straight through the proxy to llama-server's
        router and neither loads a model nor acquires the GPU, so it is safe
        to call at startup and from autocomplete.
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
        except ValueError:
            return []

        entries = data.get("data")
        return entries if isinstance(entries, list) else []

    async def list_models(self) -> list[str]:
        """
        Query the proxy /v1/models endpoint for available model IDs.

        Returns:
            List of model name strings. Returns an empty list on failure
            so that autocomplete degrades gracefully.
        """
        return [m.get("id", "") for m in await self._models_payload() if m.get("id")]

    async def model_context_sizes(self) -> dict[str, int]:
        """
        Context window per model, as llama-server itself will run them.

        In router mode each /v1/models entry carries the full argv the server
        would launch that model with, including --ctx-size, whether or not it
        is currently loaded. Reading it here means the bot's context budget
        comes from llama-server's own parse of models.ini rather than from a
        constant somebody has to remember to update.

        Returns:
            {model_id: ctx_size}. Models whose size cannot be read are omitted
            rather than guessed at, and an unreachable proxy yields {}.
        """
        sizes: dict[str, int] = {}
        for entry in await self._models_payload():
            model_id = entry.get("id")
            args = (entry.get("status") or {}).get("args")
            if not model_id or not isinstance(args, list):
                continue
            # llama-server normalises to the long form, but accept the short
            # one too rather than silently reporting no size if that changes.
            for flag in ("--ctx-size", "-c"):
                if flag not in args:
                    continue
                try:
                    sizes[model_id] = int(args[args.index(flag) + 1])
                except (IndexError, TypeError, ValueError):
                    pass
                break
        return sizes

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
