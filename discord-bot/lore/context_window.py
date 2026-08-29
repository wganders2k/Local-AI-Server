"""
How much context the agent model actually has.

The number drives every budget decision a lore thread makes — when to stop
searching, when to compact, what the footer reports — so getting it wrong is
not cosmetic. It used to be a constant in config.py that had to be kept in step
with `ctx-size` in models.ini by hand, with nothing checking the pairing.

It is now read from the proxy's /v1/models, where each entry carries the argv
llama-server would launch that model with. That is llama-server's own parse of
models.ini, so it cannot disagree with what the server will actually run, and it
is readable while the model is unloaded.

Discovery is an explicit async step at startup rather than a lazy load, because
the readers are sync — AgentMetrics.context_pct is a property, and there is no
awaiting from inside one. The pattern otherwise mirrors
lore.prompts.load_lore_context: cache in a module global, degrade to a documented
fallback, and say so in the log.
"""

import logging
from typing import Optional

from config import AGENT_CTX_LIMIT_FALLBACK, AGENT_MODEL

logger = logging.getLogger("mimic-bot.lore.context-window")

_limit: Optional[int] = None


def limit() -> int:
    """
    The agent model's context window in tokens.

    Falls back to AGENT_CTX_LIMIT_FALLBACK until discover() has succeeded, so
    this is always safe to call — a bot that cannot reach the proxy still runs,
    just on the last known-good number.
    """
    return _limit if _limit is not None else AGENT_CTX_LIMIT_FALLBACK


def describe() -> str:
    """One-line provenance for logs and diagnostics."""
    if _limit is None:
        return f"{AGENT_CTX_LIMIT_FALLBACK:,} tokens (fallback — not yet discovered)"
    return f"{_limit:,} tokens (proxy /v1/models)"


def discovered() -> bool:
    """Whether a real value has been read from the proxy."""
    return _limit is not None


async def discover(proxy_client, model: str = AGENT_MODEL) -> bool:
    """
    Read the model's context window from the proxy and cache it.

    Never raises: the proxy client degrades to an empty mapping when it cannot
    reach the backend, and every failure here leaves the fallback in place.

    Returns:
        True if a value was discovered and cached.
    """
    global _limit

    sizes = await proxy_client.model_context_sizes()
    if not sizes:
        logger.warning(
            "Could not read context sizes from the proxy — using %s. "
            "Will retry on the next /lore.",
            describe(),
        )
        return False

    size = sizes.get(model)
    if size is None:
        # A rename or typo in models.ini. /lore would fail on its first call
        # anyway, so say so now rather than leaving it to look like a backend
        # error later.
        logger.warning(
            "Model %r is not served by the backend (available: %s) — context "
            "budget falls back to %s, and /lore will fail until this is fixed.",
            model, ", ".join(sorted(sizes)), describe(),
        )
        return False

    _limit = size
    if size != AGENT_CTX_LIMIT_FALLBACK:
        # This is the check the old "keep these two in step" comment asked a
        # human to perform. Nothing breaks — the discovered value is the one in
        # use — but the constant is stale and should be updated.
        logger.warning(
            "Context window for %s is %s, but AGENT_CTX_LIMIT_FALLBACK in "
            "config.py says %s. Using the discovered value; update the constant.",
            model, f"{size:,}", f"{AGENT_CTX_LIMIT_FALLBACK:,}",
        )
    else:
        logger.info("Context window for %s: %s", model, describe())
    return True


async def ensure_discovered(proxy_client, model: str = AGENT_MODEL) -> None:
    """
    Discover the limit if startup could not.

    Cheap enough to sit at the head of every /lore turn: once a value is cached
    this returns without touching the network.
    """
    if _limit is None:
        await discover(proxy_client, model)


def reset() -> None:
    """Drop the cached value. Tests only."""
    global _limit
    _limit = None
