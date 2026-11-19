"""
How much context a /lore run actually has.

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

Two questions, two functions:

* ``window(model)`` — how much one model can take. Every check made right
  before a request is sent uses the window of the model that request goes to:
  research rounds are budgeted against the research model's, and research
  handed to a different writer is compacted to fit the writer's first.
* ``limit(speed, writer)`` — how big a thread's transcript may grow between
  turns. Both the research model and the writer receive the whole transcript on
  the next turn, so it is the smaller of their two windows.
"""

import logging
from typing import Optional

from config import AGENT_CTX_LIMIT_FALLBACK, LORE_DEFAULT_WRITER, LORE_RESEARCH_MODELS
from lore.models import all_models, research_model, writer_model

logger = logging.getLogger("mimic-bot.lore.context-window")

# {model: ctx_size} for the models a run can use, once read from the proxy.
_sizes: Optional[dict[str, int]] = None


def window(model: str) -> int:
    """
    The context window in tokens of one model.

    AGENT_CTX_LIMIT_FALLBACK until discover() has succeeded, and for a model
    the backend does not serve — the request to it fails on its own and says
    why.
    """
    if _sizes is None:
        return AGENT_CTX_LIMIT_FALLBACK
    return _sizes.get(model, AGENT_CTX_LIMIT_FALLBACK)


def limit(speed: str, writer: str = LORE_DEFAULT_WRITER) -> int:
    """
    The context a thread's transcript may fill between turns: the smaller
    window of its research model and its writer.

    Falls back to AGENT_CTX_LIMIT_FALLBACK until discover() has succeeded, so
    this is always safe to call — a bot that cannot reach the proxy still runs,
    just on the last known-good number.
    """
    return min(window(research_model(speed)), window(writer_model(speed, writer)))


def describe() -> str:
    """One-line provenance for logs and diagnostics."""
    if _sizes is None:
        return f"{AGENT_CTX_LIMIT_FALLBACK:,} tokens (fallback — not yet discovered)"
    sizes = ", ".join(f"{m} {n:,}" for m, n in _sizes.items())
    return f"{sizes} tokens (proxy /v1/models)"


def discovered() -> bool:
    """Whether real values have been read from the proxy."""
    return _sizes is not None


async def discover(proxy_client) -> bool:
    """
    Read the lore models' context windows from the proxy and cache them.

    Never raises: the proxy client degrades to an empty mapping when it cannot
    reach the backend, and every failure here leaves the fallback in place.

    Returns:
        True if every research model's window was discovered and cached.
    """
    global _sizes

    sizes = await proxy_client.model_context_sizes()
    if not sizes:
        logger.warning(
            "Could not read context sizes from the proxy — using %s. "
            "Will retry on the next /lore.",
            describe(),
        )
        return False

    missing = [m for m in LORE_RESEARCH_MODELS.values() if m not in sizes]
    if missing:
        # A rename or typo in models.ini. /lore would fail on its first call
        # anyway, so say so now rather than leaving it to look like a backend
        # error later.
        logger.warning(
            "Research model(s) %s not served by the backend (available: %s) — "
            "context budget falls back to %s, and /lore will fail until this is fixed.",
            ", ".join(missing), ", ".join(sorted(sizes)), describe(),
        )
        return False
    for m in all_models():
        if m not in sizes:
            logger.warning(
                "Writer model %r is not served by the backend — choosing it as "
                "/lore's writer will fail until models.ini has it.", m,
            )

    _sizes = {m: sizes[m] for m in all_models() if m in sizes}
    smallest = min(_sizes.values())
    if smallest != AGENT_CTX_LIMIT_FALLBACK:
        # Nothing breaks — the discovered values are the ones in use — but the
        # constant is stale and should be updated.
        logger.warning(
            "Smallest lore context window is %s, but AGENT_CTX_LIMIT_FALLBACK in "
            "config.py says %s. Using the discovered values; update the constant.",
            f"{smallest:,}", f"{AGENT_CTX_LIMIT_FALLBACK:,}",
        )
    logger.info("Lore context windows: %s", describe())
    return True


async def ensure_discovered(proxy_client) -> None:
    """
    Discover the limits if startup could not.

    Cheap enough to sit at the head of every /lore turn: once values are cached
    this returns without touching the network.
    """
    if _sizes is None:
        await discover(proxy_client)


def reset() -> None:
    """Drop the cached values. Tests only."""
    global _sizes
    _sizes = None
