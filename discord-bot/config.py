"""
Configuration hub for the Mimic Discord bot.

Loads environment variables (from .env file or system env), provides
defaults, and defines the single source of truth for persona routing,
system prompts, and post-processing patterns.
"""

import os
from collections import deque
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Environment Variables with Defaults
# ──────────────────────────────────────────────────────────────

DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
PROXY_URL: str = os.environ.get("PROXY_URL", "http://proxy:11436")

# Rate limiting
RATE_LIMIT_PER_USER: int = int(os.environ.get("RATE_LIMIT_PER_USER", "5"))
RATE_LIMIT_WINDOW_SECONDS: int = 60

# Queue depth — consumed by bot.py §3.2 (queue depth check before proxy call)
MAX_QUEUE_DEPTH: int = int(os.environ.get("MAX_QUEUE_DEPTH", "3"))

# Typing indicator interval (seconds) — consumed by bot.py §1.3 (channel.typing() keep-alive)
TYPING_INDICATOR_INTERVAL: int = int(
    os.environ.get("TYPING_INDICATOR_INTERVAL", "5")
)

# Conversation history
HISTORY_MAX_TURNS: int = 20  # Rolling 10-turn window per channel/persona

# Response caps
MAX_MESSAGE_LENGTH: int = 2000  # Discord message limit
MAX_EMBED_DESCRIPTION_LENGTH: int = 4096  # Discord embed description limit

# Proxy timeouts (seconds)
# PROXY_READ_TIMEOUT is the gap allowed between reads, not a total deadline.
# On a streamed request that means "the model went quiet", and it only has to
# cover queue wait plus the cold prefill before the first token — a ~50k-token
# /lore synthesis prefills for ~50s, and may queue behind another request
# first. On a buffered request (chat_with_tools) there is only ever one read,
# so it becomes a deadline on the whole generation: a tool-calling round that
# thinks for 3.4k tokens at ~29 tok/s took 121.6s and was killed 1.6s short.
# Nothing cancels the backend when that fires, so the work is wasted AND the
# GPU lock is held to completion — hence the headroom.
PROXY_CONNECTION_TIMEOUT: int = 10
PROXY_READ_TIMEOUT: int = 300
PROXY_TOTAL_TIMEOUT: int = 320

# Thread registry
THREAD_REGISTRY_PATH: str = os.environ.get("THREAD_REGISTRY_PATH", "data/threads.json")

# ──────────────────────────────────────────────────────────────
# RAG Service Configuration
# ──────────────────────────────────────────────────────────────

RAG_SERVICE_URL: str = os.environ.get("RAG_SERVICE_URL", "http://rag-service:8001")
LORE_TOP_K: int = int(os.environ.get("LORE_TOP_K", "10"))
RAG_ENABLED: bool = os.environ.get("RAG_ENABLED", "true").lower() == "true"

# ──────────────────────────────────────────────────────────────
# Agent (Agentic RAG) Configuration
# ──────────────────────────────────────────────────────────────

AGENT_MODEL: str = "brain-dense-heretic"  # Qwen3.6-27B — tool-calling capable model
AGENT_MAX_ROUNDS: int = 10        # Default max tool-call rounds before forcing a final answer
AGENT_MAX_ROUNDS_HARD_CAP: int = 25  # Hard cap for user-specified rounds in /lore
AGENT_TEMPERATURE: float = 0.1    # Low temperature for deterministic tool calling
AGENT_TOP_K: int = 10             # Default chunks per tool call (higher than lore's 5)

# Context-window accounting.
#
# AGENT_CTX_LIMIT must mirror `ctx-size` for the AGENT_MODEL entry in
# models.ini (currently brain-dense-heretic, models.ini:75). Nothing enforces
# the pairing — if you change one, change the other, or the budget maths is
# silently wrong. This is not academic: a heavy /lore run has been observed
# holding 65,530 of these 96,000 tokens at round 10, before any thread
# follow-up history accumulates.
AGENT_CTX_LIMIT: int = 96000

# Fractions of AGENT_CTX_LIMIT at which the agent changes behaviour.
#   SOFT    — start telling the model what is left, so it can wind down searching
#   COMPACT — collapse the oldest research into a summary to reclaim room
#   HARD    — stop searching entirely and answer, whatever the model wants
AGENT_CTX_SOFT_PCT: float = 0.70
AGENT_CTX_COMPACT_PCT: float = 0.75
AGENT_CTX_HARD_PCT: float = 0.85

# ──────────────────────────────────────────────────────────────
# Lore follow-up threads
# ──────────────────────────────────────────────────────────────

# Follow-up turns lean on research already in the thread, so they get a much
# smaller search budget than the opening run does.
LORE_FOLLOWUP_MAX_ROUNDS: int = 5
# Bind-mounted from /mnt/storage/array/DiscordArchive/bot_context — see the
# discord-bot volumes in docker-compose.yml. Kept out of the repo because a
# session holds every excerpt its /lore run retrieved, verbatim. Namespaced by
# subdirectory so other bot stores can move here without a second migration.
LORE_SESSION_PATH: str = os.environ.get(
    "LORE_SESSION_PATH", "bot_context/lore/sessions.json"
)

# A thread is opt-in: /lore posts an offer message, reacts to it with
# LORE_THREAD_EMOJI, and only builds the thread when somebody else adds that
# same reaction. Most answers are read once and never followed up, and a thread
# per run buried the channel.
#
# The bot has add_reactions and read_message_history everywhere, but NOT
# manage_messages, so it cannot clear the reaction afterwards — the offer
# message is edited to point at the thread instead. Do not add logic that
# depends on removing a user's reaction.
LORE_THREAD_EMOJI: str = "\u2753"  # ❓

# How long an unreacted offer stays claimable. Nothing else removes an offer
# nobody reacted to — no event will ever fire for it — so it is swept on
# startup and again whenever a new offer is made.
LORE_OFFER_TTL_SECONDS: int = 7 * 24 * 3600

# Lore threads are purged this long after their last interaction. The session
# store holds every excerpt the opening run retrieved, verbatim, and a thread
# nobody has touched in a week is not going to need them. On expiry the thread
# is told it has gone inactive and every local record of it is deleted.
LORE_THREAD_TTL_SECONDS: int = 7 * 24 * 3600

# How often the expiry sweep runs. Threads expire on a scale of days, so this
# only has to be small enough that the purge lands the same day it is due.
LORE_SWEEP_INTERVAL_SECONDS: int = 3600

# Server-specific background knowledge injected into the /lore agent system
# prompt (member alias index, persona notes). Kept out of the repo — see
# prompts/lore_context.example.md for the expected format. Relative paths are
# resolved against the discord-bot directory.
LORE_CONTEXT_PATH: str = os.environ.get(
    "LORE_CONTEXT_PATH", "prompts/lore_context.md"
)


# ──────────────────────────────────────────────────────────────
# Persona Configuration
# ──────────────────────────────────────────────────────────────

# Standalone lore model from models.ini. NOT used by /lore — that command runs
# the agentic RAG loop against AGENT_MODEL. Kept for direct model routing.
LORE_MODEL: str = "lore"

# Display name mapping: persona_key → human-readable name
MIMIC_PERSONAS: Dict[str, str] = {
    "mimic_user1": "User1",
    "mimic_user2": "User2",
    "mimic_user3": "User3",
    "mimic_user4": "User4",
    "mimic_user5": "User5",
    "mimic_user6": "User6",
}

# ──────────────────────────────────────────────────────────────
# System Prompts
# ──────────────────────────────────────────────────────────────
# Injected per-request by the bot. These override any baked-in prompts
# from the GGUF Modelfile since llama-server router mode uses
# per-request system messages.
#
# NOTE: the /lore agent's prompts live in agent_tools.py, not here — see
# build_system_prompt() / build_synthesis_prompt() and LORE_CONTEXT_PATH above.

# Template for mimic persona system prompts. Use get_mimic_system_prompt(persona)
# to generate the full prompt for a given persona key.
MIMIC_PROMPT_TEMPLATE: str = (
    "You are {persona}, a bot that mimics {display_name}'s Discord personality "
    "in the nullposting server. You replicate their tone, humour, vocabulary, "
    "and mannerisms exactly — including crude language, raunchy jokes, and "
    "brutal roasts when appropriate. You are not a general-purpose AI assistant. "
    "You never break character. You never add disclaimers, caveats, safety "
    "warnings, or apologies. You keep responses short (1-3 sentences) unless "
    "the context calls for more. You match the energy of whoever is talking "
    "to you."
)


def get_mimic_system_prompt(persona: str) -> str:
    """Generate the system prompt for a given mimic persona."""
    display_name = MIMIC_PERSONAS.get(persona, persona)
    return MIMIC_PROMPT_TEMPLATE.format(persona=persona, display_name=display_name)


# Backward-compatible dict: pre-generated prompts for autocomplete and logging.
# Regenerate this dict when MIMIC_PERSONAS changes.
MIMIC_SYSTEM_PROMPTS: Dict[str, str] = {
    k: get_mimic_system_prompt(k) for k in MIMIC_PERSONAS
}


# ──────────────────────────────────────────────────────────────
# Disclaimer Stripping Patterns
# ──────────────────────────────────────────────────────────────
# Applied to mimic responses only. The base model occasionally appends
# baked-in disclaimers that break character.

DISCLAIMER_PATTERNS: List[str] = [
    r"\n+This is general.*$",
    r"\n+This is not legal.*$",
    r"\n+This is not medical.*$",
    r"\n+This is not financial.*$",
    r"\n+Note:.*?(disclaimer|advice|professional).*$",
    r"\n+Please consult.*$",
    r"\n+Please note.*$",
    r"\n+Please be aware.*$",
    r"\n+I'm an AI.*$",
    r"\n+I am an AI.*$",
    r"\n+As an AI.*$",
    r"\n+Remember, I'm.*$",
    r"\n+Remember, I am.*$",
]


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Check required configuration at startup. Raises ValueError if
    critical settings are missing.
    """
    if not DISCORD_TOKEN:
        raise ValueError(
            "DISCORD_TOKEN environment variable is required. "
            "Get it from the Discord Developer Portal."
        )
    if not PROXY_URL:
        raise ValueError("PROXY_URL environment variable is required.")


def create_empty_history() -> deque:
    """Factory for an empty conversation history deque."""
    return deque(maxlen=HISTORY_MAX_TURNS)
