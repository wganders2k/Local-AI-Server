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
PROXY_CONNECTION_TIMEOUT: int = 10
PROXY_READ_TIMEOUT: int = 60
PROXY_TOTAL_TIMEOUT: int = 70

# Thread registry
THREAD_REGISTRY_PATH: str = os.environ.get("THREAD_REGISTRY_PATH", "data/threads.json")

# ──────────────────────────────────────────────────────────────
# RAG Service Configuration
# ──────────────────────────────────────────────────────────────

RAG_SERVICE_URL: str = os.environ.get("RAG_SERVICE_URL", "http://rag-service:8001")
LORE_TOP_K: int = int(os.environ.get("LORE_TOP_K", "5"))
RAG_ENABLED: bool = os.environ.get("RAG_ENABLED", "true").lower() == "true"


# ──────────────────────────────────────────────────────────────
# Persona Configuration
# ──────────────────────────────────────────────────────────────

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

LORE_SYSTEM_PROMPT: str = (
    "You are the nullposting lore assistant. You have access to a curated database of "
    "server history, in-jokes, memes, and member events. When answering questions, "
    "cite your sources from the retrieved context. Be factual and concise. "
    "If the retrieved context does not contain the answer, say so clearly rather than "
    "guessing. Never invent lore, events, or quotes."
)

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
