"""
Configuration for the Mimic Discord bot.

Environment reads and tuning constants, and nothing else. Prompt text lives in
prompts/*.md (see prompt_loader.py); post-processing patterns live with the
code that applies them (formatters.py).
"""

import os
from typing import Optional
from dotenv import load_dotenv

from persona_roster import DEFAULT_ADAPTER_SCALE

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Environment Variables with Defaults
# ──────────────────────────────────────────────────────────────

DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
PROXY_URL: str = os.environ.get("PROXY_URL", "http://proxy:11436")

# Rate limiting
RATE_LIMIT_PER_USER: int = int(os.environ.get("RATE_LIMIT_PER_USER", "5"))
RATE_LIMIT_WINDOW_SECONDS: int = 60

# Backend queue depth at which a command refuses rather than queue behind it
MAX_QUEUE_DEPTH: int = int(os.environ.get("MAX_QUEUE_DEPTH", "3"))

# Conversation history
HISTORY_MAX_MESSAGES: int = 20  # A /chat thread's rolling window: 10 exchanges

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

# Thread registry: thread_id -> model, and whether a thread is chat or lore.
#
# On the bot_context volume with the lore sessions, NOT in the repo. It used to
# live in discord-bot/data/, which put runtime state inside the working tree —
# and a branch switch that crossed the commit removing the (then-tracked) file
# deleted it out from under the running container's bind mount, taking every
# thread binding with it. Runtime state does not belong anywhere git writes.
THREAD_REGISTRY_PATH: str = os.environ.get(
    "THREAD_REGISTRY_PATH", "bot_context/threads.json"
)

# ──────────────────────────────────────────────────────────────
# RAG Service Configuration
# ──────────────────────────────────────────────────────────────

RAG_SERVICE_URL: str = os.environ.get("RAG_SERVICE_URL", "http://rag-service:8001")
# The guild the archive was exported from. /lore citations link to messages
# there; unset, answers are posted without citation links.
LORE_ARCHIVE_GUILD_ID: str = os.environ.get("LORE_ARCHIVE_GUILD_ID", "")
RAG_ENABLED: bool = os.environ.get("RAG_ENABLED", "true").lower() == "true"

# ──────────────────────────────────────────────────────────────
# Agent (Agentic RAG) Configuration
# ──────────────────────────────────────────────────────────────

AGENT_MODEL: str = "brain-dense-heretic"  # Qwen3.6-27B — tool-calling capable model
AGENT_MAX_ROUNDS: int = 10        # Default max tool-call rounds before forcing a final answer
AGENT_MAX_ROUNDS_HARD_CAP: int = 25  # Hard cap for user-specified rounds in /lore
# Identical tool calls are refused rather than re-run; after this many refusals
# in one run, searching stops and the answer is written from what was found.
AGENT_MAX_REPEATED_SEARCHES: int = 2
# simulate_conversation runs per /lore run or thread turn. Each is a model swap
# away from AGENT_MODEL and back, and a re-prefill of the whole research context.
LORE_MAX_SIMULATIONS: int = 1
AGENT_TEMPERATURE: float = 0.1    # Low temperature for deterministic tool calling
AGENT_TOP_K: int = 10             # Default chunks per tool call (higher than lore's 5)

# Context-window accounting.
#
# The real limit is read from the proxy at startup — see
# lore/context_window.py, which asks /v1/models for the argv llama-server would
# launch AGENT_MODEL with and takes --ctx-size from it. Call
# context_window.limit(), never this constant.
#
# This is only the fallback for a bot that cannot reach the proxy, and it is
# also the value discovery is checked against: a mismatch is logged as a stale
# constant. Keep it equal to `ctx-size` for the AGENT_MODEL entry in models.ini
# (currently the [brain-dense-heretic] section).
#
# The number matters. A heavy /lore run has been observed holding 65,530 of
# these 96,000 tokens at round 10, before any thread follow-up accumulates.
AGENT_CTX_LIMIT_FALLBACK: int = 96000

# Fractions of the discovered context limit at which the agent changes behaviour.
#   SOFT    — start telling the model what is left, so it can wind down searching
#   COMPACT — collapse the oldest research into a summary to reclaim room,
#             both between thread turns and between rounds of a single run
#   HARD    — stop searching entirely and answer, whatever the model wants
#             (a thread checks before a turn; a run checks after compacting)
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
# prompt (member alias index, persona notes). Holds real names, so it is never
# committed — it lives on the bot_context volume beside the session store, NOT
# in prompts/, which is ordinary committed code baked into the image. See
# prompts/lore_context.example.md for the expected format. Relative paths are
# resolved against the discord-bot directory.
LORE_CONTEXT_PATH: str = os.environ.get(
    "LORE_CONTEXT_PATH", "bot_context/lore/context.md"
)

# The title on /lore's answer embeds. Set in .env to name the server; the repo
# names no server.
LORE_EMBED_TITLE: str = os.environ.get("LORE_EMBED_TITLE", "\U0001f4da Lore")


# ──────────────────────────────────────────────────────────────
# Persona Configuration
# ──────────────────────────────────────────────────────────────

# Who the personas are is not configured here: the roster (personas.json, on
# the archive disk) names them, and [mimic-multi]'s lora list in models.ini
# gives each its adapter id, read from the proxy at startup. See personas.py.

# Every persona is served from one loaded base with its adapter picked per
# request — see [mimic-multi] in models.ini — so a change of speaker never
# reloads the model.
MIMIC_MODEL: str = "mimic-multi"

# How strongly a persona's adapter is applied — see DEFAULT_ADAPTER_SCALE in
# shared/persona_roster.py, which training's sampler uses too.
MIMIC_ADAPTER_SCALE: float = float(os.environ.get("MIMIC_ADAPTER_SCALE", DEFAULT_ADAPTER_SCALE))

# How far back /mimic and /mimic-chat read a channel: enough raw messages to
# fill a persona's trained context_n merged turns after dropping bot/command
# noise. Two Discord history pages. How much of it a persona reads is its own
# training window (mimic.transcript.build_persona_prompt), not set here.
MIMIC_HISTORY_FETCH: int = 200
# Turns a persona keeps however long the silence before them, so a chat picked
# up after a break still has a little of what came before. Training cut at the
# silence, so keep it small: past a few turns the old topic outweighs the new.
MIMIC_MIN_CONTEXT_TURNS: int = 4
# In a one-on-one /mimic-chat, what the persona sees its own member called.
# Under their own name their lines would read as the persona's own, merge with
# them into one turn, and the persona would carry on the member's thought
# instead of answering it.
MIMIC_SELF_LABEL: str = "member"
# A human message starting with one of these resets what the personas see:
# from then on their windows reach back no further than that many turns,
# counting the message itself. The prefix is stripped before anything reads
# the message. Longest first, so ">>>>" is not read as ">>" plus ">>".
# ">>>" is Discord's multi-line quote, so it resets nothing (None) and stays
# as typed: quoting a passage must not wipe the personas' context.
MIMIC_RESET_PREFIXES: tuple[tuple[str, Optional[int]], ...] = ((">>>>", 1), (">>>", None), (">>", 4))

# One webhook per channel posts every persona's lines under its own name. The
# bot needs manage_webhooks in any channel /mimic or /mimic-chat is used in.
MIMIC_WEBHOOK_NAME: str = "mimic-chat"
# Appended to a persona's name on every post, so no one mistakes it for the member.
MIMIC_WEBHOOK_SUFFIX: str = " (mimic)"
# Every webhook name a persona has posted under (they follow nicknames), so its
# earlier lines still read back as the persona after a nickname change.
PERSONA_NAMES_PATH: str = os.environ.get("PERSONA_NAMES_PATH", "bot_context/persona_names.json")

# /mimic-chat. One slash-command option per persona (persona1..persona7), and
# the same cap on lore's simulate_conversation.
MIMIC_CHAT_MAX_PERSONAS: int = 7
# Bot turns in a row, with nobody else talking, before the chat waits for a
# human whatever the controller wants. Keeps one thread from holding the GPU.
MIMIC_CHAT_MAX_BURST: int = 15
# A persona following up on its own message is part of the same turn and costs
# nothing against the burst or `length`, up to this many follow-ups in a row;
# past that each one counts as a new turn.
MIMIC_CHAT_FREE_FOLLOWUPS: int = 3
# Ceiling on the `length` option (in turns), and on lore's simulate_conversation.
MIMIC_CHAT_MAX_LENGTH: int = 20
MIMIC_CHAT_DEFAULT_SIM_LENGTH: int = 8
# A human message waits this long for follow-ups before the bots answer, so
# three quick messages get one reply rather than three.
MIMIC_CHAT_DEBOUNCE_SECONDS: float = 3.0
# How much of the transcript the controller reads. It only decides who speaks,
# so it can see further back than a persona's training window.
MIMIC_CHAT_CONTROLLER_LINES: int = 60
MIMIC_CHAT_CONTROLLER_TEMPERATURE: float = 0.3
# The speaker label for the premise in a simulation nobody started in Discord.
MIMIC_SIM_SEED_NAME: str = "someone"

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


# ──────────────────────────────────────────────────────────────
# Vision (vision.py) — describing images posted to the bot
# ──────────────────────────────────────────────────────────────

# Images larger than this are not read. Discord's non-boosted upload cap is
# 25 MB; screenshots and memes are almost always under a few.
VISION_MAX_BYTES: int = 15 * 1024 * 1024
# Longest side sent to the model; the encoders resize anyway, so more is only
# a bigger request body.
VISION_MAX_SIDE: int = 1536
# Captions remembered by attachment id. A mimic chat re-reads its whole thread
# every burst, so without this each image would be captioned once per burst.
VISION_CACHE_SIZE: int = 512
# At most this many uncaptioned images are described per mimic burst. Each is
# a few seconds on the resident model; a dump of twenty screenshots should not
# hold the reply for a minute. Later bursts pick up the rest via the cache.
VISION_MAX_NEW_PER_BURST: int = 4
# Ceiling on one caption: 1-3 sentences plus any transcribed text.
VISION_MAX_TOKENS: int = 200
# Live captions for the mimic chat controller. Off until [mimic-multi] has a
# working vision projector: the 12B base's needs a newer llama.cpp than the
# image we run (see models.ini), and an image request to a preset without one
# fails. Lore captions need no switch — the agent model has its projector.
VISION_MIMIC_ENABLED: bool = os.environ.get("VISION_MIMIC_ENABLED", "false").lower() == "true"


# ──────────────────────────────────────────────────────────────
# /imagine (cogs/imagine.py)
# ──────────────────────────────────────────────────────────────

# Registers /imagine. Off until the proxy's image server is deployed
# (IMAGEGEN_ENABLED there); otherwise the command would only ever fail.
IMAGINE_ENABLED: bool = os.environ.get("IMAGINE_ENABLED", "false").lower() == "true"
# The model named in each image's footer. The proxy does not report which model
# the image server runs, so this follows the IMAGEGEN_* choice in .env by hand.
IMAGINE_MODEL_LABEL: str = os.environ.get("IMAGINE_MODEL_LABEL", "Z-Image-Turbo")
# Shapes offered, as the proxy's allowed sizes (proxy/config.py IMAGEGEN_SIZES).
# All ~1 megapixel, so all cost about the same.
IMAGINE_SHAPES: dict[str, str] = {
    "Square": "1024x1024",
    "Portrait": "832x1216",
    "Landscape": "1216x832",
}
# Discord's slash-command string cap is 6000; the proxy refuses past 2000.
IMAGINE_MAX_PROMPT_CHARS: int = 1000
# Each of /imagine's detail options (style, mood, framing). The prompt plus all
# three, labelled, stays under the proxy's 2000.
IMAGINE_MAX_DETAIL_CHARS: int = 200
# /imagine's `count`: images from one prompt in one request, each with its own
# seed. ~7 s each warm. Must not exceed the proxy's IMAGEGEN_MAX_IMAGES.
IMAGINE_MAX_IMAGES: int = 3
# /imagine's `rewrite` option (required, so users choose each time): the mimic base, no adapter,
# expands the prompt into a long concrete description
# (prompts/imagine_rewrite.md). The current image model needs the whole card,
# so the mimic base cannot stay loaded beside it (IMAGEGEN_COEXIST_MODELS is
# empty) and a rewrite costs a model swap: ~10 s on top of the image. Chosen
# over the 27B in a side-by-side on 2026-11-13: images as good, and the 12B
# loads faster.
IMAGINE_REWRITE_MODEL: str = MIMIC_MODEL
IMAGINE_REWRITE_TEMPERATURE: float = 0.7
IMAGINE_REWRITE_MAX_TOKENS: int = 300
# Past this the rewrite is not the one paragraph asked for; the original goes instead.
IMAGINE_REWRITE_MAX_CHARS: int = 1500
