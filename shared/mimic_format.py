"""
The mimic adapters' input format, shared by the code that trains them and the
code that serves them.

lora-training/build_dataset.py turns the archive into training prompts with
these rules, and discord-bot renders live channel history with the same ones.
An adapter served anything else — a server nickname, raw `<@123>` markup, one
message per turn where training merged them — is reading a format it never
saw. Keeping one copy is what stops the two drifting apart.

    messages  ->  is_conversation  ->  merge_turns  ->  context_window  ->  context_line

The persona system prompt (mimic_persona.md, beside this file) is part of the
same contract: an adapter is trained on one rendering of it and must be served
that rendering exactly, so both sides render it here.

Pure stdlib: it is copied into the bot's slim image and the torch trainer image
alike, and both test venvs import it from ../shared (see each pytest.ini).
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

_URL_RE = re.compile(r"https?://\S+")

# Links are rewritten to what they point at, never kept. Discord's own URLs are
# nothing but snowflakes (cdn.discordapp.com/attachments/<channel>/<message>/…,
# discord.com/channels/<guild>/<channel>/<message>), and most others carry one
# too (twitter.com/x/status/<id>). With ~1.5k of them in one member's context,
# the adapter learned that a long run of digits is something people write, and
# started writing its own. The domain stays: "they posted a tweet" is voice.
_DISCORD_FILE_RE = re.compile(r"https?://(?:cdn\.discordapp\.com|media\.discordapp\.net)/\S+")
_DISCORD_LINK_RE = re.compile(r"https?://(?:\w+\.)?discord(?:app)?\.com/channels/\S+")
_URL_DOMAIN_RE = re.compile(r"https?://(?:www\.)?([^/\s?#:]+)\S*")
# What the three above leave behind, so the drop checks can still see a link.
_LINK_MARKER_RE = re.compile(r"\[(?:attachment|message link|link: [^\]\s]+)\]")
# A typed run this long is an id someone pasted or keyboard-mash, never a number
# anyone reads. 15 digits is below the shortest snowflake (17).
_LONG_NUMBER_RE = re.compile(r"\d{15,}")
LONG_NUMBER = "[long number]"

_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")

# PII. Deliberately conservative patterns: a false positive costs one odd token
# in one example, a false negative puts someone's phone number in the weights.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Separators required, so a bare run of digits (a Discord id, a score) survives.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place|Cres|Crescent)\b\.?"
)

# Bot invocations are talking to a bot, not to people. They are dropped from the
# timeline entirely — as targets they teach nothing, and as context they are noise.
# The prefixes are the ones this server's bots actually use (counted from the
# archive): mostly music bots. `-`/`--` are also ordinary punctuation, so those
# count only when followed by a music-bot verb.
_COMMAND_RE = re.compile(
    r"^(?:[/!]|;;|[mtp]!)\w"
    r"|^--?(?:p|play|skip|s|leave|np|queue|q|remove|rewind|loop|vol|volume|shuffle|fs|ps)\b",
    re.IGNORECASE,
)

# The archive keeps Discord's own events (pins, thread creation, poll results)
# as messages from whoever triggered them, tagged with their type. Nobody typed
# them: "Pinned a message." was the single most common target in the first
# mimic dataset (225 of 20,887 replies). Only conversation is kept. DCE writes a
# named type by name and an unnamed one as Discord's number; records archived
# before history-service recorded a type have none, and are conversation.
_CONVERSATION_TYPES = {None, "Default", "Reply", "0", "19"}

# Defaults of build_dataset's Options, and so of every adapter whose stats.json
# predates a flag. The bot falls back to these for an unbuilt persona.
CONTEXT_N = 24
GAP_MINUTES = 120.0
MERGE_SECONDS = 600.0


# -- system prompt --

PERSONA_PROMPT_PATH = Path(__file__).with_name("mimic_persona.md")


def render_persona_prompt(persona: str, display_name: str) -> str:
    """
    The system prompt a persona is trained on and served with. str.format, and
    the file's trailing newline stripped — the rendering every adapter so far
    was trained on, so changing either (or the file) retrains every persona.
    """
    template = PERSONA_PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    return template.format(persona=persona, display_name=display_name)


# -- text --

def scrub_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    return _ADDRESS_RE.sub("[address]", text)


def replace_links(text: str) -> str:
    """Every URL -> a marker naming what it was."""
    text = _DISCORD_FILE_RE.sub("[attachment]", text)
    text = _DISCORD_LINK_RE.sub("[message link]", text)
    return _URL_DOMAIN_RE.sub(lambda m: f"[link: {m.group(1).lower()}]", text)


def normalise(text: str, usernames: Mapping[str, str], channels: Mapping[str, str]) -> str:
    """Raw Discord markup -> what a person reading the channel would see. No id survives."""
    # Links first: an email pattern would otherwise bite into a URL's userinfo.
    text = replace_links(text)
    text = _USER_MENTION_RE.sub(lambda m: "@" + usernames.get(m.group(1), "someone"), text)
    text = _CHANNEL_MENTION_RE.sub(lambda m: "#" + channels.get(m.group(1), "channel"), text)
    text = _ROLE_MENTION_RE.sub("@role", text)
    text = _CUSTOM_EMOJI_RE.sub(r":\1:", text)
    # Last: mention and emoji markup carry snowflakes that the lines above resolve.
    text = _LONG_NUMBER_RE.sub(LONG_NUMBER, text)
    return scrub_pii(text).strip()


def without_urls(text: str) -> str:
    """Raw URLs and the markers replace_links leaves in their place, for the drop checks."""
    return _LINK_MARKER_RE.sub("", _URL_RE.sub("", text)).strip()


def is_command(content: str) -> bool:
    return bool(_COMMAND_RE.match(content.strip()))


def is_conversation_type(message_type: Optional[str]) -> bool:
    return message_type in _CONVERSATION_TYPES


def drop_reason(content: str, is_bot: bool, message_type: Optional[str], has_attachments: bool) -> Optional[str]:
    """Why a message stays out of the timeline, or None if it is conversation. In build_dataset's order."""
    if is_bot:
        return "bot_message"
    if is_command(content):
        return "command"
    if not is_conversation_type(message_type):
        return "system_message"
    if not content.strip() and not has_attachments:
        return "empty"
    return None


# -- junk --
#
# Spam storms, pictures drawn in characters and copypasta are nobody's voice:
# 1,300 copies of one emoji in a single message, Braille art, a 5,000-token
# wall. As a reply they teach the adapter to post them; as context they bury
# the conversation. Measured on the archive, these two rules catch ~470 of
# ~158k messages and nothing ordinary.

_CUSTOM_EMOJI_NAME_RE = re.compile(r":\w+:")
# Squeezed before measuring, so "BUSS!!!!!!!!" or "fuck......" stay short.
_CHAR_RUN_RE = re.compile(r"(.)\1{3,}")
REPEAT_MIN_TOKENS = 8
REPEAT_SHARE = 0.5
ART_MIN_CHARS = 40
ART_MAX_LATIN = 0.3
# In context a junk message keeps this much, so the persona still sees that
# someone spammed; any other message is cut at CONTEXT_MAX_CHARS.
JUNK_STUB_CHARS = 60
CONTEXT_MAX_CHARS = 1000
_CUT = " […]"


def _latin_letter(c: str) -> bool:
    return c.isalpha() and "LATIN" in unicodedata.name(c, "")


def junk_reason(text: str) -> Optional[str]:
    """
    "repetitive": 8+ words or emoji, one of them at least half.
    "art": 40+ visible characters, under 30% Latin letters — character art,
    emoji pictures, fancy fonts, long pastes in another script.
    """
    body = _LINK_MARKER_RE.sub(" ", text)
    tokens = body.split()
    if len(tokens) >= REPEAT_MIN_TOKENS and Counter(tokens).most_common(1)[0][1] / len(tokens) >= REPEAT_SHARE:
        return "repetitive"
    chars = [c for c in _CHAR_RUN_RE.sub(r"\1\1\1", _CUSTOM_EMOJI_NAME_RE.sub(" ", body)) if not c.isspace()]
    if len(chars) >= ART_MIN_CHARS and sum(map(_latin_letter, chars)) / len(chars) < ART_MAX_LATIN:
        return "art"
    return None


def context_text(text: str) -> str:
    """A message as context shows it: junk cut to a stub, anything else to CONTEXT_MAX_CHARS."""
    limit = JUNK_STUB_CHARS if junk_reason(text) else CONTEXT_MAX_CHARS
    return text if len(text) <= limit else text[:limit].rstrip() + _CUT


# -- names --

def aliases_from(entries: Mapping) -> dict[str, str]:
    """
    {user_id: alias} from the hand-filled names file's contents. Keys starting
    with "_" are notes, and an empty alias means "keep their handle".
    """
    return {
        uid: entry["alias"].strip()
        for uid, entry in entries.items()
        if not uid.startswith("_") and isinstance(entry, dict) and (entry.get("alias") or "").strip()
    }


# -- turns --

@dataclass
class Msg:
    """One message, its text already normalised, its author already named."""
    user_id: str
    username: str
    ts: datetime
    text: str
    has_attachments: bool = False


@dataclass
class Turn:
    """Consecutive messages from one author, close enough together to be one utterance."""
    user_id: str
    username: str
    start: datetime
    end: datetime
    texts: list[str]
    attachment_only: bool  # every message in the turn was an attachment with no text

    @property
    def text(self) -> str:
        return "\n".join(t for t in self.texts if t)


def merge_turns(msgs: Iterable[Msg], merge_seconds: float = MERGE_SECONDS) -> list[Turn]:
    """Messages in time order -> turns. Same author within `merge_seconds` of their last message is one turn."""
    turns: list[Turn] = []
    for m in msgs:
        att_only = not m.text and m.has_attachments
        prev = turns[-1] if turns else None
        if (
            prev is not None
            and prev.user_id == m.user_id
            and (m.ts - prev.end).total_seconds() <= merge_seconds
        ):
            prev.texts.append(m.text)
            prev.end = m.ts
            prev.attachment_only = prev.attachment_only and att_only
        else:
            turns.append(Turn(m.user_id, m.username, m.ts, m.ts, [m.text], att_only))
    return turns


def context_window(
    prior: list[Turn],
    context_n: int = CONTEXT_N,
    gap_seconds: float = GAP_MINUTES * 60.0,
    anchor: Optional[datetime] = None,
) -> list[Turn]:
    """
    The last `context_n` of `prior` (oldest first), cut at the first silence
    longer than `gap_seconds`. `anchor` is when the reply starts: training
    passes the target turn's start, so a reply after a long silence has no
    context. Serving passes None — the reply is being asked for now, about the
    conversation that just happened, however long ago it ended.
    """
    context: list[Turn] = []
    later = anchor
    for prev in reversed(prior[-context_n:] if context_n > 0 else []):
        if later is not None and (later - prev.end).total_seconds() > gap_seconds:
            break
        context.append(prev)
        later = prev.start
    context.reverse()
    return context


def context_line(turn: Turn) -> str:
    # The only trace an attachment leaves. Never its caption — see build_dataset's docstring.
    body = "\n".join(context_text(t) for t in turn.texts if t) or "[attachment]"
    return f"[{turn.username}]: {body}"


def render_context(turns: Iterable[Turn]) -> str:
    """The user turn of a training prompt."""
    return "\n".join(context_line(t) for t in turns)
