"""
Personas in Discord: posting as one, and reading their posts back.

Every persona line goes out through one webhook per channel, under the
mimicked member's current server nickname and avatar, suffixed so nobody
mistakes it for the member. That makes a multi-persona thread readable, and it
is what lets HistoryReader read a persona's earlier lines back as that persona
rather than dropping them as bot noise.

The webhook name follows the member's nickname, so every name a persona has
posted under is recorded (PersonaNameRegistry): without that, a nickname change
would turn its earlier lines into an unknown webhook's, and they would drop out
of the conversation.

Needs manage_webhooks and read_message_history in the channel. A thread posts
through its parent's webhook with `thread=`.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

import discord

import mimic_format
import vision
from config import MIMIC_HISTORY_FETCH, MIMIC_WEBHOOK_NAME, MIMIC_WEBHOOK_SUFFIX
from formatters import split_message, strip_disclaimers
from mimic.transcript import Cast, Line, strip_reset
from mimic_format import Msg
from personas import Persona, PersonaCatalog, member_name

logger = logging.getLogger("mimic-bot.persona-posting")

Postable = Union[discord.TextChannel, discord.Thread]

# Persona text is model output: it must never ping anyone, whatever it writes.
_NO_MENTIONS = discord.AllowedMentions.none()
_WEBHOOK_NAME_MAX = 80  # Discord's limit on a webhook post's username


# ---- names ------------------------------------------------------------------

def persona_member(persona: Persona, guild: Optional[discord.Guild]) -> Optional[discord.Member]:
    """The guild member a persona mimics, from the member cache, if they are still in the server."""
    if guild is None:
        return None
    for uid in persona.user_ids:
        member = guild.get_member(int(uid)) if uid.isdigit() else None
        if member is not None:
            return member
    return None


def webhook_username(persona: Persona, guild: Optional[discord.Guild] = None) -> str:
    """
    The name a persona posts under: its member's current server nickname, or
    with no guild or member to look up, its display name. Always suffixed.
    """
    member = persona_member(persona, guild)
    name = member.display_name if member else persona.display_name
    return f"{name}{MIMIC_WEBHOOK_SUFFIX}"[-_WEBHOOK_NAME_MAX:]


def webhook_avatar(persona: Persona, guild: Optional[discord.Guild] = None) -> Optional[str]:
    """Its member's current server avatar (or account avatar), or None for the webhook's own."""
    member = persona_member(persona, guild)
    return member.display_avatar.url if member else None


def without_suffix(name: str) -> str:
    return name[: -len(MIMIC_WEBHOOK_SUFFIX)] if name.endswith(MIMIC_WEBHOOK_SUFFIX) else name


class PersonaNameRegistry:
    """JSON-backed {webhook username: persona}. A name reused by another persona goes to the latest."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self.names: dict[str, str] = {}
        if self._path.exists():
            try:
                self.names = dict(json.loads(self._path.read_text()))
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                logger.warning("Could not read %s (%s); starting fresh", self._path, e)

    def record(self, name: str, persona: str) -> None:
        """Remember that `persona` posted as `name`. Writes only when that is news."""
        if self.names.get(name) == persona:
            return
        self.names[name] = persona
        self._save()
        logger.info("Recorded persona name %r -> %s", name, persona)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.names, indent=2, ensure_ascii=False))
            os.replace(tmp, self._path)
        except OSError as e:
            logger.error("Failed to save %s: %s", self._path, e)


# ---- reading ----------------------------------------------------------------

class _ChannelNames:
    """The `channels` mapping normalise wants, looked up on demand rather than built for the whole guild."""

    def __init__(self, guild: Optional[discord.Guild]):
        self.guild = guild

    def get(self, channel_id: str, default: str) -> str:
        channel = self.guild.get_channel_or_thread(int(channel_id)) if self.guild else None
        return getattr(channel, "name", None) or default


class HistoryReader:
    """
    Turns discord.Messages into transcript Lines, and knows the Cast they were
    posted by. Built per request: the roster, names.json and the stats files
    are read fresh each time.
    """

    def __init__(
        self,
        catalog: PersonaCatalog,
        guild: Optional[discord.Guild] = None,
        handle_for_id: Callable[[str], Optional[str]] = lambda uid: None,
        known_names: Mapping[str, str] = {},
    ):
        self.guild = guild
        self.aliases = catalog.aliases()
        self.channels = _ChannelNames(guild)
        personas = {p.key: p for p in catalog.all()}
        # Posts go out under the member's current nickname; earlier ones went
        # out under older nicknames (known_names, from PersonaNameRegistry) or
        # the display name. Any of them reads back as the persona; current wins.
        self.by_webhook_name: dict[str, str] = {n: p for n, p in known_names.items() if p in personas}
        self.by_webhook_name.update({webhook_username(p): k for k, p in personas.items()})
        if guild is not None:
            self.by_webhook_name.update({webhook_username(p, guild): k for k, p in personas.items()})
        older: dict[str, set[str]] = {}
        for name, key in self.by_webhook_name.items():
            older.setdefault(key, set()).add(without_suffix(name))
        self.cast = Cast.build(
            catalog, self.aliases, handle_for_id,
            shown_as={k: without_suffix(webhook_username(p, guild)) for k, p in personas.items()},
            other_names=older,
        )

    def _replied_persona(self, message: discord.Message) -> Optional[str]:
        """The persona whose post `message` replies to, when Discord resolved the reference."""
        ref = getattr(message, "reference", None)
        target = getattr(ref, "resolved", None)
        if getattr(target, "webhook_id", None) is None:
            return None  # not a reply, a reply to a person, or the original is gone
        return self.by_webhook_name.get(target.author.name)

    def to_line(self, message: discord.Message) -> Optional[Line]:
        content = message.content or ""
        persona, reset = None, None
        if message.webhook_id is not None:
            persona = self.by_webhook_name.get(message.author.name)
            if persona is None:
                return None  # someone else's webhook: a bot, as far as training knew
            speaker = self.cast.speakers[persona]
            user_id, label = speaker.user_id, speaker.label
        else:
            # Only a person can reset; a persona's ">>" is just text.
            content, reset = strip_reset(content)
            if mimic_format.drop_reason(
                content, message.author.bot, str(message.type.value), bool(message.attachments)
            ):
                return None
            user_id = str(message.author.id)
            label = member_name(user_id, message.author.name, self.aliases)
        usernames = {str(u.id): member_name(u.id, u.name, self.aliases) for u in message.mentions}
        msg = Msg(
            user_id=user_id,
            username=label,
            ts=message.created_at,
            text=mimic_format.normalise(content, usernames, self.channels),
            has_attachments=bool(message.attachments),
        )
        images = tuple(a for a in message.attachments if vision.is_image(a))
        return Line(msg, persona, None if persona else self._replied_persona(message), images=images, reset=reset)

    async def fetch_lines(self, channel: discord.abc.Messageable, limit: int = MIMIC_HISTORY_FETCH) -> list[Line]:
        """The channel's recent conversation, oldest first."""
        raw = [m async for m in channel.history(limit=limit)]
        return [line for line in (self.to_line(m) for m in reversed(raw)) if line is not None]

    async def fetch(self, channel: discord.abc.Messageable, limit: int = MIMIC_HISTORY_FETCH) -> list[Msg]:
        return [line.msg for line in await self.fetch_lines(channel, limit)]


# ---- posting ----------------------------------------------------------------

class WebhookPoster:
    """One cached webhook per parent channel, created on first use."""

    def __init__(self, catalog: PersonaCatalog, names: PersonaNameRegistry):
        self._hooks: dict[int, discord.Webhook] = {}
        self.catalog = catalog
        self.names = names

    def reader(
        self, guild: Optional[discord.Guild], handle_for_id: Callable[[str], Optional[str]] = lambda uid: None
    ) -> HistoryReader:
        """A HistoryReader that knows every name these personas have posted under."""
        return HistoryReader(self.catalog, guild, handle_for_id, self.names.names)

    def remember_current(self, guild: discord.Guild) -> None:
        """Record every persona's name as it stands, so posts made under it survive a later nickname change."""
        for persona in self.catalog.all():
            self.names.record(webhook_username(persona, guild), persona.key)

    async def _webhook(self, parent: discord.TextChannel) -> discord.Webhook:
        hook = self._hooks.get(parent.id)
        if hook is not None:
            return hook
        # Only a webhook this bot created comes back with a token it can send with.
        for existing in await parent.webhooks():
            if existing.name == MIMIC_WEBHOOK_NAME and existing.token:
                hook = existing
                break
        else:
            hook = await parent.create_webhook(name=MIMIC_WEBHOOK_NAME, reason="mimic persona posts")
            logger.info("Created %s webhook in #%s", MIMIC_WEBHOOK_NAME, parent.name)
        self._hooks[parent.id] = hook
        return hook

    async def post(self, key: str, text: str, channel: Postable) -> list[discord.WebhookMessage]:
        """Post `text` as persona `key`, split to Discord's limit. Returns what was sent."""
        persona = self.catalog.get(key)
        if persona is None:
            raise KeyError(f"{key} is not an available persona")
        parent = channel.parent if isinstance(channel, discord.Thread) else channel
        kwargs = {
            "username": webhook_username(persona, channel.guild),
            "avatar_url": webhook_avatar(persona, channel.guild),
            "allowed_mentions": _NO_MENTIONS,
            "wait": True,
        }
        if isinstance(channel, discord.Thread):
            kwargs["thread"] = channel
        self.names.record(kwargs["username"], key)
        chunks = split_message(strip_disclaimers(text))
        sent: list[discord.WebhookMessage] = []
        for attempt in (1, 2):
            hook = await self._webhook(parent)
            try:
                for chunk in chunks[len(sent):]:
                    sent.append(await hook.send(chunk, **kwargs))
                return sent
            except discord.NotFound:
                # Deleted from the channel settings since it was cached: make a new one, once.
                self._hooks.pop(parent.id, None)
                if attempt == 2:
                    raise
        return sent
