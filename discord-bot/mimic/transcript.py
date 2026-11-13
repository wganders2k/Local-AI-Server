"""
A conversation as the persona adapters read it, and the prompts built from it.

An adapter learned to reply to one user turn holding merged `[name]: text`
turns, with bots, commands and system messages gone
(lora-training/build_dataset.py). Everything here builds exactly that, with the
same code — shared/mimic_format.py — and the persona's own build arguments
(personas.Persona).

A persona's own lines carry the id and label of the member it mimics, which is
how the member's lines appeared in their training context. Line keeps which
persona actually posted, for the code that needs to tell them apart.

No discord here: persona_posting.HistoryReader turns Discord messages into Lines
and builds the Cast, and lore's simulations build a Cast with no guild at all.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Optional

import mimic_format
from config import MIMIC_MIN_CONTEXT_TURNS, MIMIC_RESET_PREFIXES, MIMIC_SELF_LABEL
from mimic_format import Msg
from personas import Persona, PersonaCatalog, member_name

# The base model, unadapted, continues a transcript by writing the next
# speaker's label and sometimes the line after. An adapter rarely does, but one
# stray label in a webhook post reads as the persona speaking for someone else.
_LEADING_LABEL_RE = re.compile(r"^\[[^\]\n]{1,40}\]:\s*")
_NEXT_SPEAKER_RE = re.compile(r"\n\[[^\]\n]{1,40}\]:")


def clean_persona_reply(text: str) -> str:
    """One line from one speaker: no label in front, nothing after a second speaker's label."""
    text = _LEADING_LABEL_RE.sub("", text.strip())
    return _NEXT_SPEAKER_RE.split(text, maxsplit=1)[0].strip()


def strip_reset(content: str) -> tuple[str, Optional[int]]:
    """A human message's text without its reset prefix, and the turns that prefix keeps (None: no prefix)."""
    for prefix, keep in MIMIC_RESET_PREFIXES:
        if content.startswith(prefix):
            return (content, None) if keep is None else (content[len(prefix):].lstrip(), keep)
    return content, None


@dataclass(frozen=True)
class Speaker:
    """How a persona appears in a transcript: as the member it mimics."""
    persona: str
    user_id: str
    label: str


@dataclass(frozen=True)
class Line:
    """
    A transcript entry: the message as the adapter reads it, and which persona
    posted it (None for a person). The Msg alone cannot say — a persona's lines
    carry its member's id, so a member chatting beside their own persona is
    indistinguishable from it there, by design. `reply_to` is the persona whose
    message this one used Discord's reply on, if any; the adapters never see it.
    """
    msg: Msg
    persona: Optional[str] = None
    reply_to: Optional[str] = None
    # The message's image attachments, and what vision.Captioner made of them.
    # Bot-side only: the chat controller reads captions, the persona prompt
    # never does (see vision.py) — which is why they live here and not on Msg,
    # whose shape is the training contract in shared/.
    images: tuple = ()
    captions: tuple[str, ...] = ()
    # A human's reset prefix (MIMIC_RESET_PREFIXES): persona windows reach back
    # at most this many turns from this line, counting it. The controller
    # still reads the whole transcript.
    reset: Optional[int] = None


def since_reset(lines: list[Line], merge_seconds: float) -> list[Line]:
    """
    `lines` from the latest reset on: the reset line and the reset-1 turns
    before it, then everything after. The reset line starts a turn of its own,
    so ">>>>" keeps nothing its sender said before it.
    """
    at = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].reset is not None), None)
    if at is None:
        return lines
    before = mimic_format.merge_turns([line.msg for line in lines[:at]], merge_seconds)
    kept = before[-(lines[at].reset - 1):] if lines[at].reset > 1 else []
    return lines[at - sum(len(t.texts) for t in kept):]


def persona_speaker(
    persona: Persona,
    aliases: dict[str, str],
    handle_for_id: Callable[[str], Optional[str]] = lambda uid: None,
) -> Speaker:
    """
    The member's id and label, as their training context named them: alias,
    else Discord handle, else (no handle to hand) the persona's display name.
    """
    ids = persona.user_ids
    uid = next((u for u in ids if u in aliases), ids[0] if ids else f"persona:{persona.key}")
    return Speaker(persona.key, uid, member_name(uid, handle_for_id(uid) or persona.display_name, aliases))


@dataclass
class Cast:
    """
    Every available persona, as one conversation sees them: who they are, the
    label their lines carry, every name a human might address them by, and the
    name Discord shows their posts under right now.
    """
    catalog: PersonaCatalog
    personas: dict[str, Persona]
    speakers: dict[str, Speaker]
    names: dict[str, set[str]] = field(default_factory=dict)
    shown_as: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        catalog: PersonaCatalog,
        aliases: Optional[dict[str, str]] = None,
        handle_for_id: Callable[[str], Optional[str]] = lambda uid: None,
        shown_as: Optional[Mapping[str, str]] = None,
        other_names: Optional[Mapping[str, set[str]]] = None,
    ) -> "Cast":
        """
        `shown_as` is each persona's current Discord name (default: its display
        name); `other_names` any older names it has posted under.
        """
        aliases = catalog.aliases() if aliases is None else aliases
        personas = {p.key: p for p in catalog.all()}
        speakers = {k: persona_speaker(p, aliases, handle_for_id) for k, p in personas.items()}
        shown = {k: (shown_as or {}).get(k) or p.display_name for k, p in personas.items()}
        names = {
            k: {n for n in {speakers[k].label, p.display_name, shown[k], *(other_names or {}).get(k, ())} if n}
            for k, p in personas.items()
        }
        return cls(catalog, personas, speakers, names, shown)

    def label(self, persona: str) -> str:
        return self.speakers[persona].label

    def persona_line(self, persona: str, text: str, ts: datetime) -> Line:
        """A persona's own line, for a transcript that is not (yet) read back from Discord."""
        speaker = self.speakers[persona]
        return Line(Msg(speaker.user_id, speaker.label, ts, text), persona)

    def seen_by(self, persona: str, lines: list[Line]) -> list[Msg]:
        """
        The transcript as `persona` reads it in a one-on-one chat: its member's
        own messages come from MIMIC_SELF_LABEL — a different speaker from the
        persona, whose lines carry the member's name.
        """
        ids = set(self.personas[persona].user_ids)
        return [
            Msg(f"self:{line.msg.user_id}", MIMIC_SELF_LABEL, line.msg.ts, line.msg.text, line.msg.has_attachments)
            if line.persona is None and line.msg.user_id in ids else line.msg
            for line in lines
        ]

    def prompt(self, persona: str, lines: list[Line], one_on_one: bool = False) -> Optional[list[dict]]:
        """
        build_persona_prompt for one of this cast, from the latest reset on.
        One on one, the persona is talking with its own member, who must read
        as someone else; with others in the chat, everyone keeps their own name.
        """
        p = self.personas[persona]
        lines = since_reset(lines, p.merge_seconds)
        msgs = self.seen_by(persona, lines) if one_on_one else [line.msg for line in lines]
        return build_persona_prompt(self.catalog, p, msgs)


def build_persona_prompt(catalog: PersonaCatalog, persona: Persona, msgs: list[Msg]) -> Optional[list[dict]]:
    """
    The prompt `persona` was trained on, for the conversation in `msgs`: its
    system prompt (if it was trained with one), then its last context_n merged turns, cut at the first
    silence longer than its gap_minutes — the window training cut, and no
    wider — except that the cut never leaves fewer than
    MIMIC_MIN_CONTEXT_TURNS. None when there is nothing to reply to.

    It used to be wider (every turn from the last 30 minutes, up to 48) and
    not cut at silences, so a chat resumed after a break kept its past. The
    adapters, trained on at most 24 turns of one unbroken conversation, then
    followed whatever dominated the window and hardly answered the newest
    line: they could not change topic. Nothing in the format marks a turn as
    new, so the only lever on what they treat as current is what is shown.
    """
    turns = mimic_format.merge_turns(msgs, persona.merge_seconds)
    if not turns:
        return None
    context = mimic_format.context_window(turns, persona.context_n, persona.gap_minutes * 60.0)
    floor = min(MIMIC_MIN_CONTEXT_TURNS, persona.context_n)
    if len(context) < floor:
        context = turns[-floor:]
    system = catalog.system_prompt(persona)
    return [
        *([{"role": "system", "content": system}] if system is not None else []),
        {"role": "user", "content": mimic_format.render_context(context)},
    ]
