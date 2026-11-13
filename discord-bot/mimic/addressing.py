"""
Who a human is talking to, decided in code before the controller is asked.

A human addresses a persona by replying to its post, or by opening their
message with its name or an @mention of the member it imitates. Only the first
word counts: a name further in is usually talked about, not talked to. A
persona addressed this way answers without the controller's say.

In a chat with several personas, one never answers its own member directly,
unless that member @mentions or replies to it: a bot echoing the person it
imitates, straight after them, reads as mockery rather than conversation. Alone
in a chat, it always may — talking to your own mimic is the point of that chat.
"""

from dataclasses import dataclass
from typing import Collection, Mapping, Optional

from mimic.transcript import Line

# Punctuation a name is followed by when someone addresses a person: "dana," "dana:" "dana?"
_TRAILING_PUNCT = ".,:;!?"


@dataclass(frozen=True)
class Address:
    personas: tuple[str, ...]  # who it could be, in chat order
    direct: bool  # a reply or an @mention, not just a name


def _first_word(text: str) -> tuple[str, bool]:
    word = text.split(maxsplit=1)[0] if text.strip() else ""
    mention = word.startswith("@")
    return word.lstrip("@").rstrip(_TRAILING_PUNCT).casefold(), mention


def _matches(word: str, name: str) -> bool:
    # Whole names only: a nickname's first word ("Liquid", "Sir") is too often an ordinary word.
    return word == name.casefold()


def addressed(line: Line, personas: list[str], names: Mapping[str, set[str]]) -> Optional[Address]:
    """Who `line` is aimed at among `personas`, or None. A reply outranks the first word."""
    if line.persona is not None:
        return None
    if line.reply_to in personas:
        return Address((line.reply_to,), direct=True)
    word, mention = _first_word(line.msg.text)
    if not word:
        return None
    hits = tuple(p for p in personas if any(_matches(word, n) for n in names.get(p, ())))
    return Address(hits, direct=mention) if hits else None


def who_may_answer(
    personas: list[str],
    lines: list[Line],
    names: Mapping[str, set[str]],
    member_ids: Mapping[str, Collection[str]] = {},
) -> tuple[list[str], Optional[str]]:
    """
    After a human spoke: (who may answer, who must, if anyone). The newest
    address in the run of human lines since the last bot line wins, so "dana"
    then "what do you think" still goes to dana. The persona of whoever spoke
    last is left out unless they addressed it directly. `member_ids` maps each
    persona to the accounts of the member it imitates.
    """
    run: list[Line] = []
    for line in reversed(lines):
        if line.persona is not None:
            break
        run.append(line)
    if not run:
        return personas, None
    last = run[0]
    found = next(((line, a) for line in run if (a := addressed(line, personas, names))), None)
    address = found[1] if found else None
    # Only the member themself can call their own persona in straight after them.
    direct = set(address.personas) if address and address.direct and found[0].msg.user_id == last.msg.user_id else set()
    allowed = personas if len(personas) == 1 else [
        p for p in personas if p in direct or last.msg.user_id not in member_ids.get(p, ())
    ]
    if address:
        targets = [p for p in address.personas if p in allowed]
        if targets:
            return targets, targets[0] if len(targets) == 1 else None
    return allowed, None
