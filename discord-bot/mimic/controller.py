"""
Who speaks next.

The controller is the mimic base with no adapter applied, on the same loaded
model as the personas ([gemma4-12b-mimic-u]), so deciding costs no model swap. Its
output is constrained to a JSON schema; anything unusable falls back to a
plain rule so a bad decision never stalls the chat.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from config import MIMIC_CHAT_CONTROLLER_LINES, MIMIC_CHAT_CONTROLLER_TEMPERATURE, MIMIC_MODEL
from mimic.transcript import Line
from mimic_format import Msg, context_line, merge_turns
from prompt_loader import render
from proxy_client import ProxyClient, ProxyError
from vision import with_captions

logger = logging.getLogger("mimic-bot.mimic.controller")

SPEAK, WAIT, END = "speak", "wait", "end"

_HUMANS = {
    True: "Real people are in this chat too, and the bots are there to talk with them.",
    False: "No real person is taking part: the personas only talk to each other.",
}
_WAIT_ACTION = (
    "- wait: stop and let the humans reply. Choose this at a natural resting point: a question "
    "or joke aimed at a human, a topic that has played out, or after a few bot turns in a row.\n"
)
_END_ACTION = "- end: the conversation has run its course and nothing natural is left to say.\n"
# Only a simulation, or a chat started before /mimic-chat dropped its premise, has one.
_PREMISE = "Premise of the chat: {premise}\n\n"
_GUIDANCE_PACED = (
    "If a human has just spoken, someone should answer them before you consider waiting. "
    "Prefer waiting over dragging the chat on with filler."
)
_GUIDANCE_LENGTH = (
    "The chat should run to about {length} bot turns in total. Keep it moving, and steer "
    "toward a natural ending as that count approaches."
)


@dataclass
class Decision:
    action: str
    speaker: Optional[str] = None
    reason: str = ""


NONE = "none"


def speaker_choices(personas: list[str], labels: dict[str, str]) -> dict[str, str]:
    """
    The controller names a speaker the way the transcript does, not by persona
    key: it reasons in names, and an opaque key it is forced to spell out loses
    to "none". Keys stay the choices only when names collide.
    """
    names = [labels[p] for p in personas]
    if len(set(names)) == len(names) and NONE not in names:
        return dict(zip(names, personas))
    return {p: p for p in personas}


def _schema(choices: list[str], actions: list[str]) -> dict:
    def branch(acts: list[str], speakers: list[str]) -> dict:
        return {
            "type": "object",
            # reason first, so the choice is made after the model has said why.
            "properties": {
                "reason": {"type": "string"},
                "action": {"type": "string", "enum": acts},
                "speaker": {"type": "string", "enum": speakers},
            },
            "required": ["reason", "action", "speaker"],
            "additionalProperties": False,
        }

    # Speaking names someone and nothing else does, so "speak" with no speaker cannot be produced.
    branches = [branch([SPEAK], choices)] if SPEAK in actions else []
    others = [a for a in actions if a != SPEAK]
    if others:
        branches.append(branch(others, [NONE]))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "decision",
            "strict": True,
            "schema": branches[0] if len(branches) == 1 else {"anyOf": branches},
        },
    }


def schema_actions(response_format: dict) -> list[str]:
    """Every action a schema from _schema allows, in order."""
    schema = response_format["json_schema"]["schema"]
    return [a for b in schema.get("anyOf", [schema]) for a in b["properties"]["action"]["enum"]]


def bot_turns_since_human(lines: list[Line]) -> int:
    """Bot turns since anyone else spoke. A persona following up on itself is still one turn."""
    n, prev = 0, None
    for line in reversed(lines):
        if line.persona is None:
            break
        if line.persona != prev:
            n += 1
        prev = line.persona
    return n


def _least_recent(personas: list[str], lines: list[Line]) -> str:
    last_spoke = {p: -1 for p in personas}
    for i, line in enumerate(lines):
        if line.persona in last_spoke:
            last_spoke[line.persona] = i
    return min(personas, key=lambda p: last_spoke[p])


def fallback(personas: list[str], lines: list[Line], actions: list[str]) -> Decision:
    """What happens when the controller's answer is unusable: rotate speakers, and leave room for humans."""
    if WAIT in actions and bot_turns_since_human(lines) >= 2:
        return Decision(WAIT, reason="fallback")
    return Decision(SPEAK, _least_recent(personas, lines), reason="fallback")


def parse(raw: str, choices: dict[str, str], actions: list[str]) -> Optional[Decision]:
    """`choices` maps each name the controller may give to its persona."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    action, speaker = data.get("action"), data.get("speaker")
    if action not in actions:
        return None
    if action == SPEAK and speaker not in choices:
        return None
    return Decision(action, choices[speaker] if action == SPEAK else None, str(data.get("reason") or ""))


BOT_TAG = " (bot)"


def _controller_msg(line: Line, labels: dict[str, str]) -> Msg:
    """
    A persona line carries the member it imitates, which is right for the
    personas but would hide from the controller whether the real person or
    their bot spoke, and merge the two into one turn. Here bots are their own
    speaker, and a Discord reply to a bot says so.
    """
    m = line.msg
    # What an image showed, where the persona prompt would only say
    # "[attachment]" — the controller has no adapter to keep in step with.
    text = with_captions(m.text, line.captions)
    if line.persona is None:
        if line.reply_to is None:
            return Msg(m.user_id, m.username, m.ts, text, m.has_attachments) if line.captions else m
        to = labels.get(line.reply_to, line.reply_to) + BOT_TAG
        return Msg(m.user_id, m.username, m.ts, f"(replying to {to}) {text}", m.has_attachments)
    return Msg(f"bot:{line.persona}", m.username + BOT_TAG, m.ts, text, m.has_attachments)


def _participant(name: str, shown_as: Optional[str]) -> str:
    entry = f"- {name}{BOT_TAG}"
    if shown_as and shown_as.casefold() != name.casefold():
        entry += f', shown in Discord as "{shown_as}"'
    return entry


def build_messages(
    personas: list[str],
    labels: dict[str, str],
    premise: Optional[str],
    lines: list[Line],
    humans: bool,
    actions: list[str],
    status: str,
    length: Optional[int],
    shown_as: Optional[dict[str, str]] = None,
) -> list[dict]:
    shown_as = shown_as or {}
    participants = "\n".join(
        _participant(name, shown_as.get(p)) for name, p in speaker_choices(personas, labels).items()
    )
    turns = merge_turns([_controller_msg(line, labels) for line in lines[-MIMIC_CHAT_CONTROLLER_LINES:]])
    transcript = "\n".join(context_line(t) for t in turns) or "(nothing yet)"
    system = render(
        "mimic_chat_controller",
        humans=_HUMANS[humans],
        participants=participants,
        premise=_PREMISE.format(premise=premise) if premise else "",
        wait_action=_WAIT_ACTION if WAIT in actions else "",
        end_action=_END_ACTION if END in actions else "",
        guidance=_GUIDANCE_LENGTH.format(length=length) if length else _GUIDANCE_PACED,
    )
    user = render("mimic_chat_controller_turn", transcript=transcript, status=status)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def decide(
    proxy: ProxyClient,
    personas: list[str],
    labels: dict[str, str],
    premise: Optional[str],
    lines: list[Line],
    *,
    humans: bool,
    actions: list[str],
    status: str,
    length: Optional[int] = None,
    shown_as: Optional[dict[str, str]] = None,
    lora: Optional[list[dict]] = None,
) -> Decision:
    """
    The next action, from `actions`. `labels` maps each persona to the name its
    lines carry in the transcript, so the controller can tell who said what.
    `lora` is [gemma4-12b-mimic-u]'s scales with no adapter applied (catalog.lora(None)).
    """
    msgs = build_messages(personas, labels, premise, lines, humans, actions, status, length, shown_as)
    choices = speaker_choices(personas, labels)
    try:
        raw = await proxy.chat(
            MIMIC_MODEL,
            msgs,
            lora=lora,
            response_format=_schema(list(choices), actions),
            temperature=MIMIC_CHAT_CONTROLLER_TEMPERATURE,
        )
    except ProxyError as e:
        logger.warning("Controller request failed (%s); using fallback", e)
        return fallback(personas, lines, actions)
    decision = parse(raw, choices, actions)
    if decision is None:
        logger.warning("Unusable controller output %r; using fallback", raw[:200])
        return fallback(personas, lines, actions)
    logger.info("Controller: %s %s (%s)", decision.action, decision.speaker or "", decision.reason)
    return decision
