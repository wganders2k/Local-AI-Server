"""
The conversation loop: decide, write, post, repeat.

Two ways in:
- run_burst: one stretch of bot talk in a /mimic-chat thread, which stops when
  the controller waits for a human, a human interrupts, or the burst cap hits.
- simulate: a whole bot-only conversation of a set length, for lore's
  simulate_conversation tool, which nobody but the lore agent reads.

Each persona's line is written from its own training-shaped window of the
transcript (mimic.transcript.build_persona_prompt), so a persona reads the
others — and its own earlier lines — exactly the way its training context did.

Who the personas are comes in as a transcript.Cast: from a Discord thread
(persona_posting.HistoryReader) or, for a simulation, from the catalog alone.
"""

import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from config import MIMIC_CHAT_FREE_FOLLOWUPS, MIMIC_CHAT_MAX_BURST, MIMIC_MODEL, MIMIC_SIM_SEED_NAME
from mimic import addressing, controller
from mimic.controller import END, SPEAK, WAIT
from mimic.transcript import Cast, Line, clean_persona_reply
from mimic_format import Msg, merge_turns, render_context
from personas import PersonaCatalog
from proxy_client import ProxyClient
from vision import Captioner, annotate

logger = logging.getLogger("mimic-bot.mimic.engine")

# persona, text -> posted. The cog posts through the persona's webhook.
Poster = Callable[[str, str], Awaitable[None]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def seed_line(premise: str, user_id: str, label: str, ts: datetime) -> Line:
    """The premise as the chat's opening line: a topic raised in chat is the shape the adapters know."""
    return Line(Msg(user_id, label, ts, premise))


async def generate_line(
    proxy: ProxyClient, cast: Cast, persona: str, lines: list[Line], one_on_one: bool = False
) -> str:
    prompt = cast.prompt(persona, lines, one_on_one)
    if prompt is None:
        return ""
    return clean_persona_reply(await proxy.chat(MIMIC_MODEL, prompt, lora=cast.catalog.lora(persona)))


async def _run(
    proxy: ProxyClient,
    cast: Cast,
    personas: list[str],
    premise: Optional[str],
    lines: list[Line],
    *,
    budget: int,
    humans: bool,
    length: Optional[int],
    post: Optional[Poster],
    interrupted: Callable[[], bool],
    captioner: Optional[Captioner] = None,
) -> list[Line]:
    """
    Up to `budget` bot turns appended to `lines` (and posted, if `post`).
    Returns the new lines. A persona following up on its own line extends its
    turn for free, MIMIC_CHAT_FREE_FOLLOWUPS times in a row; past that it costs
    a turn like anyone else, so no one persona can hold the floor forever.
    """
    labels = {p: cast.label(p) for p in personas}
    names = {p: cast.names[p] for p in personas}
    shown_as = {p: cast.shown_as[p] for p in personas}
    ids = {p: cast.personas[p].user_ids for p in personas}
    new: list[Line] = []
    turns = 0
    last, followups = None, 0  # the persona holding the floor, and its free follow-ups so far
    # Alone, every line is "following up on itself"; there is no floor to hold.
    free_followups = MIMIC_CHAT_FREE_FOLLOWUPS if len(personas) > 1 else 0
    while True:
        if interrupted():
            logger.info("Burst interrupted after %d line(s)", len(new))
            break
        # The last persona may still follow up for free once the budget is spent.
        may_follow_up = last is not None and followups < free_followups
        if turns >= budget and not may_follow_up:
            break
        follow_up_note = (
            f" {labels[last]} may follow up on their own message without using up a turn."
            if may_follow_up else ""
        )

        if length:
            # A set length is a floor as much as a ceiling: no ending in the first half.
            actions = [SPEAK, END] if turns >= length // 2 else [SPEAK]
            status = f"{turns} of about {length} bot turns are done.{follow_up_note}"
        else:
            since = controller.bot_turns_since_human(lines)
            # A human just spoke: someone answers before anyone waits.
            actions = [SPEAK] if since == 0 else [SPEAK, WAIT]
            status = (
                f"Bots have taken {since} turn(s) in a row since a human last spoke; "
                f"at most {max(budget - turns, 0)} more are allowed before the chat waits for one."
                f"{follow_up_note}"
            )
        # Out of turns: only the free follow-up is left, or stopping.
        speakers = personas if turns < budget else [last]
        if turns >= budget:
            actions = [SPEAK, END if length else WAIT]

        forced = None
        if humans and lines and lines[-1].persona is None and turns < budget:
            # A human just spoke: whoever they addressed answers, and never the
            # persona of the member who spoke, unless that member called it in.
            speakers, forced = addressing.who_may_answer(personas, lines, names, ids)
            if not speakers:
                logger.info("Nobody may answer %s; waiting", lines[-1].msg.username)
                break
        if forced is not None:
            decision = controller.Decision(SPEAK, forced, reason="addressed")
        elif actions == [SPEAK] and len(speakers) == 1:
            decision = controller.Decision(SPEAK, speakers[0])
        else:
            # Images are described only now, when the controller will read
            # them: an addressed reply never waits on a caption. In place, so
            # the lines appended below land in the caller's list as before.
            lines[:] = await annotate(captioner, lines, MIMIC_MODEL, cast.catalog.lora(None))
            decision = await controller.decide(
                proxy, speakers, labels, premise, lines,
                humans=humans, actions=actions, status=status, length=length, shown_as=shown_as,
                lora=cast.catalog.lora(None),
            )
        if decision.action != SPEAK:
            break
        if turns >= budget and decision.reason == "fallback":
            break  # an optional follow-up is not worth guessing at

        text = await generate_line(proxy, cast, decision.speaker, lines, one_on_one=len(personas) == 1)
        if interrupted():
            # A human spoke while this line was being written. Posting it would
            # put a bot line after theirs, and the next burst would no longer
            # see that a human is waiting to be answered.
            logger.info("Burst interrupted mid-line after %d line(s); line discarded", len(new))
            break
        if not text:
            logger.warning("%s produced nothing; stopping the burst", decision.speaker)
            break
        if post is not None:
            await post(decision.speaker, text)
        if decision.speaker == last and followups < free_followups:
            followups += 1
        else:
            turns += 1
            last, followups = decision.speaker, 0
        line = cast.persona_line(decision.speaker, text, _now())
        lines.append(line)
        new.append(line)
    return new


async def run_burst(
    proxy: ProxyClient,
    cast: Cast,
    personas: list[str],
    premise: Optional[str],
    lines: list[Line],
    *,
    post: Poster,
    interrupted: Callable[[], bool] = lambda: False,
    length: Optional[int] = None,
    captioner: Optional[Captioner] = None,
) -> list[Line]:
    """
    Bot talk in a thread with people in it. Paced by the controller and capped
    at MIMIC_CHAT_MAX_BURST turns, or with `length`, that many turns without waiting.
    """
    return await _run(
        proxy, cast, personas, premise, lines,
        budget=length or MIMIC_CHAT_MAX_BURST, humans=True, length=length,
        post=post, interrupted=interrupted, captioner=captioner,
    )


async def simulate(
    proxy: ProxyClient,
    catalog: PersonaCatalog,
    personas: list[str],
    premise: str,
    length: int,
) -> list[Line]:
    """A bot-only conversation opened by `premise`. Returns the transcript, seed line first."""
    cast = Cast.build(catalog)
    lines = [seed_line(premise, "seed", MIMIC_SIM_SEED_NAME, _now())]
    await _run(
        proxy, cast, personas, premise, lines,
        budget=length, humans=False, length=length, post=None, interrupted=lambda: False,
    )
    return lines


def render_transcript(lines: list[Line]) -> str:
    """One `[name]: text` line per turn, the way the personas read it."""
    return render_context(merge_turns([line.msg for line in lines]))
