"""
Who the bot can mimic, and how each persona is served.

Every fact here has one source, and none of them is a constant in this repo:

    who exists, display name, accounts   the roster, personas.json
                                         (shared/persona_roster.py)
    adapter id                           the persona's position in [gemma4-12b-mimic-u]'s
                                         lora list, read from the proxy's
                                         /v1/models at startup
    how its prompts are shaped (system   the live adapter's <persona>-lora.json
      prompt, context_n, silence gap,    beside its GGUF: the build arguments of
      merge window)                      the dataset it trained on, recorded by
                                         lora-training (persona_roster's
                                         trained_format)
    member aliases                       names.json

The roster and names live beside the training data on the archive disk
(mounted read-only) because they hold real names; the adapter manifests live
beside the adapters (also read-only). All are read on demand, so an edit or an
export takes effect without a restart.

The prompt shape is deliberately not read from the persona's stats.json: that
describes the dataset as last built, which may already be the next adapter's.

The adapter order is the dangerous one: a persona served under the wrong id
speaks in another member's voice under its own name. So it is never guessed.
Until discovery has read it, no persona is available and persona commands say
so; discovery is retried on every command, like lore/context_window.py.

Imports no discord: lore's simulate_conversation and the mimic package use this
too.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import mimic_format
import persona_roster
from config import MIMIC_ADAPTER_SCALE, MIMIC_MODEL

logger = logging.getLogger("mimic-bot.personas")

MIMIC_TRAINING_DIR: str = os.environ.get("MIMIC_TRAINING_DIR", "/archive/training")
MIMIC_LORA_DIR: str = os.environ.get("MIMIC_LORA_DIR", persona_roster.LORA_DIR)


@dataclass(frozen=True)
class Persona:
    key: str
    display_name: str  # its public name (from the roster); never part of a prompt
    user_ids: tuple[str, ...]  # every account the member posted from, old first
    adapter_id: int
    # How the live adapter's training prompts were shaped, so a served prompt is
    # shaped the same way. build_dataset's defaults, with the system prompt, for
    # an adapter exported before that was recorded.
    context_n: int = mimic_format.CONTEXT_N
    gap_minutes: float = mimic_format.GAP_MINUTES
    merge_seconds: float = mimic_format.MERGE_SECONDS
    no_system: bool = False


def member_name(user_id: int | str, handle: str, aliases: dict[str, str]) -> str:
    """A speaker's label, by build_dataset's rule: alias from names.json, else Discord handle."""
    return aliases.get(str(user_id)) or handle


class PersonaCatalog:
    """
    The personas available right now: on the roster AND among [gemma4-12b-mimic-u]'s
    adapters. Either alone is not enough — a roster entry with no adapter has
    nothing to serve it, and an adapter with no roster entry has no name.
    """

    def __init__(
        self,
        training_dir: Optional[str] = None,
        roster_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        adapters: Optional[list[str]] = None,
        model: str = MIMIC_MODEL,
        scale: float = MIMIC_ADAPTER_SCALE,
    ):
        self.training_dir = training_dir or MIMIC_TRAINING_DIR
        self.lora_dir = lora_dir or MIMIC_LORA_DIR
        self.roster_path = (
            roster_path or os.environ.get("PERSONA_ROSTER") or os.path.join(self.training_dir, "personas.json")
        )
        self.model = model
        self.scale = scale
        # [gemma4-12b-mimic-u]'s adapters in id order, or None until discovered.
        self._adapters: Optional[list[str]] = adapters
        self._roster: dict[str, dict] = {}
        self._roster_mtime: Optional[float] = None
        self._mismatch_logged: Optional[tuple] = None

    # ---- discovery ----------------------------------------------------------

    @property
    def ready(self) -> bool:
        return bool(self._adapters)

    async def discover(self, proxy) -> bool:
        """Read [gemma4-12b-mimic-u]'s adapter order from the proxy. Never raises; False leaves personas unavailable."""
        order = persona_roster.adapter_order(await proxy.models_payload(), self.model)
        if not order:
            logger.warning(
                "Could not read %s's adapters from the proxy — personas are unavailable until it can be. "
                "Will retry on the next persona command.",
                self.model,
            )
            return False
        self._adapters = order
        logger.info("%s adapters, in id order: %s", self.model, ", ".join(order))
        return True

    async def ensure_discovered(self, proxy) -> bool:
        """Cheap once discovered; the recovery path if the proxy was down at startup."""
        return self.ready or await self.discover(proxy)

    # ---- the roster ---------------------------------------------------------

    def roster(self) -> dict[str, dict]:
        """personas.json, re-read when it changes. An unreadable roster logs and means no personas."""
        try:
            mtime = os.path.getmtime(self.roster_path)
        except OSError:
            mtime = None
        if mtime != self._roster_mtime:
            self._roster_mtime = mtime
            try:
                self._roster = persona_roster.load_roster(self.roster_path)
            except (persona_roster.RosterError, OSError, ValueError) as e:
                logger.warning("No usable persona roster (%s) — no personas available", e)
                self._roster = {}
        return self._roster

    def _persona(self, key: str, entry: dict, adapter_id: int) -> Persona:
        args = persona_roster.trained_format(key, self.lora_dir)
        return Persona(
            key=key,
            display_name=str(entry["display_name"]).strip(),
            user_ids=persona_roster.user_ids(entry),
            adapter_id=adapter_id,
            context_n=int(args.get("context_n") or mimic_format.CONTEXT_N),
            gap_minutes=float(args.get("gap_minutes") or mimic_format.GAP_MINUTES),
            merge_seconds=float(args.get("merge_seconds") or mimic_format.MERGE_SECONDS),
            no_system=bool(args.get("no_system")),
        )

    # ---- lookups ------------------------------------------------------------

    def all(self) -> list[Persona]:
        """Every available persona, by key."""
        if not self._adapters:
            return []
        roster = self.roster()
        ids = {key: i for i, key in enumerate(self._adapters)}
        mismatch = (tuple(sorted(set(roster) - set(ids))), tuple(sorted(set(ids) - set(roster))))
        if any(mismatch) and mismatch != self._mismatch_logged:
            self._mismatch_logged = mismatch
            logger.warning(
                "Roster and %s disagree — on the roster without an adapter: %s; "
                "adapters without a roster entry: %s. Those are unavailable.",
                self.model, ", ".join(mismatch[0]) or "none", ", ".join(mismatch[1]) or "none",
            )
        return [self._persona(key, roster[key], ids[key]) for key in sorted(roster) if key in ids]

    def keys(self) -> list[str]:
        return [p.key for p in self.all()]

    def get(self, key: str) -> Optional[Persona]:
        return next((p for p in self.all() if p.key == key), None)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def by_name(self) -> dict[str, str]:
        """Lower-cased display name or key -> key, for anything that names a persona in prose."""
        out = {p.key: p.key for p in self.all()}
        out.update({p.display_name.lower(): p.key for p in self.all()})
        return out

    def aliases(self) -> dict[str, str]:
        """{user_id: alias} from names.json, read fresh."""
        return persona_roster.load_aliases(os.path.join(self.training_dir, "names.json"))

    # ---- serving ------------------------------------------------------------

    def lora(self, key: Optional[str]) -> list[dict]:
        """
        [gemma4-12b-mimic-u]'s per-request adapter scales: `key`'s at the configured
        scale, every other at 0. None = the bare base (the chat controller).
        """
        if not self._adapters:
            raise LookupError(f"{self.model}'s adapters have not been discovered")
        return persona_roster.lora_scales(self._adapters, key, self.scale)

    def system_prompt(self, persona: Persona) -> Optional[str]:
        """
        The system prompt the persona's adapter was trained on — the same
        rendering, by the same code — or None for one trained without.
        """
        if persona.no_system:
            return None
        return mimic_format.render_persona_prompt(persona.key)
