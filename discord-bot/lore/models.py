"""
Which model does what in a /lore run.

Two choices per run: ``speed`` picks the research model, which runs the tool
rounds, and ``writer`` picks the model that writes the final answer — by
default the research model itself. See LORE_RESEARCH_MODELS and LORE_WRITERS
in config.py.
"""

from config import LORE_DEFAULT_SPEED, LORE_RESEARCH_MODELS, LORE_WRITERS


def research_model(speed: str) -> str:
    """The model that researches at this speed (unknown speeds get the default)."""
    return LORE_RESEARCH_MODELS.get(speed, LORE_RESEARCH_MODELS[LORE_DEFAULT_SPEED])


def writer_model(speed: str, writer: str) -> str:
    """The model that writes the answer: the chosen writer, or the research
    model when ``writer`` is LORE_WRITER_SAME (or unknown)."""
    return LORE_WRITERS.get(writer) or research_model(speed)


def hands_off(speed: str, writer: str) -> bool:
    """Whether the answer is written by a different model than the research,
    which means a swap and a handover of the research."""
    return writer_model(speed, writer) != research_model(speed)


def all_models() -> tuple[str, ...]:
    """Every model a /lore run can use, research or writer, without repeats."""
    return tuple(dict.fromkeys([*LORE_RESEARCH_MODELS.values(), *LORE_WRITERS.values()]))
