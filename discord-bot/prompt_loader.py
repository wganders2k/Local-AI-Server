"""
Loader for the prompt templates in ``prompts/``.

Prompt prose lives in markdown next to the code rather than inside it. The
files are baked into the image and versioned with the modules that render them,
so a prompt change is a rebuild — which is what you want for text that is part
of the program's behaviour.

Server-specific background (real names, aliases) is NOT one of these. It is
host-local, lives on the bot_context volume, and is loaded separately by
``lore.prompts.load_lore_context``.

Templates are rendered with ``str.format``, so a literal brace in a template
must be doubled. None of the current templates contain one.
"""

import logging
from pathlib import Path

logger = logging.getLogger("mimic-bot.prompts")

PROMPTS_DIR = Path(__file__).parent / "prompts"

_cache: dict[str, str] = {}


class PromptError(RuntimeError):
    """A prompt template is missing or unreadable."""


def load(name: str) -> str:
    """
    Read one template by stem (e.g. "lore_agent"), cached for the process.

    The single trailing newline every file ends with is stripped, so templates
    compose exactly the way the string constants they replaced did.

    Raises:
        PromptError: The template is missing or unreadable. Unlike the optional
            lore context file, these are code — their absence is a bug, not a
            configuration choice, and failing at startup beats sending the
            model a prompt with a hole in it.
    """
    cached = _cache.get(name)
    if cached is not None:
        return cached

    path = PROMPTS_DIR / f"{name}.md"
    try:
        text = path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as e:
        raise PromptError(f"Could not read prompt template {path}: {e}") from e

    _cache[name] = text
    logger.debug("Loaded prompt template %s (%d chars)", name, len(text))
    return text


def render(name: str, **values: object) -> str:
    """
    Load a template and substitute its placeholders.

    Raises:
        PromptError: The template is missing, or names a placeholder the caller
            did not supply — which would otherwise reach the model as raw
            "{placeholder}" text.
    """
    template = load(name)
    try:
        return template.format(**values)
    except KeyError as e:
        raise PromptError(
            f"Prompt template {name} needs placeholder {e} — supplied: "
            f"{sorted(values)}"
        ) from e
