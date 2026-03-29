import configparser
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from config import SYSTEM_PROMPTS_PATH


class SystemPrompts:
    """
    Loads and manages system prompts from system_prompts.ini.
    Provides a simple API to retrieve prompts by model alias.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.prompts: dict[str, str] = {}
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / SYSTEM_PROMPTS_PATH
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all system prompts from the INI file."""
        if not self.config_path.exists():
            logger.info("system_prompts.ini not found — no system prompts will be injected")
            return

        try:
            config = configparser.ConfigParser()
            config.read(self.config_path, encoding="utf-8")

            for section in config.sections():
                if section.startswith("system_prompt:"):
                    model_alias = section.replace("system_prompt:", "")
                    prompt = config.get(section, "prompt", fallback="").strip()
                    if prompt:
                        self.prompts[model_alias] = prompt
                        logger.info(f"Loaded system prompt for model: {model_alias}")
                    else:
                        logger.warning(f"Empty prompt for model: {model_alias}")
        except Exception as exc:
            logger.error(f"Failed to load system_prompts.ini: {exc}")

    def get(self, model: str) -> Optional[str]:
        """Get the system prompt for a given model alias."""
        return self.prompts.get(model)

    def has(self, model: str) -> bool:
        """Check if a system prompt exists for the given model."""
        return model in self.prompts


# Module-level singleton
system_prompts = SystemPrompts()