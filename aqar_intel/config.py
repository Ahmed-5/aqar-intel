"""Central configuration.

Everything is driven by environment variables (see `.env.example`) so the same
code runs locally, in Docker, and in CI without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
CONTRACTS_DIR = DATA_DIR / "contracts"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "inventory.sqlite"


@dataclass
class Settings:
    """Runtime settings. Instantiate once and pass around (or use `get_settings`)."""

    openrouter_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY") or None
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    )
    # "auto" -> OpenRouter if a key is present, otherwise mock. "mock" forces offline mode.
    llm_mode: str = field(default_factory=lambda: os.getenv("AQAR_LLM_MODE", "auto"))
    app_name: str = field(default_factory=lambda: os.getenv("AQAR_APP_NAME", "AqarIntel"))
    app_url: str = field(default_factory=lambda: os.getenv("AQAR_APP_URL", "https://github.com/Ahmed-5"))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("AQAR_TIMEOUT", "60")))
    random_seed: int = field(default_factory=lambda: int(os.getenv("AQAR_SEED", "42")))

    @property
    def use_openrouter(self) -> bool:
        if self.llm_mode == "mock":
            return False
        if self.llm_mode == "openrouter":
            return True
        return bool(self.openrouter_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
