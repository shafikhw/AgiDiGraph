"""Runtime configuration for AgiDiGraph services."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"), override=False)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return Path(value)
    except Exception:  # pragma: no cover - defensive
        return default


@dataclass
class Settings:
    openai_api_key: Optional[str]
    openai_model: str
    use_offline_reasoner: bool
    max_batches: int
    batch_size: int
    cache_dir: Path
    cache_enabled: bool


@lru_cache()
def get_settings() -> Settings:
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        use_offline_reasoner=_bool_env("USE_OFFLINE_REASONER", True),
        max_batches=_int_env("GRAPH_MAX_BATCHES", 5),
        batch_size=_int_env("GRAPH_BATCH_SIZE", 5),
        cache_dir=_path_env("GRAPH_CACHE_DIR", Path("logs/cache")),
        cache_enabled=_bool_env("GRAPH_CACHE_ENABLED", True),
    )
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["Settings", "get_settings"]

