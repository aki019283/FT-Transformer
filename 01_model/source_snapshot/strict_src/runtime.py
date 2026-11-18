from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from config import APP_ROOT


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError("Config root must be a mapping")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (APP_ROOT / path).resolve()


def normalize_frequency(value: str) -> str:
    aliases = {"d": "daily", "day": "daily", "w": "weekly", "m": "monthly", "q": "quarterly"}
    normalized = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if normalized not in {"daily", "weekly", "monthly", "quarterly"}:
        raise ValueError(f"Unsupported frequency: {value}")
    return normalized


__all__ = ["load_yaml_config", "normalize_frequency", "resolve_path"]
