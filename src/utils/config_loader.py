"""Loads strategy parameters from config.yaml.

Single source of truth for runtime-tunable parameters.  Secrets (API keys)
stay in .env — only non-sensitive strategy config lives here.

Public API:
    load_symbols() -> list[str]
    load_funding_thresholds() -> tuple[float, float]
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

_LOG = logging.getLogger(__name__)

# config.yaml lives at the project root, two levels above this file
# (src/utils/config_loader.py → src/utils → src → project root)
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

_FALLBACK_SYMBOLS: list[str] = ["BTC", "ETH", "SOL", "ARB", "DOGE"]


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Read and parse config.yaml once; cache the result for the process lifetime."""
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        _LOG.debug("config_loader: loaded %s", _CONFIG_PATH)
        return cfg
    except FileNotFoundError:
        _LOG.warning("config_loader: %s not found — using fallback values", _CONFIG_PATH)
        return {}
    except yaml.YAMLError as exc:
        _LOG.error("config_loader: YAML parse error in %s — %s", _CONFIG_PATH, exc)
        return {}


def load_symbols() -> list[str]:
    """Return the universe of symbols to scan, from config.yaml.

    Falls back to the original five-symbol list if config.yaml is absent or
    malformed.

    Returns:
        List of coin name strings (e.g. ``["BTC", "ETH", ...]``).
    """
    cfg = _load_config()
    syms = cfg.get("symbols")
    if not isinstance(syms, list) or not syms:
        _LOG.warning("config_loader: 'symbols' missing or empty — using fallback %s", _FALLBACK_SYMBOLS)
        return list(_FALLBACK_SYMBOLS)
    return [str(s) for s in syms]


def load_funding_thresholds() -> tuple[float, float]:
    """Return ``(funding_extreme_pct, funding_moderate_pct)`` from config.yaml.

    Both values are expressed as percentages per 8h (e.g. ``0.10`` = 0.10%/8h).
    Defaults match the config.yaml values: extreme=0.10, moderate=0.05.

    Returns:
        Tuple of ``(extreme_pct, moderate_pct)``.
    """
    cfg = _load_config()
    extreme = float(cfg.get("funding_extreme_pct", 0.10))
    moderate = float(cfg.get("funding_moderate_pct", 0.05))
    return extreme, moderate
