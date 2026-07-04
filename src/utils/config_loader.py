"""Loads strategy parameters from config.yaml.

Single source of truth for runtime-tunable parameters.  Secrets (API keys)
stay in .env — only non-sensitive strategy config lives here.

Public API:
    load_symbols() -> list[str]
    load_funding_thresholds() -> tuple[float, float]
    load_trail_pct() -> float
    load_trailing_stop_config() -> TrailConfig
    load_take_profit_config() -> TakeProfitConfig
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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


def load_conviction_sizing() -> dict:
    """Return conviction-based position sizing config from config.yaml.

    Returns:
        Dict with ``tiers`` (list of ``{max_conviction, size_usd}`` dicts) and
        ``free_margin_cap_pct``.  Falls back to empty tiers when the section is
        absent (executor falls back to env-var sizing).
    """
    cfg = _load_config()
    sizing = cfg.get("conviction_sizing") or {}
    tiers = sizing.get("tiers") or []
    cap_pct = float(sizing.get("free_margin_cap_pct", 0.02))
    return {"tiers": list(tiers), "free_margin_cap_pct": cap_pct}


@dataclass
class TrailConfig:
    """Trailing stop parameters from config.yaml position_management.trailing_stop."""

    atr_period_h4: int = 14
    atr_multiplier_chandelier: float = 2.0
    stop_improvement_threshold_pct: float = 0.1
    micro_break_buffer_pct: float = 0.15
    enable_soft_trail_after_tp1: bool = False
    enable_h4_atr_15m_combo_for_final_20: bool = True


def load_trailing_stop_config() -> TrailConfig:
    """Return trailing stop config from config.yaml position_management.trailing_stop.

    Falls back to dataclass defaults when the section is absent.

    Returns:
        TrailConfig with all trailing stop parameters.
    """
    cfg = _load_config()
    trail = (cfg.get("position_management") or {}).get("trailing_stop") or {}
    return TrailConfig(
        atr_period_h4=int(trail.get("atr_period_h4", 14)),
        atr_multiplier_chandelier=float(trail.get("atr_multiplier_chandelier", 2.0)),
        stop_improvement_threshold_pct=float(trail.get("stop_improvement_threshold_pct", 0.1)),
        micro_break_buffer_pct=float(trail.get("micro_break_buffer_pct", 0.15)),
        enable_soft_trail_after_tp1=bool(trail.get("enable_soft_trail_after_tp1", False)),
        enable_h4_atr_15m_combo_for_final_20=bool(
            trail.get("enable_h4_atr_15m_combo_for_final_20", True)
        ),
    )


@dataclass
class TakeProfitConfig:
    """Resting TP order parameters from config.yaml position_management.take_profit."""

    enable_resting_tp_orders: bool = True
    tp1_rr: float = 2.0
    tp2_rr: float = 3.0
    tp1_fraction: float = 0.50
    tp2_fraction: float = 0.30


def load_take_profit_config() -> TakeProfitConfig:
    """Return take-profit config from config.yaml position_management.take_profit.

    Falls back to dataclass defaults when the section is absent.

    Returns:
        TakeProfitConfig with all TP parameters.
    """
    cfg = _load_config()
    tp = (cfg.get("position_management") or {}).get("take_profit") or {}
    return TakeProfitConfig(
        enable_resting_tp_orders=bool(tp.get("enable_resting_tp_orders", True)),
        tp1_rr=float(tp.get("tp1_rr", 2.0)),
        tp2_rr=float(tp.get("tp2_rr", 3.0)),
        tp1_fraction=float(tp.get("tp1_fraction", 0.50)),
        tp2_fraction=float(tp.get("tp2_fraction", 0.30)),
    )


def load_trail_pct() -> float:
    """Return the stop-correction trail percentage from config.yaml.

    Applied when a market fill lands below the proposal stop level (inverted
    geometry due to stale OHLCV entry price). The corrected stop is set at
    ``fill_price * (1 - trail_pct)`` so position_manager has a valid level.

    Returns:
        Trail percentage as a decimal (default 0.07 = 7%).
    """
    cfg = _load_config()
    return float(cfg.get("tier_trail_pct", 0.07))
