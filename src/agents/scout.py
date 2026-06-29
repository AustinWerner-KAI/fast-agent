"""Deterministic pullback-to-MA candidate detector.

Scans Hyperliquid crypto and FX symbols for price pullbacks to EMA-20, EMA-50,
or EMA-200, gated by a TREND regime on the entry timeframe.  No LLM calls.

Public surface:
    Candidate, DEFAULT_SYMBOLS, scan
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.harness.pit_data import LookAheadError, PITDataView
from src.agents.regime import Regime, RegimeResult, classify_regime_from_df

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS: list[str] = ["BTC", "ETH", "SOL", "ARB", "DOGE"]
MA_PERIODS: list[int] = [20, 50, 200]
ENTRY_TOLERANCE_PCT: float = 1.5   # ±% of MA value to qualify as "at MA"
ENTRY_TF: str = "1h"
TREND_TF: str = "1d"

__all__ = ["Candidate", "DEFAULT_SYMBOLS", "scan"]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class InsufficientDataError(Exception):
    """Raised when a symbol/timeframe has too few bars for a meaningful signal."""


class Candidate(BaseModel):
    """A pullback-to-MA setup candidate.

    Attributes:
        symbol: Coin name (e.g. ``"BTC"``).
        direction: Trade direction implied by trend alignment.
        ma_period: EMA period that price is pulling back to (20, 50, or 200).
        distance_to_ma_pct: Signed distance from current price to MA.
            Positive = price above MA; negative = price below MA.
        regime: Entry-timeframe regime at the time of detection.
        confidence: Score in [0.0, 1.0]; higher = closer to MA and stronger trend.
        ts: decision_ts at which this candidate was generated.
    """

    symbol: str
    direction: Literal["LONG", "SHORT"]
    ma_period: int
    distance_to_ma_pct: float
    regime: Regime
    confidence: float = Field(..., ge=0.0, le=1.0)
    ts: datetime


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ema_current(df: pd.DataFrame, period: int) -> float:
    """Return the most recent EMA value using pandas ewm with adjust=False.

    ``adjust=False`` uses the recursive definition:
    ``EMA_t = alpha * close_t + (1 - alpha) * EMA_{t-1}`` where
    ``alpha = 2 / (period + 1)``.  This matches standard charting packages.

    Args:
        df: OHLCV DataFrame with column ``close`` (oldest row first).
        period: EMA span.

    Returns:
        EMA of the last row as a float.

    Raises:
        InsufficientDataError: When ``df`` has fewer rows than ``period``.
    """
    if len(df) < period:
        raise InsufficientDataError(f"Need >= {period} bars for EMA-{period}, got {len(df)}")
    return float(df["close"].ewm(span=period, adjust=False).mean().iloc[-1])


def _distance_to_ma_pct(price: float, ma: float) -> float:
    """Compute signed percentage distance from price to MA.

    Args:
        price: Current price.
        ma: Moving average value.

    Returns:
        ``(price - ma) / ma * 100``. Positive = price above MA.
    """
    return (price - ma) / ma * 100.0


def _score_confidence(distance_pct: float, adx: float) -> float:
    """Score a pullback setup's confidence in [0.0, 1.0].

    70% weight on proximity (closer to MA = higher score) and 30% weight on
    ADX strength (stronger trend = higher score).  ADX is normalised at 50
    to give finer discrimination in the practical 20–50 range.

    Args:
        distance_pct: Absolute signed distance from ``_distance_to_ma_pct``.
            Caller guarantees ``abs(distance_pct) <= ENTRY_TOLERANCE_PCT``.
        adx: ADX value on the entry timeframe (0–100).

    Returns:
        Confidence in [0.0, 1.0].
    """
    proximity = 1.0 - abs(distance_pct) / ENTRY_TOLERANCE_PCT  # 1.0 = at MA
    adx_score = min(adx / 50.0, 1.0)
    return round(0.7 * proximity + 0.3 * adx_score, 6)


def _check_trend_alignment(df: pd.DataFrame, direction: Literal["LONG", "SHORT"], ma_period: int) -> bool:
    """Return True when the daily trend confirms the desired direction.

    LONG requires last daily close > EMA-{ma_period} on the daily TF.
    SHORT requires last daily close < EMA-{ma_period} on the daily TF.

    Args:
        df: Daily OHLCV DataFrame (oldest row first).
        direction: Candidate direction to validate.
        ma_period: EMA period to compare against.

    Returns:
        True when the daily EMA alignment matches ``direction``.

    Raises:
        InsufficientDataError: When ``df`` has fewer rows than ``ma_period``.
    """
    ema = _ema_current(df, ma_period)
    last_close = float(df["close"].iloc[-1])
    if direction == "LONG":
        return last_close > ema
    return last_close < ema


def _scan_symbol(
    pit: PITDataView,
    symbol: str,
    decision_ts: datetime,
) -> list[Candidate]:
    """Scan one symbol for pullback-to-MA setups across all MA periods.

    Fetches OHLCV once per timeframe.  Regime check uses the same entry-TF
    DataFrame to avoid a second ``pit.ohlcv()`` call.  LookAheadError is
    never swallowed — it propagates as a harness-level bug indicator.

    Args:
        pit: Point-in-time data view.
        symbol: Coin name.
        decision_ts: Current replay timestamp (included in emitted Candidates).

    Returns:
        List of Candidates (may be empty).
    """
    try:
        df_entry = pit.ohlcv(symbol, ENTRY_TF)
        df_trend = pit.ohlcv(symbol, TREND_TF)
    except LookAheadError:
        raise
    except Exception:
        return []

    if df_entry.empty or df_trend.empty:
        return []

    regime_result: RegimeResult = classify_regime_from_df(df_entry, symbol, ENTRY_TF)
    if regime_result.regime != Regime.TREND:
        return []

    current_price = float(df_entry["close"].iloc[-1])
    candidates: list[Candidate] = []

    for ma_period in MA_PERIODS:
        try:
            ema_entry = _ema_current(df_entry, ma_period)
            dist_pct = _distance_to_ma_pct(current_price, ema_entry)
        except InsufficientDataError:
            continue

        if abs(dist_pct) > ENTRY_TOLERANCE_PCT:
            continue

        # Determine direction from daily trend alignment
        direction: Literal["LONG", "SHORT"] | None = None
        try:
            if _check_trend_alignment(df_trend, "LONG", ma_period):
                direction = "LONG"
            elif _check_trend_alignment(df_trend, "SHORT", ma_period):
                direction = "SHORT"
        except InsufficientDataError:
            continue

        if direction is None:
            continue

        confidence = _score_confidence(dist_pct, regime_result.adx)
        candidates.append(
            Candidate(
                symbol=symbol,
                direction=direction,
                ma_period=ma_period,
                distance_to_ma_pct=round(dist_pct, 6),
                regime=regime_result.regime,
                confidence=confidence,
                ts=decision_ts,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan(
    pit: PITDataView,
    symbols: list[str] = DEFAULT_SYMBOLS,
    decision_ts: datetime | None = None,
) -> list[Candidate]:
    """Scan all symbols for pullback-to-MA setups, regime-gated.

    Only TREND-regime symbols produce candidates.  Results are sorted by
    confidence descending so the highest-quality setups appear first.

    Args:
        pit: Point-in-time data view scoped to the current decision timestamp.
        symbols: List of coin names to scan.
        decision_ts: Timestamp to stamp on emitted Candidates.  Defaults to
            ``pit.decision_ts`` when omitted.

    Returns:
        List of ``Candidate`` objects sorted by confidence descending.
    """
    ts = decision_ts if decision_ts is not None else pit.decision_ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    all_candidates: list[Candidate] = []
    for symbol in symbols:
        all_candidates.extend(_scan_symbol(pit, symbol, ts))

    all_candidates.sort(key=lambda c: c.confidence, reverse=True)
    return all_candidates
