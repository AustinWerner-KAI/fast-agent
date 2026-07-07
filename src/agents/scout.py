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
from src.utils.config_loader import load_symbols

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS: list[str] = load_symbols()
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
        direction: Always ``"LONG"`` — this pipeline is long-only.
        ma_period: EMA period that price is pulling back to (20, 50, or 200).
        distance_to_ma_pct: Signed distance from current price to MA.
            Positive = price above MA; negative = price below MA.
        regime: Entry-timeframe regime at the time of detection.
        confidence: Score in [0.0, 1.0]; higher = closer to MA and stronger trend.
        ts: decision_ts at which this candidate was generated.
        daily_trend_direction: 'UP' if daily close > EMA-20 and EMA-20 slope is
            positive; 'DOWN' otherwise.  None when daily data is insufficient.
        ema20_daily: Numeric value of EMA-20 on the daily timeframe at decision_ts.
            None when fewer than 20 daily bars are available.
        ema50_daily: Numeric value of EMA-50 on the daily timeframe.
            None when fewer than 50 daily bars are available.
        ema200_daily: Numeric value of EMA-200 on the daily timeframe.
            None when fewer than 200 daily bars are available (requires --days >= 200).
    """

    symbol: str
    direction: Literal["LONG"]
    ma_period: int
    distance_to_ma_pct: float
    regime: Regime
    confidence: float = Field(..., ge=0.0, le=1.0)
    ts: datetime
    daily_trend_direction: str | None = None
    ema20_daily: float | None = None
    ema50_daily: float | None = None
    ema200_daily: float | None = None


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


def _check_trend_alignment(df: pd.DataFrame, ma_period: int) -> bool:
    """Return True when the daily trend confirms a LONG setup.

    Requires last daily close > EMA-{ma_period} on the daily TF.

    Args:
        df: Daily OHLCV DataFrame (oldest row first).
        ma_period: EMA period to compare against.

    Returns:
        True when the daily close is above the EMA (LONG-aligned).

    Raises:
        InsufficientDataError: When ``df`` has fewer rows than ``ma_period``.
    """
    ema = _ema_current(df, ma_period)
    last_close = float(df["close"].iloc[-1])
    return last_close > ema


def _is_btc_bearish(df_trend: pd.DataFrame) -> bool:
    """Return True when BTC daily close is below EMA-20, EMA-50, and EMA-200.

    Used as a macro filter: when BTC is in confirmed bear territory, altcoin
    LONG candidates face a tighter daily-trend requirement.

    Args:
        df_trend: BTC daily OHLCV DataFrame (oldest row first).

    Returns:
        True when BTC close is below all three MA_PERIODS EMAs; False otherwise
        or when data is insufficient.
    """
    if df_trend.empty:
        return False
    try:
        last_close = float(df_trend["close"].iloc[-1])
        for period in MA_PERIODS:
            if last_close > _ema_current(df_trend, period):
                return False
        return True
    except InsufficientDataError:
        return False
    except Exception:
        return False


def _daily_trend_direction(df_trend: pd.DataFrame) -> str | None:
    """Classify the daily trend direction as 'UP' or 'DOWN'.

    Returns 'UP' only when both conditions hold: daily close > EMA-20 AND
    EMA-20 is rising (value 5 daily bars ago < current value).  Any other
    combination returns 'DOWN'.

    Args:
        df_trend: Daily OHLCV DataFrame (oldest row first).

    Returns:
        ``'UP'``, ``'DOWN'``, or ``None`` when data is insufficient.
    """
    if len(df_trend) < 26:  # 20 bars for EMA + 5 for slope comparison
        return None
    try:
        ema_series = df_trend["close"].ewm(span=20, adjust=False).mean()
        last_close = float(df_trend["close"].iloc[-1])
        ema_now = float(ema_series.iloc[-1])
        ema_prev = float(ema_series.iloc[-6])  # 5 daily bars ago
        if last_close > ema_now and ema_now > ema_prev:
            return "UP"
        return "DOWN"
    except Exception:
        return None


def _scan_symbol(
    pit: PITDataView,
    symbol: str,
    decision_ts: datetime,
    btc_bearish: bool = False,
) -> list[Candidate]:
    """Scan one symbol for pullback-to-MA setups across all MA periods.

    Fetches OHLCV once per timeframe.  Regime check uses the same entry-TF
    DataFrame to avoid a second ``pit.ohlcv()`` call.  LookAheadError is
    never swallowed — it propagates as a harness-level bug indicator.

    Args:
        pit: Point-in-time data view.
        symbol: Coin name.
        decision_ts: Current replay timestamp (included in emitted Candidates).
        btc_bearish: When True (BTC daily close is below EMA-20/50/200), the
            minimum daily-trend EMA check is raised to EMA-50 — a pullback to
            EMA-20 on an altcoin will only emit a candidate if that altcoin's
            daily close is also above EMA-50.

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

    daily_dir = _daily_trend_direction(df_trend)
    current_price = float(df_entry["close"].iloc[-1])

    # Compute daily EMA levels once per symbol — passed downstream as MA stack context.
    # InsufficientDataError (< N bars) yields None; never blocks candidate generation.
    ema20_daily: float | None = None
    ema50_daily: float | None = None
    ema200_daily: float | None = None
    try:
        ema20_daily = _ema_current(df_trend, 20)
    except InsufficientDataError:
        pass
    try:
        ema50_daily = _ema_current(df_trend, 50)
    except InsufficientDataError:
        pass
    try:
        ema200_daily = _ema_current(df_trend, 200)
    except InsufficientDataError:
        pass

    candidates: list[Candidate] = []

    for ma_period in MA_PERIODS:
        try:
            ema_entry = _ema_current(df_entry, ma_period)
            dist_pct = _distance_to_ma_pct(current_price, ema_entry)
        except InsufficientDataError:
            continue

        if abs(dist_pct) > ENTRY_TOLERANCE_PCT:
            continue

        # Daily trend alignment — when BTC is bearish, require daily close
        # above EMA-50 minimum instead of just EMA-20.
        check_period = max(ma_period, 50) if btc_bearish else ma_period
        try:
            if not _check_trend_alignment(df_trend, check_period):
                continue
        except InsufficientDataError:
            continue
        direction: Literal["LONG"] = "LONG"

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
                daily_trend_direction=daily_dir,
                ema20_daily=ema20_daily,
                ema50_daily=ema50_daily,
                ema200_daily=ema200_daily,
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

    # BTC macro gate: compute once before the symbol loop so each altcoin's
    # _scan_symbol call can apply a tighter daily-EMA requirement when BTC is
    # in full bear mode (daily close below EMA-20, EMA-50, and EMA-200).
    btc_bearish = False
    try:
        df_btc_daily = pit.ohlcv("BTC", TREND_TF)
        btc_bearish = _is_btc_bearish(df_btc_daily)
    except LookAheadError:
        raise
    except Exception:
        pass  # degrade gracefully — never block on missing BTC data

    all_candidates: list[Candidate] = []
    for symbol in symbols:
        apply_btc_filter = btc_bearish and symbol != "BTC"
        all_candidates.extend(_scan_symbol(pit, symbol, ts, btc_bearish=apply_btc_filter))

    all_candidates.sort(key=lambda c: c.confidence, reverse=True)
    return all_candidates
