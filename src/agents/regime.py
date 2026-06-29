"""Market regime classifier: TREND / CHOP / VOLATILE.

Priority order: VOLATILE (ATR%) > TREND (ADX) > CHOP.
All computation is deterministic from OHLCV data — no LLM calls.

Public surface:
    Regime, RegimeResult, classify_regime, classify_regime_from_df, get_regime
"""
from __future__ import annotations

from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

from src.harness.pit_data import PITDataView

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADX_PERIOD: int = 14
ATR_PERIOD: int = 14
VOLATILE_ATR_PCT_THRESHOLD: float = 3.0   # ATR% strictly above this → VOLATILE
TREND_ADX_THRESHOLD: float = 25.0         # ADX strictly above this → TREND
MIN_BARS: int = ADX_PERIOD + 1            # minimum rows for a meaningful result

__all__ = ["Regime", "RegimeResult", "classify_regime", "classify_regime_from_df", "get_regime"]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    """Market regime classification."""

    TREND = "TREND"
    CHOP = "CHOP"
    VOLATILE = "VOLATILE"


class RegimeResult(BaseModel):
    """Full output of the regime classifier, carrying raw indicator values.

    Attributes:
        regime: Classified regime.
        adx: Average Directional Index value (0–100).
        atr_pct: ATR as a percentage of the last close price.
        symbol: Coin name.
        timeframe: Candle interval used.
    """

    regime: Regime
    adx: float = Field(..., ge=0.0)
    atr_pct: float = Field(..., ge=0.0)
    symbol: str
    timeframe: str


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _wilder_smooth(values: list[float], period: int) -> float:
    """Apply Wilder's exponential smoothing to a list and return the final value.

    Seeds on the simple mean of the first ``period`` values, then applies
    the recursive formula: ``EMA_t = (EMA_{t-1} * (period - 1) + x_t) / period``.

    Args:
        values: List of numeric values (oldest first). Must have length >= period.
        period: Smoothing period.

    Returns:
        Final smoothed value, or the mean of all values if ``len(values) < period``.
    """
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    acc = sum(values[:period]) / period
    for v in values[period:]:
        acc = (acc * (period - 1) + v) / period
    return acc


def _compute_atr_pct(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Compute ATR as a percentage of the last close price.

    Uses Wilder's smoothing over True Range values.

    Args:
        df: OHLCV DataFrame with columns ``high``, ``low``, ``close`` (oldest first).
        period: ATR smoothing period.

    Returns:
        ATR% value >= 0. Returns 0.0 when fewer than ``period + 1`` rows exist.
    """
    if len(df) < period + 1:
        return 0.0

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    tr_values: list[float] = []
    for i in range(1, len(df)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_values.append(max(hl, hc, lc))

    atr = _wilder_smooth(tr_values, period)
    last_close = closes[-1]
    if last_close <= 0:
        return 0.0
    return atr / last_close * 100.0


def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """Compute Wilder's Average Directional Index from OHLCV data.

    Builds a full DX series then Wilder-smooths it.  The minimum required
    rows is ``period + 1`` (to produce at least one TR/DM bar).

    Args:
        df: OHLCV DataFrame with columns ``high``, ``low``, ``close`` (oldest first).
        period: ADX smoothing period (default 14).

    Returns:
        ADX in [0, 100]. Returns 0.0 when fewer than ``period + 1`` rows exist.
    """
    if len(df) < period + 1:
        return 0.0

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    tr_vals: list[float] = []
    dm_plus: list[float] = []
    dm_minus: list[float] = []

    for i in range(1, len(df)):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_vals.append(max(hl, hc, lc))
        dm_plus.append(up if (up > dn and up > 0) else 0.0)
        dm_minus.append(dn if (dn > up and dn > 0) else 0.0)

    # Seed Wilder accumulators on first `period` bars
    atr_s = sum(tr_vals[:period]) / period
    dmp_s = sum(dm_plus[:period]) / period
    dmm_s = sum(dm_minus[:period]) / period

    def _dx(dmp: float, dmm: float, atr: float) -> float:
        di_plus = 100.0 * dmp / atr if atr > 0 else 0.0
        di_minus = 100.0 * dmm / atr if atr > 0 else 0.0
        denom = di_plus + di_minus
        return 100.0 * abs(di_plus - di_minus) / denom if denom > 0 else 0.0

    dx_series: list[float] = [_dx(dmp_s, dmm_s, atr_s)]

    for i in range(period, len(tr_vals)):
        atr_s = (atr_s * (period - 1) + tr_vals[i]) / period
        dmp_s = (dmp_s * (period - 1) + dm_plus[i]) / period
        dmm_s = (dmm_s * (period - 1) + dm_minus[i]) / period
        dx_series.append(_dx(dmp_s, dmm_s, atr_s))

    return _wilder_smooth(dx_series, period)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_regime(adx: float, atr_pct: float) -> Regime:
    """Classify regime from pre-computed indicator values.

    Priority: VOLATILE (ATR%) > TREND (ADX) > CHOP.

    Args:
        adx: ADX value (0–100).
        atr_pct: ATR as percentage of last close.

    Returns:
        Classified ``Regime``.
    """
    if atr_pct > VOLATILE_ATR_PCT_THRESHOLD:
        return Regime.VOLATILE
    if adx > TREND_ADX_THRESHOLD:
        return Regime.TREND
    return Regime.CHOP


def classify_regime_from_df(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    adx_period: int = ADX_PERIOD,
    atr_period: int = ATR_PERIOD,
) -> RegimeResult:
    """Compute regime directly from a pre-fetched OHLCV DataFrame.

    Prefer this over ``get_regime`` when the caller already holds the DataFrame
    for other computations (e.g. EMA calculation) to avoid a second
    ``PITDataView.ohlcv()`` call.

    Args:
        df: OHLCV DataFrame (oldest row first).  Must have columns:
            ``high``, ``low``, ``close``.
        symbol: Coin name (included in the result for context).
        timeframe: Candle interval (included in the result for context).
        adx_period: ADX smoothing period.
        atr_period: ATR smoothing period.

    Returns:
        ``RegimeResult`` with regime, raw adx, atr_pct, symbol, timeframe.
        Returns CHOP with zero indicators when ``df`` has fewer than ``MIN_BARS`` rows.
    """
    if len(df) < MIN_BARS:
        return RegimeResult(regime=Regime.CHOP, adx=0.0, atr_pct=0.0, symbol=symbol, timeframe=timeframe)

    adx = _compute_adx(df, adx_period)
    atr_pct = _compute_atr_pct(df, atr_period)
    regime = classify_regime(adx, atr_pct)
    return RegimeResult(regime=regime, adx=adx, atr_pct=atr_pct, symbol=symbol, timeframe=timeframe)


def get_regime(
    pit: PITDataView,
    symbol: str,
    timeframe: str,
    adx_period: int = ADX_PERIOD,
    atr_period: int = ATR_PERIOD,
) -> RegimeResult:
    """Convenience wrapper: fetch OHLCV from PITDataView then classify regime.

    Use ``classify_regime_from_df`` directly when you already hold the DataFrame
    for a given (symbol, timeframe) to avoid a redundant ``pit.ohlcv()`` call.

    Args:
        pit: Point-in-time data view scoped to the current decision timestamp.
        symbol: Coin name.
        timeframe: Candle interval.
        adx_period: ADX smoothing period.
        atr_period: ATR smoothing period.

    Returns:
        ``RegimeResult`` for this symbol / timeframe at the current PIT.
    """
    df = pit.ohlcv(symbol, timeframe)
    return classify_regime_from_df(df, symbol, timeframe, adx_period, atr_period)
