"""Tests for src/agents/regime.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agents.regime import (
    ADX_PERIOD,
    MIN_BARS,
    TREND_ADX_THRESHOLD,
    VOLATILE_ATR_PCT_THRESHOLD,
    Regime,
    RegimeResult,
    _compute_adx,
    _compute_atr_pct,
    classify_regime,
    classify_regime_from_df,
    get_regime,
)
from src.harness.pit_data import PITDataView


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------

def _flat_df(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """All bars identical — zero DM, zero TR variation."""
    return pd.DataFrame({
        "open": [price] * n,
        "high": [price] * n,
        "low": [price] * n,
        "close": [price] * n,
        "volume": [1_000.0] * n,
    })


def _trending_df(n: int = 50, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Monotonically rising closes — strong trend, ADX should be high."""
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1_000.0] * n,
    })


def _volatile_df(n: int = 50, price: float = 100.0, swing_pct: float = 0.05) -> pd.DataFrame:
    """Wide-range alternating bars — ATR% should be high."""
    rows = []
    for i in range(n):
        swing = price * swing_pct
        high = price + swing
        low = price - swing
        close = price + (swing if i % 2 == 0 else -swing)
        rows.append({"open": price, "high": high, "low": low, "close": close, "volume": 1_000.0})
    return pd.DataFrame(rows)


def _mock_pit(df_1h: pd.DataFrame) -> MagicMock:
    pit = MagicMock(spec=PITDataView)
    pit.ohlcv.return_value = df_1h
    return pit


# ---------------------------------------------------------------------------
# classify_regime (pure function)
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_volatile_takes_priority_over_trend_adx(self) -> None:
        """ATR% above threshold beats ADX — VOLATILE wins."""
        assert classify_regime(adx=30.0, atr_pct=4.0) == Regime.VOLATILE

    def test_trend_when_adx_above_threshold_atr_below(self) -> None:
        assert classify_regime(adx=30.0, atr_pct=1.5) == Regime.TREND

    def test_chop_when_both_below_thresholds(self) -> None:
        assert classify_regime(adx=15.0, atr_pct=1.5) == Regime.CHOP

    def test_boundary_atr_exactly_threshold_not_volatile(self) -> None:
        """ATR% == threshold is NOT volatile (strictly greater required)."""
        result = classify_regime(adx=15.0, atr_pct=VOLATILE_ATR_PCT_THRESHOLD)
        assert result != Regime.VOLATILE

    def test_boundary_adx_exactly_threshold_not_trend(self) -> None:
        """ADX == threshold is NOT trend (strictly greater required)."""
        result = classify_regime(adx=TREND_ADX_THRESHOLD, atr_pct=1.0)
        assert result != Regime.TREND

    def test_returns_regime_enum(self) -> None:
        result = classify_regime(adx=30.0, atr_pct=1.0)
        assert isinstance(result, Regime)


# ---------------------------------------------------------------------------
# _compute_adx
# ---------------------------------------------------------------------------

class TestComputeAdx:
    def test_insufficient_data_returns_zero(self) -> None:
        df = _flat_df(n=ADX_PERIOD - 1)
        assert _compute_adx(df) == 0.0

    def test_exactly_min_rows_does_not_return_zero(self) -> None:
        df = _trending_df(n=ADX_PERIOD + 1)
        assert _compute_adx(df) > 0.0

    def test_flat_series_zero_or_near_zero_adx(self) -> None:
        """Identical bars → no directional movement → ADX ≈ 0."""
        df = _flat_df(n=50)
        assert _compute_adx(df) == pytest.approx(0.0, abs=1e-6)

    def test_trending_series_high_adx(self) -> None:
        df = _trending_df(n=60, step=2.0)
        assert _compute_adx(df) > TREND_ADX_THRESHOLD

    def test_output_in_range(self) -> None:
        for df in [_flat_df(), _trending_df(), _volatile_df()]:
            adx = _compute_adx(df)
            assert 0.0 <= adx <= 100.0


# ---------------------------------------------------------------------------
# _compute_atr_pct
# ---------------------------------------------------------------------------

class TestComputeAtrPct:
    def test_insufficient_data_returns_zero(self) -> None:
        df = _flat_df(n=ADX_PERIOD - 1)
        assert _compute_atr_pct(df) == 0.0

    def test_flat_bars_atr_pct_near_zero(self) -> None:
        df = _flat_df(n=50)
        assert _compute_atr_pct(df) == pytest.approx(0.0, abs=1e-6)

    def test_volatile_bars_atr_pct_above_threshold(self) -> None:
        df = _volatile_df(n=50, swing_pct=0.05)
        assert _compute_atr_pct(df) > VOLATILE_ATR_PCT_THRESHOLD

    def test_known_range_known_price(self) -> None:
        """Bars with constant H-L of 2 and price=100 → ATR% ≈ 2.0."""
        n = 30
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1_000.0] * n,
        })
        atr_pct = _compute_atr_pct(df)
        assert atr_pct == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# classify_regime_from_df
# ---------------------------------------------------------------------------

class TestClassifyRegimeFromDf:
    def test_trending_df_returns_trend(self) -> None:
        result = classify_regime_from_df(_trending_df(60, step=2.0), "BTC", "1h")
        assert result.regime == Regime.TREND

    def test_flat_df_returns_chop(self) -> None:
        result = classify_regime_from_df(_flat_df(50), "BTC", "1h")
        assert result.regime == Regime.CHOP

    def test_volatile_df_returns_volatile(self) -> None:
        result = classify_regime_from_df(_volatile_df(50, swing_pct=0.05), "BTC", "1h")
        assert result.regime == Regime.VOLATILE

    def test_result_carries_symbol_and_timeframe(self) -> None:
        result = classify_regime_from_df(_flat_df(), "ETH", "4h")
        assert result.symbol == "ETH"
        assert result.timeframe == "4h"

    def test_result_has_adx_and_atr_pct(self) -> None:
        result = classify_regime_from_df(_trending_df(60), "BTC", "1h")
        assert result.adx >= 0.0
        assert result.atr_pct >= 0.0

    def test_insufficient_data_returns_chop(self) -> None:
        df = _flat_df(n=MIN_BARS - 1)
        result = classify_regime_from_df(df, "BTC", "1h")
        assert result.regime == Regime.CHOP
        assert result.adx == 0.0
        assert result.atr_pct == 0.0

    def test_returns_regime_result_type(self) -> None:
        result = classify_regime_from_df(_flat_df(), "BTC", "1h")
        assert isinstance(result, RegimeResult)


# ---------------------------------------------------------------------------
# get_regime (pit wrapper)
# ---------------------------------------------------------------------------

class TestGetRegime:
    def test_calls_pit_ohlcv_once(self) -> None:
        pit = _mock_pit(_trending_df(60, step=2.0))
        get_regime(pit, "BTC", "1h")
        pit.ohlcv.assert_called_once_with("BTC", "1h")

    def test_regime_matches_classify_from_df(self) -> None:
        df = _trending_df(60, step=2.0)
        pit = _mock_pit(df)
        result = get_regime(pit, "BTC", "1h")
        direct = classify_regime_from_df(df, "BTC", "1h")
        assert result.regime == direct.regime
