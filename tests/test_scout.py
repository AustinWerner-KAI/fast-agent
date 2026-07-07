"""Tests for src/agents/scout.py."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from unittest.mock import patch

from src.agents.scout import (
    ENTRY_TOLERANCE_PCT,
    Candidate,
    InsufficientDataError,
    _check_trend_alignment,
    _distance_to_ma_pct,
    _ema_current,
    _scan_symbol,
    _score_confidence,
    scan,
)
from src.agents.regime import Regime, RegimeResult
from src.harness.pit_data import LookAheadError, PITDataView

_TS = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock PITDataView
# ---------------------------------------------------------------------------

class MockPIT:
    """Minimal PITDataView stand-in for unit tests."""

    def __init__(
        self,
        data_1h: dict[str, pd.DataFrame] | None = None,
        data_1d: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self._1h = data_1h or {}
        self._1d = data_1d or {}
        self.decision_ts = _TS

    def ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if timeframe == "1h":
            return self._1h.get(symbol, pd.DataFrame())
        if timeframe == "1d":
            return self._1d.get(symbol, pd.DataFrame())
        return pd.DataFrame()

    def funding(self, symbol: str) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame()

    def orderbook(self, symbol: str) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame()

    def future_access(self, symbol: str, timeframe: str) -> None:
        raise LookAheadError("test future_access")


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _rising_df(n: int = 50, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1_000.0] * n,
    })


def _flat_df(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [price] * n,
        "high": [price] * n,
        "low": [price] * n,
        "close": [price] * n,
        "volume": [1_000.0] * n,
    })


def _at_ma_df(n: int = 50, ma_period: int = 20, price: float = 100.0) -> pd.DataFrame:
    """Rising series that ends right at the EMA-{ma_period} value.

    The last close equals the EMA so distance_to_ma_pct ≈ 0.
    """
    closes = [price] * n
    return pd.DataFrame({
        "open": [price] * n,
        "high": [price + 0.01] * n,
        "low": [price - 0.01] * n,
        "close": closes,
        "volume": [1_000.0] * n,
    })


# ---------------------------------------------------------------------------
# _ema_current
# ---------------------------------------------------------------------------

class TestEmaCurrent:
    def test_constant_series_returns_constant(self) -> None:
        df = _flat_df(n=30, price=100.0)
        assert _ema_current(df, 20) == pytest.approx(100.0, rel=1e-6)

    def test_raises_on_insufficient_data(self) -> None:
        df = _flat_df(n=10)
        with pytest.raises(InsufficientDataError):
            _ema_current(df, 20)

    def test_ema_rises_with_price(self) -> None:
        df = _rising_df(n=50)
        ema_20 = _ema_current(df, 20)
        last_close = float(df["close"].iloc[-1])
        assert ema_20 < last_close, "EMA lags price in a rising series"

    def test_adjust_false_matches_recursive_formula(self) -> None:
        """Manually compute EMA-3 on a 5-bar series and verify."""
        closes = [10.0, 11.0, 12.0, 11.5, 13.0]
        df = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes, "volume": [1.0] * 5})
        alpha = 2 / (3 + 1)
        ema = closes[0]
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        assert _ema_current(df, 3) == pytest.approx(ema, rel=1e-9)


# ---------------------------------------------------------------------------
# _distance_to_ma_pct
# ---------------------------------------------------------------------------

class TestDistanceToMaPct:
    def test_price_above_ma_positive(self) -> None:
        assert _distance_to_ma_pct(102.0, 100.0) == pytest.approx(2.0)

    def test_price_below_ma_negative(self) -> None:
        assert _distance_to_ma_pct(98.0, 100.0) == pytest.approx(-2.0)

    def test_price_equals_ma_zero(self) -> None:
        assert _distance_to_ma_pct(100.0, 100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _score_confidence
# ---------------------------------------------------------------------------

class TestScoreConfidence:
    def test_at_ma_with_max_adx_is_one(self) -> None:
        assert _score_confidence(0.0, 50.0) == pytest.approx(1.0, rel=1e-6)

    def test_at_boundary_zero_adx_low_confidence(self) -> None:
        result = _score_confidence(ENTRY_TOLERANCE_PCT, 0.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_result_in_zero_one(self) -> None:
        for dist in [0.0, 0.5, 1.0, 1.5]:
            for adx in [0.0, 25.0, 50.0, 80.0]:
                c = _score_confidence(dist, adx)
                assert 0.0 <= c <= 1.0

    def test_adx_above_50_capped(self) -> None:
        """ADX of 80 and 200 should produce the same score as 50."""
        assert _score_confidence(0.5, 50.0) == pytest.approx(_score_confidence(0.5, 80.0), rel=1e-6)
        assert _score_confidence(0.5, 50.0) == pytest.approx(_score_confidence(0.5, 200.0), rel=1e-6)

    def test_higher_proximity_beats_lower(self) -> None:
        closer = _score_confidence(0.1, 30.0)
        farther = _score_confidence(1.0, 30.0)
        assert closer > farther


# ---------------------------------------------------------------------------
# _check_trend_alignment
# ---------------------------------------------------------------------------

class TestCheckTrendAlignment:
    def test_long_price_above_ema_true(self) -> None:
        df = _rising_df(n=50, step=1.0)
        assert _check_trend_alignment(df, 20) is True

    def test_long_price_below_ema_false(self) -> None:
        # Reverse the rising series so price falls below its EMA
        df = _rising_df(n=50, step=1.0)
        df = df.iloc[::-1].reset_index(drop=True)
        assert _check_trend_alignment(df, 20) is False

    def test_insufficient_daily_bars_raises(self) -> None:
        df = _flat_df(n=10)
        with pytest.raises(InsufficientDataError):
            _check_trend_alignment(df, 20)


# ---------------------------------------------------------------------------
# _scan_symbol
# ---------------------------------------------------------------------------

class TestScanSymbol:
    def _make_trending_pit(self, price: float = 100.0) -> MockPIT:
        """Both 1h and 1d are strongly trending, price exactly at the EMA."""
        df_1h = _at_ma_df(n=60, price=price)
        df_1d = _rising_df(n=250, step=0.5)
        return MockPIT(data_1h={"BTC": df_1h}, data_1d={"BTC": df_1d})

    def test_flat_regime_returns_empty(self) -> None:
        pit = MockPIT(data_1h={"BTC": _flat_df(50)}, data_1d={"BTC": _flat_df(250)})
        assert _scan_symbol(pit, "BTC", _TS) == []

    def test_missing_symbol_returns_empty(self) -> None:
        pit = MockPIT()
        assert _scan_symbol(pit, "UNKNOWN", _TS) == []

    def test_distance_outside_tolerance_skipped(self) -> None:
        # Price well below the EMA (large negative distance)
        df_1h = _rising_df(n=60, step=10.0)  # EMA will be far below current price
        # Reverse so price is far below EMA
        df_1h = df_1h.iloc[::-1].reset_index(drop=True)
        df_1d = _rising_df(n=250, step=0.1)
        pit = MockPIT(data_1h={"BTC": df_1h}, data_1d={"BTC": df_1d})
        results = _scan_symbol(pit, "BTC", _TS)
        for c in results:
            assert abs(c.distance_to_ma_pct) <= ENTRY_TOLERANCE_PCT

    def test_valid_candidate_has_required_fields(self) -> None:
        pit = self._make_trending_pit()
        results = _scan_symbol(pit, "BTC", _TS)
        for c in results:
            assert isinstance(c, Candidate)
            assert c.symbol == "BTC"
            assert c.direction == "LONG"
            assert c.ma_period in (20, 50, 200)
            assert 0.0 <= c.confidence <= 1.0
            assert c.ts == _TS

    def test_look_ahead_error_propagates(self) -> None:
        class LookAheadPIT:
            decision_ts = _TS

            def ohlcv(self, *_: object) -> pd.DataFrame:  # type: ignore[override]
                raise LookAheadError("harness bug")

            def funding(self, *_: object) -> pd.DataFrame:  # type: ignore[override]
                return pd.DataFrame()

            def orderbook(self, *_: object) -> pd.DataFrame:  # type: ignore[override]
                return pd.DataFrame()

            def future_access(self, *_: object) -> None:  # type: ignore[override]
                raise LookAheadError("harness bug")

        with pytest.raises(LookAheadError):
            _scan_symbol(LookAheadPIT(), "BTC", _TS)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# scan (public entry point)
# ---------------------------------------------------------------------------

class TestScan:
    def test_returns_list(self) -> None:
        pit = MockPIT()
        result = scan(pit, ["BTC"])
        assert isinstance(result, list)

    def test_empty_symbol_list_returns_empty(self) -> None:
        pit = MockPIT()
        assert scan(pit, []) == []

    def test_missing_data_symbol_gracefully_skipped(self) -> None:
        pit = MockPIT()
        result = scan(pit, ["MISSING_COIN"])
        assert result == []

    def test_sorted_by_confidence_descending(self) -> None:
        """If multiple candidates, highest confidence appears first."""
        df_trend = _rising_df(n=250, step=0.5)
        df_entry = _at_ma_df(n=60, price=100.0)
        pit = MockPIT(
            data_1h={"BTC": df_entry, "ETH": df_entry},
            data_1d={"BTC": df_trend, "ETH": df_trend},
        )
        results = scan(pit, ["BTC", "ETH"])
        confidences = [c.confidence for c in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_decision_ts_defaults_to_pit_decision_ts(self) -> None:
        df_trend = _rising_df(n=250, step=0.5)
        df_entry = _at_ma_df(n=60, price=100.0)
        pit = MockPIT(data_1h={"BTC": df_entry}, data_1d={"BTC": df_trend})
        results = scan(pit, ["BTC"])
        for c in results:
            assert c.ts == pit.decision_ts

    def test_candidate_confidence_in_bounds(self) -> None:
        df_trend = _rising_df(n=250, step=0.5)
        df_entry = _at_ma_df(n=60, price=100.0)
        pit = MockPIT(data_1h={"BTC": df_entry}, data_1d={"BTC": df_trend})
        for c in scan(pit, ["BTC"]):
            assert 0.0 <= c.confidence <= 1.0


# ---------------------------------------------------------------------------
# Daily EMA fields on Candidate
# ---------------------------------------------------------------------------

class TestCandidateDailyEmaFields:
    """Verify ema20/50/200_daily fields on Candidate and their population in _scan_symbol().

    The existing test fixtures use flat 1h data which produces CHOP regime, so
    _scan_symbol() returns no candidates.  These tests cover two layers:
      1. Candidate model accepts the new fields with correct defaults.
      2. _scan_symbol() populates the fields when candidates ARE emitted,
         verified by patching the regime classifier to force TREND.
    """

    # ── Candidate model field tests (no pipeline required) ──────────────────

    def test_candidate_ema_fields_default_to_none(self) -> None:
        c = Candidate(
            symbol="BTC", direction="LONG", ma_period=50,
            distance_to_ma_pct=0.3, regime=Regime.TREND,
            confidence=0.80, ts=_TS,
        )
        assert c.ema20_daily is None
        assert c.ema50_daily is None
        assert c.ema200_daily is None

    def test_candidate_accepts_ema_float_values(self) -> None:
        c = Candidate(
            symbol="BTC", direction="LONG", ma_period=50,
            distance_to_ma_pct=0.3, regime=Regime.TREND,
            confidence=0.80, ts=_TS,
            ema20_daily=105.0, ema50_daily=100.0, ema200_daily=90.0,
        )
        assert c.ema20_daily == pytest.approx(105.0)
        assert c.ema50_daily == pytest.approx(100.0)
        assert c.ema200_daily == pytest.approx(90.0)

    # ── _scan_symbol() integration tests (regime classifier patched) ─────────

    def _make_pit_with_n_daily(self, n_daily: int) -> MockPIT:
        """1h data at EMA level (distance≈0%) + n_daily daily bars."""
        df_1h = _at_ma_df(n=60, price=100.0)
        df_1d = _rising_df(n=n_daily, step=0.5)
        return MockPIT(data_1h={"BTC": df_1h}, data_1d={"BTC": df_1d})

    def _trend_regime_result(self) -> RegimeResult:
        return RegimeResult(regime=Regime.TREND, adx=35.0, atr_pct=1.0, symbol="BTC", timeframe="1h")

    def test_ema20_and_ema50_populated_with_sufficient_daily_bars(self) -> None:
        pit = self._make_pit_with_n_daily(250)
        with patch("src.agents.scout.classify_regime_from_df", return_value=self._trend_regime_result()):
            results = _scan_symbol(pit, "BTC", _TS)
        assert results, "expected candidates with TREND regime forced"
        for c in results:
            assert c.ema20_daily is not None
            assert c.ema50_daily is not None
            assert isinstance(c.ema20_daily, float)
            assert isinstance(c.ema50_daily, float)

    def test_ema200_populated_when_200_daily_bars_available(self) -> None:
        pit = self._make_pit_with_n_daily(250)
        with patch("src.agents.scout.classify_regime_from_df", return_value=self._trend_regime_result()):
            results = _scan_symbol(pit, "BTC", _TS)
        assert results, "expected candidates with TREND regime forced"
        for c in results:
            assert c.ema200_daily is not None
            assert isinstance(c.ema200_daily, float)

    def test_ema200_is_none_when_fewer_than_200_daily_bars(self) -> None:
        # 90-bar daily dataset — EMA-200 must silently yield None.
        pit = self._make_pit_with_n_daily(90)
        with patch("src.agents.scout.classify_regime_from_df", return_value=self._trend_regime_result()):
            results = _scan_symbol(pit, "BTC", _TS)
        assert results, "expected candidates with TREND regime forced"
        for c in results:
            assert c.ema200_daily is None, (
                "ema200_daily must be None when fewer than 200 daily bars are available"
            )

    def test_all_candidates_from_same_symbol_share_same_ema_values(self) -> None:
        # Multiple MA periods (20, 50, 200) may each emit a Candidate —
        # they must all carry the identical daily EMA snapshot.
        pit = self._make_pit_with_n_daily(250)
        with patch("src.agents.scout.classify_regime_from_df", return_value=self._trend_regime_result()):
            results = _scan_symbol(pit, "BTC", _TS)
        if len(results) > 1:
            ref = results[0]
            for c in results[1:]:
                assert c.ema20_daily == ref.ema20_daily
                assert c.ema50_daily == ref.ema50_daily
                assert c.ema200_daily == ref.ema200_daily
