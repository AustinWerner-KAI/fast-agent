"""Prove that PITDataView and ReplayEngine never expose future data.

7 tests — all use synthetic in-memory parquet written to pytest's tmp_path.
No live API calls are made.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.harness.pit_data import LookAheadError, PITDataView
from src.harness.replay import ReplayEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, start: datetime) -> pd.DataFrame:
    """Build ``n`` synthetic hourly OHLCV bars starting at ``start``."""
    rows = []
    for i in range(n):
        open_time = start + timedelta(hours=i)
        close_time = open_time + timedelta(hours=1) - timedelta(milliseconds=1)
        price = 100.0 + i
        rows.append({
            "open_time": open_time,
            "close_time": close_time,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1_000.0,
        })
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


def _make_funding(n: int, start: datetime) -> pd.DataFrame:
    """Build ``n`` synthetic funding rows, one per hour starting at ``start``."""
    rows = [
        {"ts": pd.Timestamp(start + timedelta(hours=i)), "rate": 0.0001 * i}
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _write_store(tmp_path: Path, symbol: str, tf: str, ohlcv: pd.DataFrame, funding: pd.DataFrame | None = None) -> Path:
    """Persist synthetic frames to tmp_path in the expected layout. Returns data_dir."""
    sym_dir = tmp_path / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(sym_dir / f"{tf}.parquet", index=False)
    if funding is not None:
        funding.to_parquet(sym_dir / "funding.parquet", index=False)
    return tmp_path


_START = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_SYMBOL = "BTC"
_TF = "1h"


# ---------------------------------------------------------------------------
# Test 1: ohlcv() filters future rows by close_time
# ---------------------------------------------------------------------------

def test_ohlcv_filters_future_rows(tmp_path: Path) -> None:
    """With 10 bars, decision_ts = bar 5 close_time → exactly 5 rows visible."""
    ohlcv = _make_ohlcv(10, _START)
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, ohlcv)

    decision_ts = ohlcv["close_time"].iloc[4]  # 5th bar (index 4)
    pit = PITDataView(data_dir, decision_ts)
    result = pit.ohlcv(_SYMBOL, _TF)

    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
    assert (result["close_time"] <= decision_ts).all(), "Future bar leaked through"


# ---------------------------------------------------------------------------
# Test 2: close_time == decision_ts is included (≤ not <)
# ---------------------------------------------------------------------------

def test_ohlcv_close_time_inclusive(tmp_path: Path) -> None:
    """A row whose close_time equals decision_ts must be included."""
    ohlcv = _make_ohlcv(3, _START)
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, ohlcv)

    decision_ts = ohlcv["close_time"].iloc[1]  # exactly bar 2's close
    pit = PITDataView(data_dir, decision_ts)
    result = pit.ohlcv(_SYMBOL, _TF)

    assert len(result) == 2
    assert result["close_time"].iloc[-1] == decision_ts


# ---------------------------------------------------------------------------
# Test 3: close_time = decision_ts + 1ms is excluded
# ---------------------------------------------------------------------------

def test_ohlcv_one_ms_future_excluded(tmp_path: Path) -> None:
    """A row whose close_time is 1 ms after decision_ts must not appear."""
    ohlcv = _make_ohlcv(5, _START)
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, ohlcv)

    # Set decision_ts to 1 ms before bar 3 closes
    decision_ts = ohlcv["close_time"].iloc[2] - timedelta(milliseconds=1)
    pit = PITDataView(data_dir, decision_ts)
    result = pit.ohlcv(_SYMBOL, _TF)

    assert all(ct <= decision_ts for ct in result["close_time"])
    # Bar 3 (index 2) must not appear
    assert ohlcv["close_time"].iloc[2] not in result["close_time"].values


# ---------------------------------------------------------------------------
# Test 4: funding() filters correctly by ts
# ---------------------------------------------------------------------------

def test_funding_filters_correctly(tmp_path: Path) -> None:
    """Funding rows with ts > decision_ts must be absent."""
    ohlcv = _make_ohlcv(5, _START)
    funding = _make_funding(10, _START)
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, ohlcv, funding)

    decision_ts = funding["ts"].iloc[4]  # 5th funding row
    pit = PITDataView(data_dir, decision_ts)
    result = pit.funding(_SYMBOL)

    assert len(result) == 5
    assert (result["ts"] <= decision_ts).all(), "Future funding row leaked through"
    assert funding["ts"].iloc[5] not in result["ts"].values


# ---------------------------------------------------------------------------
# Test 5: future_access() always raises LookAheadError
# ---------------------------------------------------------------------------

def test_future_access_raises_lookahead(tmp_path: Path) -> None:
    """Calling future_access() must raise LookAheadError unconditionally."""
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, _make_ohlcv(5, _START))
    pit = PITDataView(data_dir, _START)

    with pytest.raises(LookAheadError):
        pit.future_access(_SYMBOL, _TF)


# ---------------------------------------------------------------------------
# Test 6: ReplayEngine.stream() yields strictly increasing decision_ts
# ---------------------------------------------------------------------------

def test_replay_monotonic_ts(tmp_path: Path) -> None:
    """Each yielded decision_ts must be strictly greater than the previous one."""
    ohlcv = _make_ohlcv(10, _START)
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, ohlcv)

    engine = ReplayEngine(data_dir, [_SYMBOL], _TF)
    timestamps = [state.decision_ts for state in engine.stream()]

    assert len(timestamps) == 10
    for prev, curr in zip(timestamps, timestamps[1:]):
        assert curr > prev, f"Non-monotonic: {prev} → {curr}"


# ---------------------------------------------------------------------------
# Test 7: At each replay step, pit.ohlcv() shows no bar beyond decision_ts
# ---------------------------------------------------------------------------

def test_replay_pit_no_future_bar(tmp_path: Path) -> None:
    """At every replay step, the last visible candle's close_time == decision_ts."""
    ohlcv = _make_ohlcv(8, _START)
    data_dir = _write_store(tmp_path, _SYMBOL, _TF, ohlcv)

    engine = ReplayEngine(data_dir, [_SYMBOL], _TF)
    for state in engine.stream():
        visible = state.pit.ohlcv(_SYMBOL, _TF)
        assert not visible.empty, "Expected at least one visible bar"
        last_close = visible["close_time"].max()
        assert last_close == state.decision_ts, (
            f"Future bar visible: last_close={last_close}, decision_ts={state.decision_ts}"
        )
        # Also confirm nothing beyond decision_ts slipped through
        assert (visible["close_time"] <= state.decision_ts).all()
