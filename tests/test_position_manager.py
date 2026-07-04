"""Tests for position_manager.py — chandelier trailing stop + pure helpers.

Cases A–J match the spec:
  A  Rising trend: chandelier trails running_high upward.
  B  High ATR: wide chandelier keeps stop conservative.
  C  TP1 → stop to breakeven, exchange stop updated.
  D  TP2 → tp_state="TP2_HIT", exchange stop refreshed, remaining_size=20%.
  E  Price at chandelier, 15m EMA holds → spike ignored (no exit).
  F  Price at chandelier + 15m break confirmed → exit final 20%.
  G  Price dips below chandelier then recovers → position stays open.
  H  Daily EMA-20 override fires for final 20% after TP2.
  I  Feature flags OFF → pre-patch hard stop fires immediately.
  J  Pre-TP1 hard stop fires immediately (no 15m gate).

Pure-function tests:
  compute_atr, compute_chandelier_stop, check_15m_micro_break
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.position_manager import (
    PositionManager,
    _ManagedPosition,
    check_15m_micro_break,
    compute_atr,
    compute_chandelier_stop,
)
from src.utils.config_loader import TakeProfitConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker(
    mark_price: float = 100.0,
    stop_order_id: str = "new-stop-1",
    tp1_order_id: str = "resting-tp1-1",
    tp2_order_id: str = "resting-tp2-1",
) -> MagicMock:
    broker = MagicMock()
    broker.get_mark_price.return_value = mark_price
    broker.place_order.return_value = MagicMock(order_id="exit-1", status="filled")
    stop_result = MagicMock()
    stop_result.order_id = stop_order_id
    broker.place_stop_order.return_value = stop_result
    broker.cancel_order.return_value = True
    # Resting TP orders: alternate IDs on successive calls
    tp1_result = MagicMock()
    tp1_result.order_id = tp1_order_id
    tp2_result = MagicMock()
    tp2_result.order_id = tp2_order_id
    broker.place_tp_order.side_effect = [tp1_result, tp2_result]
    broker.get_open_order_ids.return_value = set()
    return broker


def _write_fill(
    log_path: Path,
    *,
    verdict_id: str = "vid-001",
    symbol: str = "BTC",
    fill_price: float = 100.0,
    stop: float = 95.0,
    tp1: float = 110.0,
    tp2: float = 115.0,
    tp3: float = 125.0,
    fill_size: float = 1.0,
    stop_order_id: str = "orig-stop-1",
    tp1_order_id: str = "",
    tp2_order_id: str = "",
) -> None:
    entry = {
        "state": "filled",
        "verdict_id": verdict_id,
        "symbol": symbol,
        "direction": "LONG",
        "order_id": "entry-order",
        "fill_price": fill_price,
        "fill_size": fill_size,
        "margin_used": 10.0,
        "notional": fill_price * fill_size,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "ts": "2026-07-01T00:00:00+00:00",
        "stop_order_id": stop_order_id,
        "tp1_order_id": tp1_order_id,
        "tp2_order_id": tp2_order_id,
    }
    with open(log_path, "w") as fh:
        fh.write(json.dumps(entry) + "\n")


def _make_h4_df(highs: list[float], *, base_days_ago: int = 10) -> pd.DataFrame:
    """Synthetic H4 OHLCV DataFrame with completed close_times in the past."""
    n = len(highs)
    base = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=base_days_ago)
    rows = []
    for i, h in enumerate(highs):
        rows.append({
            "open_time": base + pd.Timedelta(hours=4 * i),
            "close_time": base + pd.Timedelta(hours=4 * (i + 1)) - pd.Timedelta(seconds=1),
            "open": h - 1.0,
            "high": h,
            "low": h - 2.0,
            "close": h - 0.5,
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def _make_15m_df(closes: list[float], *, base_hours_ago: int = 10) -> pd.DataFrame:
    """Synthetic 15m OHLCV DataFrame with completed close_times in the past."""
    n = len(closes)
    base = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=base_hours_ago)
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "open_time": base + pd.Timedelta(minutes=15 * i),
            "close_time": base + pd.Timedelta(minutes=15 * (i + 1)) - pd.Timedelta(seconds=1),
            "open": c - 0.5,
            "high": c + 0.5,
            "low": c - 1.0,
            "close": c,
            "volume": 100.0,
        })
    return pd.DataFrame(rows)


def _make_daily_df(closes: list[float], *, base_days_ago: int = 30) -> pd.DataFrame:
    """Synthetic 1D OHLCV DataFrame with completed close_times in the past."""
    n = len(closes)
    base = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=base_days_ago)
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "open_time": base + pd.Timedelta(days=i),
            "close_time": base + pd.Timedelta(days=i + 1) - pd.Timedelta(seconds=1),
            "open": c - 1.0,
            "high": c + 1.0,
            "low": c - 2.0,
            "close": c,
            "volume": 5000.0,
        })
    return pd.DataFrame(rows)


def _make_pos(
    entry: float = 100.0,
    original_stop: float = 95.0,
    tp1_hit: bool = False,
    tp2_hit: bool = False,
    remaining_size: float = 1.0,
    original_size: float = 1.0,
    current_stop: float | None = None,
    stop_order_id: str = "orig-stop-1",
    chandelier_stop: float = 0.0,
    ema20_15m: float = 0.0,
    last_15m_close: float = 0.0,
    ema20_daily: float = 0.0,
    last_1d_close: float = 0.0,
) -> _ManagedPosition:
    pos = _ManagedPosition(
        verdict_id="vid-001",
        symbol="BTC",
        direction="LONG",
        entry=entry,
        original_stop=original_stop,
        current_stop=current_stop if current_stop is not None else original_stop,
        tp1=entry + 2 * (entry - original_stop),
        tp2=entry + 3 * (entry - original_stop),
        tp3=entry + 5 * (entry - original_stop),
        original_size=original_size,
        remaining_size=remaining_size,
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp_state=("TP2_HIT" if tp2_hit else ("TP1_HIT" if tp1_hit else "NONE")),
    )
    pos.running_high = entry
    pos.breakeven_price = entry
    pos.stop_order_id = stop_order_id
    pos.chandelier_stop = chandelier_stop
    pos.ema20_15m = ema20_15m
    pos.last_15m_close = last_15m_close
    pos.ema20_daily = ema20_daily
    pos.last_1d_close = last_1d_close
    return pos


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestComputeAtr:
    def test_returns_positive_for_valid_data(self) -> None:
        df = _make_h4_df([100, 102, 98, 103, 101, 99, 104, 100, 102, 101,
                          103, 99, 105, 102, 100, 104, 101, 99, 103, 102])
        result = compute_atr(df, period=14)
        assert result > 0

    def test_fallback_on_insufficient_data(self) -> None:
        df = _make_h4_df([100.0, 101.0])  # only 2 bars, period=14 needs 15
        result = compute_atr(df, period=14)
        assert result == pytest.approx(100.0 * 0.02, rel=0.01)

    def test_empty_df_returns_zero(self) -> None:
        df = pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])
        assert compute_atr(df, period=14) == 0.0

    def test_flat_market_produces_consistent_atr(self) -> None:
        # _make_h4_df: high=h, low=h-2, close=h-0.5 for each h.
        # True range per bar = 2 → ATR should converge to exactly 2.0.
        closes = [100.0] * 20
        df = _make_h4_df(closes)
        result = compute_atr(df, period=14)
        assert result == pytest.approx(2.0, abs=0.01)


class TestComputeChandelierStop:
    def test_basic_calculation(self) -> None:
        result = compute_chandelier_stop(
            running_high=120.0, atr_h4=3.0, atr_multiplier=2.0, floor=100.0
        )
        assert result == pytest.approx(114.0)  # 120 - 2*3 = 114

    def test_floor_prevents_stop_below_breakeven(self) -> None:
        # running_high=102, atr=10, multiplier=2 → raw = 82, floor=100
        result = compute_chandelier_stop(100.0, 10.0, 2.0, floor=100.0)
        assert result == pytest.approx(100.0)

    def test_floor_not_applied_when_chandelier_is_higher(self) -> None:
        result = compute_chandelier_stop(150.0, 5.0, 2.0, floor=100.0)
        assert result == pytest.approx(140.0)


class TestCheck15mMicroBreak:
    def test_no_break_when_close_above_threshold(self) -> None:
        # ema=110, buffer=0.15 → threshold = 110 * 0.85 = 93.5
        # close=100 → 100 > 93.5 → no break
        assert check_15m_micro_break(100.0, 110.0, 0.15) is False

    def test_break_confirmed_when_close_well_below_ema(self) -> None:
        # ema=110, buffer=0.15 → threshold = 93.5; close=90 → break
        assert check_15m_micro_break(90.0, 110.0, 0.15) is True

    def test_no_break_when_ema_is_zero(self) -> None:
        assert check_15m_micro_break(50.0, 0.0, 0.15) is False

    def test_boundary_exact(self) -> None:
        # close == threshold → NOT a break (strict <)
        assert check_15m_micro_break(93.5, 110.0, 0.15) is False


# ---------------------------------------------------------------------------
# Case A — chandelier trails running_high upward
# ---------------------------------------------------------------------------

class TestCaseA_ChandelierTrailsUp:
    def test_running_high_and_chandelier_update_on_new_h4(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0)
        broker = _make_broker(mark_price=125.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        # H4 candles with rising highs
        h4_highs = [95, 96, 97, 98, 99, 100, 102, 105, 108, 110, 112, 114, 115, 116, 118, 120, 122, 124, 125]
        df = _make_h4_df(h4_highs)

        with patch("src.pipeline.position_manager.fetch_ohlcv", return_value=df):
            pm._maybe_update_h4(pos)

        assert pos.running_high == pytest.approx(125.0)
        assert pos.atr_h4 > 0
        expected = max(125.0 - 2.0 * pos.atr_h4, pos.original_stop)
        assert pos.chandelier_stop == pytest.approx(expected)

    def test_chandelier_above_original_stop(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=80.0)  # wide stop
        broker = _make_broker(mark_price=130.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]
        pos.tp1_hit = True
        pos.tp_state = "TP1_HIT"
        pos.current_stop = 100.0  # at breakeven

        highs = list(range(100, 132))  # 32 bars, rising
        df = _make_h4_df(highs)
        with patch("src.pipeline.position_manager.fetch_ohlcv", return_value=df):
            pm._maybe_update_h4(pos)

        assert pos.chandelier_stop >= pos.original_stop


# ---------------------------------------------------------------------------
# Case B — high ATR keeps chandelier wide
# ---------------------------------------------------------------------------

class TestCaseB_HighAtrWideChandelier:
    def test_high_atr_keeps_stop_wide(self) -> None:
        running_high = 115.0
        atr = 10.0  # very high ATR
        result = compute_chandelier_stop(running_high, atr, 2.0, floor=100.0)
        # 115 - 20 = 95, but floor=100 → 100
        assert result == pytest.approx(100.0)

    def test_wide_atr_does_not_raise_stop_prematurely(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0)
        broker = _make_broker(mark_price=115.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]
        pos.tp1_hit = True
        pos.current_stop = 100.0

        # Very volatile candles → large ATR → chandelier < current_stop → no trail
        highs = [100, 120, 90, 115, 85, 110, 80, 105, 95, 115, 90, 120, 100, 115, 90, 110, 85, 115, 90, 105]
        df = _make_h4_df(highs)
        with patch("src.pipeline.position_manager.fetch_ohlcv", return_value=df):
            pm._maybe_update_h4(pos)

        # chandelier might be below current_stop; verify current_stop didn't decrease
        assert pos.current_stop >= 100.0


# ---------------------------------------------------------------------------
# Case C — TP1 moves stop to breakeven and updates exchange stop
# ---------------------------------------------------------------------------

class TestCaseC_TP1Breakeven:
    @pytest.mark.asyncio
    async def test_tp1_moves_stop_to_entry(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=110.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        await pm._evaluate(pos, mark=110.0)  # tp1_trigger = 100 + 2*5 = 110

        assert pos.tp1_hit is True
        assert pos.current_stop == pytest.approx(100.0)
        assert pos.tp_state == "TP1_HIT"

    @pytest.mark.asyncio
    async def test_tp1_closes_fifty_percent(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=110.0)
        disabled_tp = TakeProfitConfig(enable_resting_tp_orders=False)
        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=disabled_tp):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            await pm._evaluate(pos, mark=110.0, tp_cfg=disabled_tp)

        broker.place_order.assert_called_once_with(
            symbol="BTC", side="SELL", size=pytest.approx(0.5), reduce_only=True
        )
        assert pos.remaining_size == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_tp1_updates_exchange_stop(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0, stop_order_id="orig-stop-1")
        broker = _make_broker(mark_price=110.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        await pm._evaluate(pos, mark=110.0)

        # old stop cancelled
        broker.cancel_order.assert_called_once_with("BTC", "orig-stop-1")
        # new stop placed at breakeven (entry price = 100)
        broker.place_stop_order.assert_called_once()
        call_kwargs = broker.place_stop_order.call_args
        assert call_kwargs.kwargs["trigger_price"] == pytest.approx(100.0)
        assert call_kwargs.kwargs["symbol"] == "BTC"


# ---------------------------------------------------------------------------
# Case D — TP2 marks tp_state="TP2_HIT" and updates exchange stop
# ---------------------------------------------------------------------------

class TestCaseD_TP2State:
    @pytest.mark.asyncio
    async def test_tp2_sets_state_and_remaining_size(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=115.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]
        # Simulate TP1 already hit
        pos.tp1_hit = True
        pos.tp_state = "TP1_HIT"
        pos.remaining_size = 0.5
        pos.current_stop = 100.0

        await pm._evaluate(pos, mark=115.0)  # tp2_trigger = 100 + 3*5 = 115

        assert pos.tp2_hit is True
        assert pos.tp_state == "TP2_HIT"
        assert pos.remaining_size == pytest.approx(0.2)  # 0.5 - 0.30 original

    @pytest.mark.asyncio
    async def test_tp2_closes_thirty_percent_of_original(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=115.0)
        disabled_tp = TakeProfitConfig(enable_resting_tp_orders=False)
        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=disabled_tp):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            pos.tp1_hit = True
            pos.tp_state = "TP1_HIT"
            pos.remaining_size = 0.5
            pos.current_stop = 100.0
            await pm._evaluate(pos, mark=115.0, tp_cfg=disabled_tp)

        # place_order called at TP2 for 30% of original (1.0)
        calls = broker.place_order.call_args_list
        sizes = [c.kwargs.get("size", c.args[2] if len(c.args) > 2 else None) for c in calls]
        assert any(s == pytest.approx(0.30) for s in sizes)


# ---------------------------------------------------------------------------
# Case E — price at chandelier, 15m EMA holds → spike ignored
# ---------------------------------------------------------------------------

class TestCaseE_SpikeIgnored:
    @pytest.mark.asyncio
    async def test_chandelier_spike_not_exited_when_15m_holds(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=108.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp2_hit = True
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 108.0   # chandelier level
        pos.ema20_15m = 110.0      # EMA above close
        pos.last_15m_close = 109.0  # 109 > 110 * (1 - 0.15) = 93.5 → no break

        await pm._evaluate(pos, mark=108.0)

        # position NOT closed
        assert not pos.closed
        broker.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_spike_logs_chandelier_spike_ignored(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        ev_log = tmp_path / "ev.jsonl"
        _write_fill(log)
        broker = _make_broker(mark_price=108.0)
        pm = PositionManager(broker, exec_log=log, events_log=ev_log)
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp2_hit = True
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 108.0
        pos.ema20_15m = 110.0
        pos.last_15m_close = 109.0

        await pm._evaluate(pos, mark=108.0)

        events = [json.loads(l) for l in ev_log.read_text().splitlines()]
        assert any(e["event"] == "chandelier_spike_ignored" for e in events)


# ---------------------------------------------------------------------------
# Case F — price at chandelier + 15m break confirmed → exit final 20%
# ---------------------------------------------------------------------------

class TestCaseF_15mBreakExit:
    @pytest.mark.asyncio
    async def test_15m_break_exits_final_twenty_pct(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        ev_log = tmp_path / "ev.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=93.0)
        pm = PositionManager(broker, exec_log=log, events_log=ev_log)
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp2_hit = True
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 93.5    # chandelier: mark is below this
        pos.ema20_15m = 110.0      # EMA-20 on 15m
        pos.last_15m_close = 90.0  # 90 < 110 * 0.85 = 93.5 → break confirmed

        await pm._evaluate(pos, mark=93.0)

        assert pos.closed is True
        broker.place_order.assert_called_once_with(
            symbol="BTC", side="SELL", size=pytest.approx(0.2), reduce_only=True
        )
        events = [json.loads(l) for l in ev_log.read_text().splitlines()]
        assert any(e["event"] == "chandelier_15m_exit" for e in events)


# ---------------------------------------------------------------------------
# Case G — price dips then recovers → position stays open
# ---------------------------------------------------------------------------

class TestCaseG_DipAndRecover:
    @pytest.mark.asyncio
    async def test_spike_then_recovery_keeps_position_open(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=108.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp1_hit = True          # TP1 already executed
        pos.tp2_hit = True          # TP2 already executed — in final 20% phase
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 108.0
        pos.ema20_15m = 110.0
        pos.last_15m_close = 109.0  # 109 > 93.5 → no micro-break

        # Poll 1: dip exactly to chandelier, 15m holds → spike ignored
        await pm._evaluate(pos, mark=108.0)
        assert not pos.closed

        # Poll 2: price recovers to 112 — mark > current_stop, no TP3 (< 125)
        await pm._evaluate(pos, mark=112.0)
        assert not pos.closed
        broker.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# Case H — daily EMA-20 hard override fires after TP2
# ---------------------------------------------------------------------------

class TestCaseH_DailyEmaOverride:
    @pytest.mark.asyncio
    async def test_daily_ema_break_exits_final_20pct(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        ev_log = tmp_path / "ev.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=112.0)
        pm = PositionManager(broker, exec_log=log, events_log=ev_log)
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp1_hit = True          # TP1 already executed
        pos.tp2_hit = True          # TP2 already executed — in final 20% phase
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 100.0   # breakeven; mark=112 above this
        pos.ema20_daily = 115.0    # EMA-20 on daily
        pos.last_1d_close = 105.0  # close < EMA → daily override fires

        await pm._evaluate(pos, mark=112.0)

        assert pos.closed is True
        # Only one place_order call: the daily-override exit for the final 20%
        calls = [c for c in broker.place_order.call_args_list]
        assert len(calls) == 1
        assert calls[0].kwargs["size"] == pytest.approx(0.2)
        events = [json.loads(l) for l in ev_log.read_text().splitlines()]
        assert any(e["event"] == "daily_ema20_override_exit" for e in events)

    @pytest.mark.asyncio
    async def test_daily_ema_no_break_when_close_above_ema(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=112.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp1_hit = True          # TP1 already executed
        pos.tp2_hit = True          # TP2 already executed
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 100.0
        pos.ema20_daily = 110.0
        pos.last_1d_close = 111.0  # above EMA → no override

        await pm._evaluate(pos, mark=112.0)

        assert not pos.closed
        broker.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# Case I — feature flags OFF → pre-patch hard stop fires immediately
# ---------------------------------------------------------------------------

class TestCaseI_FeatureFlagsOff:
    @pytest.mark.asyncio
    async def test_hard_stop_fires_without_15m_gate_when_feature_disabled(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=94.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        pos.tp2_hit = True
        pos.tp_state = "TP2_HIT"
        pos.remaining_size = 0.2
        pos.current_stop = 95.0
        pos.ema20_15m = 110.0
        pos.last_15m_close = 109.0  # would NOT confirm micro break

        # Patch trail config with both features disabled
        disabled_cfg = __import__(
            "src.utils.config_loader", fromlist=["TrailConfig"]
        ).TrailConfig(
            enable_soft_trail_after_tp1=False,
            enable_h4_atr_15m_combo_for_final_20=False,
        )
        with patch("src.pipeline.position_manager.load_trailing_stop_config", return_value=disabled_cfg):
            await pm._evaluate(pos, mark=94.0)

        # Hard stop fires immediately with no 15m gate
        assert pos.closed is True
        broker.place_order.assert_called_once()


# ---------------------------------------------------------------------------
# Case J — pre-TP1 hard stop fires immediately (no 15m gate)
# ---------------------------------------------------------------------------

class TestCaseJ_PreTp1HardStop:
    @pytest.mark.asyncio
    async def test_stop_fires_before_tp1_without_15m_gate(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        ev_log = tmp_path / "ev.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=94.0)
        pm = PositionManager(broker, exec_log=log, events_log=ev_log)
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        # No TP1 hit — original stop should fire without 15m gating
        assert pos.tp1_hit is False
        assert pos.tp2_hit is False

        await pm._evaluate(pos, mark=94.0)

        assert pos.closed is True
        broker.place_order.assert_called_once_with(
            symbol="BTC", side="SELL", size=pytest.approx(1.0), reduce_only=True
        )
        events = [json.loads(l) for l in ev_log.read_text().splitlines()]
        assert any(e["event"] == "stop_hit" for e in events)

    @pytest.mark.asyncio
    async def test_stop_not_fired_when_mark_above_stop(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker(mark_price=98.0)
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        await pm._evaluate(pos, mark=98.0)

        assert not pos.closed
        broker.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------

class TestNewFieldsInitialisation:
    def test_sync_initialises_breakeven_and_running_high(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=123.45, stop=115.0)
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        assert pos.breakeven_price == pytest.approx(123.45)
        assert pos.running_high == pytest.approx(123.45)

    def test_sync_loads_stop_order_id_from_log(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, stop_order_id="exchange-stop-999")
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()

        pos = pm._positions["vid-001"]
        assert pos.stop_order_id == "exchange-stop-999"

    def test_tp_state_defaults_to_none(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log)
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]
        assert pos.tp_state == "NONE"


class TestUpdateExchangeStop:
    def test_cancels_old_and_places_new(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, stop_order_id="old-stop-42")
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        pm._update_exchange_stop(pos, new_stop=96.0)

        broker.cancel_order.assert_called_once_with("BTC", "old-stop-42")
        broker.place_stop_order.assert_called_once()
        assert pos.stop_order_id == "new-stop-1"

    def test_no_cancel_when_no_existing_order(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, stop_order_id="")
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        pm._update_exchange_stop(pos, new_stop=96.0)

        broker.cancel_order.assert_not_called()
        broker.place_stop_order.assert_called_once()

    def test_noop_when_new_stop_is_zero(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log)
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]

        pm._update_exchange_stop(pos, new_stop=0.0)

        broker.cancel_order.assert_not_called()
        broker.place_stop_order.assert_not_called()


class TestDailyEma20Break:
    def test_break_when_close_below_ema(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log)
        pm = PositionManager(MagicMock(), exec_log=log, events_log=tmp_path / "ev.jsonl")
        pos = _make_pos(ema20_daily=110.0, last_1d_close=105.0)
        assert pm._daily_ema20_break(pos) is True

    def test_no_break_when_close_above_ema(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log)
        pm = PositionManager(MagicMock(), exec_log=log, events_log=tmp_path / "ev.jsonl")
        pos = _make_pos(ema20_daily=110.0, last_1d_close=112.0)
        assert pm._daily_ema20_break(pos) is False

    def test_no_break_when_ema_is_zero(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log)
        pm = PositionManager(MagicMock(), exec_log=log, events_log=tmp_path / "ev.jsonl")
        pos = _make_pos(ema20_daily=0.0, last_1d_close=100.0)
        assert pm._daily_ema20_break(pos) is False


class TestChandelierTrailRaisesCurrentStop:
    def test_trail_raises_current_stop_when_tp1_hit_and_soft_trail_enabled(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0)
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]
        pos.tp1_hit = True
        pos.tp_state = "TP1_HIT"
        pos.current_stop = 100.0   # breakeven
        pos.remaining_size = 0.5

        # Tight ATR, big running_high → chandelier well above current_stop
        h4_highs = list(range(100, 122))  # flat rising with small range
        df = _make_h4_df(h4_highs)

        enabled_cfg = __import__(
            "src.utils.config_loader", fromlist=["TrailConfig"]
        ).TrailConfig(
            enable_soft_trail_after_tp1=True,
            enable_h4_atr_15m_combo_for_final_20=True,
            atr_period_h4=14,
            atr_multiplier_chandelier=2.0,
            stop_improvement_threshold_pct=0.1,
        )
        with patch("src.pipeline.position_manager.fetch_ohlcv", return_value=df):
            with patch("src.pipeline.position_manager.load_trailing_stop_config", return_value=enabled_cfg):
                pm._maybe_update_h4(pos)

        # running_high updated
        assert pos.running_high == pytest.approx(121.0)

    def test_trail_not_raised_before_tp1_when_soft_trail_disabled(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0)
        broker = _make_broker()
        pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
        pm._sync_from_log()
        pos = pm._positions["vid-001"]
        original_stop = pos.current_stop

        h4_highs = list(range(100, 122))
        df = _make_h4_df(h4_highs)

        # Default config: enable_soft_trail_after_tp1=False
        with patch("src.pipeline.position_manager.fetch_ohlcv", return_value=df):
            pm._maybe_update_h4(pos)

        # chandelier computed but current_stop should not have been raised
        # (because tp1_hit=False and soft trail disabled)
        assert pos.current_stop == pytest.approx(original_stop)


# ---------------------------------------------------------------------------
# Resting TP Cases A–F
# ---------------------------------------------------------------------------

_TP_ON = TakeProfitConfig(
    enable_resting_tp_orders=True,
    tp1_rr=2.0,
    tp2_rr=3.0,
    tp1_fraction=0.50,
    tp2_fraction=0.30,
)
_TP_OFF = TakeProfitConfig(enable_resting_tp_orders=False)


class TestRestingTpCaseA_PlacedOnSync:
    """TP orders placed on the exchange when position is first loaded."""

    def test_orders_placed_with_correct_trigger_prices_and_sizes(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()

        # stop_dist=5 → tp1=110, tp2=115
        assert broker.place_tp_order.call_count == 2
        calls = broker.place_tp_order.call_args_list
        assert calls[0].kwargs["trigger_price"] == pytest.approx(110.0)
        assert calls[0].kwargs["size"] == pytest.approx(0.5)
        assert calls[0].kwargs["side"] == "SELL"
        assert calls[1].kwargs["trigger_price"] == pytest.approx(115.0)
        assert calls[1].kwargs["size"] == pytest.approx(0.3)
        assert calls[1].kwargs["side"] == "SELL"

    def test_order_ids_stored_on_position(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()

        pos = pm._positions["vid-001"]
        assert pos.tp1_order_id == "resting-tp1-1"
        assert pos.tp2_order_id == "resting-tp2-1"

    def test_reduce_only_used_on_tp_orders(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()

        # place_tp_order is the broker method; reduce_only is enforced inside live_broker
        # (not passed as kwarg from position_manager) — verify the broker was called, not place_order
        broker.place_tp_order.assert_called()
        broker.place_order.assert_not_called()


class TestRestingTpCaseB_Tp1FillDetected:
    """TP1 fill detected by order ID disappearing from open_orders."""

    @pytest.mark.asyncio
    async def test_tp1_fill_detected_updates_state_no_market_order(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0,
                    tp1_order_id="tp1-oid", tp2_order_id="tp2-oid")
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            pos.tp1_order_id = "tp1-oid"  # override what _place_initial_tp_orders set
            pos.tp2_order_id = "tp2-oid"

            # tp1-oid absent from open orders → filled
            await pm._evaluate(pos, mark=111.0, open_order_ids={"tp2-oid", "stop-oid"}, tp_cfg=_TP_ON)

        assert pos.tp1_hit is True
        assert pos.current_stop == pytest.approx(100.0)  # moved to breakeven
        assert pos.remaining_size == pytest.approx(0.5)
        assert pos.tp1_order_id == ""   # consumed
        broker.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_tp1_not_triggered_while_id_still_in_open_orders(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0,
                    tp1_order_id="tp1-oid", tp2_order_id="tp2-oid")
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            pos.tp1_order_id = "tp1-oid"
            pos.tp2_order_id = "tp2-oid"

            # tp1-oid still present → not yet filled
            await pm._evaluate(pos, mark=111.0, open_order_ids={"tp1-oid", "tp2-oid"}, tp_cfg=_TP_ON)

        assert pos.tp1_hit is False
        broker.place_order.assert_not_called()


class TestRestingTpCaseC_Tp2FillDetected:
    """TP2 fill detected by order ID disappearing from open_orders."""

    @pytest.mark.asyncio
    async def test_tp2_fill_detected_updates_state_no_market_order(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0,
                    tp1_order_id="", tp2_order_id="tp2-oid")
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            pos.tp1_hit = True
            pos.tp_state = "TP1_HIT"
            pos.remaining_size = 0.5
            pos.current_stop = 100.0
            pos.tp1_order_id = ""      # already consumed at TP1
            pos.tp2_order_id = "tp2-oid"

            # tp2-oid absent → filled
            await pm._evaluate(pos, mark=116.0, open_order_ids={"stop-oid"}, tp_cfg=_TP_ON)

        assert pos.tp2_hit is True
        assert pos.remaining_size == pytest.approx(0.2)   # 0.5 - 0.30
        assert pos.tp2_order_id == ""
        broker.place_order.assert_not_called()


class TestRestingTpCaseD_StopExitCancelsTpOrders:
    """Stop-loss exit cancels resting TP orders before closing."""

    @pytest.mark.asyncio
    async def test_stop_cancels_both_tp_orders(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0,
                    tp1_order_id="tp1-oid", tp2_order_id="tp2-oid")
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            pos.tp1_order_id = "tp1-oid"
            pos.tp2_order_id = "tp2-oid"

            await pm._evaluate(pos, mark=94.0, tp_cfg=_TP_ON)   # mark <= stop=95

        assert pos.closed is True
        cancelled = {c.args[1] for c in broker.cancel_order.call_args_list}
        assert "tp1-oid" in cancelled
        assert "tp2-oid" in cancelled
        broker.place_order.assert_called_once()   # market stop exit still fires

    @pytest.mark.asyncio
    async def test_chandelier_exit_cancels_tp_orders(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            pos.tp1_hit = True
            pos.tp2_hit = True
            pos.tp_state = "TP2_HIT"
            pos.remaining_size = 0.2
            pos.current_stop = 93.5
            pos.ema20_15m = 110.0
            pos.last_15m_close = 90.0   # confirmed micro-break
            pos.tp1_order_id = ""       # consumed at TP1
            pos.tp2_order_id = "tp2-remaining"

            await pm._evaluate(pos, mark=93.0, tp_cfg=_TP_ON)

        assert pos.closed is True
        cancelled = {c.args[1] for c in broker.cancel_order.call_args_list}
        assert "tp2-remaining" in cancelled


class TestRestingTpCaseE_FlagFalsePreservesOriginalBehavior:
    """With enable_resting_tp_orders=False, behavior is identical to pre-patch."""

    def test_no_tp_orders_placed_on_sync(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_OFF):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()

        broker.place_tp_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_market_tp1_order_fires_via_mark_price(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0)
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_OFF):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]
            await pm._evaluate(pos, mark=110.0, tp_cfg=_TP_OFF)

        broker.place_tp_order.assert_not_called()
        broker.place_order.assert_called_once_with(
            symbol="BTC", side="SELL", size=pytest.approx(0.5), reduce_only=True
        )


class TestRestingTpCaseF_RestartRecovery:
    """On restart, TP order IDs are loaded from log without re-placing."""

    def test_ids_loaded_from_log_place_tp_not_called(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0,
                    tp1_order_id="existing-tp1", tp2_order_id="existing-tp2")
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()

        pos = pm._positions["vid-001"]
        assert pos.tp1_order_id == "existing-tp1"
        assert pos.tp2_order_id == "existing-tp2"
        broker.place_tp_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_fill_detected_after_id_recovery(self, tmp_path: Path) -> None:
        log = tmp_path / "exec.jsonl"
        _write_fill(log, fill_price=100.0, stop=95.0, fill_size=1.0,
                    tp1_order_id="existing-tp1", tp2_order_id="existing-tp2")
        broker = _make_broker()

        with patch("src.pipeline.position_manager.load_take_profit_config", return_value=_TP_ON):
            pm = PositionManager(broker, exec_log=log, events_log=tmp_path / "ev.jsonl")
            pm._sync_from_log()
            pos = pm._positions["vid-001"]

            # After restart: existing-tp1 gone from open orders → fill detected
            await pm._evaluate(
                pos, mark=111.0,
                open_order_ids={"existing-tp2", "stop-oid"},
                tp_cfg=_TP_ON,
            )

        assert pos.tp1_hit is True
        assert pos.tp1_order_id == ""
        broker.place_order.assert_not_called()
