"""Background position manager — monitors bot-owned positions and fires exits.

Runs as an asyncio background task, polling mark prices every 30 seconds.
Reads the active position set from execution.jsonl (filled state, not closed)
and checks each position against its TP and stop levels.

Exit logic (LONG only, all exits use reduce_only=True):
  TP1: mark >= entry + 2 × (entry - stop) → close 50%, move stop to breakeven
  TP2: mark >= entry + 3 × (entry - stop) → close 30% of original size
  TP3: mark >= entry + 5 × (entry - stop) → close remaining 20%
  Stop: mark <= current_stop → close 100%

Phase 1 / 2 additions (H4 ATR Chandelier + 15m micro-structure):
  After TP1: optionally trail stop up to chandelier level (enable_soft_trail_after_tp1).
  After TP2: exit final 20% only when BOTH chandelier is breached AND the most
             recent 15m candle close confirms a micro-structure break below EMA-20.
Phase 3: daily EMA-20 override — if last 1D close < EMA-20 daily while the
         final 20% is still open, exit immediately regardless of ATR/15m state.

All events are appended to logs/position_events.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from src.harness.ingest import fetch_ohlcv
from src.utils.config_loader import (
    TakeProfitConfig,
    TrailConfig,
    load_take_profit_config,
    load_trailing_stop_config,
)

__all__ = ["PositionManager"]

_LOG = logging.getLogger(__name__)

_DEFAULT_EXEC_LOG = Path("logs/execution.jsonl")
_DEFAULT_EVENTS_LOG = Path("logs/position_events.jsonl")
_POLL_INTERVAL = 30  # seconds

_H4_MS = 14_400_000   # 4 hours in milliseconds
_15M_MS = 900_000     # 15 minutes in milliseconds
_1D_MS = 86_400_000   # 1 day in milliseconds


# ---------------------------------------------------------------------------
# Pure helper functions (independently testable, no side effects)
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Wilder ATR from an OHLCV DataFrame.

    Uses pandas ewm with alpha=1/period (equivalent to Wilder's smoothing).

    Args:
        df: OHLCV DataFrame with columns ``high``, ``low``, ``close`` (oldest first).
        period: ATR period.

    Returns:
        Absolute ATR value; falls back to 2% of last close when data is insufficient.
    """
    if df.empty:
        return 0.0
    if len(df) < period + 1:
        return float(df["close"].iloc[-1]) * 0.02

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


def compute_chandelier_stop(
    running_high: float,
    atr_h4: float,
    atr_multiplier: float,
    floor: float,
) -> float:
    """Return the Chandelier Exit stop level, clamped to a minimum floor.

    Args:
        running_high: Highest H4 candle high since position entry.
        atr_h4: ATR(14) computed from H4 candles.
        atr_multiplier: Chandelier multiplier (e.g. 2.0).
        floor: Minimum stop level (original_stop before TP1; breakeven after).

    Returns:
        Stop price.  Never below ``floor``.
    """
    raw = running_high - atr_multiplier * atr_h4
    return max(raw, floor)


def check_15m_micro_break(last_15m_close: float, ema20_15m: float, buffer_pct: float) -> bool:
    """Return True when the last 15m candle confirms a micro-structure break.

    A break is confirmed when the close is more than ``buffer_pct`` (as a
    decimal fraction, e.g. 0.15 = 15%) below the 15m EMA-20.

    Args:
        last_15m_close: Most recent completed 15m candle close price.
        ema20_15m: EMA-20 computed on 15m closes.
        buffer_pct: Required gap below EMA (e.g. 0.15 = 15%).

    Returns:
        True when close < ema20 × (1 − buffer_pct).
    """
    if ema20_15m <= 0:
        return False
    return last_15m_close < ema20_15m * (1.0 - buffer_pct)


# ---------------------------------------------------------------------------
# Internal position state
# ---------------------------------------------------------------------------

@dataclass
class _ManagedPosition:
    verdict_id: str
    symbol: str
    direction: Literal["LONG"]
    entry: float
    original_stop: float
    current_stop: float
    tp1: float
    tp2: float
    tp3: float
    original_size: float
    remaining_size: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    closed: bool = False
    # ── Chandelier trailing stop state ────────────────────────────────────
    running_high: float = 0.0            # highest H4 high since entry
    atr_h4: float = 0.0                  # ATR(14) on H4 candles
    chandelier_stop: float = 0.0         # running_high − k × atr_h4
    last_h4_candle_ts: datetime | None = None
    last_15m_close: float = 0.0          # close of last completed 15m candle
    ema20_15m: float = 0.0               # EMA-20 on 15m closes
    last_15m_candle_ts: datetime | None = None
    last_1d_close: float = 0.0           # close of last completed daily candle
    ema20_daily: float = 0.0             # EMA-20 on daily closes
    last_1d_candle_ts: datetime | None = None
    tp_state: str = "NONE"               # "NONE" | "TP1_HIT" | "TP2_HIT"
    breakeven_price: float = 0.0         # fill price; floor for stop after TP1
    stop_order_id: str = ""              # exchange stop order ID (cancel+replace on trail)
    tp1_order_id: str = ""               # exchange TP1 trigger order ID ("" = not placed)
    tp2_order_id: str = ""               # exchange TP2 trigger order ID


# ---------------------------------------------------------------------------
# PositionManager
# ---------------------------------------------------------------------------

class PositionManager:
    """Background task that monitors and exits bot-owned positions.

    Args:
        broker: LiveBroker instance.
        exec_log: Path to execution.jsonl.
        events_log: Path to position_events.jsonl.
        poll_interval: Seconds between mark-price polls.
    """

    def __init__(
        self,
        broker: object,
        exec_log: Path | None = None,
        events_log: Path | None = None,
        poll_interval: int = _POLL_INTERVAL,
    ) -> None:
        self._broker = broker
        self._exec_log = exec_log or _DEFAULT_EXEC_LOG
        self._events_log = events_log or _DEFAULT_EVENTS_LOG
        self._poll_interval = poll_interval
        self._positions: dict[str, _ManagedPosition] = {}  # verdict_id → position

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Infinite poll loop — runs as asyncio.create_task()."""
        _LOG.warning("position_manager: started (poll_interval=%ds)", self._poll_interval)
        while True:
            try:
                self._sync_from_log()
                await self._check_all()
            except Exception as exc:
                _LOG.error("position_manager: unhandled error in loop: %s", exc)
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Private — position sync
    # ------------------------------------------------------------------

    def _sync_from_log(self) -> None:
        """Load active filled positions from execution.jsonl into memory."""
        if not self._exec_log.exists():
            return

        latest: dict[str, dict] = {}
        try:
            with self._exec_log.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        entry = json.loads(raw.strip())
                        vid = entry.get("verdict_id", "")
                        if vid:
                            latest[vid] = entry
                    except json.JSONDecodeError:
                        pass
        except OSError:
            return

        closed_states = {"closed", "stop_hit", "tp3_hit", "error", "dry_run",
                         "chandelier_15m_exit", "daily_ema20_override_exit"}

        for vid, entry in latest.items():
            state = entry.get("state", "")
            symbol = entry.get("symbol", "")
            if not symbol or not vid:
                continue

            if state in closed_states:
                self._positions.pop(vid, None)
                continue

            if state == "filled" and vid not in self._positions:
                try:
                    size = float(entry.get("fill_size", 0))
                    pos = _ManagedPosition(
                        verdict_id=vid,
                        symbol=symbol,
                        direction="LONG",
                        entry=float(entry.get("fill_price") or entry.get("entry", 0)),
                        original_stop=float(entry.get("stop", 0)),
                        current_stop=float(entry.get("stop", 0)),
                        tp1=float(entry.get("tp1", 0)),
                        tp2=float(entry.get("tp2", 0)),
                        tp3=float(entry.get("tp3", 0)),
                        original_size=size,
                        remaining_size=size,
                    )
                    # Initialise chandelier state from fill entry
                    pos.running_high = pos.entry
                    pos.breakeven_price = pos.entry
                    pos.stop_order_id = str(entry.get("stop_order_id", ""))
                    pos.tp1_order_id = str(entry.get("tp1_order_id", ""))
                    pos.tp2_order_id = str(entry.get("tp2_order_id", ""))
                    self._positions[vid] = pos
                    # Place resting TP orders if not already persisted from a previous run
                    tp_cfg = load_take_profit_config()
                    if tp_cfg.enable_resting_tp_orders and not pos.tp1_order_id:
                        self._place_initial_tp_orders(pos, tp_cfg)
                    _LOG.info(
                        "position_manager: tracking %s %s entry=%.6g stop=%.6g",
                        symbol, vid[:8], pos.entry, pos.current_stop,
                    )
                except (ValueError, TypeError) as exc:
                    _LOG.warning("position_manager: could not parse filled entry %s: %s", vid, exc)

    # ------------------------------------------------------------------
    # Private — candle state updates (called once per poll, per position)
    # ------------------------------------------------------------------

    def _maybe_update_h4(self, pos: _ManagedPosition) -> None:
        """Fetch H4 candles when a new bar has closed; update ATR + chandelier."""
        trail_cfg = load_trailing_stop_config()
        if not trail_cfg.enable_h4_atr_15m_combo_for_final_20 and not trail_cfg.enable_soft_trail_after_tp1:
            return

        # Skip if the next H4 bar cannot have closed yet
        if pos.last_h4_candle_ts is not None:
            next_h4 = pos.last_h4_candle_ts + timedelta(hours=4)
            if datetime.now(tz=timezone.utc) < next_h4:
                return

        period = trail_cfg.atr_period_h4
        n_bars = period + 5
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - (n_bars + 2) * _H4_MS

        try:
            df = fetch_ohlcv(pos.symbol, "4h", start_ms, now_ms)
        except Exception as exc:
            _LOG.warning("position_manager: H4 fetch failed %s: %s", pos.symbol, exc)
            return

        if df.empty:
            return

        now_pd = pd.Timestamp.now(tz="UTC")
        completed = df[df["close_time"] < now_pd].copy()
        if completed.empty:
            return

        last_close_time: pd.Timestamp = completed["close_time"].iloc[-1]

        if pos.last_h4_candle_ts is not None:
            stored_ts = pd.Timestamp(pos.last_h4_candle_ts)
            if last_close_time <= stored_ts:
                return  # no new H4 bar

        pos.last_h4_candle_ts = last_close_time.to_pydatetime().replace(tzinfo=timezone.utc)

        # Update running_high: max of all completed H4 highs in window
        pos.running_high = max(pos.running_high, float(completed["high"].max()))

        # Recompute ATR(14) on H4
        pos.atr_h4 = compute_atr(completed, period)
        if pos.atr_h4 <= 0:
            return

        # Floor: original_stop before TP1; breakeven (entry price) after TP1
        floor = pos.breakeven_price if pos.tp1_hit else pos.original_stop
        new_chandelier = compute_chandelier_stop(
            pos.running_high, pos.atr_h4, trail_cfg.atr_multiplier_chandelier, floor,
        )
        pos.chandelier_stop = new_chandelier

        # Raise current_stop when chandelier has improved by the threshold
        threshold_abs = pos.current_stop * (trail_cfg.stop_improvement_threshold_pct / 100.0)
        if new_chandelier > pos.current_stop + threshold_abs:
            should_trail = (
                (pos.tp2_hit and trail_cfg.enable_h4_atr_15m_combo_for_final_20)
                or (pos.tp1_hit and not pos.tp2_hit and trail_cfg.enable_soft_trail_after_tp1)
            )
            if should_trail:
                pos.current_stop = new_chandelier
                self._update_exchange_stop(pos, new_chandelier)
                _LOG.info(
                    "position_manager: CHANDELIER_TRAIL %s stop=%.6g chandelier=%.6g",
                    pos.symbol, new_chandelier, new_chandelier,
                )

    def _maybe_update_15m(self, pos: _ManagedPosition) -> None:
        """Fetch 15m candles when a new bar has closed; update ema20_15m."""
        trail_cfg = load_trailing_stop_config()
        if not trail_cfg.enable_h4_atr_15m_combo_for_final_20:
            return
        if not pos.tp2_hit:
            return  # only relevant for the final 20% gating

        if pos.last_15m_candle_ts is not None:
            next_15m = pos.last_15m_candle_ts + timedelta(minutes=15)
            if datetime.now(tz=timezone.utc) < next_15m:
                return

        n_bars = 25  # 20 + warmup buffer
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - (n_bars + 2) * _15M_MS

        try:
            df = fetch_ohlcv(pos.symbol, "15m", start_ms, now_ms)
        except Exception as exc:
            _LOG.warning("position_manager: 15m fetch failed %s: %s", pos.symbol, exc)
            return

        if df.empty:
            return

        now_pd = pd.Timestamp.now(tz="UTC")
        completed = df[df["close_time"] < now_pd].copy()
        if completed.empty:
            return

        last_close_time: pd.Timestamp = completed["close_time"].iloc[-1]

        if pos.last_15m_candle_ts is not None:
            stored_ts = pd.Timestamp(pos.last_15m_candle_ts)
            if last_close_time <= stored_ts:
                return

        pos.last_15m_candle_ts = last_close_time.to_pydatetime().replace(tzinfo=timezone.utc)
        pos.last_15m_close = float(completed["close"].iloc[-1])

        if len(completed) >= 20:
            pos.ema20_15m = float(
                completed["close"].astype(float).ewm(span=20, adjust=False).mean().iloc[-1]
            )

    def _maybe_update_daily(self, pos: _ManagedPosition) -> None:
        """Fetch 1D candles when a new bar has closed; update ema20_daily."""
        trail_cfg = load_trailing_stop_config()
        if not trail_cfg.enable_h4_atr_15m_combo_for_final_20:
            return
        if not pos.tp2_hit:
            return

        if pos.last_1d_candle_ts is not None:
            next_1d = pos.last_1d_candle_ts + timedelta(days=1)
            if datetime.now(tz=timezone.utc) < next_1d:
                return

        n_bars = 25
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - (n_bars + 2) * _1D_MS

        try:
            df = fetch_ohlcv(pos.symbol, "1d", start_ms, now_ms)
        except Exception as exc:
            _LOG.warning("position_manager: 1D fetch failed %s: %s", pos.symbol, exc)
            return

        if df.empty:
            return

        now_pd = pd.Timestamp.now(tz="UTC")
        completed = df[df["close_time"] < now_pd].copy()
        if completed.empty:
            return

        last_close_time: pd.Timestamp = completed["close_time"].iloc[-1]

        if pos.last_1d_candle_ts is not None:
            stored_ts = pd.Timestamp(pos.last_1d_candle_ts)
            if last_close_time <= stored_ts:
                return

        pos.last_1d_candle_ts = last_close_time.to_pydatetime().replace(tzinfo=timezone.utc)
        pos.last_1d_close = float(completed["close"].iloc[-1])

        if len(completed) >= 20:
            pos.ema20_daily = float(
                completed["close"].astype(float).ewm(span=20, adjust=False).mean().iloc[-1]
            )

    # ------------------------------------------------------------------
    # Private — exchange stop management
    # ------------------------------------------------------------------

    def _update_exchange_stop(self, pos: _ManagedPosition, new_stop: float) -> None:
        """Cancel the existing exchange stop and place a new one at new_stop.

        No-op when new_stop <= 0 or broker is unavailable.  Updates
        ``pos.stop_order_id`` to the new order ID on success.

        Args:
            pos: Managed position to update.
            new_stop: New trigger price for the stop order.
        """
        if new_stop <= 0:
            return

        if pos.stop_order_id:
            try:
                self._broker.cancel_order(pos.symbol, pos.stop_order_id)  # type: ignore[union-attr]
                _LOG.info(
                    "position_manager: STOP_CANCELLED %s oid=%s",
                    pos.symbol, pos.stop_order_id,
                )
            except Exception as exc:
                _LOG.warning(
                    "position_manager: cancel_stop failed %s oid=%s: %s",
                    pos.symbol, pos.stop_order_id, exc,
                )
            pos.stop_order_id = ""

        try:
            stop_side: Literal["BUY", "SELL"] = "SELL" if pos.direction == "LONG" else "BUY"
            result = self._broker.place_stop_order(  # type: ignore[union-attr]
                symbol=pos.symbol,
                side=stop_side,
                size=pos.remaining_size,
                trigger_price=new_stop,
            )
            pos.stop_order_id = str(result.order_id)
            _LOG.info(
                "position_manager: STOP_PLACED %s trigger=%.6g size=%.6g order_id=%s",
                pos.symbol, new_stop, pos.remaining_size, result.order_id,
            )
        except Exception as exc:
            _LOG.error(
                "position_manager: place_stop_order failed %s @ %.6g: %s",
                pos.symbol, new_stop, exc,
            )

    def _place_initial_tp_orders(self, pos: _ManagedPosition, tp_cfg: TakeProfitConfig) -> None:
        """Place resting TP1 and TP2 trigger orders on the exchange.

        Called once per position when first loaded from the fill log and
        enable_resting_tp_orders=True.  Persists order IDs to execution.jsonl
        so they survive restart without re-placement.

        Args:
            pos: Position for which to place TP orders.
            tp_cfg: Take-profit configuration.
        """
        stop_dist = pos.entry - pos.original_stop
        if stop_dist <= 0:
            _LOG.warning(
                "position_manager: _place_initial_tp_orders: stop_dist<=0 for %s, skipping",
                pos.symbol,
            )
            return

        tp1_price = pos.entry + tp_cfg.tp1_rr * stop_dist
        tp2_price = pos.entry + tp_cfg.tp2_rr * stop_dist
        tp1_size = round(pos.original_size * tp_cfg.tp1_fraction, 8)
        tp2_size = round(pos.original_size * tp_cfg.tp2_fraction, 8)
        side: Literal["BUY", "SELL"] = "SELL" if pos.direction == "LONG" else "BUY"

        try:
            r1 = self._broker.place_tp_order(  # type: ignore[union-attr]
                symbol=pos.symbol, side=side, size=tp1_size, trigger_price=tp1_price,
            )
            pos.tp1_order_id = str(r1.order_id)
            _LOG.info(
                "position_manager: TP1_ORDER_PLACED %s trigger=%.6g size=%.6g oid=%s",
                pos.symbol, tp1_price, tp1_size, pos.tp1_order_id,
            )
        except Exception as exc:
            _LOG.error("position_manager: place_tp_order(TP1) failed %s: %s", pos.symbol, exc)
            return

        try:
            r2 = self._broker.place_tp_order(  # type: ignore[union-attr]
                symbol=pos.symbol, side=side, size=tp2_size, trigger_price=tp2_price,
            )
            pos.tp2_order_id = str(r2.order_id)
            _LOG.info(
                "position_manager: TP2_ORDER_PLACED %s trigger=%.6g size=%.6g oid=%s",
                pos.symbol, tp2_price, tp2_size, pos.tp2_order_id,
            )
        except Exception as exc:
            _LOG.error("position_manager: place_tp_order(TP2) failed %s: %s", pos.symbol, exc)

        # Persist TP order IDs to execution.jsonl so restart recovers them
        persistence_entry = {
            "state": "filled",
            "verdict_id": pos.verdict_id,
            "symbol": pos.symbol,
            "direction": pos.direction,
            "fill_price": pos.entry,
            "fill_size": pos.original_size,
            "stop": pos.original_stop,
            "tp1": pos.tp1,
            "tp2": pos.tp2,
            "tp3": pos.tp3,
            "stop_order_id": pos.stop_order_id,
            "tp1_order_id": pos.tp1_order_id,
            "tp2_order_id": pos.tp2_order_id,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            self._exec_log.parent.mkdir(parents=True, exist_ok=True)
            with self._exec_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(persistence_entry) + "\n")
        except OSError as exc:
            _LOG.error("position_manager: failed to persist TP order IDs: %s", exc)

    def _cancel_tp_orders(self, pos: _ManagedPosition) -> None:
        """Cancel any active resting TP orders before closing the position.

        Best-effort: errors are logged but do not block the closing exit.

        Args:
            pos: Position whose TP orders should be cancelled.
        """
        for attr, oid in [("tp1_order_id", pos.tp1_order_id), ("tp2_order_id", pos.tp2_order_id)]:
            if oid:
                try:
                    self._broker.cancel_order(pos.symbol, oid)  # type: ignore[union-attr]
                    _LOG.info(
                        "position_manager: TP_ORDER_CANCELLED %s oid=%s", pos.symbol, oid,
                    )
                except Exception as exc:
                    _LOG.warning(
                        "position_manager: cancel_tp_order failed %s oid=%s: %s",
                        pos.symbol, oid, exc,
                    )
                setattr(pos, attr, "")

    # ------------------------------------------------------------------
    # Private — price check + exit logic
    # ------------------------------------------------------------------

    async def _check_all(self) -> None:
        if not self._positions:
            return

        # Fetch open order IDs once per cycle when any position has resting TPs
        tp_cfg = load_take_profit_config()
        open_order_ids: set[str] | None = None
        if tp_cfg.enable_resting_tp_orders and any(
            p.tp1_order_id or p.tp2_order_id
            for p in self._positions.values()
            if not p.closed
        ):
            try:
                open_order_ids = self._broker.get_open_order_ids()  # type: ignore[union-attr]
            except Exception as exc:
                _LOG.warning("position_manager: get_open_order_ids failed: %s", exc)
                open_order_ids = None  # fall back to mark-price state update

        for vid, pos in list(self._positions.items()):
            if pos.closed:
                continue
            mark = self._get_mark(pos.symbol)
            if mark is None:
                continue
            # Update candle-derived state when new bars have closed
            self._maybe_update_h4(pos)
            self._maybe_update_15m(pos)
            self._maybe_update_daily(pos)
            await self._evaluate(pos, mark, open_order_ids=open_order_ids, tp_cfg=tp_cfg)

    def _get_mark(self, symbol: str) -> float | None:
        try:
            return self._broker.get_mark_price(symbol)  # type: ignore[union-attr]
        except Exception:
            return None

    def _daily_ema20_break(self, pos: _ManagedPosition) -> bool:
        """Return True when last daily close < EMA-20 daily."""
        return pos.last_1d_close > 0 and pos.ema20_daily > 0 and pos.last_1d_close < pos.ema20_daily

    async def _evaluate(
        self,
        pos: _ManagedPosition,
        mark: float,
        open_order_ids: set[str] | None = None,
        tp_cfg: TakeProfitConfig | None = None,
    ) -> None:
        stop_dist = pos.entry - pos.original_stop  # always positive for LONG
        trail_cfg = load_trailing_stop_config()
        if tp_cfg is None:
            tp_cfg = load_take_profit_config()

        # ── Hard stop (with 15m gate for final 20% when trail is active) ─
        if mark <= pos.current_stop and not pos.closed:
            if (
                pos.tp2_hit
                and trail_cfg.enable_h4_atr_15m_combo_for_final_20
                and pos.ema20_15m > 0
            ):
                # 15m micro-structure confirmation required
                if check_15m_micro_break(pos.last_15m_close, pos.ema20_15m, trail_cfg.micro_break_buffer_pct):
                    close_size = pos.remaining_size
                    pnl = (mark - pos.entry) * close_size
                    self._cancel_tp_orders(pos)
                    self._place_exit(pos, close_size, mark)
                    self._log_event(pos, "chandelier_15m_exit", close_pct=1.0, close_price=mark, pnl_usd=pnl)
                    self._mark_closed_in_exec_log(pos.verdict_id, "chandelier_15m_exit")
                    pos.closed = True
                    _LOG.info(
                        "position_manager: CHANDELIER_15M_EXIT %s @ %.6g pnl=%.2f",
                        pos.symbol, mark, pnl,
                    )
                    self._positions.pop(pos.verdict_id, None)
                else:
                    # Wick rejected — log and wait for next poll
                    self._log_event(pos, "chandelier_spike_ignored", close_pct=0.0, close_price=mark, pnl_usd=0.0)
                    _LOG.info(
                        "position_manager: chandelier_spike_ignored %s mark=%.6g stop=%.6g ema15=%.6g",
                        pos.symbol, mark, pos.current_stop, pos.ema20_15m,
                    )
            else:
                # Pre-TP2 or trail disabled: original hard stop logic
                close_size = pos.remaining_size
                pnl = (mark - pos.entry) * close_size
                self._cancel_tp_orders(pos)
                self._place_exit(pos, close_size, mark)
                self._log_event(pos, "stop_hit", close_pct=1.0, close_price=mark, pnl_usd=pnl)
                self._mark_closed_in_exec_log(pos.verdict_id, "stop_hit")
                pos.closed = True
                _LOG.info("position_manager: STOP %s @ %.6g pnl=%.2f", pos.symbol, mark, pnl)
                self._positions.pop(pos.verdict_id, None)
            return

        # ── TP1 ─────────────────────────────────────────────────────
        # With resting TPs: detect fill by order ID disappearing from open_orders.
        # Without resting TPs (or open_order_ids unavailable): use mark-price check
        # but do NOT fire a market order when resting orders are active.
        if not pos.tp1_hit:
            tp1_filled = False
            if tp_cfg.enable_resting_tp_orders and pos.tp1_order_id:
                if open_order_ids is not None:
                    # Order ID gone from open orders → filled by exchange
                    tp1_filled = pos.tp1_order_id not in open_order_ids
                # If open_order_ids is None (API failure): use mark as fallback signal
                # but skip market order placement (resting order will still fire)
                elif mark >= pos.entry + tp_cfg.tp1_rr * stop_dist:
                    tp1_filled = True
            elif not tp_cfg.enable_resting_tp_orders:
                tp1_filled = mark >= pos.entry + 2 * stop_dist

            if tp1_filled:
                close_size = round(pos.original_size * tp_cfg.tp1_fraction, 8)
                pnl = (mark - pos.entry) * close_size
                if not tp_cfg.enable_resting_tp_orders:
                    self._place_exit(pos, close_size, mark)
                pos.tp1_order_id = ""  # consumed
                self._log_event(pos, "tp1_hit", close_pct=tp_cfg.tp1_fraction, close_price=mark, pnl_usd=pnl)
                pos.remaining_size = round(pos.remaining_size - close_size, 8)
                pos.current_stop = pos.entry   # move stop to breakeven
                pos.tp1_hit = True
                pos.tp_state = "TP1_HIT"
                self._update_exchange_stop(pos, pos.entry)
                _LOG.info(
                    "position_manager: TP1 %s @ %.6g pnl=%.2f remaining=%.6g",
                    pos.symbol, mark, pnl, pos.remaining_size,
                )

        # ── TP2 ─────────────────────────────────────────────────────
        if pos.tp1_hit and not pos.tp2_hit:
            tp2_filled = False
            if tp_cfg.enable_resting_tp_orders and pos.tp2_order_id:
                if open_order_ids is not None:
                    tp2_filled = pos.tp2_order_id not in open_order_ids
                elif mark >= pos.entry + tp_cfg.tp2_rr * stop_dist:
                    tp2_filled = True
            elif not tp_cfg.enable_resting_tp_orders:
                tp2_filled = mark >= pos.entry + 3 * stop_dist

            if tp2_filled:
                close_size = round(pos.original_size * tp_cfg.tp2_fraction, 8)
                pnl = (mark - pos.entry) * close_size
                if not tp_cfg.enable_resting_tp_orders:
                    self._place_exit(pos, close_size, mark)
                pos.tp2_order_id = ""  # consumed
                self._log_event(pos, "tp2_hit", close_pct=tp_cfg.tp2_fraction, close_price=mark, pnl_usd=pnl)
                pos.remaining_size = round(pos.remaining_size - close_size, 8)
                pos.tp2_hit = True
                pos.tp_state = "TP2_HIT"
                stop_level = pos.chandelier_stop if (
                    trail_cfg.enable_h4_atr_15m_combo_for_final_20 and pos.chandelier_stop > pos.current_stop
                ) else pos.current_stop
                self._update_exchange_stop(pos, stop_level)
                _LOG.info(
                    "position_manager: TP2 %s @ %.6g pnl=%.2f remaining=%.6g",
                    pos.symbol, mark, pnl, pos.remaining_size,
                )

        # ── TP3 (full close) ─────────────────────────────────────────
        tp3_trigger = pos.entry + 5 * stop_dist
        if pos.tp2_hit and not pos.tp3_hit and mark >= tp3_trigger:
            close_size = pos.remaining_size
            pnl = (mark - pos.entry) * close_size
            self._cancel_tp_orders(pos)
            self._place_exit(pos, close_size, mark)
            self._log_event(pos, "tp3_hit", close_pct=0.20, close_price=mark, pnl_usd=pnl)
            self._mark_closed_in_exec_log(pos.verdict_id, "tp3_hit")
            pos.tp3_hit = True
            pos.closed = True
            _LOG.info("position_manager: TP3 (full close) %s @ %.6g pnl=%.2f", pos.symbol, mark, pnl)
            self._positions.pop(pos.verdict_id, None)
            return

        # ── Phase 3: daily EMA-20 hard override ─────────────────────
        if pos.tp2_hit and not pos.tp3_hit and not pos.closed:
            if (
                trail_cfg.enable_h4_atr_15m_combo_for_final_20
                and self._daily_ema20_break(pos)
            ):
                close_size = pos.remaining_size
                pnl = (mark - pos.entry) * close_size
                self._cancel_tp_orders(pos)
                self._place_exit(pos, close_size, mark)
                self._log_event(pos, "daily_ema20_override_exit", close_pct=1.0, close_price=mark, pnl_usd=pnl)
                self._mark_closed_in_exec_log(pos.verdict_id, "daily_ema20_override_exit")
                pos.tp3_hit = True
                pos.closed = True
                _LOG.info(
                    "position_manager: DAILY_EMA20_OVERRIDE %s @ %.6g last_1d_close=%.6g ema=%.6g",
                    pos.symbol, mark, pos.last_1d_close, pos.ema20_daily,
                )
                self._positions.pop(pos.verdict_id, None)

    def _place_exit(self, pos: _ManagedPosition, size: float, mark: float) -> None:
        try:
            self._broker.place_order(  # type: ignore[union-attr]
                symbol=pos.symbol,
                side="SELL",
                size=size,
                reduce_only=True,
            )
        except Exception as exc:
            _LOG.error(
                "position_manager: exit order failed %s size=%.6g: %s",
                pos.symbol, size, exc,
            )

    def _log_event(
        self,
        pos: _ManagedPosition,
        event: str,
        close_pct: float,
        close_price: float,
        pnl_usd: float,
    ) -> None:
        entry = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "verdict_id": pos.verdict_id,
            "symbol": pos.symbol,
            "event": event,
            "close_pct": close_pct,
            "close_price": close_price,
            "pnl_usd": round(pnl_usd, 4),
        }
        try:
            self._events_log.parent.mkdir(parents=True, exist_ok=True)
            with self._events_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            _LOG.error("position_manager: failed to write events log: %s", exc)

    def _mark_closed_in_exec_log(self, verdict_id: str, final_state: str) -> None:
        """Append a terminal state entry to execution.jsonl."""
        entry = {
            "state": final_state,
            "verdict_id": verdict_id,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            self._exec_log.parent.mkdir(parents=True, exist_ok=True)
            with self._exec_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            _LOG.error("position_manager: failed to write exec log: %s", exc)
