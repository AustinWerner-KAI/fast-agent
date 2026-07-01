"""Background position manager — monitors bot-owned positions and fires exits.

Runs as an asyncio background task, polling mark prices every 30 seconds.
Reads the active position set from execution.jsonl (filled state, not closed)
and checks each position against its TP and stop levels.

Exit logic (LONG only, all exits use reduce_only=True):
  TP1: mark >= entry + 2 × (entry - stop) → close 50%, move stop to breakeven
  TP2: mark >= entry + 3 × (entry - stop) → close 30% of original size
  TP3: mark >= entry + 5 × (entry - stop) → close remaining 20%
  Stop: mark <= current_stop → close 100%

All events are appended to logs/position_events.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

__all__ = ["PositionManager"]

_LOG = logging.getLogger(__name__)

_DEFAULT_EXEC_LOG = Path("logs/execution.jsonl")
_DEFAULT_EVENTS_LOG = Path("logs/position_events.jsonl")
_POLL_INTERVAL = 30  # seconds


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
        _LOG.info("position_manager: started (poll_interval=%ds)", self._poll_interval)
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

        closed_states = {"closed", "stop_hit", "tp3_hit", "error", "dry_run"}

        for vid, entry in latest.items():
            state = entry.get("state", "")
            symbol = entry.get("symbol", "")
            if not symbol or not vid:
                continue

            if state in closed_states:
                # Remove from managed set if previously tracked.
                self._positions.pop(vid, None)
                continue

            if state == "filled" and vid not in self._positions:
                try:
                    size = float(entry.get("fill_size", 0))
                    self._positions[vid] = _ManagedPosition(
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
                    _LOG.info(
                        "position_manager: tracking %s %s entry=%.6g stop=%.6g",
                        symbol, vid[:8], self._positions[vid].entry,
                        self._positions[vid].current_stop,
                    )
                except (ValueError, TypeError) as exc:
                    _LOG.warning("position_manager: could not parse filled entry %s: %s", vid, exc)

    # ------------------------------------------------------------------
    # Private — price check + exit logic
    # ------------------------------------------------------------------

    async def _check_all(self) -> None:
        if not self._positions:
            return

        for vid, pos in list(self._positions.items()):
            if pos.closed:
                continue
            mark = self._get_mark(pos.symbol)
            if mark is None:
                continue
            await self._evaluate(pos, mark)

    def _get_mark(self, symbol: str) -> float | None:
        try:
            return self._broker.get_mark_price(symbol)  # type: ignore[union-attr]
        except Exception:
            return None

    async def _evaluate(self, pos: _ManagedPosition, mark: float) -> None:
        stop_dist = pos.entry - pos.original_stop  # always positive for LONG

        # ── Hard stop ───────────────────────────────────────────────
        if mark <= pos.current_stop and not pos.closed:
            close_size = pos.remaining_size
            pnl = (mark - pos.entry) * close_size
            self._place_exit(pos, close_size, mark)
            self._log_event(pos, "stop_hit", close_pct=1.0, close_price=mark, pnl_usd=pnl)
            self._mark_closed_in_exec_log(pos.verdict_id, "stop_hit")
            pos.closed = True
            _LOG.info("position_manager: STOP %s @ %.6g pnl=%.2f", pos.symbol, mark, pnl)
            self._positions.pop(pos.verdict_id, None)
            return

        # ── TP1 ─────────────────────────────────────────────────────
        tp1_trigger = pos.entry + 2 * stop_dist
        if not pos.tp1_hit and mark >= tp1_trigger:
            close_size = round(pos.original_size * 0.50, 8)
            pnl = (mark - pos.entry) * close_size
            self._place_exit(pos, close_size, mark)
            self._log_event(pos, "tp1_hit", close_pct=0.50, close_price=mark, pnl_usd=pnl)
            pos.remaining_size = round(pos.remaining_size - close_size, 8)
            pos.current_stop = pos.entry   # move stop to breakeven
            pos.tp1_hit = True
            _LOG.info("position_manager: TP1 %s @ %.6g pnl=%.2f remaining=%.6g", pos.symbol, mark, pnl, pos.remaining_size)

        # ── TP2 ─────────────────────────────────────────────────────
        tp2_trigger = pos.entry + 3 * stop_dist
        if pos.tp1_hit and not pos.tp2_hit and mark >= tp2_trigger:
            close_size = round(pos.original_size * 0.30, 8)
            pnl = (mark - pos.entry) * close_size
            self._place_exit(pos, close_size, mark)
            self._log_event(pos, "tp2_hit", close_pct=0.30, close_price=mark, pnl_usd=pnl)
            pos.remaining_size = round(pos.remaining_size - close_size, 8)
            pos.tp2_hit = True
            _LOG.info("position_manager: TP2 %s @ %.6g pnl=%.2f remaining=%.6g", pos.symbol, mark, pnl, pos.remaining_size)

        # ── TP3 (full close) ─────────────────────────────────────────
        tp3_trigger = pos.entry + 5 * stop_dist
        if pos.tp2_hit and not pos.tp3_hit and mark >= tp3_trigger:
            close_size = pos.remaining_size
            pnl = (mark - pos.entry) * close_size
            self._place_exit(pos, close_size, mark)
            self._log_event(pos, "tp3_hit", close_pct=0.20, close_price=mark, pnl_usd=pnl)
            self._mark_closed_in_exec_log(pos.verdict_id, "tp3_hit")
            pos.tp3_hit = True
            pos.closed = True
            _LOG.info("position_manager: TP3 (full close) %s @ %.6g pnl=%.2f", pos.symbol, mark, pnl)
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
