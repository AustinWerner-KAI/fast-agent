"""Candle-by-candle replay engine.

Streams the parquet store in chronological order, advancing ``decision_ts``
one closed candle at a time.  At each step it exposes a ``PITDataView`` so
the Scout→Arbiter chain sees exactly the data a live system would have had.

Fee and slippage are placeholder constants; plug in real exchange specs before
connecting this to a live paper-trading loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd

from src.harness.pit_data import PITDataView

FEE_RATE: float = 0.0006   # 6 bps taker (Hyperliquid default)
SLIPPAGE_BPS: float = 2.0  # 2 bps market-impact placeholder


@dataclass
class Position:
    """An open simulated position."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    size: float          # notional in USD
    entry_ts: datetime


@dataclass
class ReplayState:
    """Snapshot of world state at one replay step.

    Attributes:
        decision_ts: The close_time of the bar that just completed.
            All PITDataView calls are bounded to this timestamp.
        pit: Point-in-time data view; every accessor respects decision_ts.
        open_positions: Positions currently held by the simulated agent.
        equity: Current account equity in USD (updated by mark_to_market).
    """

    decision_ts: datetime
    pit: PITDataView
    open_positions: list[Position] = field(default_factory=list)
    equity: float = 100_000.0


class ReplayEngine:
    """Drives candle-by-candle replay over the parquet store.

    Args:
        data_dir: Root directory of the parquet store produced by ingest.py.
        symbols: Coins to include in the replay universe.
        timeframe: Candle interval to replay on (e.g. ``"1h"``).
        initial_equity: Starting account equity in USD.
    """

    def __init__(
        self,
        data_dir: Path,
        symbols: list[str],
        timeframe: str = "1h",
        initial_equity: float = 100_000.0,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._symbols = symbols
        self._timeframe = timeframe
        self._initial_equity = initial_equity

    def _all_close_times(self) -> list[datetime]:
        """Return the sorted union of all close_times across symbols."""
        seen: set[datetime] = set()
        for sym in self._symbols:
            path = self._data_dir / sym / f"{self._timeframe}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["close_time"])
            if df["close_time"].dt.tz is None:
                df["close_time"] = df["close_time"].dt.tz_localize("UTC")
            else:
                df["close_time"] = df["close_time"].dt.tz_convert("UTC")
            seen.update(df["close_time"].tolist())
        return sorted(seen)

    def stream(self) -> Iterator[ReplayState]:
        """Yield one ReplayState per closed candle bar in chronological order.

        decision_ts advances strictly forward — it equals the close_time of
        the bar that just completed, which is the latest timestamp any
        PITDataView call will ever return for that step.

        Yields:
            ReplayState with a fresh PITDataView at each bar boundary.
        """
        close_times = self._all_close_times()
        if not close_times:
            return

        state = ReplayState(
            decision_ts=close_times[0],
            pit=PITDataView(self._data_dir, close_times[0]),
            equity=self._initial_equity,
        )

        for ts in close_times:
            state = ReplayState(
                decision_ts=ts,
                pit=PITDataView(self._data_dir, ts),
                open_positions=list(state.open_positions),
                equity=state.equity,
            )
            yield state

    def mark_to_market(self, state: ReplayState) -> float:
        """Compute current equity including unrealised PnL on open positions.

        Uses the last close price visible in the PITDataView at the current
        ``decision_ts``.  Entry fills are assumed to have already had fee and
        slippage applied (placeholders — not yet deducted in this version).

        Args:
            state: Current replay state.

        Returns:
            Updated equity value in USD.
        """
        equity = state.equity
        for pos in state.open_positions:
            df = state.pit.ohlcv(pos.symbol, self._timeframe)
            if df.empty:
                continue
            last_price = float(df["close"].iloc[-1])
            entry = _fill_price(pos.entry_price, pos.direction)
            if pos.direction == "LONG":
                pnl = (last_price - entry) / entry * pos.size
            else:
                pnl = (entry - last_price) / entry * pos.size
            equity += pnl
        return equity


def _fill_price(price: float, direction: Literal["LONG", "SHORT"]) -> float:
    """Apply slippage to a simulated fill price.

    Longs pay a slightly higher price; shorts receive a slightly lower price.
    SLIPPAGE_BPS is a placeholder — replace with exchange-specific logic.
    """
    slip = price * SLIPPAGE_BPS / 10_000
    return price + slip if direction == "LONG" else price - slip
