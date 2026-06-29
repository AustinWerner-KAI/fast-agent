"""Point-in-time data view — the look-ahead safety layer.

Every data access goes through PITDataView. Callers supply a ``decision_ts``
(the timestamp at which a hypothetical decision is being made); the view
returns only rows whose ``close_time`` (or equivalent timestamp) is <=
``decision_ts``.  Accessing data keyed on ``open_time`` is never done because
that would leak the open price of a bar that has not yet closed.

A deliberate ``future_access()`` method always raises ``LookAheadError`` so
the test suite can prove the guard is in place.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class LookAheadError(Exception):
    """Raised when code attempts to access data beyond decision_ts."""


class PITDataView:
    """Read-only, point-in-time slice of the parquet data store.

    Args:
        data_dir: Root directory of the parquet store produced by ingest.py.
        decision_ts: The timestamp representing "now" for this decision step.
            Only rows with close_time <= decision_ts are returned.
    """

    def __init__(self, data_dir: Path, decision_ts: datetime) -> None:
        self._data_dir = Path(data_dir)
        if decision_ts.tzinfo is None:
            decision_ts = decision_ts.replace(tzinfo=timezone.utc)
        self._decision_ts = decision_ts

    @property
    def decision_ts(self) -> datetime:
        """The cut-off timestamp for this view (inclusive)."""
        return self._decision_ts

    def ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Return OHLCV rows with close_time <= decision_ts.

        Args:
            symbol: Coin name (e.g. ``"BTC"``).
            timeframe: Candle interval (e.g. ``"1h"``).

        Returns:
            DataFrame with columns: open_time, close_time, open, high, low, close, volume.
            Empty DataFrame if the parquet file does not exist or no rows qualify.
        """
        path = self._data_dir / symbol / f"{timeframe}.parquet"
        if not path.exists():
            return _empty_ohlcv()

        df = pd.read_parquet(path)
        df = _ensure_utc(df, "close_time")
        mask = df["close_time"] <= self._decision_ts
        return df.loc[mask].reset_index(drop=True)

    def funding(self, symbol: str) -> pd.DataFrame:
        """Return funding rows with ts <= decision_ts.

        Args:
            symbol: Coin name (e.g. ``"BTC"``).

        Returns:
            DataFrame with columns: ts, rate.
            Empty DataFrame if no data exists or no rows qualify.
        """
        path = self._data_dir / symbol / "funding.parquet"
        if not path.exists():
            return _empty_funding()

        df = pd.read_parquet(path)
        df = _ensure_utc(df, "ts")
        mask = df["ts"] <= self._decision_ts
        return df.loc[mask].reset_index(drop=True)

    def orderbook(self, symbol: str) -> pd.DataFrame:  # noqa: ARG002
        """Return an empty DataFrame — orderbook snapshots are not stored historically.

        Args:
            symbol: Coin name (unused — no historical orderbook in the store).

        Returns:
            Empty DataFrame with columns: ts, side, price, size.
        """
        return pd.DataFrame(columns=["ts", "side", "price", "size"])

    def future_access(self, symbol: str, timeframe: str) -> None:  # noqa: ARG002
        """Always raises LookAheadError.

        This method exists solely so the test suite can assert that the
        look-ahead guard is reachable and functioning.  Production code must
        never call it.

        Args:
            symbol: Ignored.
            timeframe: Ignored.

        Raises:
            LookAheadError: unconditionally.
        """
        raise LookAheadError(
            f"Attempted to access future data beyond decision_ts={self._decision_ts.isoformat()}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce a datetime column to timezone-aware UTC in place (returns same df)."""
    if df[col].dtype == object:
        df[col] = pd.to_datetime(df[col], utc=True)
    elif df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize("UTC")
    else:
        df[col] = df[col].dt.tz_convert("UTC")
    return df


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "rate"])
