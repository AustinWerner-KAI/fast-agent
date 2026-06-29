"""Ingest historical OHLCV and funding data from Hyperliquid into parquet.

Usage:
    python -m src.harness.ingest --symbols BTC ETH --tf 1h --days 90
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

_HL_URL = "https://api.hyperliquid.xyz/info"

# Hyperliquid interval strings supported by the candleSnapshot endpoint
VALID_TF = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}

# Maximum candles per request (Hyperliquid limit)
_BATCH = 5_000


def _hl_post(payload: dict[str, Any], timeout: float = 30.0) -> Any:
    """POST to Hyperliquid info endpoint and return parsed JSON."""
    resp = httpx.post(_HL_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Hyperliquid for a single time window.

    Args:
        symbol: Coin name as used by Hyperliquid (e.g. ``"BTC"``).
        timeframe: Interval string (e.g. ``"1h"``).
        start_ms: Window start in Unix milliseconds (inclusive).
        end_ms: Window end in Unix milliseconds (inclusive).

    Returns:
        DataFrame with columns: open_time, close_time, open, high, low, close, volume.
        Both timestamp columns are timezone-aware UTC.
    """
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": symbol, "interval": timeframe, "startTime": start_ms, "endTime": end_ms},
    }
    raw: list[dict[str, Any]] = _hl_post(payload)
    if not raw:
        return _empty_ohlcv()

    df = pd.DataFrame(raw)
    df = df.rename(columns={"t": "open_time", "T": "close_time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.sort_values("open_time").reset_index(drop=True)


def fetch_funding(symbol: str, start_ms: int) -> pd.DataFrame:
    """Fetch funding rate history from Hyperliquid.

    Args:
        symbol: Coin name (e.g. ``"BTC"``).
        start_ms: Fetch from this Unix millisecond timestamp onward.

    Returns:
        DataFrame with columns: ts (UTC datetime), rate (float).
    """
    payload = {"type": "fundingHistory", "coin": symbol, "startTime": start_ms}
    raw: list[dict[str, Any]] = _hl_post(payload)
    if not raw:
        return _empty_funding()

    df = pd.DataFrame(raw)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    return df[["ts", "rate"]].sort_values("ts").reset_index(drop=True)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "rate"])


def ingest_symbol(
    symbol: str,
    timeframe: str,
    days: int,
    data_dir: Path,
) -> None:
    """Fetch and persist OHLCV + funding for one symbol.

    Args:
        symbol: Coin name (e.g. ``"BTC"``).
        timeframe: Candle interval (e.g. ``"1h"``).
        days: How many calendar days of history to fetch.
        data_dir: Root directory for the parquet store.
    """
    sym_dir = data_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    end_ms = int(time.time() * 1_000)
    start_ms = end_ms - days * 86_400_000

    # --- OHLCV ---
    ohlcv_path = sym_dir / f"{timeframe}.parquet"
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        batch_end = min(cursor + _BATCH * _tf_ms(timeframe), end_ms)
        chunk = fetch_ohlcv(symbol, timeframe, cursor, batch_end)
        if chunk.empty:
            break
        frames.append(chunk)
        last_close_ms = int(chunk["close_time"].iloc[-1].timestamp() * 1_000)
        if last_close_ms <= cursor:
            break
        cursor = last_close_ms + 1
        time.sleep(0.1)

    if frames:
        ohlcv = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        ohlcv.to_parquet(ohlcv_path, index=False)
        print(f"  {symbol}/{timeframe}: {len(ohlcv)} candles → {ohlcv_path}")
    else:
        print(f"  {symbol}/{timeframe}: no data returned")

    # --- Funding ---
    funding_path = sym_dir / "funding.parquet"
    funding = fetch_funding(symbol, start_ms)
    if not funding.empty:
        funding.to_parquet(funding_path, index=False)
        print(f"  {symbol}/funding: {len(funding)} rows → {funding_path}")


def _tf_ms(timeframe: str) -> int:
    """Return candle duration in milliseconds for a given timeframe string."""
    _map = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
        "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
    }
    if timeframe not in _map:
        raise ValueError(f"Unknown timeframe: {timeframe!r}. Valid: {sorted(_map)}")
    return _map[timeframe]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest Hyperliquid data to parquet")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "ARB", "DOGE"])
    parser.add_argument("--tf", default="1h", help="Candle timeframe (e.g. 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--data-dir", default="./data_store", help="Parquet store root directory")
    args = parser.parse_args()

    if args.tf not in VALID_TF:
        parser.error(f"--tf must be one of {sorted(VALID_TF)}")

    data_dir = Path(args.data_dir)
    print(f"Ingesting {args.symbols} / {args.tf} / {args.days}d → {data_dir}")
    for sym in args.symbols:
        ingest_symbol(sym, args.tf, args.days, data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
