"""Async CoinGlass API client with 5-minute TTL cache.

All fetch functions are non-fatal: on any error they log a WARNING and
return None so the calling stage continues without CoinGlass data.
Liquidation fetches bypass the cache (always fresh per spec).

Auth: ``COINGLASS_API_KEY`` env var passed as ``CG-API-KEY`` HTTP header.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

import aiohttp

from src.data.coinglass_models import (
    FundingSnapshot,
    LiquidationMap,
    LSRatio,
    OISnapshot,
)

_log = logging.getLogger(__name__)

_BASE_URL = "https://open-api-v4.coinglass.com"
_API_KEY_ENV = "COINGLASS_API_KEY"
_CACHE_TTL: float = 300.0  # 5 minutes

_cache: dict[str, tuple[object, float]] = {}


class CoinGlassError(Exception):
    """Raised by the client; callers catch this to degrade gracefully."""


def _api_key() -> str:
    key = os.getenv(_API_KEY_ENV)
    if not key:
        raise CoinGlassError(f"{_API_KEY_ENV} not set")
    return key


def _cache_get(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if entry and entry[1] > time.monotonic():
        return entry[0]
    return None


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (value, time.monotonic() + _CACHE_TTL)


async def _get(path: str, params: dict) -> dict:
    """Perform a single authenticated GET request against the CoinGlass API.

    Args:
        path: URL path, e.g. ``"/api/futures/funding-rate/history"``.
        params: Query parameters.

    Returns:
        Parsed JSON response dict.

    Raises:
        CoinGlassError: On HTTP error or a non-success API response.
    """
    url = f"{_BASE_URL}{path}"
    headers = {"CG-API-KEY": _api_key()}
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise CoinGlassError(f"HTTP {resp.status}: {body[:200]}")
            data = await resp.json()
    if not data.get("success", True):
        raise CoinGlassError(f"API error: {data.get('msg', 'unknown')}")
    return data


def _normalize_funding_rate(raw: float) -> float:
    """Convert funding rate to decimal fraction.

    CoinGlass v4 returns rates as percentages (e.g. 0.01 = 0.01% per 8h).
    Values > 0.1 are almost certainly already in percent; divide by 100.

    Args:
        raw: Raw value from the API.

    Returns:
        Rate as a decimal fraction (e.g. 0.0001 = 0.01% per 8h).
    """
    if abs(raw) > 0.1:
        return raw / 100.0
    return raw


async def get_funding(symbol: str) -> Optional[FundingSnapshot]:
    """Fetch the latest 8h perpetual funding rate for *symbol*.

    Args:
        symbol: Coin name, e.g. ``"BTC"``.

    Returns:
        :class:`FundingSnapshot` with ``rate`` as a decimal fraction, or
        None when the API is unavailable or returns no data.
    """
    key = f"funding:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        data = await _get(
            "/api/futures/funding-rate/history",
            {"symbol": symbol, "exchange": "Binance", "interval": "h8", "limit": "1"},
        )
        rows = data.get("data") or []
        if not rows:
            _log.warning("coinglass_funding: empty response for %s", symbol)
            return None
        row = rows[-1] if isinstance(rows, list) else rows
        raw_rate = float(row.get("c", row.get("close", row.get("fundingRate", 0))))
        snap = FundingSnapshot(
            symbol=symbol,
            rate=_normalize_funding_rate(raw_rate),
            timestamp=datetime.now(timezone.utc),
        )
        _cache_set(key, snap)
        return snap
    except Exception as exc:  # noqa: BLE001
        _log.warning("coinglass_funding: %s for %s — %s", type(exc).__name__, symbol, exc)
        return None


async def get_oi(symbol: str) -> Optional[OISnapshot]:
    """Fetch current open interest and 24h change for *symbol*.

    Aggregates OI across all exchanges returned by the API.

    Args:
        symbol: Coin name, e.g. ``"BTC"``.

    Returns:
        :class:`OISnapshot`, or None when unavailable.
    """
    key = f"oi:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        data = await _get(
            "/api/futures/open-interest/exchange-list",
            {"symbol": symbol},
        )
        rows = data.get("data") or []
        if not rows:
            _log.warning("coinglass_oi: empty response for %s", symbol)
            return None
        total_oi = 0.0
        for row in rows:
            amt = float(row.get("openInterestAmount", row.get("openInterest", 0)))
            price = float(row.get("price", 1.0))
            total_oi += amt * price
        # 24h change is typically on the first/aggregated row
        change_field = (
            "openInterestChangePercent24h",
            "open_interest_change_percent_24h",
            "changePercent24h",
        )
        change_24h = 0.0
        for field in change_field:
            if field in rows[0]:
                change_24h = float(rows[0][field])
                break
        snap = OISnapshot(
            symbol=symbol,
            oi_usd=total_oi,
            oi_change_24h_pct=change_24h,
            timestamp=datetime.now(timezone.utc),
        )
        _cache_set(key, snap)
        return snap
    except Exception as exc:  # noqa: BLE001
        _log.warning("coinglass_oi: %s for %s — %s", type(exc).__name__, symbol, exc)
        return None


async def get_liquidations(symbol: str, reference_price: float) -> Optional[LiquidationMap]:
    """Fetch liquidation heatmap and split clusters above/below *reference_price*.

    Liquidations bypass the TTL cache (always fetched fresh).

    Args:
        symbol: Coin name, e.g. ``"BTC"``.
        reference_price: Entry price used to partition the heatmap.

    Returns:
        :class:`LiquidationMap`, or None when the API is unavailable.
    """
    try:
        data = await _get(
            "/api/futures/liquidation/heatmap/model1",
            {"symbol": symbol, "exchange": "All", "range": "3d"},
        )
        heat = data.get("data") or {}
        price_list = heat.get("priceList") or heat.get("y") or []
        liq_list = heat.get("liqAmountList") or heat.get("data") or []

        below = 0.0
        above = 0.0
        for price_raw, liq_raw in zip(price_list, liq_list):
            try:
                price = float(price_raw)
                if isinstance(liq_raw, list):
                    amount = sum(float(x) for x in liq_raw)
                else:
                    amount = float(liq_raw)
                if price < reference_price:
                    below += amount
                else:
                    above += amount
            except (TypeError, ValueError):
                continue

        return LiquidationMap(
            symbol=symbol,
            liquidations_below_usd=below,
            liquidations_above_usd=above,
            reference_price=reference_price,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("coinglass_liquidations: %s for %s — %s", type(exc).__name__, symbol, exc)
        return None


async def get_ls_ratio(symbol: str) -> Optional[LSRatio]:
    """Fetch the latest global long/short account ratio for *symbol*.

    Args:
        symbol: Coin name, e.g. ``"BTC"``.

    Returns:
        :class:`LSRatio`, or None when unavailable.
    """
    key = f"ls_ratio:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        data = await _get(
            "/api/futures/global-long-short-account-ratio/history",
            {"symbol": symbol, "exchange": "Binance", "interval": "h4", "limit": "1"},
        )
        rows = data.get("data") or []
        if not rows:
            _log.warning("coinglass_ls_ratio: empty response for %s", symbol)
            return None
        row = rows[-1] if isinstance(rows, list) else rows
        long_raw = float(row.get("longAccount", row.get("longRatio", row.get("long", 0.5))))
        short_raw = float(row.get("shortAccount", row.get("shortRatio", row.get("short", 0.5))))
        # Normalize: if values are in [0, 1] convert to percent
        if long_raw <= 1.0:
            long_raw *= 100.0
            short_raw *= 100.0
        snap = LSRatio(
            symbol=symbol,
            long_pct=long_raw,
            short_pct=short_raw,
            timestamp=datetime.now(timezone.utc),
        )
        _cache_set(key, snap)
        return snap
    except Exception as exc:  # noqa: BLE001
        _log.warning("coinglass_ls_ratio: %s for %s — %s", type(exc).__name__, symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Synchronous batch fetch (for use in the synchronous main.py pipeline)
# ---------------------------------------------------------------------------

@dataclass
class CoinGlassSnapshot:
    """All four CoinGlass data points for one symbol at one point in time.

    All fields are Optional — any individual fetch failure leaves that field
    None while the rest are still populated.
    """

    symbol: str
    funding: Optional[FundingSnapshot] = None
    oi: Optional[OISnapshot] = None
    liquidations: Optional[LiquidationMap] = None
    ls_ratio: Optional[LSRatio] = None

    # Convenience accessors ------------------------------------------------

    @property
    def funding_rate(self) -> Optional[float]:
        return self.funding.rate if self.funding else None

    @property
    def oi_change_24h_pct(self) -> Optional[float]:
        return self.oi.oi_change_24h_pct if self.oi else None

    @property
    def liquidations_below_usd(self) -> Optional[float]:
        return self.liquidations.liquidations_below_usd if self.liquidations else None

    @property
    def liquidations_above_usd(self) -> Optional[float]:
        return self.liquidations.liquidations_above_usd if self.liquidations else None

    @property
    def ls_long_pct(self) -> Optional[float]:
        return self.ls_ratio.long_pct if self.ls_ratio else None

    @property
    def ls_short_pct(self) -> Optional[float]:
        return self.ls_ratio.short_pct if self.ls_ratio else None

    def to_dict(self) -> dict:
        """Serialise to a plain dict for injection into LLM prompts."""
        return {
            "symbol": self.symbol,
            "funding_rate_8h_pct": (
                round(self.funding_rate * 100, 5) if self.funding_rate is not None else None
            ),
            "oi_change_24h_pct": (
                round(self.oi_change_24h_pct, 2) if self.oi_change_24h_pct is not None else None
            ),
            "liquidations_below_usd": (
                round(self.liquidations_below_usd, 0) if self.liquidations_below_usd is not None else None
            ),
            "liquidations_above_usd": (
                round(self.liquidations_above_usd, 0) if self.liquidations_above_usd is not None else None
            ),
            "ls_long_pct": (
                round(self.ls_long_pct, 1) if self.ls_long_pct is not None else None
            ),
            "ls_short_pct": (
                round(self.ls_short_pct, 1) if self.ls_short_pct is not None else None
            ),
        }


async def _fetch_all_async(symbol: str, reference_price: float) -> CoinGlassSnapshot:
    """Fetch all four data points concurrently."""
    results = await asyncio.gather(
        get_funding(symbol),
        get_oi(symbol),
        get_liquidations(symbol, reference_price),
        get_ls_ratio(symbol),
        return_exceptions=True,
    )
    funding, oi, liq, ls = (
        r if not isinstance(r, Exception) else None for r in results
    )
    return CoinGlassSnapshot(
        symbol=symbol,
        funding=funding,  # type: ignore[arg-type]
        oi=oi,  # type: ignore[arg-type]
        liquidations=liq,  # type: ignore[arg-type]
        ls_ratio=ls,  # type: ignore[arg-type]
    )


def fetch_all_sync(symbol: str, reference_price: float) -> CoinGlassSnapshot:
    """Synchronous wrapper — runs the async batch fetch in a new event loop.

    Safe to call from synchronous pipeline code.  Returns a CoinGlassSnapshot
    where any failed individual fetch is None (non-fatal).

    Args:
        symbol: Coin name (e.g. "BTC").
        reference_price: Entry price used to partition the liquidation heatmap.

    Returns:
        CoinGlassSnapshot with whichever fields succeeded.
    """
    try:
        return asyncio.run(_fetch_all_async(symbol, reference_price))
    except Exception as exc:  # noqa: BLE001
        _log.warning("coinglass_fetch_all: failed for %s — %s", symbol, exc)
        return CoinGlassSnapshot(symbol=symbol)
