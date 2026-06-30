"""Pydantic v2 models for CoinGlass market microstructure data."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FundingSnapshot(BaseModel):
    """Current 8h funding rate for a perpetual futures symbol."""

    symbol: str
    rate: float  # per-8h funding rate as a decimal fraction (e.g. 0.0001 = 0.01%)
    timestamp: datetime


class OISnapshot(BaseModel):
    """Aggregate open interest snapshot for a symbol."""

    symbol: str
    oi_usd: float  # total open interest in USD notional
    oi_change_24h_pct: float  # 24-hour OI change in percent
    timestamp: datetime


class LiquidationMap(BaseModel):
    """Liquidation clusters split above and below a reference price."""

    symbol: str
    liquidations_below_usd: float  # USD value of liq clusters below reference_price
    liquidations_above_usd: float  # USD value of liq clusters above reference_price
    reference_price: float
    timestamp: datetime


class LSRatio(BaseModel):
    """Global long/short account ratio for a symbol."""

    symbol: str
    long_pct: float  # percent of accounts that are long (0–100)
    short_pct: float  # percent of accounts that are short (0–100)
    timestamp: datetime
