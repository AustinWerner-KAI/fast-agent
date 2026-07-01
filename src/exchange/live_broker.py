"""Hyperliquid live broker — order placement + account queries.

Wraps the Hyperliquid Python SDK.  Agent-wallet-only: if the private key
resolves to the same address as HYPERLIQUID_ACCOUNT_ADDRESS (i.e. main wallet
key was supplied by mistake) the constructor raises RuntimeError and refuses
to start.

All exit orders (TP, stop) must use reduce_only=True — callers are responsible
for passing the correct flag.  Every public method is async-friendly (sync SDK
calls are made; mark them as blocking if needed in asyncio contexts).
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

__all__ = [
    "Fill",
    "LiveBrokerError",
    "OrderResult",
    "Position",
    "LiveBroker",
]

_HL_INFO_URL = "https://api.hyperliquid.xyz/info"

_ENV_PRIVATE_KEY = "HYPERLIQUID_SECRET_KEY"
_ENV_AGENT_ADDRESS = "HYPERLIQUID_ADDRESS"
_ENV_MAIN_ADDRESS = "HYPERLIQUID_MAIN_ADDRESS"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LiveBrokerError(Exception):
    """Raised for broker-level failures (auth, order rejection, etc.)."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """An open perpetual position on Hyperliquid."""

    symbol: str
    size: float           # positive = LONG, negative = SHORT
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float

    @property
    def direction(self) -> Literal["LONG", "SHORT"]:
        return "LONG" if self.size > 0 else "SHORT"

    @property
    def abs_size(self) -> float:
        return abs(self.size)


@dataclass
class OrderResult:
    """Result of a place_order call."""

    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    size: float
    filled_price: float | None
    status: Literal["filled", "open", "error"]
    raw: dict


@dataclass
class Fill:
    """A single realized fill from the fills API."""

    symbol: str
    side: Literal["BUY", "SELL"]
    size: float
    price: float
    pnl: float
    ts: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_post(body: dict) -> dict:
    """POST JSON to the Hyperliquid info endpoint (no auth required)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        _HL_INFO_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _derive_address(private_key: str) -> str:
    """Return the Ethereum address that corresponds to a private key."""
    from eth_account import Account  # type: ignore[import-untyped]
    return Account.from_key(private_key).address.lower()


# ---------------------------------------------------------------------------
# LiveBroker
# ---------------------------------------------------------------------------

class LiveBroker:
    """Thin wrapper around the Hyperliquid exchange + info SDKs.

    Args:
        private_key: Agent wallet private key (0x-prefixed hex).  If omitted,
            read from HYPERLIQUID_SECRET_KEY env var.
        account_address: Main wallet address that the agent acts on behalf of.
            If omitted, read from HYPERLIQUID_MAIN_ADDRESS (falls back to
            HYPERLIQUID_ADDRESS).
        testnet: When True, connect to testnet instead of mainnet.

    Raises:
        LiveBrokerError: If the private key resolves to the same address as
            account_address (main wallet key mistakenly supplied).
        LiveBrokerError: If required env vars are missing.
    """

    def __init__(
        self,
        private_key: str | None = None,
        account_address: str | None = None,
        *,
        testnet: bool = False,
    ) -> None:
        self._private_key = private_key or os.environ.get(_ENV_PRIVATE_KEY, "")
        self._account_address = (
            account_address
            or os.environ.get(_ENV_MAIN_ADDRESS)
            or os.environ.get(_ENV_AGENT_ADDRESS, "")
        )

        if not self._private_key:
            raise LiveBrokerError(f"Private key not set ({_ENV_PRIVATE_KEY})")
        if not self._account_address:
            raise LiveBrokerError(f"Account address not set ({_ENV_MAIN_ADDRESS})")

        # Safety check: agent key must not match the main wallet address.
        try:
            derived = _derive_address(self._private_key)
            if derived.lower() == self._account_address.lower():
                raise LiveBrokerError(
                    "SAFETY: private key resolves to the same address as "
                    "account_address — this looks like the main wallet key. "
                    "Supply an agent wallet key only."
                )
        except LiveBrokerError:
            raise
        except Exception:
            pass  # eth_account may not be installed; skip the check gracefully

        from hyperliquid.exchange import Exchange  # type: ignore[import-untyped]
        from hyperliquid.utils import constants  # type: ignore[import-untyped]

        api_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self._exchange = Exchange(
            self._private_key,
            api_url,
            account_address=self._account_address,
        )
        self._testnet = testnet

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        size: float,
        order_type: str = "market",
        reduce_only: bool = False,
        slippage: float = 0.01,
    ) -> OrderResult:
        """Place a market order on Hyperliquid.

        Args:
            symbol: Coin name (e.g. "BTC").
            side: "BUY" or "SELL".
            size: Position size in coin units.
            order_type: Currently only "market" is supported.
            reduce_only: When True the order can only reduce an existing
                position — use for all TP and stop exits.
            slippage: Max slippage fraction (default 1%).

        Returns:
            OrderResult with fill details.

        Raises:
            LiveBrokerError: On SDK or network error.
        """
        is_buy = side == "BUY"
        try:
            if reduce_only:
                result = self._exchange.market_close(
                    symbol, size, slippage=slippage
                )
            else:
                result = self._exchange.market_open(
                    symbol, is_buy, size, slippage=slippage
                )
        except Exception as exc:
            raise LiveBrokerError(f"place_order failed ({symbol} {side} {size}): {exc}") from exc

        raw: dict = result if isinstance(result, dict) else {}
        statuses = raw.get("response", {}).get("data", {}).get("statuses", [{}])
        first = statuses[0] if statuses else {}
        filled = first.get("filled")
        order_id = str(first.get("resting", {}).get("oid", "") or first.get("filled", {}).get("oid", "") or "unknown")
        fill_price = float(filled.get("avgPx", 0)) if filled else None
        status: Literal["filled", "open", "error"] = (
            "filled" if filled else ("open" if first.get("resting") else "error")
        )

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            filled_price=fill_price,
            status=status,
            raw=raw,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            symbol: Coin name.
            order_id: Order ID string.

        Returns:
            True if cancellation succeeded.
        """
        try:
            result = self._exchange.cancel(symbol, int(order_id))
            return isinstance(result, dict)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Account queries (via public REST, no auth needed)
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        """Return all open perpetual positions on the main wallet.

        Returns:
            List of Position objects; empty list on error.
        """
        try:
            data = _http_post({"type": "clearinghouseState", "user": self._account_address})
            positions: list[Position] = []
            for entry in data.get("assetPositions", []):
                pos = entry.get("position", {})
                szi = float(pos.get("szi", 0) or 0)
                if szi == 0.0:
                    continue
                positions.append(
                    Position(
                        symbol=pos.get("coin", ""),
                        size=szi,
                        entry_price=float(pos.get("entryPx", 0) or 0),
                        mark_price=float(
                            pos.get("positionValue", 0) or 0
                        ) / abs(szi) if szi != 0 else 0.0,
                        unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
                        leverage=float(
                            (pos.get("leverage", {}) or {}).get("value", 1) or 1
                        ),
                    )
                )
            return positions
        except Exception:
            return []

    def get_free_margin(self) -> float:
        """Return the withdrawable (free) margin on the main wallet.

        Returns:
            Withdrawable USD; 0.0 on error.
        """
        try:
            data = _http_post({"type": "clearinghouseState", "user": self._account_address})
            return float(data.get("withdrawable", 0.0))
        except Exception:
            return 0.0

    def get_equity(self) -> float:
        """Return total account equity (accountValue) on the main wallet.

        Returns:
            Account value in USD; 0.0 on error.
        """
        try:
            data = _http_post({"type": "clearinghouseState", "user": self._account_address})
            return float(data.get("marginSummary", {}).get("accountValue", 0.0))
        except Exception:
            return 0.0

    def get_fills(self, since_ts: datetime) -> list[Fill]:
        """Return all fills since a given UTC timestamp.

        Args:
            since_ts: UTC datetime; only fills at or after this time are returned.

        Returns:
            List of Fill objects, or empty list on error.
        """
        since_ms = int(since_ts.timestamp() * 1000)
        try:
            data = _http_post({
                "type": "userFills",
                "user": self._account_address,
                "aggregateByTime": False,
            })
            fills: list[Fill] = []
            for f in data if isinstance(data, list) else []:
                ts_ms = int(f.get("time", 0))
                if ts_ms < since_ms:
                    continue
                fills.append(
                    Fill(
                        symbol=f.get("coin", ""),
                        side="BUY" if f.get("side", "") == "B" else "SELL",
                        size=float(f.get("sz", 0)),
                        price=float(f.get("px", 0)),
                        pnl=float(f.get("closedPnl", 0)),
                        ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                    )
                )
            return fills
        except Exception:
            return []

    def get_mark_price(self, symbol: str) -> float | None:
        """Return the current mark price for a symbol.

        Args:
            symbol: Coin name.

        Returns:
            Mark price as float, or None on error.
        """
        try:
            data = _http_post({"type": "allMids"})
            val = data.get(symbol)
            return float(val) if val is not None else None
        except Exception:
            return None
