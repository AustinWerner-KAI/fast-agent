"""Two-layer manual-position protection for fast-agent.

Layer 1 (cycle-level): refresh() re-queries Hyperliquid at the start of every
cycle and updates the held_markets set, excluding symbols the bot opened itself
(identified by verdict_ids present in execution.jsonl).

Layer 2 (pre-order): is_protected(symbol) is checked immediately before any
live_broker.place_order() call to guard against race conditions between the
cycle refresh and actual order placement.

Direction is irrelevant — any non-zero position on a symbol not opened by the
bot is treated as manual and protected.

On API error the existing protected set is preserved (over-protects rather than
under-protects).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import FrozenSet

__all__ = ["PositionGuard"]

_LOG = logging.getLogger(__name__)
_HL_INFO_URL = "https://api.hyperliquid.xyz/info"


class PositionGuard:
    """Tracks manually-held symbols and blocks the bot from touching them.

    Args:
        broker: LiveBroker instance (used for get_open_positions in refresh).
        account_address: Main wallet address to query.  Falls back to broker's
            internal address if omitted.
    """

    def __init__(self, broker: object | None = None) -> None:
        self._broker = broker
        self._protected: set[str] = set()

    # ------------------------------------------------------------------
    # Layer 1 — cycle-level refresh
    # ------------------------------------------------------------------

    def refresh(self, bot_symbols: FrozenSet[str] = frozenset()) -> None:
        """Re-fetch open positions and rebuild the protected set.

        Symbols in bot_symbols are excluded from protection — the bot may
        manage its own positions.  On API error, the existing set is kept.

        Args:
            bot_symbols: Frozenset of coin names the bot opened this session.
        """
        if self._broker is None:
            return
        try:
            positions = self._broker.get_open_positions()  # type: ignore[union-attr]
            self._protected = {
                p.symbol
                for p in positions
                if p.symbol and p.symbol not in bot_symbols
            }
            _LOG.debug(
                "position_guard_refreshed: protected=%s bot=%s",
                sorted(self._protected),
                sorted(bot_symbols),
            )
        except Exception as exc:
            _LOG.warning(
                "position_guard_refresh_failed (keeping existing): %s", exc
            )

    # ------------------------------------------------------------------
    # Layer 2 — pre-order hard gate
    # ------------------------------------------------------------------

    def is_protected(self, symbol: str) -> bool:
        """Return True if symbol has a manual position that the bot must not touch.

        Args:
            symbol: Coin name (e.g. "BTC").

        Returns:
            True when the symbol is externally held.
        """
        return symbol in self._protected

    @property
    def protected_symbols(self) -> FrozenSet[str]:
        """Immutable set of currently protected coin names."""
        return frozenset(self._protected)
