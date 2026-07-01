"""Daily PnL circuit breaker — halts trading when realized loss exceeds -3%.

Reads realized PnL from Hyperliquid's fills API (not from local logs) so the
check is ground-truth even if the bot restarts mid-day.

Rules:
- Realized PnL from fills since today's UTC midnight is summed.
- If pnl_pct < -3.0% of current equity → HALT; log to circuit_breaker.jsonl.
- Resets automatically at UTC midnight (next call on a new day re-evaluates).
- Manual override: CIRCUIT_BREAKER_OVERRIDE=true skips the PnL check (logs
  a WARNING on every call when active).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["CircuitBreaker"]

_LOG = logging.getLogger(__name__)
_HALT_PCT = -3.0
_ENV_OVERRIDE = "CIRCUIT_BREAKER_OVERRIDE"
_DEFAULT_LOG = Path("logs/circuit_breaker.jsonl")


class CircuitBreaker:
    """Evaluates realized daily PnL against the halt threshold.

    Args:
        broker: LiveBroker instance.
        log_path: Path to the circuit-breaker JSONL log file.
        halt_pct: Daily loss % that triggers a halt (default -3.0).
    """

    def __init__(
        self,
        broker: object,
        log_path: Path | None = None,
        halt_pct: float = _HALT_PCT,
    ) -> None:
        self._broker = broker
        self._log_path = log_path or _DEFAULT_LOG
        self._halt_pct = halt_pct
        self._halted: bool = False
        self._halted_date: str = ""  # "YYYY-MM-DD" — auto-resets on new day

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> bool:
        """Return True if trading is allowed; False if the breaker is tripped.

        Queries Hyperliquid fills since today's UTC midnight.  Automatically
        resets the halted flag when called on a new calendar day.

        Returns:
            True → trading allowed. False → halted.
        """
        today = datetime.now(tz=timezone.utc).date().isoformat()

        # Auto-reset at midnight UTC.
        if self._halted and self._halted_date != today:
            _LOG.info("circuit_breaker: auto-reset on new day (%s)", today)
            self._halted = False
            self._halted_date = ""

        if self._halted:
            return False

        # Manual override.
        if os.environ.get(_ENV_OVERRIDE, "false").strip().lower() == "true":
            _LOG.warning(
                "circuit_breaker: OVERRIDE active — PnL check skipped"
            )
            return True

        return self._evaluate()

    @property
    def halted(self) -> bool:
        """True when the circuit breaker is currently tripped."""
        return self._halted

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evaluate(self) -> bool:
        """Fetch fills, compute PnL, compare against threshold."""
        today_midnight = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        try:
            fills = self._broker.get_fills(since_ts=today_midnight)  # type: ignore[union-attr]
        except Exception as exc:
            _LOG.warning("circuit_breaker: get_fills failed — allowing trade: %s", exc)
            return True

        realized_pnl = sum(f.pnl for f in fills)

        try:
            equity = self._broker.get_equity()  # type: ignore[union-attr]
        except Exception as exc:
            _LOG.warning("circuit_breaker: get_equity failed — allowing trade: %s", exc)
            equity = 0.0

        if equity <= 0:
            return True  # can't evaluate — allow trading

        pnl_pct = (realized_pnl / equity) * 100.0

        if pnl_pct < self._halt_pct:
            self._halt(pnl_pct, equity, today_midnight.date().isoformat())
            return False

        return True

    def _halt(self, pnl_pct: float, equity: float, date: str) -> None:
        """Trip the breaker and log the event."""
        self._halted = True
        self._halted_date = date
        entry = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "pnl_pct": round(pnl_pct, 4),
            "equity": round(equity, 2),
            "halt_threshold_pct": self._halt_pct,
            "action": "HALT",
        }
        _LOG.error(
            "circuit_breaker: HALT — daily PnL %.2f%% < %.2f%%",
            pnl_pct,
            self._halt_pct,
        )
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            _LOG.error("circuit_breaker: failed to write log: %s", exc)
