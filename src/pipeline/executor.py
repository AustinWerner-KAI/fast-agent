"""Trade executor — called after Arbiter returns GO.

Full order lifecycle with nine gates and three-state logging:

  intent → submitted → filled

Each state is an append to logs/execution.jsonl.  The verdict_id (decision
memory UUID) is the idempotency key — a duplicate verdict_id is silently
suppressed.

Gates (evaluated in order):
  1. TRADING_ENABLED env var
  2. Circuit breaker daily PnL check
  3. PositionGuard — manual position protection
  4. Idempotency — duplicate verdict_id
  5. Concurrent open bot positions cap
  6. Free margin check (MIN_FREE_MARGIN)
  7. Position guard refresh (final pre-order check)
  8. Place order
  9. Await fill or mark submitted

Sizing:
  margin_used = min(free_margin × MAX_POSITION_PCT, MAX_POSITION_USD / MAX_LEVERAGE)
  notional    = margin_used × MAX_LEVERAGE
  units       = notional / entry_price
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config_loader import load_conviction_sizing, load_trail_pct

__all__ = ["Executor", "ExecutorConfig"]

_LOG = logging.getLogger(__name__)

_DEFAULT_EXEC_LOG = Path("logs/execution.jsonl")

# Env var names
_ENV_TRADING_ENABLED = "TRADING_ENABLED"
_ENV_MAX_POS_PCT = "MAX_POSITION_PCT"
_ENV_MAX_POS_USD = "MAX_POSITION_USD"
_ENV_MAX_LEVERAGE = "MAX_LEVERAGE"
_ENV_MIN_MARGIN = "MIN_FREE_MARGIN"
_ENV_MAX_CONCURRENT = "MAX_CONCURRENT_POSITIONS"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ExecutorConfig:
    """Runtime parameters read from environment variables.

    All values have safe defaults that correspond to the plan spec.
    """

    def __init__(self) -> None:
        self.trading_enabled: bool = (
            os.environ.get(_ENV_TRADING_ENABLED, "false").strip().lower() == "true"
        )
        self.max_position_pct: float = float(os.environ.get(_ENV_MAX_POS_PCT, "0.02"))
        self.max_position_usd: float = float(os.environ.get(_ENV_MAX_POS_USD, "20.0"))
        self.max_leverage: float = float(os.environ.get(_ENV_MAX_LEVERAGE, "10.0"))
        self.min_free_margin: float = float(os.environ.get(_ENV_MIN_MARGIN, "50.0"))
        self.max_concurrent_positions: int = int(os.environ.get(_ENV_MAX_CONCURRENT, "2"))


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Places real orders on Hyperliquid after all gates pass.

    Args:
        broker: LiveBroker instance.
        position_guard: PositionGuard instance.
        circuit_breaker: CircuitBreaker instance.
        exec_log: Path to the execution JSONL log.
    """

    def __init__(
        self,
        broker: object,
        position_guard: object,
        circuit_breaker: object,
        exec_log: Path | None = None,
    ) -> None:
        self._broker = broker
        self._guard = position_guard
        self._breaker = circuit_breaker
        self._exec_log = exec_log or _DEFAULT_EXEC_LOG

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        verdict_id: str,
        proposal: Any,
        candidate: Any,
    ) -> dict[str, Any]:
        """Execute one GO verdict.

        Args:
            verdict_id: UUID from decision_memory (idempotency key).
            proposal: TradeProposal from the Proposer.
            candidate: Scout Candidate (for symbol / direction metadata).

        Returns:
            Dict describing the final state and any relevant context.
        """
        cfg = ExecutorConfig()
        symbol: str = proposal.symbol
        direction: str = proposal.direction
        entry: float = float(proposal.entry)
        stop: float = float(proposal.stop)
        tp1: float = float(proposal.tp1)
        tp2: float = float(proposal.tp2)
        tp3: float = float(proposal.tp3)

        # ── Gate 1: TRADING_ENABLED ──────────────────────────────────
        if not cfg.trading_enabled:
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="dry_run",
                reason="TRADING_ENABLED=false",
            )

        # ── Gate 2: Circuit breaker ──────────────────────────────────
        try:
            allowed = self._breaker.check()  # type: ignore[union-attr]
        except Exception as exc:
            _LOG.warning("executor: circuit_breaker.check() failed — skipping: %s", exc)
            allowed = False
        if not allowed:
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="halted",
                reason="Circuit breaker tripped (daily PnL < -3%)",
            )

        # ── Gate 3: PositionGuard ────────────────────────────────────
        try:
            protected = self._guard.is_protected(symbol)  # type: ignore[union-attr]
        except Exception:
            protected = False
        if protected:
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="blocked_manual",
                reason=f"{symbol} has a manual position — PositionGuard blocked order",
            )

        # ── Gate 4: Idempotency ──────────────────────────────────────
        if self._verdict_id_exists(verdict_id):
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="duplicate_suppressed",
                reason=f"verdict_id {verdict_id} already in execution.jsonl",
            )

        # ── Gate 5: Concurrent position cap ─────────────────────────
        active_count = self._count_active_positions()
        if active_count >= cfg.max_concurrent_positions:
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="max_positions_reached",
                reason=f"Open bot positions ({active_count}) >= max ({cfg.max_concurrent_positions})",
            )

        # ── Gate 6: Free margin ──────────────────────────────────────
        try:
            free_margin = self._broker.get_free_margin()  # type: ignore[union-attr]
        except Exception as exc:
            _LOG.warning("executor: get_free_margin failed — skipping: %s", exc)
            free_margin = 0.0

        if free_margin < cfg.min_free_margin:
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="skipped_low_margin",
                reason=f"Free margin ${free_margin:.2f} < min ${cfg.min_free_margin:.2f}",
                free_margin=free_margin,
            )

        # ── Conviction-based sizing ──────────────────────────────────
        try:
            conviction = float(proposal.conviction if hasattr(proposal, "conviction") else proposal.confidence)
        except (TypeError, ValueError, AttributeError):
            conviction = 0.5

        conv_cfg = load_conviction_sizing()
        tiers = conv_cfg.get("tiers", [])
        if tiers:
            notional_target = _lookup_conviction_usd(conviction, tiers)
            cap_pct = float(conv_cfg.get("free_margin_cap_pct", cfg.max_position_pct))
            margin_for_trade = notional_target / max(cfg.max_leverage, 1.0)
            margin_cap_by_pct = free_margin * cap_pct
            margin_used = min(margin_for_trade, margin_cap_by_pct)
        else:
            margin_cap = cfg.max_position_usd / max(cfg.max_leverage, 1.0)
            margin_used = min(free_margin * cfg.max_position_pct, margin_cap)
        notional = margin_used * cfg.max_leverage
        units = notional / entry if entry > 0 else 0.0

        if units <= 0:
            return self._log_and_return(
                verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                state="skipped_zero_size",
                reason="Computed units <= 0 after sizing",
            )

        # ── Log intent ───────────────────────────────────────────────
        self._append_log({
            "state": "intent",
            "verdict_id": verdict_id,
            "symbol": symbol,
            "direction": direction,
            "units": round(units, 8),
            "margin_used": round(margin_used, 4),
            "notional": round(notional, 4),
            "entry": entry,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "free_margin": round(free_margin, 4),
            "ts": _now_iso(),
        })

        # ── Gate 7: Pre-order position guard refresh ─────────────────
        try:
            bot_syms = frozenset(self._active_bot_symbols())
            self._guard.refresh(bot_symbols=bot_syms)  # type: ignore[union-attr]
            if self._guard.is_protected(symbol):  # type: ignore[union-attr]
                return self._log_and_return(
                    verdict_id, symbol, direction, entry, stop, tp1, tp2, tp3,
                    state="blocked_manual",
                    reason=f"{symbol} became manually held between scan and order",
                )
        except Exception as exc:
            _LOG.warning("executor: pre-order guard refresh failed (continuing): %s", exc)

        # ── Place order ──────────────────────────────────────────────
        side = "BUY" if direction == "LONG" else "SELL"
        try:
            result = self._broker.place_order(  # type: ignore[union-attr]
                symbol=symbol,
                side=side,
                size=units,
                reduce_only=False,
            )
        except Exception as exc:
            self._append_log({
                "state": "error",
                "verdict_id": verdict_id,
                "symbol": symbol,
                "error": str(exc),
                "ts": _now_iso(),
            })
            _LOG.error("executor: place_order failed (%s): %s", symbol, exc)
            return {"state": "error", "verdict_id": verdict_id, "error": str(exc)}

        # ── Log submitted ────────────────────────────────────────────
        self._append_log({
            "state": "submitted",
            "verdict_id": verdict_id,
            "symbol": symbol,
            "order_id": result.order_id,
            "ts": _now_iso(),
        })

        # ── Log filled ───────────────────────────────────────────────
        if result.status == "filled":
            fill_price: float = result.filled_price or 0.0

            # Post-fill geometry guard: a market order can fill below the
            # proposal stop when OHLCV entry data is stale relative to the
            # live market. Recompute a trailing stop rather than reject —
            # the trade is real and must be tracked.
            effective_stop = stop
            if fill_price > 0 and fill_price < stop:
                trail_pct = load_trail_pct()
                effective_stop = round(fill_price * (1.0 - trail_pct), 8)
                _LOG.warning(
                    "executor: GEOMETRY_CORRECTED %s fill_price=%.6g "
                    "original_stop=%.6g new_stop=%.6g (trail_pct=%.1f%%)",
                    symbol, fill_price, stop, effective_stop, trail_pct * 100,
                )

            fill_entry: dict[str, Any] = {
                "state": "filled",
                "verdict_id": verdict_id,
                "symbol": symbol,
                "direction": direction,
                "order_id": result.order_id,
                "fill_price": fill_price,
                "fill_size": result.size,
                "margin_used": round(margin_used, 4),
                "notional": round(notional, 4),
                "stop": effective_stop,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "ts": _now_iso(),
            }
            if effective_stop != stop:
                fill_entry["original_stop"] = stop

            # Place exchange stop order immediately — before logging so the
            # stop_order_id is captured in the fill record atomically.
            stop_side = "SELL" if direction == "LONG" else "BUY"
            try:
                stop_result = self._broker.place_stop_order(  # type: ignore[union-attr]
                    symbol=symbol,
                    side=stop_side,
                    size=result.size,
                    trigger_price=effective_stop,
                )
                fill_entry["stop_order_id"] = stop_result.order_id
                _LOG.info(
                    "executor: STOP_PLACED %s trigger=%.6g order_id=%s",
                    symbol, effective_stop, stop_result.order_id,
                )
            except Exception as exc:
                _LOG.error(
                    "executor: STOP_ORDER_FAILED %s trigger=%.6g: %s",
                    symbol, effective_stop, exc,
                )

            self._append_log(fill_entry)
            _LOG.info(
                "executor: FILLED %s %s %.6f @ %s (margin=$%.2f notional=$%.2f)",
                direction, symbol, units, result.filled_price, margin_used, notional,
            )
            return {"state": "filled", **fill_entry}

        # Submitted but not yet confirmed filled (resting / partial).
        _LOG.info(
            "executor: SUBMITTED %s %s order_id=%s (status=%s)",
            direction, symbol, result.order_id, result.status,
        )
        return {"state": "submitted", "verdict_id": verdict_id, "order_id": result.order_id}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append_log(self, entry: dict) -> None:
        try:
            self._exec_log.parent.mkdir(parents=True, exist_ok=True)
            with self._exec_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            _LOG.error("executor: failed to write execution log: %s", exc)

    def _verdict_id_exists(self, verdict_id: str) -> bool:
        if not self._exec_log.exists():
            return False
        try:
            with self._exec_log.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        entry = json.loads(raw)
                        if entry.get("verdict_id") == verdict_id:
                            return True
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return False

    def _count_active_positions(self) -> int:
        """Count bot-opened positions currently in 'filled' state."""
        return len(self._active_bot_symbols())

    def _active_bot_symbols(self) -> set[str]:
        """Return symbols with an active (filled, not closed) bot position."""
        if not self._exec_log.exists():
            return set()
        latest: dict[str, dict] = {}
        try:
            with self._exec_log.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        entry = json.loads(raw)
                        vid = entry.get("verdict_id", "")
                        if vid:
                            latest[vid] = entry
                    except json.JSONDecodeError:
                        pass
        except OSError:
            return set()

        closed_states = {"closed", "stop_hit", "tp1_hit_full", "tp2_hit_full", "tp3_hit"}
        active: set[str] = set()
        for entry in latest.values():
            if entry.get("state") == "filled" and entry.get("state") not in closed_states:
                sym = entry.get("symbol", "")
                if sym:
                    active.add(sym)
        return active

    def _log_and_return(
        self,
        verdict_id: str,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        tp1: float,
        tp2: float,
        tp3: float,
        state: str,
        reason: str,
        **extra: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "state": state,
            "verdict_id": verdict_id,
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "reason": reason,
            "ts": _now_iso(),
            **extra,
        }
        self._append_log(record)
        _LOG.info("executor: %s %s — %s", state.upper(), symbol, reason)
        return record


def _lookup_conviction_usd(conviction: float, tiers: list[dict]) -> float:
    """Return the target notional USD for a given conviction score.

    Tiers must have ``max_conviction`` and ``size_usd`` keys; sorted ascending
    by ``max_conviction``.  The first tier whose ``max_conviction`` strictly
    exceeds ``conviction`` wins.  If conviction is above all thresholds the
    last tier applies.
    """
    sorted_tiers = sorted(tiers, key=lambda t: float(t["max_conviction"]))
    for tier in sorted_tiers:
        if conviction < float(tier["max_conviction"]):
            return float(tier["size_usd"])
    return float(sorted_tiers[-1]["size_usd"]) if sorted_tiers else 50.0


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
