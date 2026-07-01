"""Startup position reconciliation — compares Hyperliquid state with execution.jsonl.

Runs once at startup before the executor is armed.  Classifies every open
position as:

- matched:              on-chain AND in execution.jsonl (filled state)
- manual:               on-chain but NOT in execution.jsonl → position guard will protect
- orphaned_bot:         in execution.jsonl (filled) but NOT on-chain (closed externally)

The reconciliation result is written to logs/reconciliation.jsonl.  The
executor checks report.ok before running.  report.ok is False only when the
broker itself is unreachable (network failure, bad keys) — the presence of
manual or orphaned positions is logged but does NOT block execution.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

__all__ = ["ReconciliationReport", "reconcile"]

_LOG = logging.getLogger(__name__)
_DEFAULT_EXEC_LOG = Path("logs/execution.jsonl")
_DEFAULT_RECON_LOG = Path("logs/reconciliation.jsonl")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ReconciliationReport(BaseModel):
    """Result of a startup reconciliation run."""

    ts: str
    ok: bool
    on_chain_symbols: list[str]
    bot_claimed_symbols: list[str]
    manual_symbols: list[str]
    orphaned_symbols: list[str]
    matched_symbols: list[str]
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bot_positions(exec_log: Path) -> set[str]:
    """Return symbols of all positions the bot opened that are in 'filled' state
    and have no corresponding 'closed' entry in execution.jsonl.

    A position is "active" if the most recent state entry for its verdict_id
    is 'filled' (not 'closed' or 'stop_hit' etc.).
    """
    if not exec_log.exists():
        return set()

    # Group lines by verdict_id; track the latest state per verdict_id.
    latest_state: dict[str, dict] = {}
    with exec_log.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                vid = entry.get("verdict_id", "")
                if vid:
                    latest_state[vid] = entry
            except json.JSONDecodeError:
                pass

    active: set[str] = set()
    closed_states = {"closed", "stop_hit", "tp3_hit"}
    for entry in latest_state.values():
        state = entry.get("state", "")
        symbol = entry.get("symbol", "")
        if symbol and state == "filled" and state not in closed_states:
            active.add(symbol)
    return active


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconcile(
    broker: object,
    exec_log: Path | None = None,
    recon_log: Path | None = None,
) -> ReconciliationReport:
    """Compare Hyperliquid open positions with the bot's execution log.

    Args:
        broker: LiveBroker instance.
        exec_log: Path to execution.jsonl (defaults to logs/execution.jsonl).
        recon_log: Path to reconciliation output JSONL.

    Returns:
        ReconciliationReport.  report.ok is False only on broker connectivity
        failure.
    """
    exec_path = exec_log or _DEFAULT_EXEC_LOG
    recon_path = recon_log or _DEFAULT_RECON_LOG

    ts = datetime.now(tz=timezone.utc).isoformat()

    try:
        positions = broker.get_open_positions()  # type: ignore[union-attr]
        on_chain = {p.symbol for p in positions}
    except Exception as exc:
        report = ReconciliationReport(
            ts=ts,
            ok=False,
            on_chain_symbols=[],
            bot_claimed_symbols=[],
            manual_symbols=[],
            orphaned_symbols=[],
            matched_symbols=[],
            error=str(exc),
        )
        _write_report(report, recon_path)
        _LOG.error("reconciliation: broker unreachable — executor will NOT start: %s", exc)
        return report

    bot_active = _load_bot_positions(exec_path)

    matched = sorted(on_chain & bot_active)
    manual = sorted(on_chain - bot_active)
    orphaned = sorted(bot_active - on_chain)

    report = ReconciliationReport(
        ts=ts,
        ok=True,
        on_chain_symbols=sorted(on_chain),
        bot_claimed_symbols=sorted(bot_active),
        manual_symbols=manual,
        orphaned_symbols=orphaned,
        matched_symbols=matched,
    )
    _write_report(report, recon_path)

    if manual:
        _LOG.info(
            "reconciliation: %d manual position(s) detected — PositionGuard will protect: %s",
            len(manual),
            manual,
        )
    if orphaned:
        _LOG.warning(
            "reconciliation: %d orphaned bot position(s) — closed externally? %s",
            len(orphaned),
            orphaned,
        )

    _LOG.info(
        "reconciliation OK: on_chain=%s bot=%s manual=%s orphaned=%s",
        sorted(on_chain),
        sorted(bot_active),
        manual,
        orphaned,
    )
    return report


def _write_report(report: ReconciliationReport, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(report.model_dump_json() + "\n")
    except OSError as exc:
        _LOG.error("reconciliation: failed to write report: %s", exc)
