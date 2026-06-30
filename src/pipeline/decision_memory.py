"""Append-only decision memory store for the fast-agent Arbiter.

Persists GO / NO-GO decisions to a JSONL file so past outcomes can
inform future Critic LLM prompts.  Also auto-matches closed-trade
outcomes written by BOTZACHARY so ``outcome_pct`` is back-filled
automatically on the next ``get_history()`` call.

Public API:
    log_decision(candidate, decision, funding_rate, memory_path) -> str
    get_history(symbol, n, memory_path, outcomes_path) -> list[dict]
    update_outcome(decision_id, pnl_pct, reflection, memory_path) -> None
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = ["log_decision", "get_history", "update_outcome"]

_LOG = logging.getLogger(__name__)

_ENV_MEMORY_LOG = "FAST_AGENT_DECISION_MEMORY"
_DEFAULT_MEMORY_LOG = "/opt/fast-agent/logs/decision_memory.jsonl"
_ENV_OUTCOMES_LOG = "FAST_AGENT_TRADE_OUTCOMES"
_DEFAULT_OUTCOMES_LOG = "/opt/fast-agent/logs/trade_outcomes.jsonl"

_OUTCOME_MATCH_WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# Entry model
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    """One record in the append-only decision memory store.

    Attributes:
        id: UUID4 assigned at write time.
        ts: ISO-8601 UTC timestamp of the Arbiter decision.
        symbol: Coin name (e.g. ``"BTC"``).
        direction: ``"LONG"`` or ``"SHORT"``.
        fib_level: MA period stored as float (fast-agent uses MA, not Fib).
            E.g. ``50.0`` for EMA-50.  None when Candidate is unavailable.
        regime: Regime string at decision time (e.g. ``"TREND"``).
        funding_rate: Perpetual funding rate as a decimal, or None.
        rsi: Not computed by fast-agent; always None (reserved for future use).
        confidence: Scout candidate confidence in [0.0, 1.0].
        decision: ``"GO"`` or ``"NO_GO"``.
        kill_code: First kill code that fired, or None for a clean decision.
        outcome_pct: Realised PnL % once the trade closes; None until filled.
        reflection: Free-text post-trade analysis; None until filled.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    ts: str
    symbol: str
    direction: str
    fib_level: float | None = None
    regime: str | None = None
    funding_rate: float | None = None
    rsi: float | None = None
    confidence: float
    decision: str
    kill_code: str | None = None
    outcome_pct: float | None = None
    reflection: str | None = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _memory_path() -> Path:
    return Path(os.environ.get(_ENV_MEMORY_LOG, _DEFAULT_MEMORY_LOG))


def _outcomes_path() -> Path:
    return Path(os.environ.get(_ENV_OUTCOMES_LOG, _DEFAULT_OUTCOMES_LOG))


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _read_all(path: Path) -> list[MemoryEntry]:
    """Read every entry from a JSONL file; skip malformed lines."""
    if not path.exists():
        return []
    entries: list[MemoryEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(MemoryEntry.model_validate_json(raw))
            except Exception as exc:
                _LOG.warning("decision_memory: skipping malformed line %d: %s", lineno, exc)
    return entries


def _write_all(entries: list[MemoryEntry], path: Path) -> None:
    """Atomically rewrite the entire JSONL file via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(entry.model_dump_json() + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_outcomes(path: Path) -> list[dict[str, Any]]:
    """Read closed-trade outcome lines from BOTZACHARY's JSONL file."""
    if not path.exists():
        return []
    outcomes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                outcomes.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return outcomes


def _parse_dt(ts_str: str) -> datetime:
    """Parse an ISO timestamp string into a timezone-aware datetime."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Auto-matching logic
# ---------------------------------------------------------------------------

def _sync_outcomes(entries: list[MemoryEntry], outcomes_path: Path) -> bool:
    """Fill null ``outcome_pct`` fields by matching against trade_outcomes.jsonl.

    Matching rules:
    - symbol and direction must match exactly.
    - ``closed_ts`` must be after the decision ``ts``.
    - The gap between decision and close must be ≤ 30 days.
    - Each trade outcome is matched at most once per call.

    Args:
        entries: Subset of MemoryEntry objects (modified in-place).
        outcomes_path: Path to BOTZACHARY's trade_outcomes.jsonl.

    Returns:
        True if at least one entry was updated.
    """
    unresolved = [e for e in entries if e.decision == "GO" and e.outcome_pct is None]
    if not unresolved:
        return False

    outcomes = _read_outcomes(outcomes_path)
    if not outcomes:
        return False

    claimed: set[str] = set()  # outcome keys consumed in this pass
    updated = False

    for entry in unresolved:
        try:
            entry_dt = _parse_dt(entry.ts)
        except Exception:
            continue

        best: dict[str, Any] | None = None
        best_key: str = ""
        best_delta = float("inf")

        for outcome in outcomes:
            try:
                if outcome.get("symbol") != entry.symbol:
                    continue
                if outcome.get("direction") != entry.direction:
                    continue
                closed_str: str = outcome.get("closed_ts", "")
                if not closed_str:
                    continue
                closed_dt = _parse_dt(closed_str)
                if closed_dt <= entry_dt:
                    continue
                delta = (closed_dt - entry_dt).total_seconds()
                if delta > _OUTCOME_MATCH_WINDOW_DAYS * 86_400:
                    continue
                key = f"{outcome.get('symbol')}|{outcome.get('direction')}|{closed_str}"
                if key in claimed:
                    continue
                if delta < best_delta:
                    best = outcome
                    best_delta = delta
                    best_key = key
            except Exception:
                continue

        if best is not None:
            claimed.add(best_key)
            entry.outcome_pct = float(best["pnl_pct"])
            updated = True

    return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_decision(
    candidate: Any,
    decision: Any,
    funding_rate: float | None = None,
    memory_path: Path | None = None,
) -> str:
    """Append one GO / NO-GO decision to the memory store.

    Args:
        candidate: ``src.agents.scout.Candidate`` — provides symbol, direction,
            ma_period, regime, and confidence.
        decision: ``src.agents.arbiter.ArbiterDecision`` — provides verdict,
            kill_codes_fired, and ts.
        funding_rate: Perpetual funding rate at decision time, if available.
        memory_path: Override the default log path (useful in tests).

    Returns:
        The UUID ``id`` of the newly written entry.
    """
    path = memory_path or _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    kill_code: str | None = (
        decision.kill_codes_fired[0].value if decision.kill_codes_fired else None
    )
    regime_val: str | None = (
        candidate.regime.value
        if hasattr(candidate.regime, "value")
        else str(candidate.regime)
        if candidate is not None
        else None
    )

    entry = MemoryEntry(
        ts=decision.ts.isoformat(),
        symbol=candidate.symbol,
        direction=candidate.direction,
        fib_level=float(candidate.ma_period),
        regime=regime_val,
        funding_rate=funding_rate,
        rsi=None,
        confidence=float(candidate.confidence),
        decision=decision.verdict.value,
        kill_code=kill_code,
    )

    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")

    return entry.id


def get_history(
    symbol: str,
    n: int = 5,
    memory_path: Path | None = None,
    outcomes_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the last N decisions for a symbol, most recent first.

    Before returning, auto-matches any null ``outcome_pct`` fields against
    BOTZACHARY's trade_outcomes.jsonl and persists the updates.  All errors
    are non-fatal — returns an empty list if the file is unreadable.

    Args:
        symbol: Coin name to filter by.
        n: Maximum entries to return (default 5).
        memory_path: Override the default memory log path.
        outcomes_path: Override the default outcomes log path.

    Returns:
        List of decision dicts, most recent first.
    """
    mpath = memory_path or _memory_path()
    opath = outcomes_path or _outcomes_path()

    try:
        all_entries = _read_all(mpath)
    except Exception as exc:
        _LOG.warning("decision_memory: get_history could not read log: %s", exc)
        return []

    # symbol_entries shares the same MemoryEntry objects as all_entries
    symbol_entries = [e for e in all_entries if e.symbol == symbol]
    if not symbol_entries:
        return []

    try:
        if _sync_outcomes(symbol_entries, opath):
            _write_all(all_entries, mpath)
    except Exception as exc:
        _LOG.warning("decision_memory: outcome sync failed (non-fatal): %s", exc)

    symbol_entries.sort(key=lambda e: e.ts, reverse=True)
    return [e.model_dump() for e in symbol_entries[:n]]


def update_outcome(
    decision_id: str,
    pnl_pct: float,
    reflection: str | None = None,
    memory_path: Path | None = None,
) -> None:
    """Rewrite a single entry's ``outcome_pct`` and ``reflection`` fields.

    Atomically rewrites the JSONL file; all other entries are preserved
    byte-for-byte (re-serialised).

    Args:
        decision_id: UUID of the entry to update.
        pnl_pct: Realised PnL percentage (e.g. ``-1.8`` for a 1.8% loss).
        reflection: Optional free-text post-trade analysis.
        memory_path: Override the default log path.

    Raises:
        ValueError: If no entry with ``decision_id`` exists.
        OSError: If the file cannot be read or written.
    """
    path = memory_path or _memory_path()
    entries = _read_all(path)

    found = False
    for entry in entries:
        if entry.id == decision_id:
            entry.outcome_pct = pnl_pct
            entry.reflection = reflection
            found = True
            break

    if not found:
        raise ValueError(f"decision_memory: no entry with id={decision_id!r}")

    _write_all(entries, path)


# ---------------------------------------------------------------------------
# Prompt formatting helper (used by critic.py)
# ---------------------------------------------------------------------------

def format_history_block(symbol: str, history: list[dict[str, Any]]) -> str:
    """Format decision history as a structured block for LLM injection.

    Args:
        symbol: Coin name (used in the header line).
        history: List of decision dicts from ``get_history()``.

    Returns:
        Multi-line string ready to embed in a prompt.
    """
    lines = [f"DECISION MEMORY — last {len(history)} decisions for {symbol}:"]
    for h in history:
        date_str = h.get("ts", "")[:10]
        direction = h.get("direction", "?")
        fib = h.get("fib_level")
        fib_str = f"MA-{int(fib)}" if fib is not None else "?"
        regime = h.get("regime", "?")
        funding = h.get("funding_rate")
        verdict = h.get("decision", "?")
        outcome = h.get("outcome_pct")
        kill = h.get("kill_code")

        parts: list[str] = [f"[{date_str}] {direction} @ {fib_str} | Regime: {regime}"]

        if funding is not None:
            sign = "+" if funding >= 0 else ""
            parts.append(f"Funding: {sign}{funding * 100:.3f}%")

        if verdict == "NO_GO" and kill:
            parts.append(f"Decision: NO-GO | Kill: {kill}")
        else:
            outcome_str = f"{outcome:+.1f}%" if outcome is not None else "pending"
            parts.append(f"Decision: {verdict} | Outcome: {outcome_str}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)
