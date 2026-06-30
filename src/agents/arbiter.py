"""Arbiter: final GO / NO-GO decision engine.

Applies deterministic rules to a TradeProposal + CriticReport and produces
an ArbiterDecision.  No LLM calls — every decision is fully auditable.

Rules (evaluated in priority order):
  1. Any HIGH-severity objection  → NO_GO
  2. Two or more MEDIUM objections → NO_GO
  3. Otherwise                    → GO

On a GO decision the full context is appended to the KILL log
(``kill_log.jsonl`` by default, one JSON object per line).  The log is
append-only and must never be overwritten — it is the system's primary
learning artefact: which kill codes fired on approved trades, and
(once outcome data is added) whether those trades succeeded.

Public surface:
    ArbiterVerdict, KillLogEntry, ArbiterDecision, ArbiterError, arbitrate
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.agents.critic import CriticReport, KillCode, Severity
from src.agents.proposer import TradeProposal

if TYPE_CHECKING:
    from src.agents.scout import Candidate

_LOG = logging.getLogger(__name__)

__all__ = [
    "ArbiterVerdict", "KillLogEntry", "ArbiterDecision",
    "ArbiterError", "arbitrate",
]

_ENV_LOG_PATH = "FAST_AGENT_KILL_LOG"
_DEFAULT_LOG_NAME = "kill_log.jsonl"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ArbiterError(Exception):
    """Raised when the Arbiter cannot produce a valid decision."""


# ---------------------------------------------------------------------------
# Enums / types
# ---------------------------------------------------------------------------

class ArbiterVerdict(str, Enum):
    """Final GO / NO-GO outcome from the Arbiter."""

    GO = "GO"
    NO_GO = "NO_GO"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class KillLogEntry(BaseModel):
    """One record in the append-only KILL log.

    Written on every GO decision so the system can learn which kill codes
    co-occurred with approved trades and (later) correlate against outcomes.

    Attributes:
        ts: UTC timestamp of the decision.
        symbol: Coin name.
        direction: LONG or SHORT.
        verdict: Always GO for log entries (NO_GO decisions are not logged).
        kill_codes_fired: All objection codes raised by the Critic, regardless
            of severity.  Empty list means the proposal was fully clean.
        confidence: Scout confidence score propagated from the Candidate.
        reason: Human-readable explanation of the GO decision.
    """

    ts: datetime
    symbol: str
    direction: Literal["LONG"]
    verdict: ArbiterVerdict
    kill_codes_fired: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class ArbiterDecision(BaseModel):
    """The Arbiter's complete output for one proposal/report pair.

    Attributes:
        proposal: The evaluated TradeProposal (passed through unchanged).
        critic_report: The CriticReport (passed through unchanged).
        verdict: GO or NO_GO.
        reason: Plain-English explanation of the verdict.
        kill_codes_fired: All kill codes raised by the Critic (all severities).
        ts: UTC timestamp at the moment the decision was made.
    """

    proposal: TradeProposal
    critic_report: CriticReport
    verdict: ArbiterVerdict
    reason: str
    kill_codes_fired: list[KillCode]
    ts: datetime


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_rules(report: CriticReport) -> tuple[ArbiterVerdict, str]:
    """Apply the three Arbiter rules and return (verdict, reason).

    Evaluation order:
      1. Any HIGH → NO_GO (highest priority).
      2. 2+ MEDIUM → NO_GO.
      3. Otherwise → GO.

    Args:
        report: Critic output containing the objection list.

    Returns:
        Tuple of (ArbiterVerdict, human-readable reason string).
    """
    highs = [o for o in report.objections if o.severity == Severity.HIGH]
    mediums = [o for o in report.objections if o.severity == Severity.MEDIUM]

    if highs:
        codes = ", ".join(o.kill_code.value for o in highs)
        return ArbiterVerdict.NO_GO, f"HIGH severity objection(s): {codes}"

    if len(mediums) >= 2:
        codes = ", ".join(o.kill_code.value for o in mediums)
        return ArbiterVerdict.NO_GO, f"2+ MEDIUM severity objections: {codes}"

    if mediums:
        return ArbiterVerdict.GO, f"1 MEDIUM objection (non-blocking): {mediums[0].kill_code.value}"

    return ArbiterVerdict.GO, "no objections — clean proposal"


def _make_log_entry(decision: ArbiterDecision) -> KillLogEntry:
    """Build a KillLogEntry from a GO ArbiterDecision.

    Args:
        decision: An ArbiterDecision with verdict == GO.

    Returns:
        KillLogEntry ready to be serialised.
    """
    return KillLogEntry(
        ts=decision.ts,
        symbol=decision.proposal.symbol,
        direction=decision.proposal.direction,
        verdict=decision.verdict,
        kill_codes_fired=[kc.value for kc in decision.kill_codes_fired],
        confidence=decision.proposal.confidence,
        reason=decision.reason,
    )


def _append_log(entry: KillLogEntry, log_path: Path) -> None:
    """Append one JSON line to the KILL log.  Creates the file if absent.

    The file is opened in append mode ('a') on every call so concurrent
    writers do not corrupt earlier entries.  Each line is a self-contained
    JSON object terminated by a newline (JSONL format).

    Args:
        entry: The log entry to serialise.
        log_path: Path to the KILL log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")


def _resolve_log_path(log_path: Path | None) -> Path:
    """Return the effective log path from the argument, env var, or default."""
    if log_path is not None:
        return log_path
    return Path(os.environ.get(_ENV_LOG_PATH, _DEFAULT_LOG_NAME))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def arbitrate(
    proposal: TradeProposal,
    report: CriticReport,
    log_path: Path | None = None,
    candidate: "Candidate | None" = None,
    funding_rate: float | None = None,
    memory_log_path: Path | None = None,
) -> ArbiterDecision:
    """Apply Arbiter rules and produce a final GO / NO-GO decision.

    Rules are evaluated in priority order:
      1. Any HIGH-severity Critic objection → NO_GO.
      2. Two or more MEDIUM-severity objections → NO_GO.
      3. Otherwise → GO.

    A GO decision is appended to the KILL log (append-only JSONL file) so
    the system accumulates a record of every approved trade alongside the
    kill codes that fired, for future outcome-correlation analysis.

    NO_GO decisions are not written to the log — they never reached the
    execution layer and carry no outcome signal.

    Every decision (GO and NO_GO alike) is also appended to the decision
    memory store when ``candidate`` is provided, so past outcomes can
    inform future Critic LLM prompts.

    Args:
        proposal: The TradeProposal from the Proposer.
        report: The CriticReport from the Critic.
        log_path: Path to the KILL log file.  Resolved from the
            ``FAST_AGENT_KILL_LOG`` env var if omitted, defaulting to
            ``kill_log.jsonl`` in the current working directory.
        candidate: Scout Candidate for the decision (used by decision memory).
            When None, decision memory logging is skipped.
        funding_rate: Funding rate at decision time (passed to decision memory).
        memory_log_path: Override path for the decision memory JSONL file.
            When None, the default from env / hardcoded path is used.

    Returns:
        An ``ArbiterDecision`` with verdict, reason, and full proposal /
        report context.

    Raises:
        ArbiterError: If the log file cannot be written on a GO decision.
    """
    effective_log_path = _resolve_log_path(log_path)
    verdict, reason = _apply_rules(report)
    kill_codes_fired = [o.kill_code for o in report.objections]

    decision = ArbiterDecision(
        proposal=proposal,
        critic_report=report,
        verdict=verdict,
        reason=reason,
        kill_codes_fired=kill_codes_fired,
        ts=datetime.now(tz=timezone.utc),
    )

    if verdict == ArbiterVerdict.GO:
        entry = _make_log_entry(decision)
        try:
            _append_log(entry, effective_log_path)
        except OSError as exc:
            raise ArbiterError(f"Failed to write KILL log at {effective_log_path}: {exc}") from exc

    if candidate is not None:
        try:
            from src.pipeline.decision_memory import log_decision
            log_decision(candidate, decision, funding_rate=funding_rate, memory_path=memory_log_path)
        except Exception as exc:
            _LOG.warning("decision_memory: log_decision failed (non-fatal): %s", exc)

    return decision
