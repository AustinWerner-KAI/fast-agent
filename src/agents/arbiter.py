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
from typing import TYPE_CHECKING, Any, Literal

import anthropic
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
_OPUS_MODEL = "claude-sonnet-4-6"
_OPUS_MAX_TOKENS = 512


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


# ---------------------------------------------------------------------------
# Opus 4.8 final review (called only when deterministic rules would GO)
# ---------------------------------------------------------------------------

_OPUS_TOOL: list[dict] = [
    {
        "name": "submit_arbiter_verdict",
        "description": "Submit the final GO or NO_GO verdict with reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["GO", "NO_GO"],
                    "description": "Final execution decision.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "1–3 sentences explaining the decision, referencing specific data points.",
                },
            },
            "required": ["verdict", "reasoning"],
        },
    }
]


def _opus_review(
    proposal: TradeProposal,
    report: CriticReport,
    coinglass: dict,
    pre_filter_reason: str,
    client: anthropic.Anthropic,
    funding_crowded_severity: "Severity | None" = None,
) -> tuple[ArbiterVerdict, str]:
    """Call Opus 4.8 for the final GO / NO_GO decision with full CoinGlass context.

    Only called when deterministic pre-filter would produce GO.  Opus can
    override to NO_GO based on microstructure hostility, but must respect
    the pre-computed FUNDING_CROWDED severity rather than re-assessing it.

    Args:
        proposal: TradeProposal from the Proposer.
        report: CriticReport (HIGH/2+MEDIUM already filtered out).
        coinglass: CoinGlassSnapshot.to_dict() output.
        pre_filter_reason: Human-readable deterministic outcome.
        client: Anthropic client.
        funding_crowded_severity: Pre-computed severity from the bracket rules.
            None means funding is not a concern (negative or below threshold).

    Returns:
        Tuple of (ArbiterVerdict, reason string).
    """
    p = proposal
    obj_lines = (
        "\n".join(
            f"  [{o.severity.value}] {o.kill_code.value}: {o.reasoning}"
            for o in report.objections
        )
        or "  (none)"
    )

    from src.agents.critic import Severity as _Severity  # local import to avoid circular

    cg = coinglass
    funding_rate_raw = cg.get("funding_rate_8h_pct")
    funding_str = (
        f"{funding_rate_raw:+.5f}%"
        if funding_rate_raw is not None else "unknown"
    )
    oi_str = (
        f"{cg['oi_change_24h_pct']:+.2f}%"
        if cg.get("oi_change_24h_pct") is not None else "unknown"
    )
    liq_below = (
        f"${cg['liquidations_below_usd']:,.0f}"
        if cg.get("liquidations_below_usd") is not None else "unknown"
    )
    liq_above = (
        f"${cg['liquidations_above_usd']:,.0f}"
        if cg.get("liquidations_above_usd") is not None else "unknown"
    )
    ls_str = (
        f"LONG {cg['ls_long_pct']:.1f}% / SHORT {cg['ls_short_pct']:.1f}%"
        if cg.get("ls_long_pct") is not None and cg.get("ls_short_pct") is not None
        else "unknown"
    )

    # Build the pre-computed funding assessment line shown to Opus
    if funding_rate_raw is None:
        funding_assessment = "unknown — not assessed"
    elif funding_rate_raw < 0:
        funding_assessment = (
            f"FAVOURABLE for this LONG ({funding_rate_raw:+.5f}%/8h) — "
            "longs are being paid; do NOT treat negative funding as hostile"
        )
    elif funding_crowded_severity == _Severity.HIGH:
        funding_assessment = (
            f"HIGH — {funding_rate_raw:+.5f}%/8h exceeds the 0.10% extreme threshold "
            "(already caught by pre-filter if applicable)"
        )
    elif funding_crowded_severity == _Severity.MEDIUM:
        funding_assessment = (
            f"MEDIUM — {funding_rate_raw:+.5f}%/8h is elevated (0.05–0.10% range); "
            "caution warranted but not a veto"
        )
    else:
        funding_assessment = (
            f"NEUTRAL — {funding_rate_raw:+.5f}%/8h is below the 0.05% moderate threshold"
        )

    prompt = (
        "You are the final Arbiter for a crypto swing-trading system.\n"
        "The deterministic pre-filter has already blocked any HIGH-severity or 2+ MEDIUM "
        "objections.  Your job is to make the final GO / NO_GO decision for a proposal that "
        "passed those hard gates, using the full market microstructure context below.\n\n"
        "TRADE PROPOSAL:\n"
        f"  Symbol:      {p.symbol}\n"
        f"  Direction:   {p.direction}\n"
        f"  Entry:       {p.entry:.6g}\n"
        f"  Stop:        {p.stop:.6g}\n"
        f"  TP1:         {p.tp1:.6g}\n"
        f"  TP2:         {p.tp2:.6g}\n"
        f"  TP3:         {p.tp3:.6g}\n"
        f"  Confidence:  {p.confidence:.3f}\n"
        f"  Notional:    ${p.position_size_usd:,.0f}\n"
        f"  Risk:        ${p.risk_usd:,.0f}\n"
        f"  Reasoning:   {p.reasoning}\n\n"
        "CRITIC OBJECTIONS (remaining after pre-filter):\n"
        f"{obj_lines}\n\n"
        "CRITIC ASSESSMENT:\n"
        f"  {report.overall_assessment}\n\n"
        "MARKET MICROSTRUCTURE (CoinGlass — live data):\n"
        f"  Funding rate (8h):         {funding_str}\n"
        f"  Funding assessment:        {funding_assessment}\n"
        f"  OI change 24h:             {oi_str}\n"
        f"  Liq clusters below entry:  {liq_below}\n"
        f"  Liq clusters above entry:  {liq_above}\n"
        f"  Long/Short account ratio:  {ls_str}\n\n"
        "PRE-FILTER OUTCOME:\n"
        f"  {pre_filter_reason}\n\n"
        "DECISION GUIDANCE:\n"
        "  FUNDING: the 'Funding assessment' line above is pre-computed and authoritative — "
        "do not override it.  Negative funding FAVOURS this LONG and must never be cited as "
        "hostile.  Only positive rates above 0.10%/8h are extreme; 0.05–0.10% is moderate.\n"
        "  Vote NO_GO when OTHER microstructure signals are hostile: rapidly rising OI "
        "signalling a crowded trade, thin liquidation support below entry relative to "
        "position risk, or a heavily lopsided L/S ratio indicating dangerous positioning.\n"
        "  Vote GO when the trade is clean and microstructure is neutral or supportive.\n"
        "  An empty objection list with neutral microstructure should be GO.\n\n"
        "Call submit_arbiter_verdict with your final verdict and a 1–3 sentence "
        "rationale that references specific CoinGlass data points."
    )

    try:
        response = client.messages.create(
            model=_OPUS_MODEL,
            max_tokens=_OPUS_MAX_TOKENS,
            tools=_OPUS_TOOL,  # type: ignore[arg-type]
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        _LOG.warning("arbiter: Opus call failed — falling back to pre-filter result: %s", exc)
        return ArbiterVerdict.GO, f"{pre_filter_reason} (Opus unavailable: {exc})"

    tool_input: dict | None = None
    for block in response.content:
        if hasattr(block, "type") and block.type == "tool_use" and block.name == "submit_arbiter_verdict":
            tool_input = block.input  # type: ignore[attr-defined]
            break

    if tool_input is None:
        _LOG.warning("arbiter: Opus returned no tool call — defaulting to pre-filter GO")
        return ArbiterVerdict.GO, f"{pre_filter_reason} (Opus: no tool call)"

    raw_verdict = tool_input.get("verdict", "GO")
    reasoning: str = tool_input.get("reasoning", "")
    verdict = ArbiterVerdict.GO if raw_verdict == "GO" else ArbiterVerdict.NO_GO
    _LOG.info(
        "arbiter: Opus verdict=%s symbol=%s reasoning=%s",
        raw_verdict, proposal.symbol, reasoning[:120],
    )
    return verdict, f"[Opus] {reasoning}"


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
    coinglass_snapshot: "dict | None" = None,
    client: "anthropic.Anthropic | None" = None,
    funding_crowded_severity: "Severity | None" = None,
) -> ArbiterDecision:
    """Apply Arbiter rules, then Opus 4.8 review, to produce a final GO / NO_GO.

    Flow:
      1. Deterministic pre-filter: any HIGH objection → NO_GO immediately.
         Two or more MEDIUM objections → NO_GO immediately.
      2. If the pre-filter would GO and an Anthropic client is provided,
         call Opus 4.8 with full context (proposal + critic report + CoinGlass
         microstructure).  Opus makes the final call and may override to NO_GO.
      3. GO decisions are appended to the KILL log.
      4. All decisions are appended to decision memory when a candidate is given.

    Args:
        proposal: The TradeProposal from the Proposer.
        report: The CriticReport from the Critic.
        log_path: Path to the KILL log file.  Resolved from
            ``FAST_AGENT_KILL_LOG`` env var if omitted.
        candidate: Scout Candidate (used by decision memory).
        funding_rate: Funding rate at decision time (for memory logging).
        memory_log_path: Override path for decision memory JSONL.
        coinglass_snapshot: CoinGlassSnapshot.to_dict() for the symbol.
            Passed to Opus so it sees live microstructure.  When None,
            Opus still runs but sees all CoinGlass fields as "unknown".
        client: Anthropic client for the Opus 4.8 call.  When None, the
            pipeline runs deterministic-only (no LLM in the Arbiter).
        funding_crowded_severity: Pre-computed FUNDING_CROWDED severity from
            the bracket rules.  Passed to Opus so it does not re-assess the
            funding rate independently.  None = not a concern.

    Returns:
        ArbiterDecision with verdict, reason, and full context.

    Raises:
        ArbiterError: If the KILL log cannot be written on a GO decision.
    """
    effective_log_path = _resolve_log_path(log_path)

    # ── Step 1: deterministic hard gates ────────────────────────────────
    pre_verdict, pre_reason = _apply_rules(report)
    kill_codes_fired = [o.kill_code for o in report.objections]

    # ── Step 2: Opus 4.8 final review (only when pre-filter says GO) ────
    if pre_verdict == ArbiterVerdict.GO and client is not None:
        cg = coinglass_snapshot or {}
        verdict, reason = _opus_review(proposal, report, cg, pre_reason, client, funding_crowded_severity)
    else:
        verdict, reason = pre_verdict, pre_reason

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
