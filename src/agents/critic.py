"""Critic agent: adversarial reviewer of TradeProposals.

Calls Claude Haiku via tool_use to generate structured objections against a
TradeProposal.  Each objection carries a KILL code from a fixed taxonomy,
a severity (HIGH / MEDIUM / LOW), and a reasoning string.

Verdict is computed deterministically from the objection list — the model is
never asked to decide pass/fail:
  - Any HIGH-severity objection → KILL
  - Only MEDIUM / LOW objections, or empty list → PASS

Public surface:
    KillCode, Severity, Objection, CriticInput, CriticReport, CriticError, critique
"""
from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from typing import Any

import anthropic
from pydantic import BaseModel, Field

from src.agents.proposer import TradeProposal

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

__all__ = [
    "KillCode", "Severity", "Verdict",
    "Objection", "CriticInput", "CriticReport",
    "CriticError", "critique",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CriticError(Exception):
    """Raised when the Critic cannot produce a valid report."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class KillCode(str, Enum):
    """Taxonomy of reasons a trade proposal may be vetoed or flagged."""

    THIN_LIQUIDITY = "THIN_LIQUIDITY"
    """Insufficient market depth to fill the position without damaging slippage."""

    FUNDING_CROWDED = "FUNDING_CROWDED"
    """Funding rate signals a crowded trade — elevated squeeze / fade risk."""

    BOOK_IMBALANCE_AGAINST = "BOOK_IMBALANCE_AGAINST"
    """Orderbook is stacked against the proposed direction."""

    REGIME_MISMATCH = "REGIME_MISMATCH"
    """Current price structure contradicts the stated TREND regime."""

    RR_INADEQUATE = "RR_INADEQUATE"
    """Risk:reward ratio to TP1 falls below the acceptable minimum."""

    FOMC_WINDOW = "FOMC_WINDOW"
    """An FOMC announcement is imminent, creating outsized macro uncertainty."""

    CHOP_STRUCTURE = "CHOP_STRUCTURE"
    """Price action appears choppy / range-bound despite the TREND classification."""


class Severity(str, Enum):
    """How seriously an objection should be weighted."""

    HIGH = "HIGH"
    """Veto-level: blocks the trade outright."""

    MEDIUM = "MEDIUM"
    """Notable concern: proceed with caution or at reduced size."""

    LOW = "LOW"
    """Minor flag: worth noting but not a blocker."""


class Verdict(str, Enum):
    """Final pass/fail decision derived from the objection list."""

    PASS = "PASS"
    KILL = "KILL"


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

class Objection(BaseModel):
    """A single structured objection raised by the Critic.

    Attributes:
        kill_code: Category from the fixed KILL taxonomy.
        severity: How seriously this objection should be weighted.
        reasoning: 1–3 sentence explanation grounded in the proposal data.
    """

    kill_code: KillCode
    severity: Severity
    reasoning: str


class CriticInput(BaseModel):
    """All context the Critic needs to evaluate a TradeProposal.

    Attributes:
        proposal: The TradeProposal produced by the Proposer.
        funding_rate: Current perpetual funding rate as a decimal
            (e.g. 0.0001 = 0.01% per 8h).  None if unavailable.
        book_bid_depth_usd: Total USD bid depth visible in the orderbook.
            None if unavailable.
        book_ask_depth_usd: Total USD ask depth visible in the orderbook.
            None if unavailable.
        fomc_hours_until: Hours until the next FOMC announcement.
            None if not known.
        min_rr: Minimum acceptable R:R ratio to TP1.
    """

    proposal: TradeProposal
    funding_rate: float | None = None
    book_bid_depth_usd: float | None = None
    book_ask_depth_usd: float | None = None
    fomc_hours_until: float | None = None
    min_rr: float = Field(default=1.5, gt=0)
    liquidation_below_usd: float | None = None  # CoinGlass: USD liq support below entry
    oi_change_24h_pct: float | None = None       # CoinGlass: OI 24h change %


class CriticReport(BaseModel):
    """The Critic's full output for one TradeProposal.

    Attributes:
        proposal: The evaluated proposal (passed through unchanged).
        objections: All objections raised (may be empty).
        verdict: KILL if any objection is HIGH severity, otherwise PASS.
        overall_assessment: Claude's 1–2 sentence summary of the critique.
        ts: Timestamp propagated from the proposal.
    """

    proposal: TradeProposal
    objections: list[Objection]
    verdict: Verdict
    overall_assessment: str
    ts: datetime


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_CRITIC_TOOL: dict[str, Any] = {
    "name": "submit_critic_report",
    "description": (
        "Submit a structured critique of the trade proposal. "
        "Raise objections only when genuinely warranted by the data provided. "
        "Use an empty objections array if the proposal looks clean."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "objections": {
                "type": "array",
                "description": (
                    "List of structured objections. Empty array means no concerns. "
                    "Each objection must use one of the approved kill_code values."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kill_code": {
                            "type": "string",
                            "enum": [k.value for k in KillCode],
                            "description": "Category from the KILL taxonomy.",
                        },
                        "severity": {
                            "type": "string",
                            "enum": [s.value for s in Severity],
                            "description": "HIGH = veto, MEDIUM = caution, LOW = flag.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "1–3 sentence explanation grounded in the proposal data.",
                        },
                    },
                    "required": ["kill_code", "severity", "reasoning"],
                },
            },
            "overall_assessment": {
                "type": "string",
                "description": "1–2 sentence summary of the critique overall.",
            },
        },
        "required": ["objections", "overall_assessment"],
    },
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fmt_optional(value: float | None, unit: str = "", fmt: str = ".4f") -> str:
    """Format an optional float for the prompt, falling back to 'unknown'."""
    if value is None:
        return "unknown"
    return f"{value:{fmt}}{unit}"


def _build_prompt(inp: CriticInput) -> str:
    """Construct the adversarial review prompt sent to Claude.

    Args:
        inp: Critic input containing the proposal and market context.

    Returns:
        Formatted prompt string.
    """
    p = inp.proposal
    stop_dist_pct = abs(p.entry - p.stop) / p.entry * 100

    # Orderbook imbalance string
    if inp.book_bid_depth_usd is not None and inp.book_ask_depth_usd is not None:
        total = inp.book_bid_depth_usd + inp.book_ask_depth_usd
        bid_pct = inp.book_bid_depth_usd / total * 100 if total > 0 else 50.0
        book_str = (
            f"bid ${inp.book_bid_depth_usd:,.0f} ({bid_pct:.0f}%) / "
            f"ask ${inp.book_ask_depth_usd:,.0f} ({100 - bid_pct:.0f}%)"
        )
    else:
        book_str = "unknown"

    # FOMC string
    if inp.fomc_hours_until is not None:
        fomc_str = f"{inp.fomc_hours_until:.1f}h away"
    else:
        fomc_str = "timing unknown"

    funding_str = _fmt_optional(inp.funding_rate, unit="", fmt=".6f")
    if inp.funding_rate is not None:
        funding_str += f"  ({inp.funding_rate * 100:.4f}% per 8h)"

    return (
        "You are an adversarial trade critic. Review the proposal below and raise objections "
        "for any genuine concerns using the KILL taxonomy provided.\n\n"
        "PROPOSAL:\n"
        f"  Symbol:      {p.symbol}\n"
        f"  Direction:   {p.direction}\n"
        f"  Entry:       {p.entry:.6g}\n"
        f"  Stop:        {p.stop:.6g}  ({stop_dist_pct:.2f}% from entry)\n"
        f"  TP1:         {p.tp1:.6g}  (R:R to TP1: {p.risk_reward:.2f})\n"
        f"  TP2:         {p.tp2:.6g}\n"
        f"  TP3:         {p.tp3:.6g}\n"
        f"  Position:    ${p.position_size_usd:,.0f} notional\n"
        f"  Risk:        ${p.risk_usd:,.0f}\n"
        f"  Proposer reasoning: {p.reasoning}\n\n"
        "MARKET CONTEXT:\n"
        f"  Funding rate:       {funding_str}\n"
        f"  Orderbook depth:    {book_str}\n"
        f"  Next FOMC:          {fomc_str}\n"
        f"  Min R:R required:   {inp.min_rr:.2f}\n\n"
        "MARKET MICROSTRUCTURE (CoinGlass):\n"
        f"  Liquidation support below entry: "
        f"{'${:,.0f}'.format(inp.liquidation_below_usd) if inp.liquidation_below_usd is not None else 'unknown'}\n"
        f"  OI change 24h:      "
        f"{'{:+.1f}%'.format(inp.oi_change_24h_pct) if inp.oi_change_24h_pct is not None else 'unknown'}\n\n"
        "KILL CODE TAXONOMY:\n"
        "  THIN_LIQUIDITY         — insufficient depth to fill without damaging slippage\n"
        "  FUNDING_CROWDED        — extreme funding rate signals a crowded trade\n"
        "  BOOK_IMBALANCE_AGAINST — orderbook stacked against the direction\n"
        "  REGIME_MISMATCH        — structure contradicts the stated TREND regime\n"
        "  RR_INADEQUATE          — R:R to TP1 is below the minimum threshold\n"
        "  FOMC_WINDOW            — FOMC within 48h creates outsized macro uncertainty\n"
        "  CHOP_STRUCTURE         — price action appears range-bound despite TREND label\n\n"
        "SEVERITY:\n"
        "  HIGH   — veto-level: blocks the trade\n"
        "  MEDIUM — notable: proceed at reduced size or with extra caution\n"
        "  LOW    — minor flag: worth noting, not a blocker\n\n"
        "Raise an objection only when the data above gives you genuine grounds for concern. "
        "An empty objections list is a valid response for a clean proposal.\n"
        "Call submit_critic_report with your objections and a 1–2 sentence overall_assessment."
    )


def _parse_response(
    response: anthropic.types.Message,
    inp: CriticInput,
) -> CriticReport:
    """Extract the tool call and build a CriticReport.

    Verdict is computed deterministically: any HIGH objection → KILL.

    Args:
        response: Raw Anthropic API response.
        inp: Original critic input (for proposal passthrough and timestamp).

    Returns:
        Validated ``CriticReport``.

    Raises:
        CriticError: If the response lacks a valid tool call or contains
            unrecognised kill_code / severity values.
    """
    tool_input: dict[str, Any] | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_critic_report":
            tool_input = block.input
            break

    if tool_input is None:
        raise CriticError(
            f"Claude did not call submit_critic_report. "
            f"stop_reason={response.stop_reason!r}, "
            f"content_types={[b.type for b in response.content]}"
        )

    raw_objections = tool_input.get("objections")
    overall_assessment = str(tool_input.get("overall_assessment", ""))

    if not isinstance(raw_objections, list):
        raise CriticError(
            f"'objections' must be a list, got {type(raw_objections).__name__}"
        )

    objections: list[Objection] = []
    for i, raw in enumerate(raw_objections):
        if not isinstance(raw, dict):
            raise CriticError(f"objections[{i}] is not a dict: {raw!r}")
        try:
            objections.append(
                Objection(
                    kill_code=KillCode(raw["kill_code"]),
                    severity=Severity(raw["severity"]),
                    reasoning=str(raw["reasoning"]),
                )
            )
        except (KeyError, ValueError) as exc:
            raise CriticError(f"Invalid objection at index {i}: {exc}") from exc

    has_high = any(obj.severity == Severity.HIGH for obj in objections)
    verdict = Verdict.KILL if has_high else Verdict.PASS

    return CriticReport(
        proposal=inp.proposal,
        objections=objections,
        verdict=verdict,
        overall_assessment=overall_assessment,
        ts=inp.proposal.ts,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def critique(
    inp: CriticInput,
    client: anthropic.Anthropic | None = None,
) -> CriticReport:
    """Generate an adversarial critique of a TradeProposal via Claude Haiku.

    Calls the Anthropic API using tool_use so the objection list is always
    structured JSON.  Verdict is derived deterministically from severity —
    the model is not asked to decide pass/fail.

    Args:
        inp: Critic input — the proposal plus market context.
        client: Optional pre-constructed ``anthropic.Anthropic`` instance.
            When omitted, a client is created from ``ANTHROPIC_API_KEY`` env var.

    Returns:
        A ``CriticReport`` with objections and a PASS / KILL verdict.

    Raises:
        CriticError: If the API call fails, the model does not call the tool,
            or the response contains invalid kill_code / severity values.
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise CriticError("ANTHROPIC_API_KEY not set and no client provided")
        client = anthropic.Anthropic(api_key=api_key)

    prompt = _build_prompt(inp)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[_CRITIC_TOOL],
            tool_choice={"type": "tool", "name": "submit_critic_report"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise CriticError(f"Anthropic API error: {exc}") from exc

    return _parse_response(response, inp)
