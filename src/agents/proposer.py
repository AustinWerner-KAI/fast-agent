"""Proposer agent: converts a Scout Candidate into a structured trade proposal.

Calls Claude Haiku via tool_use to generate entry, stop, TP1/TP2/TP3 levels
and a reasoning narrative.  Position sizing and R:R are computed deterministically
after the LLM returns its levels — the model is not asked to do arithmetic.

Public surface:
    ProposerInput, TradeProposal, ProposerError, propose
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal

import logging

import anthropic
from pydantic import BaseModel, Field, model_validator

from src.agents.scout import Candidate

_LOG = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

__all__ = ["ProposerInput", "TradeProposal", "ProposerError", "propose", "_adjust_confidence_for_ma_stack"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProposerError(Exception):
    """Raised when the Proposer cannot produce a valid trade proposal."""


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

class ProposerInput(BaseModel):
    """All context the Proposer needs to generate a trade proposal.

    Attributes:
        candidate: Scout output — symbol, direction, MA level, confidence.
        current_price: Last close price visible at decision_ts.
        atr: Average True Range on the entry timeframe (absolute price units).
        account_equity: Total account equity in USD for position sizing.
        risk_pct: Percentage of equity to risk on this trade (e.g. 1.0 = 1%).
        liquidation_below_usd: Optional CoinGlass USD value of liquidation clusters
            below the current price.  Passed through to the prompt so the
            Proposer can adjust its stop placement if support is thin.
    """

    candidate: Candidate
    current_price: float = Field(..., gt=0)
    atr: float = Field(..., gt=0)
    account_equity: float = Field(default=100_000.0, gt=0)
    risk_pct: float = Field(default=1.0, gt=0, le=5.0)
    liquidation_below_usd: float | None = None  # CoinGlass liq support below entry


class TradeProposal(BaseModel):
    """Structured trade proposal produced by the Proposer.

    Geometric validity is enforced by a model validator:
    - LONG:  stop < entry < tp1 < tp2 < tp3
    - SHORT: stop > entry > tp1 > tp2 > tp3

    Attributes:
        symbol: Coin name.
        direction: LONG or SHORT.
        entry: Proposed entry price.
        stop: Stop loss price.
        tp1: Take profit 1 (~2:1 R:R).
        tp2: Take profit 2 (~1:2 R:R).
        tp3: Take profit 3 (~1:3 R:R).
        position_size_usd: Notional position size in USD.
        risk_usd: Maximum dollar loss if stop is hit.
        risk_reward: R:R ratio measured to TP1.
        reasoning: Claude's narrative explaining the proposal (2–4 sentences).
        confidence: Propagated from the Scout Candidate.
        ts: decision_ts from the Candidate.
    """

    symbol: str
    direction: Literal["LONG"]
    entry: float = Field(..., gt=0)
    stop: float = Field(..., gt=0)
    tp1: float = Field(..., gt=0)
    tp2: float = Field(..., gt=0)
    tp3: float = Field(..., gt=0)
    position_size_usd: float = Field(..., ge=0)
    risk_usd: float = Field(..., ge=0)
    risk_reward: float = Field(..., ge=0)
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    ts: datetime

    @model_validator(mode="after")
    def validate_geometry(self) -> "TradeProposal":
        """Enforce LONG geometry: stop < entry < tp1 < tp2 < tp3."""
        if not self.stop < self.entry:
            raise ValueError(
                f"stop ({self.stop}) must be strictly below entry ({self.entry})"
            )
        if not (self.entry < self.tp1 < self.tp2 < self.tp3):
            raise ValueError(
                f"TPs must be strictly increasing above entry "
                f"({self.entry}): tp1={self.tp1}, tp2={self.tp2}, tp3={self.tp3}"
            )
        return self


# ---------------------------------------------------------------------------
# Tool definition (forces Claude to emit structured output)
# ---------------------------------------------------------------------------

_PROPOSAL_TOOL: dict[str, Any] = {
    "name": "submit_trade_proposal",
    "description": (
        "Submit a structured trade proposal with entry, stop-loss, and three "
        "take-profit levels.  All prices must be positive.  For a LONG trade "
        "stop < entry < tp1 < tp2 < tp3; for SHORT stop > entry > tp1 > tp2 > tp3."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entry": {
                "type": "number",
                "description": "Proposed entry price (limit or market).",
            },
            "stop": {
                "type": "number",
                "description": (
                    "Stop-loss price. Must be below entry by 0.5–1.0 × ATR. "
                    "Keep stops tight — this is a mean-reversion entry, not a swing stop."
                ),
            },
            "tp1": {
                "type": "number",
                "description": "Take-profit 1: approximately 2:1 R:R from entry. Must achieve at least 1.5:1 R:R.",
            },
            "tp2": {
                "type": "number",
                "description": "Take-profit 2: approximately 1:2 R:R from entry.",
            },
            "tp3": {
                "type": "number",
                "description": "Take-profit 3: approximately 1:3 R:R from entry.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Concise rationale for this proposal (2–4 sentences). "
                    "Reference the MA pullback, regime, and key levels."
                ),
            },
        },
        "required": ["entry", "stop", "tp1", "tp2", "tp3", "reasoning"],
    },
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ma_stack_section(inp: ProposerInput) -> str:
    """Build the daily MA stack context block for the prompt.

    Returns an empty string when no daily EMA values are available.

    Args:
        inp: Proposer input whose candidate carries optional daily EMA values.

    Returns:
        Formatted multi-line string (trailing newline included) or ``""``.
    """
    c = inp.candidate
    price = inp.current_price
    rows: list[str] = []
    for period, val in ((20, c.ema20_daily), (50, c.ema50_daily), (200, c.ema200_daily)):
        if val is not None:
            rel = "ABOVE" if price > val else "BELOW"
            rows.append(f"  EMA-{period}: {val:.6g}  — price is {rel}")
    if not rows:
        return ""
    if c.ema20_daily is not None and c.ema50_daily is not None and c.ema200_daily is not None:
        if c.ema20_daily > c.ema50_daily > c.ema200_daily:
            structure = "BULLISH (EMA-20 > EMA-50 > EMA-200)"
        elif c.ema20_daily < c.ema50_daily < c.ema200_daily:
            structure = "BEARISH (EMA-20 < EMA-50 < EMA-200)"
        else:
            structure = "MIXED"
        rows.append(f"  MA structure: {structure}")
    return "Daily MA stack:\n" + "\n".join(rows) + "\n\n"


def _build_prompt(inp: ProposerInput) -> str:
    """Construct the user message sent to Claude.

    Args:
        inp: Proposer input containing the candidate and market context.

    Returns:
        Formatted prompt string.
    """
    c = inp.candidate
    ma_label = f"EMA-{c.ma_period}"
    dist_sign = "above" if c.distance_to_ma_pct >= 0 else "below"
    dist_abs = abs(c.distance_to_ma_pct)

    return (
        f"You are a quantitative trade proposer. Generate a LONG trade proposal "
        f"for {c.symbol} based on the following context.\n\n"
        f"Signal summary:\n"
        f"  Symbol:        {c.symbol}\n"
        f"  Direction:     LONG\n"
        f"  Setup:         Pullback to {ma_label} in a TREND regime\n"
        f"  Current price: {inp.current_price:.6g}\n"
        f"  Distance to {ma_label}: {dist_abs:.3f}% {dist_sign} the MA\n"
        f"  ATR (1h):      {inp.atr:.6g}  (use as stop-sizing reference)\n"
        f"  Regime:        {c.regime.value}\n"
        f"  Scout confidence: {c.confidence:.3f}\n\n"
        + _ma_stack_section(inp)
        + (
            f"Market microstructure:\n"
            f"  Liquidation support below entry: "
            f"{'${:,.0f}'.format(inp.liquidation_below_usd) if inp.liquidation_below_usd is not None else 'unknown'}\n\n"
            if inp.liquidation_below_usd is not None else ""
        )
        + f"Rules:\n"
        f"  - Entry should be at or near the current price / {ma_label} level.\n"
        f"  - Stop must be BELOW entry by 0.5–1.0 × ATR (keep it tight).\n"
        f"  - TP1 at 2:1 R:R (reward = 2 × stop distance above entry).\n"
        f"  - TP2 at 3:1 R:R, TP3 at 4:1 R:R — all above TP1.\n"
        f"  - Geometry: stop < entry < tp1 < tp2 < tp3 (all positive).\n"
        f"Daily MA alignment (LONG only):\n"
        f"  - Price ABOVE EMA-200 daily: full conviction entry permitted.\n"
        f"  - Price BELOW EMA-200 but ABOVE EMA-50: note as reduced conviction in reasoning.\n"
        f"  - Price BELOW EMA-50 daily: note as low conviction in reasoning.\n"
        f"  - Full BULLISH stack (price > EMA-20 > EMA-50 > EMA-200): note as high conviction.\n"
        f"  - Do not propose a LONG entry if price is below all three daily MAs.\n\n"
        f"Example: entry=100, stop=99 (1 ATR), tp1=102 (2:1), tp2=103 (3:1), tp3=104 (4:1).\n\n"
        f"Call submit_trade_proposal with your proposed levels and a brief reasoning."
    )


def _adjust_confidence_for_ma_stack(
    confidence: float,
    current_price: float,
    ema20_daily: float | None,
    ema50_daily: float | None,
    ema200_daily: float | None,
) -> float:
    """Deterministically adjust Scout confidence based on daily MA alignment.

    Adjustments (applied once, in order):
    - EMA-200 unavailable: no adjustment (insufficient daily history).
    - Price below EMA-200 AND below EMA-50: −0.20 (low conviction against trend).
    - Price below EMA-200 only: −0.10 (reduced conviction).
    - Full bullish stack (price > EMA-20 > EMA-50 > EMA-200): +0.05 bonus.
    Result is clamped to [0.0, 1.0].

    Args:
        confidence: Raw Scout confidence in [0.0, 1.0].
        current_price: Current market price (1h close).
        ema20_daily: Daily EMA-20 value, or None when unavailable.
        ema50_daily: Daily EMA-50 value, or None when unavailable.
        ema200_daily: Daily EMA-200 value, or None when unavailable.

    Returns:
        Adjusted confidence clamped to [0.0, 1.0].
    """
    if ema200_daily is None:
        return confidence
    adjusted = confidence
    if current_price < ema200_daily:
        if ema50_daily is not None and current_price < ema50_daily:
            adjusted -= 0.20
        else:
            adjusted -= 0.10
    elif (
        ema20_daily is not None
        and ema50_daily is not None
        and current_price > ema20_daily
        and ema20_daily > ema50_daily
        and ema50_daily > ema200_daily
    ):
        adjusted += 0.05
    return round(max(0.0, min(1.0, adjusted)), 6)


def _parse_response(
    response: anthropic.types.Message,
    inp: ProposerInput,
) -> TradeProposal:
    """Extract the tool call from Claude's response and build a TradeProposal.

    Args:
        response: Raw response from the Anthropic API.
        inp: Original proposer input (for sizing and metadata).

    Returns:
        Validated ``TradeProposal``.

    Raises:
        ProposerError: If the response does not contain a valid tool call.
    """
    tool_input: dict[str, Any] | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_trade_proposal":
            tool_input = block.input
            break

    if tool_input is None:
        raise ProposerError(
            f"Claude did not call submit_trade_proposal. "
            f"stop_reason={response.stop_reason!r}, "
            f"content_types={[b.type for b in response.content]}"
        )

    try:
        entry: float = float(tool_input["entry"])
        stop: float = float(tool_input["stop"])
        tp1: float = float(tool_input["tp1"])
        tp2: float = float(tool_input["tp2"])
        tp3: float = float(tool_input["tp3"])
        reasoning: str = str(tool_input["reasoning"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProposerError(f"Malformed tool arguments: {exc}") from exc

    # Clamp stop if Haiku placed it closer than 0.5 × ATR to entry.
    # When clamped, recompute all TPs at 2R/3R/4R from the new stop so
    # geometry remains consistent and TP levels don't diverge from the stop.
    min_stop_dist = 0.5 * inp.atr
    actual_stop_dist = abs(entry - stop)
    if actual_stop_dist < min_stop_dist:
        _LOG.warning(
            "proposer: STOP_TOO_TIGHT %s distance=%.6f min_required=%.6f (0.5×ATR=%.6f) "
            "— clamping stop and recomputing TPs at 2R/3R/4R",
            inp.candidate.symbol, actual_stop_dist, min_stop_dist, inp.atr,
        )
        stop = round(entry - min_stop_dist, 8)
        tp1 = round(entry + 2.0 * min_stop_dist, 8)
        tp2 = round(entry + 3.0 * min_stop_dist, 8)
        tp3 = round(entry + 4.0 * min_stop_dist, 8)

    risk_usd, position_size_usd, rr = _compute_sizing(
        entry=entry,
        stop=stop,
        tp1=tp1,
        account_equity=inp.account_equity,
        risk_pct=inp.risk_pct,
    )

    adjusted_confidence = _adjust_confidence_for_ma_stack(
        confidence=inp.candidate.confidence,
        current_price=inp.current_price,
        ema20_daily=inp.candidate.ema20_daily,
        ema50_daily=inp.candidate.ema50_daily,
        ema200_daily=inp.candidate.ema200_daily,
    )

    try:
        return TradeProposal(
            symbol=inp.candidate.symbol,
            direction=inp.candidate.direction,
            entry=entry,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            position_size_usd=position_size_usd,
            risk_usd=risk_usd,
            risk_reward=rr,
            reasoning=reasoning,
            confidence=adjusted_confidence,
            ts=inp.candidate.ts,
        )
    except Exception as exc:
        raise ProposerError(f"TradeProposal validation failed: {exc}") from exc


def _compute_sizing(
    entry: float,
    stop: float,
    tp1: float,
    account_equity: float,
    risk_pct: float,
) -> tuple[float, float, float]:
    """Compute LONG position sizing and R:R deterministically.

    Args:
        entry: Entry price.
        stop: Stop-loss price (must be below entry).
        tp1: First take-profit price (must be above entry).
        account_equity: Total equity in USD.
        risk_pct: Percentage of equity to risk.

    Returns:
        Tuple of (risk_usd, position_size_usd, risk_reward).
        Returns (0, 0, 0) when stop == entry (degenerate input guard).
    """
    stop_distance = abs(entry - stop)
    if stop_distance == 0 or entry == 0:
        return 0.0, 0.0, 0.0

    risk_usd = account_equity * risk_pct / 100.0
    stop_pct = stop_distance / entry
    position_size_usd = risk_usd / stop_pct
    reward = tp1 - entry
    risk_reward = reward / stop_distance if stop_distance > 0 else 0.0

    return round(risk_usd, 2), round(position_size_usd, 2), round(risk_reward, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose(
    inp: ProposerInput,
    client: anthropic.Anthropic | None = None,
) -> TradeProposal:
    """Generate a trade proposal for a Scout Candidate via Claude Haiku.

    Calls the Anthropic API using tool_use to guarantee structured output.
    Position sizing is computed deterministically after the LLM returns levels.

    Args:
        inp: Proposer input — candidate plus market context.
        client: Optional pre-constructed ``anthropic.Anthropic`` instance.
            When omitted, a client is created from ``ANTHROPIC_API_KEY`` env var.

    Returns:
        A validated ``TradeProposal``.

    Raises:
        ProposerError: If the API call fails, the model does not call the tool,
            or the returned levels fail geometric validation.
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProposerError("ANTHROPIC_API_KEY not set and no client provided")
        client = anthropic.Anthropic(api_key=api_key)

    prompt = _build_prompt(inp)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[_PROPOSAL_TOOL],
            tool_choice={"type": "tool", "name": "submit_trade_proposal"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise ProposerError(f"Anthropic API error: {exc}") from exc

    return _parse_response(response, inp)
