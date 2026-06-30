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

import anthropic
from pydantic import BaseModel, Field, model_validator

from src.agents.scout import Candidate

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

__all__ = ["ProposerInput", "TradeProposal", "ProposerError", "propose"]


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
    direction: Literal["LONG", "SHORT"]
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
        """Enforce that stop, entry, and TPs are on the correct sides."""
        if self.direction == "LONG":
            if not self.stop < self.entry:
                raise ValueError(
                    f"LONG stop ({self.stop}) must be strictly below entry ({self.entry})"
                )
            if not (self.entry < self.tp1 < self.tp2 < self.tp3):
                raise ValueError(
                    f"LONG TPs must be strictly increasing above entry "
                    f"({self.entry}): tp1={self.tp1}, tp2={self.tp2}, tp3={self.tp3}"
                )
        else:
            if not self.stop > self.entry:
                raise ValueError(
                    f"SHORT stop ({self.stop}) must be strictly above entry ({self.entry})"
                )
            if not (self.entry > self.tp1 > self.tp2 > self.tp3):
                raise ValueError(
                    f"SHORT TPs must be strictly decreasing below entry "
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
                    "Stop-loss price. LONG: below entry by ~0.5–1 ATR. Keep stops tight — mean-reversion entry, not a swing stop. "
                    "SHORT: above entry by ~0.5–1 ATR."
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

def _build_prompt(inp: ProposerInput) -> str:
    """Construct the user message sent to Claude.

    Args:
        inp: Proposer input containing the candidate and market context.

    Returns:
        Formatted prompt string.
    """
    c = inp.candidate
    direction_word = "long" if c.direction == "LONG" else "short"
    ma_label = f"EMA-{c.ma_period}"
    dist_sign = "above" if c.distance_to_ma_pct >= 0 else "below"
    dist_abs = abs(c.distance_to_ma_pct)

    return (
        f"You are a quantitative trade proposer. Generate a {direction_word} trade proposal "
        f"for {c.symbol} based on the following context.\n\n"
        f"Signal summary:\n"
        f"  Symbol:        {c.symbol}\n"
        f"  Direction:     {c.direction}\n"
        f"  Setup:         Pullback to {ma_label} in a TREND regime\n"
        f"  Current price: {inp.current_price:.6g}\n"
        f"  Distance to {ma_label}: {dist_abs:.3f}% {dist_sign} the MA\n"
        f"  ATR (1h):      {inp.atr:.6g}  (use as stop-sizing reference)\n"
        f"  Regime:        {c.regime.value}\n"
        f"  Scout confidence: {c.confidence:.3f}\n\n"
        + (
            f"Market microstructure:\n"
            f"  Liquidation support below entry: "
            f"{'${:,.0f}'.format(inp.liquidation_below_usd) if inp.liquidation_below_usd is not None else 'unknown'}\n\n"
            if inp.liquidation_below_usd is not None else ""
        )
        + f"Rules:\n"
        f"  - Entry should be at or near the current price / {ma_label} level.\n"
        f"  - Stop should be 1.0–1.5 × ATR on the wrong side of the MA.\n"
        f"  - TP1 at ~1:1 R:R, TP2 at ~1:2 R:R, TP3 at ~1:3 R:R.\n"
        f"  - All prices must be positive and geometrically valid for a {c.direction}.\n\n"
        f"Call submit_trade_proposal with your proposed levels and a brief reasoning."
    )


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

    risk_usd, position_size_usd, rr = _compute_sizing(
        entry=entry,
        stop=stop,
        tp1=tp1,
        direction=inp.candidate.direction,
        account_equity=inp.account_equity,
        risk_pct=inp.risk_pct,
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
            confidence=inp.candidate.confidence,
            ts=inp.candidate.ts,
        )
    except Exception as exc:
        raise ProposerError(f"TradeProposal validation failed: {exc}") from exc


def _compute_sizing(
    entry: float,
    stop: float,
    tp1: float,
    direction: Literal["LONG", "SHORT"],
    account_equity: float,
    risk_pct: float,
) -> tuple[float, float, float]:
    """Compute position sizing and R:R deterministically.

    Args:
        entry: Entry price.
        stop: Stop-loss price.
        tp1: First take-profit price.
        direction: LONG or SHORT.
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

    if direction == "LONG":
        reward = tp1 - entry
    else:
        reward = entry - tp1
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
