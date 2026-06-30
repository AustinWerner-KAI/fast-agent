"""Tests for src/agents/proposer.py.

All tests mock the Anthropic client — no live API calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from src.agents.proposer import (
    MODEL,
    ProposerError,
    ProposerInput,
    TradeProposal,
    _build_prompt,
    _compute_sizing,
    _parse_response,
    propose,
)
from src.agents.regime import Regime
from src.agents.scout import Candidate

_TS = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _candidate(
    symbol: str = "BTC",
    ma_period: int = 50,
    distance_to_ma_pct: float = 0.3,
    confidence: float = 0.85,
) -> Candidate:
    return Candidate(
        symbol=symbol,
        direction="LONG",
        ma_period=ma_period,
        distance_to_ma_pct=distance_to_ma_pct,
        regime=Regime.TREND,
        confidence=confidence,
        ts=_TS,
    )


def _inp(
    candidate: Candidate | None = None,
    current_price: float = 50_000.0,
    atr: float = 800.0,
    account_equity: float = 100_000.0,
    risk_pct: float = 1.0,
) -> ProposerInput:
    return ProposerInput(
        candidate=candidate or _candidate(),
        current_price=current_price,
        atr=atr,
        account_equity=account_equity,
        risk_pct=risk_pct,
    )


def _mock_response(
    entry: float = 50_000.0,
    stop: float = 49_000.0,
    tp1: float = 51_000.0,
    tp2: float = 52_000.0,
    tp3: float = 53_000.0,
    reasoning: str = "Test reasoning.",
    stop_reason: str = "tool_use",
) -> MagicMock:
    """Build a mock Anthropic Message with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_trade_proposal"
    tool_block.input = {
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "reasoning": reasoning,
    }
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [tool_block]
    return response


def _mock_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# ProposerInput validation
# ---------------------------------------------------------------------------

class TestProposerInput:
    def test_valid_input_constructs(self) -> None:
        inp = _inp()
        assert inp.current_price == 50_000.0
        assert inp.risk_pct == 1.0

    def test_current_price_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            _inp(current_price=0.0)

    def test_atr_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            _inp(atr=0.0)

    def test_risk_pct_capped_at_5(self) -> None:
        with pytest.raises(Exception):
            _inp(risk_pct=10.0)

    def test_account_equity_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            _inp(account_equity=-1.0)


# ---------------------------------------------------------------------------
# TradeProposal validation
# ---------------------------------------------------------------------------

class TestTradeProposal:
    def _valid_long(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = dict(
            symbol="BTC", direction="LONG",
            entry=100.0, stop=98.0, tp1=102.0, tp2=104.0, tp3=106.0,
            position_size_usd=5_000.0, risk_usd=100.0, risk_reward=1.0,
            reasoning="ok", confidence=0.8, ts=_TS,
        )
        base.update(overrides)
        return base

    def test_valid_long_constructs(self) -> None:
        p = TradeProposal(**self._valid_long())
        assert p.direction == "LONG"

    def test_stop_above_entry_rejected(self) -> None:
        with pytest.raises(Exception, match="stop"):
            TradeProposal(**self._valid_long(stop=101.0))

    def test_tp1_below_entry_rejected(self) -> None:
        with pytest.raises(Exception):
            TradeProposal(**self._valid_long(tp1=99.0))

    def test_tps_not_ascending_rejected(self) -> None:
        with pytest.raises(Exception):
            TradeProposal(**self._valid_long(tp2=101.5))  # tp2 < tp1=102

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(Exception):
            TradeProposal(**self._valid_long(confidence=1.5))


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_symbol(self) -> None:
        prompt = _build_prompt(_inp(candidate=_candidate(symbol="ETH")))
        assert "ETH" in prompt

    def test_contains_long_direction(self) -> None:
        prompt = _build_prompt(_inp())
        assert "LONG" in prompt or "long" in prompt

    def test_contains_current_price(self) -> None:
        prompt = _build_prompt(_inp(current_price=42_000.0))
        assert "42000" in prompt

    def test_contains_atr(self) -> None:
        prompt = _build_prompt(_inp(atr=500.0))
        assert "500" in prompt

    def test_contains_ma_period(self) -> None:
        prompt = _build_prompt(_inp(candidate=_candidate(ma_period=200)))
        assert "EMA-200" in prompt

    def test_contains_regime(self) -> None:
        prompt = _build_prompt(_inp())
        assert "TREND" in prompt

    def test_mentions_tool(self) -> None:
        prompt = _build_prompt(_inp())
        assert "submit_trade_proposal" in prompt


# ---------------------------------------------------------------------------
# _compute_sizing
# ---------------------------------------------------------------------------

class TestComputeSizing:
    def test_long_risk_usd(self) -> None:
        risk_usd, _, _ = _compute_sizing(
            entry=100.0, stop=98.0, tp1=102.0,
            account_equity=100_000.0, risk_pct=1.0,
        )
        assert risk_usd == pytest.approx(1_000.0)

    def test_long_position_size(self) -> None:
        # stop_pct = 2/100 = 0.02; size = 1000/0.02 = 50_000
        _, position_size, _ = _compute_sizing(
            entry=100.0, stop=98.0, tp1=102.0,
            account_equity=100_000.0, risk_pct=1.0,
        )
        assert position_size == pytest.approx(50_000.0)

    def test_long_risk_reward(self) -> None:
        # reward = 102-100=2, risk = 100-98=2 → RR=1.0
        _, _, rr = _compute_sizing(
            entry=100.0, stop=98.0, tp1=102.0,
            account_equity=100_000.0, risk_pct=1.0,
        )
        assert rr == pytest.approx(1.0)

    def test_two_to_one_rr(self) -> None:
        _, _, rr = _compute_sizing(
            entry=100.0, stop=98.0, tp1=104.0,
            account_equity=100_000.0, risk_pct=1.0,
        )
        assert rr == pytest.approx(2.0)

    def test_zero_stop_distance_returns_zeros(self) -> None:
        risk, size, rr = _compute_sizing(
            entry=100.0, stop=100.0, tp1=102.0,
            account_equity=100_000.0, risk_pct=1.0,
        )
        assert risk == 0.0
        assert size == 0.0
        assert rr == 0.0

    def test_higher_risk_pct_scales_size(self) -> None:
        _, size_1, _ = _compute_sizing(
            entry=100.0, stop=98.0, tp1=102.0,
            account_equity=100_000.0, risk_pct=1.0,
        )
        _, size_2, _ = _compute_sizing(
            entry=100.0, stop=98.0, tp1=102.0,
            account_equity=100_000.0, risk_pct=2.0,
        )
        assert size_2 == pytest.approx(size_1 * 2.0)


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_long_response_parsed(self) -> None:
        inp = _inp()
        response = _mock_response(
            entry=50_000.0, stop=49_200.0,
            tp1=50_800.0, tp2=51_600.0, tp3=52_400.0,
        )
        proposal = _parse_response(response, inp)
        assert isinstance(proposal, TradeProposal)
        assert proposal.symbol == "BTC"
        assert proposal.direction == "LONG"
        assert proposal.entry == 50_000.0
        assert proposal.stop == 49_200.0

    def test_missing_tool_call_raises_proposer_error(self) -> None:
        inp = _inp()
        response = MagicMock()
        response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        response.content = [text_block]
        with pytest.raises(ProposerError, match="submit_trade_proposal"):
            _parse_response(response, inp)

    def test_wrong_tool_name_raises_proposer_error(self) -> None:
        inp = _inp()
        wrong_block = MagicMock()
        wrong_block.type = "tool_use"
        wrong_block.name = "some_other_tool"
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [wrong_block]
        with pytest.raises(ProposerError):
            _parse_response(response, inp)

    def test_malformed_tool_input_raises_proposer_error(self) -> None:
        inp = _inp()
        bad_block = MagicMock()
        bad_block.type = "tool_use"
        bad_block.name = "submit_trade_proposal"
        bad_block.input = {"entry": "not_a_number", "stop": None}
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [bad_block]
        with pytest.raises(ProposerError):
            _parse_response(response, inp)

    def test_invalid_geometry_raises_proposer_error(self) -> None:
        inp = _inp()
        # LONG with stop above entry — geometrically invalid
        response = _mock_response(
            entry=50_000.0, stop=51_000.0,
            tp1=52_000.0, tp2=53_000.0, tp3=54_000.0,
        )
        with pytest.raises(ProposerError):
            _parse_response(response, inp)

    def test_proposal_carries_candidate_metadata(self) -> None:
        c = _candidate(confidence=0.92)
        inp = _inp(candidate=c)
        response = _mock_response()
        proposal = _parse_response(response, inp)
        assert proposal.confidence == pytest.approx(0.92)
        assert proposal.ts == _TS

    def test_reasoning_preserved(self) -> None:
        inp = _inp()
        response = _mock_response(reasoning="Strong EMA-50 bounce expected.")
        proposal = _parse_response(response, inp)
        assert proposal.reasoning == "Strong EMA-50 bounce expected."

    def test_sizing_fields_populated(self) -> None:
        inp = _inp(account_equity=100_000.0, risk_pct=1.0)
        response = _mock_response(
            entry=50_000.0, stop=49_000.0,
            tp1=51_000.0, tp2=52_000.0, tp3=53_000.0,
        )
        proposal = _parse_response(response, inp)
        assert proposal.risk_usd > 0
        assert proposal.position_size_usd > 0
        assert proposal.risk_reward > 0


# ---------------------------------------------------------------------------
# propose (integration — client fully mocked)
# ---------------------------------------------------------------------------

class TestPropose:
    def test_returns_trade_proposal(self) -> None:
        client = _mock_client(_mock_response())
        result = propose(_inp(), client=client)
        assert isinstance(result, TradeProposal)

    def test_calls_correct_model(self) -> None:
        client = _mock_client(_mock_response())
        propose(_inp(), client=client)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == MODEL

    def test_tool_choice_forces_tool(self) -> None:
        client = _mock_client(_mock_response())
        propose(_inp(), client=client)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"]["type"] == "tool"
        assert call_kwargs["tool_choice"]["name"] == "submit_trade_proposal"

    def test_api_error_raises_proposer_error(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIStatusError(
            "rate limit",
            response=MagicMock(status_code=429),
            body={},
        )
        with pytest.raises(ProposerError, match="API error"):
            propose(_inp(), client=client)

    def test_no_client_no_key_raises_proposer_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove key if present
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ProposerError, match="ANTHROPIC_API_KEY"):
                propose(_inp(), client=None)

    def test_messages_list_has_user_role(self) -> None:
        client = _mock_client(_mock_response())
        propose(_inp(), client=client)
        messages = client.messages.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"
