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
    _adjust_confidence_for_ma_stack,
    _build_prompt,
    _compute_sizing,
    _ma_stack_section,
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

    # ── ATR minimum stop clamp ────────────────────────────────────────────

    def test_stop_too_tight_is_clamped_to_half_atr(self) -> None:
        # Haiku proposes stop 0.0006 from entry; ATR=0.004 → min=0.002.
        # Mirrors the TRX trade that triggered this fix.
        inp = _inp(current_price=0.327576, atr=0.004)
        response = _mock_response(
            entry=0.3262, stop=0.3256,           # stop distance = 0.0006 < 0.002
            tp1=0.3374, tp2=0.3486, tp3=0.3598,
        )
        proposal = _parse_response(response, inp)
        expected_stop = round(0.3262 - 0.002, 8)
        assert proposal.stop == pytest.approx(expected_stop, abs=1e-7)

    def test_clamped_tps_are_at_2r_3r_4r(self) -> None:
        inp = _inp(current_price=0.327576, atr=0.004)
        response = _mock_response(
            entry=0.3262, stop=0.3256,
            tp1=0.3374, tp2=0.3486, tp3=0.3598,
        )
        proposal = _parse_response(response, inp)
        stop_dist = proposal.entry - proposal.stop
        assert proposal.tp1 == pytest.approx(proposal.entry + 2.0 * stop_dist, abs=1e-7)
        assert proposal.tp2 == pytest.approx(proposal.entry + 3.0 * stop_dist, abs=1e-7)
        assert proposal.tp3 == pytest.approx(proposal.entry + 4.0 * stop_dist, abs=1e-7)

    def test_clamped_proposal_passes_geometry_validator(self) -> None:
        inp = _inp(current_price=0.327576, atr=0.004)
        response = _mock_response(
            entry=0.3262, stop=0.3256,
            tp1=0.3374, tp2=0.3486, tp3=0.3598,
        )
        proposal = _parse_response(response, inp)
        # Geometry must hold after clamping: stop < entry < tp1 < tp2 < tp3
        assert proposal.stop < proposal.entry
        assert proposal.entry < proposal.tp1 < proposal.tp2 < proposal.tp3

    def test_stop_at_exactly_half_atr_is_not_clamped(self) -> None:
        # stop distance = exactly 0.5 × ATR → no warning, no change.
        atr = 0.004
        entry = 0.3262
        stop = round(entry - 0.5 * atr, 8)   # = 0.3242, exactly at boundary
        tp1 = round(entry + 2 * 0.5 * atr, 8)
        tp2 = round(entry + 3 * 0.5 * atr, 8)
        tp3 = round(entry + 4 * 0.5 * atr, 8)
        inp = _inp(current_price=entry, atr=atr)
        response = _mock_response(entry=entry, stop=stop, tp1=tp1, tp2=tp2, tp3=tp3)
        proposal = _parse_response(response, inp)
        # Stop must pass through unchanged
        assert proposal.stop == pytest.approx(stop, abs=1e-8)

    def test_stop_above_half_atr_is_not_clamped(self) -> None:
        # Stop at 1.0 × ATR — well within bounds, should pass through unchanged.
        atr = 800.0
        entry = 50_000.0
        stop = entry - atr   # 49200 — 1×ATR below entry
        inp = _inp(current_price=entry, atr=atr)
        response = _mock_response(
            entry=entry, stop=stop,
            tp1=entry + 2 * atr, tp2=entry + 3 * atr, tp3=entry + 4 * atr,
        )
        proposal = _parse_response(response, inp)
        assert proposal.stop == pytest.approx(stop, abs=1e-6)


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


# ---------------------------------------------------------------------------
# _adjust_confidence_for_ma_stack
# ---------------------------------------------------------------------------

class TestAdjustConfidenceForMaStack:
    """Deterministic daily-MA confidence adjustment logic."""

    def test_no_ema200_returns_unchanged(self) -> None:
        result = _adjust_confidence_for_ma_stack(0.80, 100.0, 95.0, 90.0, None)
        assert result == pytest.approx(0.80)

    def test_price_above_all_three_no_bonus_without_full_structure(self) -> None:
        # EMA-20 < EMA-50 (mixed structure) — bullish bonus NOT applied.
        result = _adjust_confidence_for_ma_stack(0.80, 110.0, 90.0, 95.0, 85.0)
        assert result == pytest.approx(0.80)

    def test_full_bullish_stack_adds_bonus(self) -> None:
        # price (110) > EMA-20 (105) > EMA-50 (100) > EMA-200 (90)
        result = _adjust_confidence_for_ma_stack(0.80, 110.0, 105.0, 100.0, 90.0)
        assert result == pytest.approx(0.85)

    def test_price_below_ema200_reduces_by_0_10(self) -> None:
        # price below EMA-200 but above EMA-50
        result = _adjust_confidence_for_ma_stack(0.80, 85.0, 95.0, 80.0, 90.0)
        assert result == pytest.approx(0.70)

    def test_price_below_ema200_and_ema50_reduces_by_0_20(self) -> None:
        # price below both EMA-200 and EMA-50
        result = _adjust_confidence_for_ma_stack(0.80, 70.0, 95.0, 80.0, 75.0)
        assert result == pytest.approx(0.60)

    def test_result_clamped_to_zero_minimum(self) -> None:
        # Very low confidence pushed below zero by the −0.20 penalty.
        result = _adjust_confidence_for_ma_stack(0.10, 70.0, 95.0, 80.0, 75.0)
        assert result == pytest.approx(0.0)

    def test_result_clamped_to_one_maximum(self) -> None:
        # Confidence near 1.0 with bullish bonus must not exceed 1.0.
        result = _adjust_confidence_for_ma_stack(0.98, 110.0, 105.0, 100.0, 90.0)
        assert result == pytest.approx(1.0)

    def test_ema200_none_skips_all_rules_including_penalty(self) -> None:
        # Even when price is "below" all EMAs, None EMA-200 means no adjustment.
        result = _adjust_confidence_for_ma_stack(0.60, 50.0, 95.0, 80.0, None)
        assert result == pytest.approx(0.60)

    def test_parse_response_uses_adjusted_confidence(self) -> None:
        # Full bullish stack: price=110 > EMA-20=105 > EMA-50=100 > EMA-200=90.
        # Base confidence=0.80 → adjusted=0.85.
        # atr=2.0 → min_stop_dist=1.0; stop distance=2.0 is within bounds.
        c = Candidate(
            symbol="BTC", direction="LONG", ma_period=50,
            distance_to_ma_pct=0.3, regime=Regime.TREND,
            confidence=0.80, ts=_TS,
            ema20_daily=105.0, ema50_daily=100.0, ema200_daily=90.0,
        )
        inp = ProposerInput(candidate=c, current_price=110.0, atr=2.0)
        response = _mock_response(entry=110.0, stop=108.0, tp1=114.0, tp2=116.0, tp3=118.0)
        proposal = _parse_response(response, inp)
        assert proposal.confidence == pytest.approx(0.85)

    def test_parse_response_penalises_below_ema200(self) -> None:
        # price=85 < EMA-200=90 but > EMA-50=80 → −0.10 penalty.
        # atr=2.0 → min_stop_dist=1.0; stop distance=2.0 is within bounds.
        c = Candidate(
            symbol="BTC", direction="LONG", ma_period=50,
            distance_to_ma_pct=0.3, regime=Regime.TREND,
            confidence=0.80, ts=_TS,
            ema20_daily=95.0, ema50_daily=80.0, ema200_daily=90.0,
        )
        inp = ProposerInput(candidate=c, current_price=85.0, atr=2.0)
        response = _mock_response(entry=85.0, stop=83.0, tp1=89.0, tp2=91.0, tp3=93.0)
        proposal = _parse_response(response, inp)
        assert proposal.confidence == pytest.approx(0.70)

    def test_parse_response_no_adjustment_when_ema200_none(self) -> None:
        # atr=2.0 → min_stop_dist=1.0; stop distance=2.0 is within bounds.
        c = Candidate(
            symbol="BTC", direction="LONG", ma_period=50,
            distance_to_ma_pct=0.3, regime=Regime.TREND,
            confidence=0.75, ts=_TS,
            ema20_daily=95.0, ema50_daily=80.0, ema200_daily=None,
        )
        inp = ProposerInput(candidate=c, current_price=50.0, atr=2.0)
        response = _mock_response(entry=50.0, stop=48.0, tp1=54.0, tp2=56.0, tp3=58.0)
        proposal = _parse_response(response, inp)
        assert proposal.confidence == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _ma_stack_section
# ---------------------------------------------------------------------------

class TestMaStackSection:
    def _inp_with_emas(
        self,
        ema20: float | None = None,
        ema50: float | None = None,
        ema200: float | None = None,
        price: float = 100.0,
    ) -> ProposerInput:
        c = Candidate(
            symbol="BTC", direction="LONG", ma_period=50,
            distance_to_ma_pct=0.3, regime=Regime.TREND,
            confidence=0.80, ts=_TS,
            ema20_daily=ema20, ema50_daily=ema50, ema200_daily=ema200,
        )
        return ProposerInput(candidate=c, current_price=price, atr=800.0)

    def test_empty_when_all_none(self) -> None:
        assert _ma_stack_section(self._inp_with_emas()) == ""

    def test_contains_ema_values_when_set(self) -> None:
        section = _ma_stack_section(self._inp_with_emas(ema20=95.0, ema50=90.0, ema200=85.0))
        assert "EMA-20" in section
        assert "EMA-50" in section
        assert "EMA-200" in section

    def test_above_below_labels_correct(self) -> None:
        section = _ma_stack_section(self._inp_with_emas(ema200=90.0, price=100.0))
        assert "ABOVE" in section

        section_below = _ma_stack_section(self._inp_with_emas(ema200=110.0, price=100.0))
        assert "BELOW" in section_below

    def test_bullish_structure_label(self) -> None:
        section = _ma_stack_section(self._inp_with_emas(ema20=105.0, ema50=100.0, ema200=90.0, price=110.0))
        assert "BULLISH" in section

    def test_bearish_structure_label(self) -> None:
        section = _ma_stack_section(self._inp_with_emas(ema20=85.0, ema50=90.0, ema200=95.0, price=80.0))
        assert "BEARISH" in section

    def test_mixed_structure_label(self) -> None:
        # EMA-20 > EMA-200 but EMA-50 in between in wrong order → MIXED
        section = _ma_stack_section(self._inp_with_emas(ema20=100.0, ema50=110.0, ema200=90.0, price=105.0))
        assert "MIXED" in section

    def test_build_prompt_includes_ma_stack_when_emas_set(self) -> None:
        inp = self._inp_with_emas(ema20=95.0, ema50=90.0, ema200=85.0, price=100.0)
        prompt = _build_prompt(inp)
        assert "Daily MA stack" in prompt
        assert "EMA-20" in prompt
        assert "EMA-200" in prompt

    def test_build_prompt_no_ma_stack_when_all_none(self) -> None:
        inp = self._inp_with_emas()
        prompt = _build_prompt(inp)
        assert "Daily MA stack" not in prompt
