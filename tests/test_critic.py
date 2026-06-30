"""Tests for src/agents/critic.py.

All tests mock the Anthropic client — no live API calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from src.agents.critic import (
    MODEL,
    CriticError,
    CriticInput,
    CriticReport,
    KillCode,
    Objection,
    Severity,
    Verdict,
    _build_prompt,
    _parse_response,
    critique,
)
from src.agents.proposer import TradeProposal
from src.agents.regime import Regime
from src.agents.scout import Candidate

_TS = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _candidate() -> Candidate:
    return Candidate(
        symbol="BTC",
        direction="LONG",
        ma_period=50,
        distance_to_ma_pct=0.3,
        regime=Regime.TREND,
        confidence=0.85,
        ts=_TS,
    )


def _proposal(
    entry: float = 50_000.0,
    stop: float = 49_000.0,
    tp1: float = 51_000.0,
    tp2: float = 52_000.0,
    tp3: float = 53_000.0,
    risk_reward: float = 1.0,
    reasoning: str = "EMA-50 pullback with strong ADX.",
) -> TradeProposal:
    return TradeProposal(
        symbol="BTC",
        direction="LONG",
        entry=entry,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        position_size_usd=50_000.0,
        risk_usd=1_000.0,
        risk_reward=risk_reward,
        reasoning=reasoning,
        confidence=0.85,
        ts=_TS,
    )


def _inp(
    proposal: TradeProposal | None = None,
    funding_rate: float | None = None,
    book_bid_depth_usd: float | None = None,
    book_ask_depth_usd: float | None = None,
    fomc_hours_until: float | None = None,
    min_rr: float = 1.5,
) -> CriticInput:
    return CriticInput(
        proposal=proposal or _proposal(),
        funding_rate=funding_rate,
        book_bid_depth_usd=book_bid_depth_usd,
        book_ask_depth_usd=book_ask_depth_usd,
        fomc_hours_until=fomc_hours_until,
        min_rr=min_rr,
    )


def _mock_response(
    objections: list[dict[str, str]] | None = None,
    overall_assessment: str = "No major concerns.",
    stop_reason: str = "tool_use",
) -> MagicMock:
    """Build a mock Anthropic Message with a submit_critic_report tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_critic_report"
    tool_block.input = {
        "objections": objections if objections is not None else [],
        "overall_assessment": overall_assessment,
    }
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [tool_block]
    return response


def _mock_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _objection_dict(
    kill_code: str = "RR_INADEQUATE",
    severity: str = "HIGH",
    reasoning: str = "R:R is below threshold.",
) -> dict[str, str]:
    return {"kill_code": kill_code, "severity": severity, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------

class TestKillCode:
    def test_all_seven_codes_exist(self) -> None:
        expected = {
            "THIN_LIQUIDITY", "FUNDING_CROWDED", "BOOK_IMBALANCE_AGAINST",
            "REGIME_MISMATCH", "RR_INADEQUATE", "FOMC_WINDOW", "CHOP_STRUCTURE",
        }
        assert {k.value for k in KillCode} == expected

    def test_is_string_enum(self) -> None:
        assert isinstance(KillCode.THIN_LIQUIDITY, str)

    def test_construction_from_string(self) -> None:
        assert KillCode("FOMC_WINDOW") == KillCode.FOMC_WINDOW


class TestSeverity:
    def test_three_levels(self) -> None:
        assert {s.value for s in Severity} == {"HIGH", "MEDIUM", "LOW"}

    def test_construction_from_string(self) -> None:
        assert Severity("HIGH") == Severity.HIGH


class TestVerdict:
    def test_two_outcomes(self) -> None:
        assert {v.value for v in Verdict} == {"PASS", "KILL"}


# ---------------------------------------------------------------------------
# Objection model
# ---------------------------------------------------------------------------

class TestObjection:
    def test_valid_high_objection(self) -> None:
        obj = Objection(
            kill_code=KillCode.RR_INADEQUATE,
            severity=Severity.HIGH,
            reasoning="R:R is 0.8.",
        )
        assert obj.kill_code == KillCode.RR_INADEQUATE
        assert obj.severity == Severity.HIGH

    def test_invalid_kill_code_raises(self) -> None:
        with pytest.raises(Exception):
            Objection(kill_code="BAD_CODE", severity=Severity.LOW, reasoning="x")  # type: ignore[arg-type]

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(Exception):
            Objection(kill_code=KillCode.FOMC_WINDOW, severity="CRITICAL", reasoning="x")  # type: ignore[arg-type]

    def test_reasoning_preserved(self) -> None:
        obj = Objection(
            kill_code=KillCode.FOMC_WINDOW,
            severity=Severity.MEDIUM,
            reasoning="FOMC in 18h.",
        )
        assert obj.reasoning == "FOMC in 18h."


# ---------------------------------------------------------------------------
# CriticInput validation
# ---------------------------------------------------------------------------

class TestCriticInput:
    def test_minimal_input_constructs(self) -> None:
        inp = _inp()
        assert inp.min_rr == 1.5
        assert inp.funding_rate is None
        assert inp.fomc_hours_until is None

    def test_min_rr_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            _inp(min_rr=0.0)

    def test_full_context_constructs(self) -> None:
        inp = _inp(
            funding_rate=0.0003,
            book_bid_depth_usd=1_000_000.0,
            book_ask_depth_usd=500_000.0,
            fomc_hours_until=12.0,
        )
        assert inp.funding_rate == pytest.approx(0.0003)
        assert inp.fomc_hours_until == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# CriticReport verdict logic
# ---------------------------------------------------------------------------

class TestCriticReport:
    def _make_report(self, objections: list[Objection], verdict: Verdict) -> CriticReport:
        return CriticReport(
            proposal=_proposal(),
            objections=objections,
            verdict=verdict,
            overall_assessment="test",
            ts=_TS,
        )

    def test_pass_with_no_objections(self) -> None:
        report = self._make_report([], Verdict.PASS)
        assert report.verdict == Verdict.PASS

    def test_kill_with_high_objection(self) -> None:
        obj = Objection(kill_code=KillCode.RR_INADEQUATE, severity=Severity.HIGH, reasoning="low rr")
        report = self._make_report([obj], Verdict.KILL)
        assert report.verdict == Verdict.KILL

    def test_pass_with_only_medium_objections(self) -> None:
        obj = Objection(kill_code=KillCode.FOMC_WINDOW, severity=Severity.MEDIUM, reasoning="caution")
        report = self._make_report([obj], Verdict.PASS)
        assert report.verdict == Verdict.PASS

    def test_pass_with_only_low_objections(self) -> None:
        obj = Objection(kill_code=KillCode.THIN_LIQUIDITY, severity=Severity.LOW, reasoning="minor")
        report = self._make_report([obj], Verdict.PASS)
        assert report.verdict == Verdict.PASS

    def test_ts_from_proposal(self) -> None:
        report = self._make_report([], Verdict.PASS)
        assert report.ts == _TS

    def test_proposal_passed_through(self) -> None:
        p = _proposal()
        report = self._make_report([], Verdict.PASS)
        assert report.proposal.symbol == p.symbol


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_symbol(self) -> None:
        assert "BTC" in _build_prompt(_inp())

    def test_contains_direction(self) -> None:
        assert "LONG" in _build_prompt(_inp())

    def test_contains_entry_price(self) -> None:
        assert "50000" in _build_prompt(_inp())

    def test_contains_rr(self) -> None:
        prompt = _build_prompt(_inp())
        assert "R:R" in prompt or "risk_reward" in prompt.lower() or "1.00" in prompt

    def test_contains_min_rr(self) -> None:
        assert "1.5" in _build_prompt(_inp(min_rr=1.5))

    def test_contains_all_kill_codes(self) -> None:
        prompt = _build_prompt(_inp())
        for code in KillCode:
            assert code.value in prompt, f"{code.value} missing from prompt"

    def test_contains_funding_rate_when_provided(self) -> None:
        prompt = _build_prompt(_inp(funding_rate=0.0003))
        assert "0.0003" in prompt

    def test_unknown_funding_when_none(self) -> None:
        prompt = _build_prompt(_inp(funding_rate=None))
        assert "unknown" in prompt

    def test_contains_fomc_hours_when_provided(self) -> None:
        prompt = _build_prompt(_inp(fomc_hours_until=12.0))
        assert "12.0" in prompt

    def test_fomc_unknown_when_none(self) -> None:
        prompt = _build_prompt(_inp(fomc_hours_until=None))
        assert "unknown" in prompt

    def test_orderbook_shown_when_both_depths_provided(self) -> None:
        prompt = _build_prompt(_inp(book_bid_depth_usd=1_000_000.0, book_ask_depth_usd=500_000.0))
        assert "1,000,000" in prompt or "1000000" in prompt

    def test_mentions_submit_tool(self) -> None:
        assert "submit_critic_report" in _build_prompt(_inp())

    def test_long_direction_in_prompt(self) -> None:
        prompt = _build_prompt(_inp())
        assert "LONG" in prompt


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_empty_objections_gives_pass(self) -> None:
        report = _parse_response(_mock_response(objections=[]), _inp())
        assert report.verdict == Verdict.PASS
        assert report.objections == []

    def test_high_objection_gives_kill(self) -> None:
        raw = [_objection_dict("RR_INADEQUATE", "HIGH", "R:R too low.")]
        report = _parse_response(_mock_response(objections=raw), _inp())
        assert report.verdict == Verdict.KILL

    def test_medium_only_gives_pass(self) -> None:
        raw = [_objection_dict("FOMC_WINDOW", "MEDIUM", "FOMC soon.")]
        report = _parse_response(_mock_response(objections=raw), _inp())
        assert report.verdict == Verdict.PASS

    def test_low_only_gives_pass(self) -> None:
        raw = [_objection_dict("THIN_LIQUIDITY", "LOW", "Minor depth issue.")]
        report = _parse_response(_mock_response(objections=raw), _inp())
        assert report.verdict == Verdict.PASS

    def test_mixed_high_medium_gives_kill(self) -> None:
        raw = [
            _objection_dict("RR_INADEQUATE", "HIGH", "R:R 0.8."),
            _objection_dict("FOMC_WINDOW", "MEDIUM", "FOMC in 20h."),
        ]
        report = _parse_response(_mock_response(objections=raw), _inp())
        assert report.verdict == Verdict.KILL
        assert len(report.objections) == 2

    def test_all_seven_kill_codes_accepted(self) -> None:
        for code in KillCode:
            raw = [_objection_dict(code.value, "LOW", "test")]
            report = _parse_response(_mock_response(objections=raw), _inp())
            assert report.objections[0].kill_code == code

    def test_all_three_severities_accepted(self) -> None:
        for sev in Severity:
            raw = [_objection_dict("CHOP_STRUCTURE", sev.value, "test")]
            report = _parse_response(_mock_response(objections=raw), _inp())
            assert report.objections[0].severity == sev

    def test_invalid_kill_code_raises_critic_error(self) -> None:
        raw = [_objection_dict("UNKNOWN_CODE", "HIGH", "bad code")]
        with pytest.raises(CriticError, match="Invalid objection"):
            _parse_response(_mock_response(objections=raw), _inp())

    def test_invalid_severity_raises_critic_error(self) -> None:
        raw = [_objection_dict("FOMC_WINDOW", "CRITICAL", "bad severity")]
        with pytest.raises(CriticError, match="Invalid objection"):
            _parse_response(_mock_response(objections=raw), _inp())

    def test_missing_tool_call_raises_critic_error(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [text_block]
        with pytest.raises(CriticError, match="submit_critic_report"):
            _parse_response(response, _inp())

    def test_wrong_tool_name_raises_critic_error(self) -> None:
        wrong = MagicMock()
        wrong.type = "tool_use"
        wrong.name = "some_other_tool"
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [wrong]
        with pytest.raises(CriticError):
            _parse_response(response, _inp())

    def test_non_list_objections_raises_critic_error(self) -> None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_critic_report"
        tool_block.input = {"objections": "not a list", "overall_assessment": "x"}
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [tool_block]
        with pytest.raises(CriticError, match="list"):
            _parse_response(response, _inp())

    def test_non_dict_objection_element_raises_critic_error(self) -> None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_critic_report"
        tool_block.input = {"objections": ["not_a_dict"], "overall_assessment": "x"}
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [tool_block]
        with pytest.raises(CriticError, match="not a dict"):
            _parse_response(response, _inp())

    def test_overall_assessment_preserved(self) -> None:
        report = _parse_response(
            _mock_response(overall_assessment="Clean setup with adequate R:R."), _inp()
        )
        assert report.overall_assessment == "Clean setup with adequate R:R."

    def test_proposal_passed_through_unchanged(self) -> None:
        p = _proposal()
        inp = _inp(proposal=p)
        report = _parse_response(_mock_response(), inp)
        assert report.proposal.entry == p.entry
        assert report.proposal.symbol == p.symbol

    def test_ts_from_proposal(self) -> None:
        report = _parse_response(_mock_response(), _inp())
        assert report.ts == _TS

    def test_multiple_objections_all_present(self) -> None:
        raw = [
            _objection_dict("RR_INADEQUATE", "HIGH", "low rr"),
            _objection_dict("FOMC_WINDOW", "MEDIUM", "fomc close"),
            _objection_dict("THIN_LIQUIDITY", "LOW", "shallow book"),
        ]
        report = _parse_response(_mock_response(objections=raw), _inp())
        assert len(report.objections) == 3
        codes = {obj.kill_code for obj in report.objections}
        assert KillCode.RR_INADEQUATE in codes
        assert KillCode.FOMC_WINDOW in codes
        assert KillCode.THIN_LIQUIDITY in codes


# ---------------------------------------------------------------------------
# critique (integration — client fully mocked)
# ---------------------------------------------------------------------------

class TestCritique:
    def test_returns_critic_report(self) -> None:
        client = _mock_client(_mock_response())
        result = critique(_inp(), client=client)
        assert isinstance(result, CriticReport)

    def test_calls_correct_model(self) -> None:
        client = _mock_client(_mock_response())
        critique(_inp(), client=client)
        assert client.messages.create.call_args.kwargs["model"] == MODEL

    def test_tool_choice_forces_submit_critic_report(self) -> None:
        client = _mock_client(_mock_response())
        critique(_inp(), client=client)
        tc = client.messages.create.call_args.kwargs["tool_choice"]
        assert tc["type"] == "tool"
        assert tc["name"] == "submit_critic_report"

    def test_messages_has_user_role(self) -> None:
        client = _mock_client(_mock_response())
        critique(_inp(), client=client)
        messages = client.messages.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"

    def test_api_error_raises_critic_error(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIStatusError(
            "rate limit",
            response=MagicMock(status_code=429),
            body={},
        )
        with pytest.raises(CriticError, match="API error"):
            critique(_inp(), client=client)

    def test_no_client_no_key_raises_critic_error(self) -> None:
        import os
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(CriticError, match="ANTHROPIC_API_KEY"):
                critique(_inp(), client=None)

    def test_clean_proposal_returns_pass(self) -> None:
        client = _mock_client(_mock_response(objections=[]))
        result = critique(_inp(), client=client)
        assert result.verdict == Verdict.PASS

    def test_high_objection_returns_kill(self) -> None:
        raw = [_objection_dict("REGIME_MISMATCH", "HIGH", "Structure is choppy.")]
        client = _mock_client(_mock_response(objections=raw))
        result = critique(_inp(), client=client)
        assert result.verdict == Verdict.KILL

    def test_long_proposal_critique(self) -> None:
        client = _mock_client(_mock_response(objections=[]))
        result = critique(_inp(proposal=_proposal()), client=client)
        assert result.proposal.direction == "LONG"
        assert result.verdict == Verdict.PASS
