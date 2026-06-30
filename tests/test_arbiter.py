"""Tests for src/agents/arbiter.py.

No LLM calls, no live I/O outside tmp_path fixtures.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.arbiter import (
    ArbiterDecision,
    ArbiterError,
    ArbiterVerdict,
    KillLogEntry,
    _append_log,
    _apply_rules,
    _make_log_entry,
    _resolve_log_path,
    arbitrate,
)
from src.agents.critic import (
    CriticReport,
    KillCode,
    Objection,
    Severity,
    Verdict,
)
from src.agents.proposer import TradeProposal
from src.agents.regime import Regime
from src.agents.scout import Candidate

_TS = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _proposal(
    symbol: str = "BTC",
    direction: str = "LONG",
    confidence: float = 0.85,
) -> TradeProposal:
    if direction == "LONG":
        return TradeProposal(
            symbol=symbol, direction="LONG",
            entry=50_000.0, stop=49_000.0,
            tp1=51_000.0, tp2=52_000.0, tp3=53_000.0,
            position_size_usd=50_000.0, risk_usd=1_000.0,
            risk_reward=1.0, reasoning="EMA-50 bounce.",
            confidence=confidence, ts=_TS,
        )
    return TradeProposal(
        symbol=symbol, direction="SHORT",
        entry=50_000.0, stop=51_000.0,
        tp1=49_000.0, tp2=48_000.0, tp3=47_000.0,
        position_size_usd=50_000.0, risk_usd=1_000.0,
        risk_reward=1.0, reasoning="EMA-50 short bounce.",
        confidence=confidence, ts=_TS,
    )


def _objection(
    code: KillCode = KillCode.RR_INADEQUATE,
    severity: Severity = Severity.HIGH,
    reasoning: str = "test",
) -> Objection:
    return Objection(kill_code=code, severity=severity, reasoning=reasoning)


def _report(
    objections: list[Objection] | None = None,
    verdict: Verdict = Verdict.PASS,
) -> CriticReport:
    objs = objections or []
    has_high = any(o.severity == Severity.HIGH for o in objs)
    v = Verdict.KILL if has_high else Verdict.PASS
    return CriticReport(
        proposal=_proposal(),
        objections=objs,
        verdict=v,
        overall_assessment="test",
        ts=_TS,
    )


def _read_log_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# ArbiterVerdict enum
# ---------------------------------------------------------------------------

class TestArbiterVerdict:
    def test_go_and_no_go_exist(self) -> None:
        assert {v.value for v in ArbiterVerdict} == {"GO", "NO_GO"}

    def test_is_string_enum(self) -> None:
        assert isinstance(ArbiterVerdict.GO, str)

    def test_construction_from_string(self) -> None:
        assert ArbiterVerdict("NO_GO") == ArbiterVerdict.NO_GO


# ---------------------------------------------------------------------------
# _apply_rules
# ---------------------------------------------------------------------------

class TestApplyRules:
    # ---- GO cases ----

    def test_no_objections_is_go(self) -> None:
        verdict, reason = _apply_rules(_report([]))
        assert verdict == ArbiterVerdict.GO
        assert "no objections" in reason.lower()

    def test_one_low_is_go(self) -> None:
        verdict, _ = _apply_rules(_report([_objection(severity=Severity.LOW)]))
        assert verdict == ArbiterVerdict.GO

    def test_two_low_is_go(self) -> None:
        objs = [
            _objection(KillCode.THIN_LIQUIDITY, Severity.LOW),
            _objection(KillCode.CHOP_STRUCTURE, Severity.LOW),
        ]
        verdict, _ = _apply_rules(_report(objs))
        assert verdict == ArbiterVerdict.GO

    def test_one_medium_is_go(self) -> None:
        verdict, reason = _apply_rules(_report([_objection(severity=Severity.MEDIUM)]))
        assert verdict == ArbiterVerdict.GO
        assert "MEDIUM" in reason

    def test_one_medium_one_low_is_go(self) -> None:
        objs = [
            _objection(KillCode.FOMC_WINDOW, Severity.MEDIUM),
            _objection(KillCode.THIN_LIQUIDITY, Severity.LOW),
        ]
        verdict, _ = _apply_rules(_report(objs))
        assert verdict == ArbiterVerdict.GO

    # ---- NO_GO cases ----

    def test_one_high_is_no_go(self) -> None:
        verdict, reason = _apply_rules(_report([_objection(severity=Severity.HIGH)]))
        assert verdict == ArbiterVerdict.NO_GO
        assert "HIGH" in reason

    def test_two_medium_is_no_go(self) -> None:
        objs = [
            _objection(KillCode.FOMC_WINDOW, Severity.MEDIUM),
            _objection(KillCode.REGIME_MISMATCH, Severity.MEDIUM),
        ]
        verdict, reason = _apply_rules(_report(objs))
        assert verdict == ArbiterVerdict.NO_GO
        assert "MEDIUM" in reason

    def test_three_medium_is_no_go(self) -> None:
        objs = [
            _objection(KillCode.FOMC_WINDOW, Severity.MEDIUM),
            _objection(KillCode.REGIME_MISMATCH, Severity.MEDIUM),
            _objection(KillCode.CHOP_STRUCTURE, Severity.MEDIUM),
        ]
        verdict, _ = _apply_rules(_report(objs))
        assert verdict == ArbiterVerdict.NO_GO

    def test_high_beats_medium_count(self) -> None:
        """One HIGH + one MEDIUM: HIGH rule fires first, not the 2-MEDIUM rule."""
        objs = [
            _objection(KillCode.RR_INADEQUATE, Severity.HIGH),
            _objection(KillCode.FOMC_WINDOW, Severity.MEDIUM),
        ]
        verdict, reason = _apply_rules(_report(objs))
        assert verdict == ArbiterVerdict.NO_GO
        assert "HIGH" in reason

    def test_high_with_lows_is_still_no_go(self) -> None:
        objs = [
            _objection(KillCode.RR_INADEQUATE, Severity.HIGH),
            _objection(KillCode.THIN_LIQUIDITY, Severity.LOW),
        ]
        verdict, _ = _apply_rules(_report(objs))
        assert verdict == ArbiterVerdict.NO_GO

    # ---- Reason content ----

    def test_reason_contains_high_kill_code(self) -> None:
        objs = [_objection(KillCode.FUNDING_CROWDED, Severity.HIGH)]
        _, reason = _apply_rules(_report(objs))
        assert "FUNDING_CROWDED" in reason

    def test_reason_contains_medium_kill_codes(self) -> None:
        objs = [
            _objection(KillCode.FOMC_WINDOW, Severity.MEDIUM),
            _objection(KillCode.REGIME_MISMATCH, Severity.MEDIUM),
        ]
        _, reason = _apply_rules(_report(objs))
        assert "FOMC_WINDOW" in reason
        assert "REGIME_MISMATCH" in reason

    def test_reason_is_string(self) -> None:
        _, reason = _apply_rules(_report([]))
        assert isinstance(reason, str)
        assert reason


# ---------------------------------------------------------------------------
# KillLogEntry model
# ---------------------------------------------------------------------------

class TestKillLogEntry:
    def _valid(self, **overrides) -> KillLogEntry:
        base = dict(
            ts=_TS,
            symbol="BTC",
            direction="LONG",
            verdict=ArbiterVerdict.GO,
            kill_codes_fired=["RR_INADEQUATE"],
            confidence=0.85,
            reason="1 MEDIUM objection (non-blocking): RR_INADEQUATE",
        )
        base.update(overrides)
        return KillLogEntry(**base)

    def test_valid_constructs(self) -> None:
        entry = self._valid()
        assert entry.symbol == "BTC"
        assert entry.verdict == ArbiterVerdict.GO

    def test_empty_kill_codes_valid(self) -> None:
        entry = self._valid(kill_codes_fired=[])
        assert entry.kill_codes_fired == []

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(Exception):
            self._valid(confidence=1.5)

    def test_model_dump_json_is_valid_json(self) -> None:
        entry = self._valid()
        parsed = json.loads(entry.model_dump_json())
        assert parsed["symbol"] == "BTC"
        assert parsed["verdict"] == "GO"

    def test_all_fields_serialised(self) -> None:
        entry = self._valid()
        data = json.loads(entry.model_dump_json())
        for key in ("ts", "symbol", "direction", "verdict", "kill_codes_fired", "confidence", "reason"):
            assert key in data, f"'{key}' missing from serialised entry"


# ---------------------------------------------------------------------------
# _append_log
# ---------------------------------------------------------------------------

class TestAppendLog:
    def _entry(self, symbol: str = "BTC", codes: list[str] | None = None) -> KillLogEntry:
        return KillLogEntry(
            ts=_TS,
            symbol=symbol,
            direction="LONG",
            verdict=ArbiterVerdict.GO,
            kill_codes_fired=codes or [],
            confidence=0.85,
            reason="no objections — clean proposal",
        )

    def test_creates_file_when_absent(self, tmp_path: Path) -> None:
        log = tmp_path / "test.jsonl"
        assert not log.exists()
        _append_log(self._entry(), log)
        assert log.exists()

    def test_written_line_is_valid_json(self, tmp_path: Path) -> None:
        log = tmp_path / "test.jsonl"
        _append_log(self._entry(), log)
        lines = _read_log_lines(log)
        assert len(lines) == 1
        assert lines[0]["symbol"] == "BTC"

    def test_second_write_appends_not_overwrites(self, tmp_path: Path) -> None:
        log = tmp_path / "test.jsonl"
        _append_log(self._entry("BTC"), log)
        _append_log(self._entry("ETH"), log)
        lines = _read_log_lines(log)
        assert len(lines) == 2
        symbols = {l["symbol"] for l in lines}
        assert symbols == {"BTC", "ETH"}

    def test_each_line_independently_parseable(self, tmp_path: Path) -> None:
        log = tmp_path / "test.jsonl"
        for sym in ["BTC", "ETH", "SOL"]:
            _append_log(self._entry(sym), log)
        raw_lines = log.read_text().splitlines()
        assert len(raw_lines) == 3
        for line in raw_lines:
            json.loads(line)  # must not raise

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = tmp_path / "nested" / "dir" / "kill_log.jsonl"
        _append_log(self._entry(), log)
        assert log.exists()

    def test_kill_codes_preserved_in_file(self, tmp_path: Path) -> None:
        log = tmp_path / "test.jsonl"
        _append_log(self._entry(codes=["FOMC_WINDOW", "THIN_LIQUIDITY"]), log)
        lines = _read_log_lines(log)
        assert set(lines[0]["kill_codes_fired"]) == {"FOMC_WINDOW", "THIN_LIQUIDITY"}


# ---------------------------------------------------------------------------
# _make_log_entry
# ---------------------------------------------------------------------------

class TestMakeLogEntry:
    def _go_decision(
        self,
        codes: list[KillCode] | None = None,
        confidence: float = 0.85,
        symbol: str = "BTC",
    ) -> ArbiterDecision:
        p = _proposal(symbol=symbol, confidence=confidence)
        r = _report()
        return ArbiterDecision(
            proposal=p,
            critic_report=r,
            verdict=ArbiterVerdict.GO,
            reason="no objections — clean proposal",
            kill_codes_fired=codes or [],
            ts=_TS,
        )

    def test_symbol_copied(self) -> None:
        entry = _make_log_entry(self._go_decision(symbol="ETH"))
        assert entry.symbol == "ETH"

    def test_direction_copied(self) -> None:
        entry = _make_log_entry(self._go_decision())
        assert entry.direction == "LONG"

    def test_confidence_copied(self) -> None:
        entry = _make_log_entry(self._go_decision(confidence=0.72))
        assert entry.confidence == pytest.approx(0.72)

    def test_verdict_is_go(self) -> None:
        entry = _make_log_entry(self._go_decision())
        assert entry.verdict == ArbiterVerdict.GO

    def test_kill_codes_stringified(self) -> None:
        codes = [KillCode.FOMC_WINDOW, KillCode.THIN_LIQUIDITY]
        entry = _make_log_entry(self._go_decision(codes=codes))
        assert set(entry.kill_codes_fired) == {"FOMC_WINDOW", "THIN_LIQUIDITY"}

    def test_empty_codes_preserved(self) -> None:
        entry = _make_log_entry(self._go_decision(codes=[]))
        assert entry.kill_codes_fired == []


# ---------------------------------------------------------------------------
# arbitrate — verdict logic
# ---------------------------------------------------------------------------

class TestArbitrateVerdict:
    def test_clean_report_is_go(self, tmp_path: Path) -> None:
        result = arbitrate(_proposal(), _report([]), log_path=tmp_path / "k.jsonl")
        assert result.verdict == ArbiterVerdict.GO

    def test_one_high_is_no_go(self, tmp_path: Path) -> None:
        objs = [_objection(KillCode.RR_INADEQUATE, Severity.HIGH)]
        result = arbitrate(_proposal(), _report(objs), log_path=tmp_path / "k.jsonl")
        assert result.verdict == ArbiterVerdict.NO_GO

    def test_two_medium_is_no_go(self, tmp_path: Path) -> None:
        objs = [
            _objection(KillCode.FOMC_WINDOW, Severity.MEDIUM),
            _objection(KillCode.REGIME_MISMATCH, Severity.MEDIUM),
        ]
        result = arbitrate(_proposal(), _report(objs), log_path=tmp_path / "k.jsonl")
        assert result.verdict == ArbiterVerdict.NO_GO

    def test_one_medium_is_go(self, tmp_path: Path) -> None:
        objs = [_objection(KillCode.FOMC_WINDOW, Severity.MEDIUM)]
        result = arbitrate(_proposal(), _report(objs), log_path=tmp_path / "k.jsonl")
        assert result.verdict == ArbiterVerdict.GO

    def test_two_low_is_go(self, tmp_path: Path) -> None:
        objs = [
            _objection(KillCode.THIN_LIQUIDITY, Severity.LOW),
            _objection(KillCode.CHOP_STRUCTURE, Severity.LOW),
        ]
        result = arbitrate(_proposal(), _report(objs), log_path=tmp_path / "k.jsonl")
        assert result.verdict == ArbiterVerdict.GO


# ---------------------------------------------------------------------------
# arbitrate — KILL log behaviour
# ---------------------------------------------------------------------------

class TestArbitrateKillLog:
    def test_go_writes_to_log(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        arbitrate(_proposal(), _report([]), log_path=log)
        assert log.exists()
        lines = _read_log_lines(log)
        assert len(lines) == 1

    def test_no_go_does_not_write_to_log(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        objs = [_objection(KillCode.RR_INADEQUATE, Severity.HIGH)]
        arbitrate(_proposal(), _report(objs), log_path=log)
        assert not log.exists()

    def test_multiple_go_calls_append(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        for _ in range(3):
            arbitrate(_proposal(), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert len(lines) == 3

    def test_go_after_no_go_appends_one_line(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        # NO_GO — nothing written
        objs_high = [_objection(KillCode.RR_INADEQUATE, Severity.HIGH)]
        arbitrate(_proposal(), _report(objs_high), log_path=log)
        # GO — one line written
        arbitrate(_proposal(), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert len(lines) == 1

    def test_log_entry_has_correct_symbol(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        arbitrate(_proposal(symbol="ETH"), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert lines[0]["symbol"] == "ETH"

    def test_log_entry_has_correct_direction(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        arbitrate(_proposal(direction="SHORT"), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert lines[0]["direction"] == "SHORT"

    def test_log_entry_verdict_is_go(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        arbitrate(_proposal(), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert lines[0]["verdict"] == "GO"

    def test_log_entry_confidence_matches_proposal(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        arbitrate(_proposal(confidence=0.73), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert lines[0]["confidence"] == pytest.approx(0.73)

    def test_log_entry_kill_codes_includes_medium_on_go(self, tmp_path: Path) -> None:
        """A MEDIUM code that didn't veto the trade still appears in the log."""
        log = tmp_path / "kill_log.jsonl"
        objs = [_objection(KillCode.FOMC_WINDOW, Severity.MEDIUM)]
        arbitrate(_proposal(), _report(objs), log_path=log)
        lines = _read_log_lines(log)
        assert "FOMC_WINDOW" in lines[0]["kill_codes_fired"]

    def test_log_entry_kill_codes_empty_on_clean_go(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        arbitrate(_proposal(), _report([]), log_path=log)
        lines = _read_log_lines(log)
        assert lines[0]["kill_codes_fired"] == []

    def test_log_is_valid_jsonl(self, tmp_path: Path) -> None:
        """Every line in the file must be independently parseable JSON."""
        log = tmp_path / "kill_log.jsonl"
        for sym in ["BTC", "ETH", "SOL"]:
            arbitrate(_proposal(symbol=sym), _report([]), log_path=log)
        for line in log.read_text().splitlines():
            assert line.strip()
            json.loads(line)

    def test_log_never_overwritten(self, tmp_path: Path) -> None:
        """Pre-existing log content must be preserved across calls."""
        log = tmp_path / "kill_log.jsonl"
        log.write_text('{"existing": true}\n')
        arbitrate(_proposal(), _report([]), log_path=log)
        raw = log.read_text().splitlines()
        assert raw[0] == '{"existing": true}'
        assert len(raw) == 2

    def test_env_var_log_path_respected(self, tmp_path: Path, monkeypatch) -> None:
        log = tmp_path / "env_log.jsonl"
        monkeypatch.setenv("FAST_AGENT_KILL_LOG", str(log))
        arbitrate(_proposal(), _report([]), log_path=None)
        assert log.exists()

    def test_os_error_on_log_write_raises_arbiter_error(self, tmp_path: Path) -> None:
        log = tmp_path / "kill_log.jsonl"
        log.mkdir()  # make it a directory so open() fails
        with pytest.raises(ArbiterError, match="Failed to write KILL log"):
            arbitrate(_proposal(), _report([]), log_path=log)


# ---------------------------------------------------------------------------
# arbitrate — decision fields
# ---------------------------------------------------------------------------

class TestArbitrateDecisionFields:
    def test_proposal_passed_through(self, tmp_path: Path) -> None:
        p = _proposal(symbol="SOL")
        result = arbitrate(p, _report([]), log_path=tmp_path / "k.jsonl")
        assert result.proposal.symbol == "SOL"

    def test_report_passed_through(self, tmp_path: Path) -> None:
        r = _report([_objection(KillCode.FOMC_WINDOW, Severity.LOW)])
        result = arbitrate(_proposal(), r, log_path=tmp_path / "k.jsonl")
        assert len(result.critic_report.objections) == 1

    def test_kill_codes_fired_matches_all_objection_codes(self, tmp_path: Path) -> None:
        objs = [
            _objection(KillCode.FOMC_WINDOW, Severity.LOW),
            _objection(KillCode.THIN_LIQUIDITY, Severity.LOW),
        ]
        result = arbitrate(_proposal(), _report(objs), log_path=tmp_path / "k.jsonl")
        assert set(result.kill_codes_fired) == {KillCode.FOMC_WINDOW, KillCode.THIN_LIQUIDITY}

    def test_kill_codes_empty_for_clean_report(self, tmp_path: Path) -> None:
        result = arbitrate(_proposal(), _report([]), log_path=tmp_path / "k.jsonl")
        assert result.kill_codes_fired == []

    def test_ts_is_utc(self, tmp_path: Path) -> None:
        result = arbitrate(_proposal(), _report([]), log_path=tmp_path / "k.jsonl")
        assert result.ts.tzinfo is not None
        assert result.ts.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_reason_is_non_empty_string(self, tmp_path: Path) -> None:
        result = arbitrate(_proposal(), _report([]), log_path=tmp_path / "k.jsonl")
        assert isinstance(result.reason, str)
        assert result.reason

    def test_returns_arbiter_decision_type(self, tmp_path: Path) -> None:
        result = arbitrate(_proposal(), _report([]), log_path=tmp_path / "k.jsonl")
        assert isinstance(result, ArbiterDecision)
