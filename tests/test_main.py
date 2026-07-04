"""Tests for src/main.py — pipeline wiring, helper functions, and CLI."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import pandas as pd
import pytest

from src.main import (
    _compute_atr,
    _latest_funding,
    _print_summary,
    _run_ingest,
    _run_replay,
    main,
)
from src.agents.arbiter import ArbiterVerdict
from src.agents.critic import CriticReport, KillCode, Objection, Severity, Verdict
from src.agents.proposer import TradeProposal
from src.agents.regime import Regime
from src.agents.scout import Candidate

_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
_TS = _START


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, base_close: float = 100.0) -> pd.DataFrame:
    """Generate n hourly OHLCV bars starting at _START."""
    rows = []
    for i in range(n):
        open_t = _START + timedelta(hours=i)
        close_t = open_t + timedelta(hours=1) - timedelta(milliseconds=1)
        c = base_close + i * 0.1
        rows.append(
            dict(open_time=open_t, close_time=close_t, open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1000.0)
        )
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _setup_data(tmp_path: Path, symbol: str = "BTC", n_bars: int = 20) -> Path:
    """Write minimal parquet files; return the data_dir."""
    data_dir = tmp_path / "data_store"
    _write_parquet(_make_ohlcv(n_bars), data_dir / symbol / "1h.parquet")
    return data_dir


def _make_candidate(symbol: str = "BTC", direction: str = "LONG") -> Candidate:
    return Candidate(
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        ma_period=50,
        distance_to_ma_pct=0.3,
        regime=Regime.TREND,
        confidence=0.85,
        ts=_TS,
    )


def _make_proposal(symbol: str = "BTC", direction: str = "LONG") -> TradeProposal:
    if direction == "LONG":
        return TradeProposal(
            symbol=symbol, direction="LONG",
            entry=100.0, stop=98.0, tp1=102.0, tp2=104.0, tp3=106.0,
            position_size_usd=50_000.0, risk_usd=1_000.0, risk_reward=1.0,
            reasoning="Test reasoning.", confidence=0.85, ts=_TS,
        )
    return TradeProposal(
        symbol=symbol, direction="SHORT",
        entry=100.0, stop=102.0, tp1=98.0, tp2=96.0, tp3=94.0,
        position_size_usd=50_000.0, risk_usd=1_000.0, risk_reward=1.0,
        reasoning="Test reasoning.", confidence=0.75, ts=_TS,
    )


def _make_clean_report(proposal: TradeProposal) -> CriticReport:
    return CriticReport(
        proposal=proposal, objections=[], verdict=Verdict.PASS,
        overall_assessment="Clean.", ts=_TS,
    )


def _make_objection_report(
    proposal: TradeProposal,
    kill_code: KillCode,
    severity: Severity,
) -> CriticReport:
    return CriticReport(
        proposal=proposal,
        objections=[Objection(kill_code=kill_code, severity=severity, reasoning="Test.")],
        verdict=Verdict.KILL if severity == Severity.HIGH else Verdict.PASS,
        overall_assessment="Has objection.", ts=_TS,
    )


def _mock_client() -> MagicMock:
    return MagicMock(spec=anthropic.Anthropic)


def _scan_once(candidate: Candidate):
    """Return a scan side-effect that emits candidate once then []."""
    calls: list[int] = []

    def _inner(*args, **kwargs):
        calls.append(1)
        return [candidate] if len(calls) == 1 else []

    return _inner


# ---------------------------------------------------------------------------
# _compute_atr
# ---------------------------------------------------------------------------

class TestComputeAtr:
    def _df(self, n: int, spread: float = 10.0, close: float = 100.0) -> pd.DataFrame:
        return pd.DataFrame({
            "high": [close + spread / 2] * n,
            "low": [close - spread / 2] * n,
            "close": [close] * n,
        })

    def test_returns_positive_float(self) -> None:
        assert _compute_atr(self._df(20)) > 0

    def test_fallback_for_too_few_rows(self) -> None:
        df = self._df(5, close=200.0)
        # 5 < 14+1 → 2% of last close = 4.0
        assert _compute_atr(df, period=14) == pytest.approx(4.0)

    def test_exactly_period_plus_one_uses_formula(self) -> None:
        df = self._df(15, spread=10.0, close=100.0)
        result = _compute_atr(df, period=14)
        assert result > 0
        assert result != pytest.approx(100.0 * 0.02)  # not the fallback

    def test_constant_candles_give_atr_equal_to_spread(self) -> None:
        # high-low = 10 every bar, prev-close extension = 0 → TR = 10
        df = self._df(30, spread=10.0)
        assert _compute_atr(df, period=14) == pytest.approx(10.0, rel=1e-3)

    def test_larger_spread_gives_larger_atr(self) -> None:
        small = self._df(30, spread=2.0)
        large = self._df(30, spread=20.0)
        assert _compute_atr(large) > _compute_atr(small)


# ---------------------------------------------------------------------------
# _latest_funding
# ---------------------------------------------------------------------------

class TestLatestFunding:
    def test_returns_last_rate(self) -> None:
        pit = MagicMock()
        pit.funding.return_value = pd.DataFrame({
            "ts": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "rate": [0.0001, 0.0003],
        })
        assert _latest_funding(pit, "BTC") == pytest.approx(0.0003)

    def test_empty_dataframe_returns_none(self) -> None:
        pit = MagicMock()
        pit.funding.return_value = pd.DataFrame(columns=["ts", "rate"])
        assert _latest_funding(pit, "BTC") is None

    def test_exception_returns_none(self) -> None:
        pit = MagicMock()
        pit.funding.side_effect = RuntimeError("no data")
        assert _latest_funding(pit, "BTC") is None


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def _stats(
        self,
        go: int = 5,
        no_go: int = 3,
        kill_codes: dict | None = None,
        proposer_errors: int = 0,
        critic_errors: int = 0,
    ) -> dict:
        return {
            "bars": 100,
            "candidates": go + no_go,
            "go": go,
            "no_go": no_go,
            "proposer_errors": proposer_errors,
            "critic_errors": critic_errors,
            "kill_codes": Counter(kill_codes or {}),
        }

    def test_prints_bar_count(self, capsys) -> None:
        _print_summary(self._stats(), Path("kill.jsonl"))
        assert "100" in capsys.readouterr().out

    def test_prints_go_count(self, capsys) -> None:
        _print_summary(self._stats(go=7, no_go=1), Path("kill.jsonl"))
        assert "7" in capsys.readouterr().out

    def test_prints_kill_codes(self, capsys) -> None:
        _print_summary(
            self._stats(kill_codes={"RR_INADEQUATE": 4, "REGIME_MISMATCH": 2}),
            Path("kill.jsonl"),
        )
        out = capsys.readouterr().out
        assert "RR_INADEQUATE" in out
        assert "REGIME_MISMATCH" in out

    def test_no_kill_codes_prints_message(self, capsys) -> None:
        _print_summary(self._stats(kill_codes={}), Path("kill.jsonl"))
        assert "No kill codes" in capsys.readouterr().out

    def test_zero_decisions_no_division_error(self, capsys) -> None:
        stats = self._stats(go=0, no_go=0)
        stats["candidates"] = 0
        _print_summary(stats, Path("kill.jsonl"))
        assert "0.0%" in capsys.readouterr().out

    def test_prints_kill_log_path(self, capsys) -> None:
        _print_summary(self._stats(), Path("/custom/path/kill.jsonl"))
        assert "/custom/path/kill.jsonl" in capsys.readouterr().out

    def test_proposer_errors_shown_when_nonzero(self, capsys) -> None:
        _print_summary(self._stats(proposer_errors=3), Path("kill.jsonl"))
        assert "Proposer errors" in capsys.readouterr().out

    def test_proposer_errors_hidden_when_zero(self, capsys) -> None:
        _print_summary(self._stats(proposer_errors=0), Path("kill.jsonl"))
        assert "Proposer errors" not in capsys.readouterr().out

    def test_critic_errors_shown_when_nonzero(self, capsys) -> None:
        _print_summary(self._stats(critic_errors=2), Path("kill.jsonl"))
        assert "Critic errors" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _run_ingest
# ---------------------------------------------------------------------------

class TestRunIngest:
    def test_calls_ingest_for_each_symbol_and_both_timeframes(self, tmp_path: Path) -> None:
        with patch("src.main.ingest_symbol") as mock_ingest:
            _run_ingest(["BTC", "ETH"], days=30, data_dir=tmp_path)

        calls = [(c.args[0], c.args[1]) for c in mock_ingest.call_args_list]
        assert ("BTC", "1h") in calls
        assert ("BTC", "1d") in calls
        assert ("ETH", "1h") in calls
        assert ("ETH", "1d") in calls
        assert mock_ingest.call_count == 4


# ---------------------------------------------------------------------------
# _run_replay (pipeline smoke tests — all LLM calls mocked)
# ---------------------------------------------------------------------------

class TestRunReplay:
    def test_no_candidates_returns_zero_counts(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        with patch("src.main.scan", return_value=[]):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["candidates"] == 0
        assert stats["go"] == 0
        assert stats["no_go"] == 0
        assert stats["bars"] > 0

    def test_go_decision_increments_go(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        candidate = _make_candidate()
        proposal = _make_proposal()
        report = _make_clean_report(proposal)

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["candidates"] == 1
        assert stats["go"] == 1
        assert stats["no_go"] == 0

    def test_go_decision_written_to_kill_log(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        kill_log = tmp_path / "kill.jsonl"
        candidate = _make_candidate()
        proposal = _make_proposal()
        report = _make_clean_report(proposal)

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=kill_log,
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )

        assert kill_log.exists()
        lines = kill_log.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["verdict"] == "GO"
        assert entry["symbol"] == "BTC"

    def test_no_go_not_written_to_kill_log(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        kill_log = tmp_path / "kill.jsonl"
        candidate = _make_candidate()
        proposal = _make_proposal()
        report = _make_objection_report(proposal, KillCode.RR_INADEQUATE, Severity.HIGH)

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=kill_log,
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )

        assert stats["no_go"] == 1
        assert stats["go"] == 0
        assert not kill_log.exists()

    def test_max_candidates_caps_processing(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path, n_bars=40)
        candidate = _make_candidate()
        proposal = _make_proposal()
        report = _make_clean_report(proposal)

        # scan always returns 3 candidates — without cap we'd process many
        with patch("src.main.scan", return_value=[candidate, candidate, candidate]), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=2, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["candidates"] == 2

    def test_medium_kill_code_tracked_and_still_go(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        candidate = _make_candidate()
        proposal = _make_proposal()
        # 1 MEDIUM objection → GO, but kill code recorded
        report = _make_objection_report(proposal, KillCode.FUNDING_CROWDED, Severity.MEDIUM)

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["go"] == 1
        assert stats["kill_codes"]["FUNDING_CROWDED"] == 1

    def test_high_objection_produces_no_go(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        candidate = _make_candidate()
        proposal = _make_proposal()
        report = _make_objection_report(proposal, KillCode.CHOP_STRUCTURE, Severity.HIGH)

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["no_go"] == 1
        assert stats["go"] == 0

    def test_proposer_error_increments_counter(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        candidate = _make_candidate()

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", side_effect=ProposerError("API down")):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["proposer_errors"] == 1
        assert stats["go"] == 0

    def test_critic_error_increments_counter(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path)
        candidate = _make_candidate()
        proposal = _make_proposal()

        with patch("src.main.scan", side_effect=_scan_once(candidate)), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", side_effect=CriticError("API down")):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=tmp_path / "kill.jsonl",
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )
        assert stats["critic_errors"] == 1
        assert stats["go"] == 0

    def test_multiple_go_decisions_appended_to_log(self, tmp_path: Path) -> None:
        data_dir = _setup_data(tmp_path, n_bars=5)
        kill_log = tmp_path / "kill.jsonl"
        candidate = _make_candidate()
        proposal = _make_proposal()
        report = _make_clean_report(proposal)

        # scan returns 1 candidate on every bar
        with patch("src.main.scan", return_value=[candidate]), \
             patch("src.main.propose", return_value=proposal), \
             patch("src.main.critique", return_value=report):
            stats = _run_replay(
                symbols=["BTC"], data_dir=data_dir,
                kill_log_path=kill_log,
                initial_equity=100_000.0, risk_pct=1.0,
                max_candidates=0, client=_mock_client(),
                memory_log_path=tmp_path / "memory.jsonl",
            )

        lines = kill_log.read_text().strip().split("\n")
        assert len(lines) == stats["go"]
        assert all(json.loads(ln)["verdict"] == "GO" for ln in lines)


# ---------------------------------------------------------------------------
# main() — CLI and env-var handling
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_api_key_raises_system_exit(self, tmp_path: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict("os.environ", env, clear=True), \
             patch("sys.argv", ["main", "--skip-ingest",
                                "--data-dir", str(tmp_path),
                                "--kill-log", str(tmp_path / "kill.jsonl")]):
            with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
                main()

    def test_skip_ingest_does_not_call_ingest_symbol(self, tmp_path: Path) -> None:
        mock_stats = {
            "bars": 0, "candidates": 0, "go": 0, "no_go": 0,
            "proposer_errors": 0, "critic_errors": 0, "kill_codes": Counter(),
        }
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("src.main.ingest_symbol") as mock_ingest, \
             patch("src.main._run_replay", return_value=mock_stats), \
             patch("sys.argv", ["main", "--skip-ingest",
                                "--data-dir", str(tmp_path),
                                "--kill-log", str(tmp_path / "kill.jsonl")]):
            main()
        mock_ingest.assert_not_called()


# Deferred imports placed here to avoid circular-import issues in test discovery
from src.agents.proposer import ProposerError  # noqa: E402
from src.agents.critic import CriticError  # noqa: E402
