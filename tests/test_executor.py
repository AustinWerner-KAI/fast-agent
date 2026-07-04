"""Tests for src/pipeline/executor.py.

Covers the post-fill geometry sanity check (GEOMETRY_CORRECTED) and the
main execution gate flow. All broker/guard/breaker calls are mocked.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.executor import Executor, ExecutorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker(
    free_margin: float = 200.0,
    fill_price: float = 50_000.0,
    order_id: str = "order-1",
    status: str = "filled",
    size: float = 0.001,
    stop_order_id: str = "stop-order-1",
) -> MagicMock:
    broker = MagicMock()
    broker.get_free_margin.return_value = free_margin
    result = MagicMock()
    result.status = status
    result.order_id = order_id
    result.filled_price = fill_price
    result.size = size
    broker.place_order.return_value = result
    stop_result = MagicMock()
    stop_result.order_id = stop_order_id
    broker.place_stop_order.return_value = stop_result
    return broker


def _make_guard(protected: bool = False) -> MagicMock:
    guard = MagicMock()
    guard.is_protected.return_value = protected
    return guard


def _make_breaker(allowed: bool = True) -> MagicMock:
    breaker = MagicMock()
    breaker.check.return_value = allowed
    return breaker


def _make_proposal(
    symbol: str = "BTC",
    direction: str = "LONG",
    entry: float = 50_000.0,
    stop: float = 49_000.0,
    tp1: float = 52_000.0,
    tp2: float = 53_000.0,
    tp3: float = 54_000.0,
    confidence: float = 0.75,
) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.direction = direction
    p.entry = entry
    p.stop = stop
    p.tp1 = tp1
    p.tp2 = tp2
    p.tp3 = tp3
    p.confidence = confidence
    return p


def _make_candidate(symbol: str = "BTC") -> MagicMock:
    c = MagicMock()
    c.symbol = symbol
    return c


_ENV = {
    "TRADING_ENABLED": "true",
    "MAX_POSITION_PCT": "0.10",
    "MAX_POSITION_USD": "50.0",
    "MAX_LEVERAGE": "10.0",
    "MIN_FREE_MARGIN": "50.0",
    "MAX_CONCURRENT_POSITIONS": "2",
}


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

def test_dry_run_when_trading_disabled(tmp_path: Path) -> None:
    env = {**_ENV, "TRADING_ENABLED": "false"}
    with patch.dict(os.environ, env):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-1", _make_proposal(), _make_candidate())
    assert result["state"] == "dry_run"


def test_halted_when_circuit_breaker_tripped(tmp_path: Path) -> None:
    with patch.dict(os.environ, _ENV):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(allowed=False), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-1", _make_proposal(), _make_candidate())
    assert result["state"] == "halted"


def test_blocked_manual_when_position_guard_active(tmp_path: Path) -> None:
    with patch.dict(os.environ, _ENV):
        ex = Executor(_make_broker(), _make_guard(protected=True), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-1", _make_proposal(), _make_candidate())
    assert result["state"] == "blocked_manual"


def test_duplicate_verdict_suppressed(tmp_path: Path) -> None:
    log = tmp_path / "exec.jsonl"
    # Pre-seed the log with a matching verdict_id
    log.write_text(json.dumps({"state": "filled", "verdict_id": "vid-dup"}) + "\n")
    with patch.dict(os.environ, _ENV):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(), exec_log=log)
        result = ex.execute("vid-dup", _make_proposal(), _make_candidate())
    assert result["state"] == "duplicate_suppressed"


def test_max_positions_reached(tmp_path: Path) -> None:
    log = tmp_path / "exec.jsonl"
    # Two active filled positions
    log.write_text(
        json.dumps({"state": "filled", "verdict_id": "v1", "symbol": "BTC"}) + "\n" +
        json.dumps({"state": "filled", "verdict_id": "v2", "symbol": "ETH"}) + "\n"
    )
    with patch.dict(os.environ, {**_ENV, "MAX_CONCURRENT_POSITIONS": "2"}):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(), exec_log=log)
        result = ex.execute("vid-new", _make_proposal(symbol="SOL"), _make_candidate("SOL"))
    assert result["state"] == "max_positions_reached"


def test_skipped_low_margin(tmp_path: Path) -> None:
    broker = _make_broker(free_margin=10.0)
    with patch.dict(os.environ, {**_ENV, "MIN_FREE_MARGIN": "50.0"}):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-1", _make_proposal(), _make_candidate())
    assert result["state"] == "skipped_low_margin"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_successful_fill_normal_geometry(tmp_path: Path) -> None:
    """When fill_price > stop, stop is stored unchanged."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0)

    with patch.dict(os.environ, _ENV):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-ok", proposal, _make_candidate())

    assert result["state"] == "filled"
    assert result["stop"] == 49_000.0
    assert "original_stop" not in result


def test_successful_fill_logs_intent_then_filled(tmp_path: Path) -> None:
    log = tmp_path / "exec.jsonl"
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_000.0)

    with patch.dict(os.environ, _ENV):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=log)
        ex.execute("vid-log", proposal, _make_candidate())

    entries = [json.loads(line) for line in log.read_text().strip().splitlines()]
    states = [e["state"] for e in entries]
    assert "intent" in states
    assert "submitted" in states
    assert "filled" in states


# ---------------------------------------------------------------------------
# GEOMETRY_CORRECTED tests
# ---------------------------------------------------------------------------

def test_geometry_corrected_when_fill_below_stop(tmp_path: Path) -> None:
    """fill_price < stop → stop recomputed at fill_price * (1 - 0.07)."""
    # Proposal entry=78_000, stop=77_000 (valid geometry)
    # Market fills at 58_000 (well below stop) — simulates stale OHLCV entry
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07):
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
            result = ex.execute("vid-geo", proposal, _make_candidate())

    assert result["state"] == "filled"
    expected_stop = round(58_000.0 * (1.0 - 0.07), 8)
    assert result["stop"] == pytest.approx(expected_stop)
    assert result["original_stop"] == 77_000.0


def test_geometry_corrected_stop_is_below_fill_price(tmp_path: Path) -> None:
    """Corrected stop must always be below fill_price."""
    proposal = _make_proposal(entry=2500.0, stop=2400.0, tp1=2600.0, tp2=2700.0, tp3=2800.0)
    broker = _make_broker(free_margin=200.0, fill_price=1600.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07):
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
            result = ex.execute("vid-eth", proposal, _make_candidate("ETH"))

    assert result["state"] == "filled"
    assert result["stop"] < result["fill_price"]


def test_geometry_corrected_uses_config_trail_pct(tmp_path: Path) -> None:
    """load_trail_pct is called when fill_price < stop."""
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.05) as mock_trail:
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
            ex.execute("vid-trail", proposal, _make_candidate())

    mock_trail.assert_called_once()
    # 5% trail instead of 7%
    log_entries = [
        json.loads(l) for l in (tmp_path / "exec.jsonl").read_text().strip().splitlines()
    ]
    filled = next(e for e in log_entries if e["state"] == "filled")
    assert filled["stop"] == pytest.approx(round(58_000.0 * 0.95, 8))


def test_geometry_corrected_logged_to_jsonl(tmp_path: Path) -> None:
    """original_stop field appears in filled log entry when correction applied."""
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0)
    log = tmp_path / "exec.jsonl"

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07):
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=log)
            ex.execute("vid-log-check", proposal, _make_candidate())

    entries = [json.loads(l) for l in log.read_text().strip().splitlines()]
    filled = next(e for e in entries if e["state"] == "filled")
    assert "original_stop" in filled
    assert filled["original_stop"] == 77_000.0
    assert filled["stop"] == pytest.approx(round(58_000.0 * 0.93, 8))


def test_no_correction_when_fill_equals_stop(tmp_path: Path) -> None:
    """fill_price == stop: boundary — no correction (not strictly below)."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=49_000.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07) as mock_trail:
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
            result = ex.execute("vid-eq", proposal, _make_candidate())

    # fill == stop is not < stop, so no correction
    mock_trail.assert_not_called()
    assert result["stop"] == 49_000.0
    assert "original_stop" not in result


def test_no_correction_when_fill_above_stop(tmp_path: Path) -> None:
    """Normal fill above stop: no geometry correction."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_500.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07) as mock_trail:
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
            result = ex.execute("vid-normal", proposal, _make_candidate())

    mock_trail.assert_not_called()
    assert result["stop"] == 49_000.0
    assert "original_stop" not in result


def test_geometry_corrected_warning_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """GEOMETRY_CORRECTED warning is emitted when stop is recomputed."""
    import logging
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07):
            with caplog.at_level(logging.WARNING, logger="src.pipeline.executor"):
                ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                ex.execute("vid-warn", proposal, _make_candidate())

    assert any("GEOMETRY_CORRECTED" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# active_bot_symbols — closed-state tests
# ---------------------------------------------------------------------------

def test_closed_entry_frees_position_slot(tmp_path: Path) -> None:
    """A 'closed' entry after 'filled' removes the symbol from active set."""
    log = tmp_path / "exec.jsonl"
    log.write_text(
        json.dumps({"state": "filled", "verdict_id": "v1", "symbol": "BTC"}) + "\n" +
        json.dumps({"state": "closed", "verdict_id": "v1", "symbol": "BTC", "reason": "reconciliation_cleanup"}) + "\n" +
        json.dumps({"state": "filled", "verdict_id": "v2", "symbol": "ETH"}) + "\n" +
        json.dumps({"state": "closed", "verdict_id": "v2", "symbol": "ETH", "reason": "reconciliation_cleanup"}) + "\n"
    )
    with patch.dict(os.environ, _ENV):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(), exec_log=log)
        assert ex._count_active_positions() == 0


def test_orphaned_filled_without_closed_counts_as_active(tmp_path: Path) -> None:
    """A 'filled' entry with no subsequent terminal entry counts as open."""
    log = tmp_path / "exec.jsonl"
    log.write_text(
        json.dumps({"state": "filled", "verdict_id": "v1", "symbol": "BTC"}) + "\n"
    )
    with patch.dict(os.environ, _ENV):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(), exec_log=log)
        assert ex._count_active_positions() == 1


# ---------------------------------------------------------------------------
# Stop order placement tests
# ---------------------------------------------------------------------------

def test_stop_order_placed_immediately_after_fill(tmp_path: Path) -> None:
    """Exchange stop order is placed immediately after fill confirmation."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0, stop_order_id="stop-99")

    with patch.dict(os.environ, _ENV):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-stop", proposal, _make_candidate())

    assert result["state"] == "filled"
    broker.place_stop_order.assert_called_once()
    kwargs = broker.place_stop_order.call_args.kwargs
    assert kwargs["symbol"] == "BTC"
    assert kwargs["side"] == "SELL"
    assert kwargs["trigger_price"] == pytest.approx(49_000.0)


def test_stop_order_id_logged_in_fill_entry(tmp_path: Path) -> None:
    """stop_order_id is written to the filled log entry when stop is placed."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0, stop_order_id="stop-99")
    log = tmp_path / "exec.jsonl"

    with patch.dict(os.environ, _ENV):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=log)
        result = ex.execute("vid-stop-log", proposal, _make_candidate())

    assert result.get("stop_order_id") == "stop-99"
    entries = [json.loads(l) for l in log.read_text().strip().splitlines()]
    filled = next(e for e in entries if e["state"] == "filled")
    assert filled.get("stop_order_id") == "stop-99"


def test_fill_succeeds_when_stop_order_fails(tmp_path: Path) -> None:
    """Fill is still logged successfully even if stop order placement fails."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0)
    broker.place_stop_order.side_effect = Exception("network error")

    with patch.dict(os.environ, _ENV):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-stop-fail", proposal, _make_candidate())

    assert result["state"] == "filled"
    assert "stop_order_id" not in result


def test_stop_order_uses_corrected_stop_after_geometry_correction(tmp_path: Path) -> None:
    """When fill is below proposal stop, stop order uses the geometry-corrected level."""
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_trail_pct", return_value=0.07):
            ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
            ex.execute("vid-stop-geo", proposal, _make_candidate())

    kwargs = broker.place_stop_order.call_args.kwargs
    expected_stop = round(58_000.0 * (1.0 - 0.07), 8)
    assert kwargs["trigger_price"] == pytest.approx(expected_stop)


def test_stop_order_not_placed_on_dry_run(tmp_path: Path) -> None:
    """Stop order is not placed when TRADING_ENABLED=false."""
    env = {**_ENV, "TRADING_ENABLED": "false"}
    broker = _make_broker()

    with patch.dict(os.environ, env):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-dry", _make_proposal(), _make_candidate())

    assert result["state"] == "dry_run"
    broker.place_stop_order.assert_not_called()
