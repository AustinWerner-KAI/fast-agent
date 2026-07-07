"""Tests for src/pipeline/executor.py.

Covers the post-fill stop formula (always computed from fill_price), TP order
placement, the main execution gate flow, and stop/TP order IDs logged to
execution.jsonl.  All broker/guard/breaker calls are mocked.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
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
    tp1_order_id: str = "tp1-order-1",
    tp2_order_id: str = "tp2-order-1",
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
    tp1_result = MagicMock()
    tp1_result.order_id = tp1_order_id
    tp2_result = MagicMock()
    tp2_result.order_id = tp2_order_id
    broker.place_tp_order.side_effect = [tp1_result, tp2_result]
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


def _mock_tp_cfg(tp1_fraction: float = 0.50, tp2_fraction: float = 0.30) -> MagicMock:
    cfg = MagicMock()
    cfg.tp1_fraction = tp1_fraction
    cfg.tp2_fraction = tp2_fraction
    return cfg


_ENV = {
    "TRADING_ENABLED": "true",
    "MAX_POSITION_PCT": "0.10",
    "MAX_POSITION_USD": "50.0",
    "MAX_LEVERAGE": "10.0",
    "MIN_FREE_MARGIN": "50.0",
    "MAX_CONCURRENT_POSITIONS": "2",
}

# Patches common to all fill-path tests.
# With empty tiers: margin_used = min(free_margin*0.10, 50/10) = min(20, 5) = 5.0
_EMPTY_CONV = {"tiers": [], "free_margin_cap_pct": 0.02}


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
    log.write_text(json.dumps({"state": "filled", "verdict_id": "vid-dup"}) + "\n")
    with patch.dict(os.environ, _ENV):
        ex = Executor(_make_broker(), _make_guard(), _make_breaker(), exec_log=log)
        result = ex.execute("vid-dup", _make_proposal(), _make_candidate())
    assert result["state"] == "duplicate_suppressed"


def test_max_positions_reached(tmp_path: Path) -> None:
    log = tmp_path / "exec.jsonl"
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

def test_successful_fill_logs_intent_then_filled(tmp_path: Path) -> None:
    log = tmp_path / "exec.jsonl"
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_000.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=log)
                    ex.execute("vid-log", proposal, _make_candidate())

    entries = [json.loads(line) for line in log.read_text().strip().splitlines()]
    states = [e["state"] for e in entries]
    assert "intent" in states
    assert "submitted" in states
    assert "filled" in states


# ---------------------------------------------------------------------------
# Stop formula — always computed from fill_price
# ---------------------------------------------------------------------------

def test_stop_always_computed_from_fill_price(tmp_path: Path) -> None:
    """Stop is always fill_price - (margin * risk / size), even when fill > proposal stop."""
    # margin_used=5.0 (empty tiers), risk_pct=0.10, fill_size=0.001
    # expected_stop = 50_100 - (5 * 0.10 / 0.001) = 50_100 - 500 = 49_600
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    result = ex.execute("vid-ok", proposal, _make_candidate())

    assert result["state"] == "filled"
    expected = round(50_100.0 - (5.0 * 0.10 / 0.001), 8)
    assert result["stop"] == pytest.approx(expected)
    assert result["original_stop"] == 49_000.0


def test_stop_formula_long_stop_below_fill(tmp_path: Path) -> None:
    """LONG effective_stop is always strictly below fill_price."""
    proposal = _make_proposal(entry=2500.0, stop=2400.0, tp1=2600.0, tp2=2700.0, tp3=2800.0)
    broker = _make_broker(free_margin=200.0, fill_price=1600.0, size=0.001)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    result = ex.execute("vid-eth", proposal, _make_candidate("ETH"))

    assert result["state"] == "filled"
    assert result["stop"] < result["fill_price"]


def test_stop_formula_respects_risk_pct_config(tmp_path: Path) -> None:
    """load_risk_pct_per_trade value is used in stop calculation."""
    # margin_used=5.0, risk_pct=0.05 → stop_distance = 5*0.05/0.001 = 250
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0, size=0.001)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.05) as mock_risk:
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    ex.execute("vid-risk", proposal, _make_candidate())

    mock_risk.assert_called_once()
    expected = round(58_000.0 - (5.0 * 0.05 / 0.001), 8)
    log_entries = [
        json.loads(l) for l in (tmp_path / "exec.jsonl").read_text().strip().splitlines()
    ]
    filled = next(e for e in log_entries if e["state"] == "filled")
    assert filled["stop"] == pytest.approx(expected)


def test_stop_risk_pct_always_called(tmp_path: Path) -> None:
    """load_risk_pct_per_trade is called on every fill, regardless of fill vs proposal stop relationship."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_500.0)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10) as mock_risk:
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    ex.execute("vid-always", proposal, _make_candidate())

    mock_risk.assert_called_once()


def test_original_stop_logged_when_formula_differs(tmp_path: Path) -> None:
    """original_stop field appears in filled log entry when computed stop differs from proposal."""
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0, size=0.001)
    log = tmp_path / "exec.jsonl"

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=log)
                    ex.execute("vid-log-check", proposal, _make_candidate())

    entries = [json.loads(l) for l in log.read_text().strip().splitlines()]
    filled = next(e for e in entries if e["state"] == "filled")
    assert "original_stop" in filled
    assert filled["original_stop"] == 77_000.0
    expected = round(58_000.0 - (5.0 * 0.10 / 0.001), 8)
    assert filled["stop"] == pytest.approx(expected)


def test_stop_recalc_logged_when_formula_differs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """STOP_RECALC info log is emitted when computed stop differs from proposal stop."""
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0, size=0.001)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    with caplog.at_level(logging.INFO, logger="src.pipeline.executor"):
                        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                        ex.execute("vid-warn", proposal, _make_candidate())

    assert any("STOP_RECALC" in r.message for r in caplog.records)


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
    """Exchange stop order is placed with the formula-computed trigger price."""
    # margin_used=5.0, risk_pct=0.10, fill_size=0.001 → trigger=50_100 - 500 = 49_600
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0, stop_order_id="stop-99")

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    result = ex.execute("vid-stop", proposal, _make_candidate())

    assert result["state"] == "filled"
    broker.place_stop_order.assert_called_once()
    kwargs = broker.place_stop_order.call_args.kwargs
    assert kwargs["symbol"] == "BTC"
    assert kwargs["side"] == "SELL"
    expected_trigger = round(50_100.0 - (5.0 * 0.10 / 0.001), 8)
    assert kwargs["trigger_price"] == pytest.approx(expected_trigger)


def test_stop_order_id_logged_in_fill_entry(tmp_path: Path) -> None:
    """stop_order_id is written to the filled log entry when stop is placed."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0, stop_order_id="stop-99")
    log = tmp_path / "exec.jsonl"

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
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
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    result = ex.execute("vid-stop-fail", proposal, _make_candidate())

    assert result["state"] == "filled"
    assert "stop_order_id" not in result


def test_stop_order_uses_formula_stop_not_proposal(tmp_path: Path) -> None:
    """Stop order trigger_price uses the formula-computed stop, not proposal.stop."""
    proposal = _make_proposal(entry=78_000.0, stop=77_000.0, tp1=80_000.0, tp2=82_000.0, tp3=84_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=58_000.0, size=0.001)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    ex.execute("vid-stop-geo", proposal, _make_candidate())

    kwargs = broker.place_stop_order.call_args.kwargs
    expected_stop = round(58_000.0 - (5.0 * 0.10 / 0.001), 8)
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


# ---------------------------------------------------------------------------
# TP order placement tests
# ---------------------------------------------------------------------------

def test_tp1_and_tp2_orders_placed_after_fill(tmp_path: Path) -> None:
    """place_tp_order called twice (TP1, TP2) immediately after a fill."""
    proposal = _make_proposal(
        entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0,
    )
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0, size=0.001)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg(0.50, 0.30)):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    ex.execute("vid-tp", proposal, _make_candidate())

    assert broker.place_tp_order.call_count == 2
    calls = broker.place_tp_order.call_args_list
    # Executor recalculates TP triggers from fill_price, preserving proposal R:R.
    # fill=50_100, effective_stop=49_600 (margin_used=5*0.10/0.001=500 away),
    # stop_distance=500, proposal R:R: tp1=(52k-50k)/(50k-49k)=2.0, tp2=3.0
    # effective_tp1 = 50_100 + 2.0*500 = 51_100
    # effective_tp2 = 50_100 + 3.0*500 = 51_600
    kw1 = calls[0].kwargs
    assert kw1["symbol"] == "BTC"
    assert kw1["side"] == "SELL"
    assert kw1["trigger_price"] == pytest.approx(51_100.0)
    assert kw1["size"] == pytest.approx(0.001 * 0.50, rel=1e-6)
    # TP2: trigger=51_600, size=30% of 0.001
    kw2 = calls[1].kwargs
    assert kw2["trigger_price"] == pytest.approx(51_600.0)
    assert kw2["size"] == pytest.approx(0.001 * 0.30, rel=1e-6)


def test_tp_order_ids_in_result_and_log(tmp_path: Path) -> None:
    """tp1_order_id and tp2_order_id appear in the result dict and filled log entry."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(
        free_margin=200.0, fill_price=50_100.0,
        tp1_order_id="tp1-abc", tp2_order_id="tp2-xyz",
    )
    log = tmp_path / "exec.jsonl"

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=log)
                    result = ex.execute("vid-tp-ids", proposal, _make_candidate())

    assert result.get("tp1_order_id") == "tp1-abc"
    assert result.get("tp2_order_id") == "tp2-xyz"
    entries = [json.loads(l) for l in log.read_text().strip().splitlines()]
    filled = next(e for e in entries if e["state"] == "filled")
    assert filled.get("tp1_order_id") == "tp1-abc"
    assert filled.get("tp2_order_id") == "tp2-xyz"


def test_fill_succeeds_when_tp1_order_fails(tmp_path: Path) -> None:
    """Fill is still logged and TP2 still attempted even if TP1 placement fails."""
    proposal = _make_proposal(entry=50_000.0, stop=49_000.0, tp1=52_000.0, tp2=53_000.0, tp3=54_000.0)
    broker = _make_broker(free_margin=200.0, fill_price=50_100.0, tp2_order_id="tp2-ok")
    # TP1 raises, TP2 succeeds — side_effect: first call raises, second returns tp2_result
    tp2_result = MagicMock()
    tp2_result.order_id = "tp2-ok"
    broker.place_tp_order.side_effect = [Exception("TP1 failed"), tp2_result]

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    result = ex.execute("vid-tp-fail", proposal, _make_candidate())

    assert result["state"] == "filled"
    assert "tp1_order_id" not in result
    assert result.get("tp2_order_id") == "tp2-ok"


def test_tp_orders_use_short_side_for_short_direction(tmp_path: Path) -> None:
    """SHORT trade uses BUY side for TP orders (closing a short)."""
    proposal = _make_proposal(
        direction="SHORT", entry=50_000.0, stop=51_000.0, tp1=48_000.0, tp2=47_000.0, tp3=46_000.0,
    )
    broker = _make_broker(free_margin=200.0, fill_price=50_000.0, size=0.001)

    with patch.dict(os.environ, _ENV):
        with patch("src.pipeline.executor.load_conviction_sizing", return_value=_EMPTY_CONV):
            with patch("src.pipeline.executor.load_risk_pct_per_trade", return_value=0.10):
                with patch("src.pipeline.executor.load_take_profit_config", return_value=_mock_tp_cfg()):
                    ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
                    ex.execute("vid-short", proposal, _make_candidate())

    for call in broker.place_tp_order.call_args_list:
        assert call.kwargs["side"] == "BUY"


def test_tp_orders_not_placed_on_dry_run(tmp_path: Path) -> None:
    """TP orders are not placed when TRADING_ENABLED=false."""
    env = {**_ENV, "TRADING_ENABLED": "false"}
    broker = _make_broker()

    with patch.dict(os.environ, env):
        ex = Executor(broker, _make_guard(), _make_breaker(), exec_log=tmp_path / "exec.jsonl")
        result = ex.execute("vid-dry", _make_proposal(), _make_candidate())

    assert result["state"] == "dry_run"
    broker.place_tp_order.assert_not_called()
