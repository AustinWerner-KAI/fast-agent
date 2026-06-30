"""Main pipeline: ingest → replay → Scout → Proposer → Critic → Arbiter.

Paper-trading only — no orders are placed. Every Arbiter GO decision is
appended to the KILL log (append-only JSONL) for future outcome analysis.

Usage:
    python -m src.main [options]
    python -m src.main --skip-ingest --max-candidates 10  # quick dev run

Options:
    --symbols         Coins to scan (default: BTC ETH SOL ARB DOGE)
    --days            Days of history to fetch and replay (default: 90)
    --data-dir        Parquet store root directory (default: ./data_store)
    --kill-log        Append-only KILL log path (default: kill_log.jsonl)
    --equity          Starting account equity in USD (default: 100000)
    --risk-pct        Equity fraction to risk per trade (default: 1.0)
    --skip-ingest     Use existing parquet data; skip the fetch step
    --max-candidates  Stop after this many LLM pipeline calls (0 = unlimited)
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from src.harness.ingest import ingest_symbol
from src.harness.replay import ReplayEngine
from src.agents.scout import scan, DEFAULT_SYMBOLS, ENTRY_TF
from src.agents.proposer import ProposerInput, propose, ProposerError
from src.agents.critic import CriticInput, critique, CriticError
from src.agents.arbiter import arbitrate, ArbiterVerdict, ArbiterError

_DIVIDER = "=" * 64
_ATR_PERIOD = 14


# ---------------------------------------------------------------------------
# Market-context helpers
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> float:
    """Compute Wilder ATR from an OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with high, low, close columns (oldest row first).
        period: Smoothing period (default 14).

    Returns:
        ATR as an absolute price value. Returns 2% of last close as a
        fallback when the DataFrame has fewer than ``period + 1`` rows.
    """
    if len(df) < period + 1:
        return float(df["close"].iloc[-1]) * 0.02
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


def _latest_funding(pit: Any, symbol: str) -> float | None:
    """Return the most recent funding rate visible at decision_ts, or None.

    Args:
        pit: PITDataView — funding access is best-effort; errors return None.
        symbol: Coin name.

    Returns:
        Funding rate as a decimal (e.g. 0.0001), or None if unavailable.
    """
    try:
        df = pit.funding(symbol)
        if df.empty:
            return None
        return float(df["rate"].iloc[-1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _run_ingest(symbols: list[str], days: int, data_dir: Path) -> None:
    """Fetch and persist fresh OHLCV and funding data for all symbols.

    Fetches both the 1h (entry) and 1d (trend) timeframes required by Scout.

    Args:
        symbols: Coin names to ingest.
        days: How many calendar days of history to pull.
        data_dir: Root of the parquet store.
    """
    print(_DIVIDER)
    print(f"  INGEST  {days}d history · {len(symbols)} symbols: {', '.join(symbols)}")
    print(_DIVIDER)
    for sym in symbols:
        for tf in ["1h", "1d"]:
            ingest_symbol(sym, tf, days, data_dir)
    print("Ingest complete.\n")


def _run_replay(
    symbols: list[str],
    data_dir: Path,
    kill_log_path: Path,
    initial_equity: float,
    risk_pct: float,
    max_candidates: int,
    client: anthropic.Anthropic,
) -> dict[str, Any]:
    """Drive the Scout → Proposer → Critic → Arbiter pipeline over replay.

    Streams the parquet store bar-by-bar via ReplayEngine. At each bar
    Scout scans for candidates; each candidate is processed through the full
    LLM pipeline. GO decisions are written to the KILL log by the Arbiter.

    Args:
        symbols: Coins to include in the scan.
        data_dir: Root of the parquet store.
        kill_log_path: Append-only KILL log file.
        initial_equity: Starting account equity in USD.
        risk_pct: Percent of equity to risk per trade (e.g. 1.0 = 1%).
        max_candidates: Stop after this many total LLM calls (0 = unlimited).
        client: Pre-built Anthropic client (shared across all pipeline calls).

    Returns:
        Stats dict: bars, candidates, go, no_go, proposer_errors,
        critic_errors, kill_codes (Counter).
    """
    print(_DIVIDER)
    print(
        f"  REPLAY  {len(symbols)} symbols · equity=${initial_equity:,.0f} · "
        f"risk={risk_pct}% · kill-log={kill_log_path}"
    )
    print(_DIVIDER)

    engine = ReplayEngine(
        data_dir, symbols, timeframe=ENTRY_TF, initial_equity=initial_equity
    )

    stats: dict[str, Any] = {
        "bars": 0,
        "candidates": 0,
        "go": 0,
        "no_go": 0,
        "proposer_errors": 0,
        "critic_errors": 0,
        "kill_codes": Counter(),
    }

    for state in engine.stream():
        stats["bars"] += 1
        candidates = scan(state.pit, symbols, state.decision_ts)
        if not candidates:
            continue

        for candidate in candidates:
            if max_candidates > 0 and stats["candidates"] >= max_candidates:
                break
            stats["candidates"] += 1

            # --- Market context ---
            try:
                df_entry = state.pit.ohlcv(candidate.symbol, ENTRY_TF)
                if df_entry.empty:
                    continue
                current_price = float(df_entry["close"].iloc[-1])
                atr = _compute_atr(df_entry)
            except Exception:
                continue

            # --- Proposer ---
            try:
                proposal = propose(
                    ProposerInput(
                        candidate=candidate,
                        current_price=current_price,
                        atr=atr,
                        account_equity=state.equity,
                        risk_pct=risk_pct,
                    ),
                    client=client,
                )
            except ProposerError as exc:
                stats["proposer_errors"] += 1
                print(f"  [PROPOSER ERR] {candidate.symbol}: {exc}")
                continue

            # --- Critic ---
            try:
                report = critique(
                    CriticInput(
                        proposal=proposal,
                        funding_rate=_latest_funding(state.pit, candidate.symbol),
                    ),
                    client=client,
                )
            except CriticError as exc:
                stats["critic_errors"] += 1
                print(f"  [CRITIC ERR] {candidate.symbol}: {exc}")
                continue

            # --- Arbiter ---
            try:
                decision = arbitrate(proposal, report, log_path=kill_log_path)
            except ArbiterError as exc:
                print(f"  [ARBITER ERR] {candidate.symbol}: {exc}")
                continue

            if decision.verdict == ArbiterVerdict.GO:
                stats["go"] += 1
            else:
                stats["no_go"] += 1

            for kc in decision.kill_codes_fired:
                stats["kill_codes"][kc.value] += 1

            # Per-decision line
            obj_str = ""
            if report.objections:
                tags = ", ".join(
                    f"{o.kill_code.value}:{o.severity.value}"
                    for o in report.objections
                )
                obj_str = f"  [{tags}]"
            ts_str = state.decision_ts.strftime("%Y-%m-%d %H:%M")
            print(
                f"  {ts_str} | {candidate.symbol:<6} {candidate.direction:<5} "
                f"conf={candidate.confidence:.3f} | "
                f"{decision.verdict.value:<5} — {decision.reason}{obj_str}"
            )

        if max_candidates > 0 and stats["candidates"] >= max_candidates:
            print(f"\n  [cap reached: max-candidates={max_candidates}]")
            break

    return stats


def _print_summary(stats: dict[str, Any], kill_log_path: Path) -> None:
    """Print the end-of-run summary table.

    Args:
        stats: Dict returned by ``_run_replay``.
        kill_log_path: Path to the KILL log (shown in output).
    """
    total = stats["go"] + stats["no_go"]
    go_pct = stats["go"] / total * 100 if total else 0.0
    no_go_pct = stats["no_go"] / total * 100 if total else 0.0

    print(f"\n{_DIVIDER}")
    print("  SUMMARY")
    print(_DIVIDER)
    print(f"  Bars replayed:       {stats['bars']:>8,}")
    print(f"  Total candidates:    {stats['candidates']:>8,}")
    print(f"  GO decisions:        {stats['go']:>8,}  ({go_pct:.1f}%)")
    print(f"  NO-GO decisions:     {stats['no_go']:>8,}  ({no_go_pct:.1f}%)")
    if stats["proposer_errors"]:
        print(f"  Proposer errors:     {stats['proposer_errors']:>8,}")
    if stats["critic_errors"]:
        print(f"  Critic errors:       {stats['critic_errors']:>8,}")
    print(f"  Kill log:            {kill_log_path}")

    if stats["kill_codes"]:
        print("\n  Most common kill codes (across all decisions):")
        for code, count in stats["kill_codes"].most_common():
            bar = "█" * min(count, 32)
            print(f"    {code:<28}  {count:>3}  {bar}")
    else:
        print("\n  No kill codes fired.")
    print(_DIVIDER)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the fast-agent paper-trading pipeline."""
    parser = argparse.ArgumentParser(
        description="fast-agent: paper-trade replay — Scout → Proposer → Critic → Arbiter"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Coins to scan"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Days of history (default: 90)"
    )
    parser.add_argument(
        "--data-dir", default="./data_store", help="Parquet store root"
    )
    parser.add_argument(
        "--kill-log", default="kill_log.jsonl", help="KILL log output path"
    )
    parser.add_argument(
        "--equity", type=float, default=100_000.0, help="Starting equity in USD"
    )
    parser.add_argument(
        "--risk-pct", type=float, default=1.0, help="Risk %% per trade (default: 1.0)"
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Skip data fetch; use existing parquet"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Cap total LLM pipeline calls (0 = unlimited)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set.")

    data_dir = Path(args.data_dir)
    kill_log_path = Path(args.kill_log)

    if not args.skip_ingest:
        _run_ingest(args.symbols, args.days, data_dir)
    else:
        print("Skipping ingest (--skip-ingest).\n")

    client = anthropic.Anthropic(api_key=api_key)
    stats = _run_replay(
        symbols=args.symbols,
        data_dir=data_dir,
        kill_log_path=kill_log_path,
        initial_equity=args.equity,
        risk_pct=args.risk_pct,
        max_candidates=args.max_candidates,
        client=client,
    )
    _print_summary(stats, kill_log_path)


if __name__ == "__main__":
    main()
