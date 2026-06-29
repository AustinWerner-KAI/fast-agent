# fast-agent

LLM-agent trading system for Hyperliquid crypto and FX pairs.
Separate from BotZachary (swing bot). Shadow / propose-only mode — no live execution keys.

Agent pipeline: **Scout → Proposer → Critic → Arbiter**

---

## Structure

```
src/
  harness/
    ingest.py      — Fetch OHLCV + funding from Hyperliquid → parquet
    pit_data.py    — PITDataView: point-in-time data, look-ahead safe
    replay.py      — Candle-by-candle replay engine
    leak_test.py   — 7 pytest cases proving no look-ahead leakage
  agents/
    regime.py      — TREND / CHOP / VOLATILE classifier (ADX + ATR%)
    scout.py       — Pullback-to-EMA candidate detector, regime-gated
data_store/        — Parquet store (gitignored)
tests/
  test_regime.py
  test_scout.py
```

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch 90 days of BTC/ETH hourly data
python -m src.harness.ingest --symbols BTC ETH --tf 1h --days 90

# Run all tests
pytest -v

# Run only the look-ahead safety tests
pytest src/harness/leak_test.py -v
```

---

## Look-ahead safety

All data access goes through `PITDataView(data_dir, decision_ts)`. Every
accessor (`ohlcv`, `funding`, `orderbook`) returns only rows with
`close_time <= decision_ts`. The `open_time` column is never used as the
filter key — that would expose the open price of a bar before it closed.

The 7 leak tests in `src/harness/leak_test.py` assert:
1. `ohlcv()` returns only rows ≤ `decision_ts`
2. The boundary row (close_time == decision_ts) is included
3. A row 1 ms in the future is excluded
4. `funding()` is filtered correctly
5. `future_access()` always raises `LookAheadError`
6. `ReplayEngine.stream()` yields strictly increasing `decision_ts`
7. At each replay step, the last visible candle's `close_time == decision_ts`

---

## Regime classifier

`regime.py` classifies each (symbol, timeframe) as TREND / CHOP / VOLATILE.

| Condition | Regime |
|---|---|
| ATR% > 3.0% | VOLATILE (highest priority) |
| ADX > 25 | TREND |
| else | CHOP |

Uses Wilder's smoothed ADX (14-period) and ATR% (ATR / last close × 100).
Minimum 15 bars required; returns CHOP with zero indicators if insufficient.

---

## Scout

`scout.py` scans symbols for pullback-to-EMA setups, gated by TREND regime.

- **Entry timeframe**: 1h (regime check + EMA proximity)
- **Trend timeframe**: 1d (direction alignment)
- **MAs**: EMA-20, EMA-50, EMA-200
- **Entry tolerance**: price within ±1.5% of EMA
- **Confidence**: 70% proximity score + 30% ADX weight

Outputs a `list[Candidate]` sorted by confidence descending. No LLM calls.

---

## Data store schema

OHLCV parquet: `data_store/{symbol}/{timeframe}.parquet`
| Column | Type |
|---|---|
| open_time | datetime64[ns, UTC] |
| close_time | datetime64[ns, UTC] |
| open / high / low / close | float64 |
| volume | float64 |

Funding parquet: `data_store/{symbol}/funding.parquet`
| Column | Type |
|---|---|
| ts | datetime64[ns, UTC] |
| rate | float64 |

---

## Environment

```bash
cp .env.example .env
# Fill in HYPERLIQUID_ADDRESS if you want address-specific data
```

No trading keys are used — ingest.py calls Hyperliquid public endpoints only.
