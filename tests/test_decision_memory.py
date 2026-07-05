"""Tests for src/pipeline/decision_memory — schema, TTL gate, and log_decision."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline.decision_memory import (
    MemoryEntry,
    get_cached_keys,
    log_decision,
    get_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _entry(
    symbol: str = "BTC",
    candidate_ts: str | None = "2025-05-30T08:00:00+00:00",
    evaluated_at: str | None = None,
    decision: str = "GO",
) -> MemoryEntry:
    return MemoryEntry(
        ts=_NOW.isoformat(),
        symbol=symbol,
        direction="LONG",
        confidence=0.75,
        decision=decision,
        candidate_ts=candidate_ts,
        evaluated_at=evaluated_at,
    )


def _write_entries(path: Path, entries: list[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(e.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# MemoryEntry schema
# ---------------------------------------------------------------------------

class TestMemoryEntrySchema:
    def test_evaluated_at_defaults_none(self) -> None:
        e = _entry()
        assert e.evaluated_at is None

    def test_candidate_ts_defaults_none(self) -> None:
        e = MemoryEntry(ts=_NOW.isoformat(), symbol="ETH", direction="LONG", confidence=0.5, decision="NO_GO")
        assert e.candidate_ts is None
        assert e.evaluated_at is None

    def test_round_trip_json(self) -> None:
        ts = _NOW.isoformat()
        e = _entry(evaluated_at=ts)
        restored = MemoryEntry.model_validate_json(e.model_dump_json())
        assert restored.evaluated_at == ts
        assert restored.candidate_ts == e.candidate_ts


# ---------------------------------------------------------------------------
# get_cached_keys — TTL filtering
# ---------------------------------------------------------------------------

class TestGetCachedKeysTTL:
    def test_fresh_entry_included(self, tmp_path: Path) -> None:
        recent = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [_entry(evaluated_at=recent)])

        keys = get_cached_keys(path, cache_ttl_hours=24.0)
        assert ("BTC", "2025-05-30T08:00:00+00:00") in keys

    def test_expired_entry_excluded(self, tmp_path: Path) -> None:
        old = (datetime.now(tz=timezone.utc) - timedelta(hours=25)).isoformat()
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [_entry(evaluated_at=old)])

        keys = get_cached_keys(path, cache_ttl_hours=24.0)
        assert ("BTC", "2025-05-30T08:00:00+00:00") not in keys

    def test_missing_evaluated_at_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [_entry(evaluated_at=None)])

        keys = get_cached_keys(path, cache_ttl_hours=24.0)
        assert len(keys) == 0

    def test_missing_candidate_ts_excluded(self, tmp_path: Path) -> None:
        recent = (_NOW - timedelta(hours=1)).isoformat()
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [_entry(candidate_ts=None, evaluated_at=recent)])

        keys = get_cached_keys(path, cache_ttl_hours=24.0)
        assert len(keys) == 0

    def test_empty_file_returns_empty_frozenset(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.jsonl"
        path.write_text("")
        assert get_cached_keys(path) == frozenset()

    def test_nonexistent_file_returns_empty_frozenset(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.jsonl"
        assert get_cached_keys(path) == frozenset()

    def test_mixed_fresh_and_expired(self, tmp_path: Path) -> None:
        recent = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        old = (datetime.now(tz=timezone.utc) - timedelta(hours=25)).isoformat()
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [
            _entry(symbol="BTC", candidate_ts="2025-05-30T08:00:00+00:00", evaluated_at=recent),
            _entry(symbol="ETH", candidate_ts="2025-05-29T10:00:00+00:00", evaluated_at=old),
        ])

        keys = get_cached_keys(path, cache_ttl_hours=24.0)
        assert ("BTC", "2025-05-30T08:00:00+00:00") in keys
        assert ("ETH", "2025-05-29T10:00:00+00:00") not in keys

    def test_custom_ttl_boundary(self, tmp_path: Path) -> None:
        two_hours_ago = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [_entry(evaluated_at=two_hours_ago)])

        assert ("BTC", "2025-05-30T08:00:00+00:00") in get_cached_keys(path, cache_ttl_hours=3.0)
        assert ("BTC", "2025-05-30T08:00:00+00:00") not in get_cached_keys(path, cache_ttl_hours=1.0)

    def test_default_ttl_is_24_hours(self, tmp_path: Path) -> None:
        recent = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        path = tmp_path / "mem.jsonl"
        _write_entries(path, [_entry(evaluated_at=recent)])

        keys = get_cached_keys(path)  # no explicit TTL → default 24h
        assert ("BTC", "2025-05-30T08:00:00+00:00") in keys


# ---------------------------------------------------------------------------
# log_decision — evaluated_at is set
# ---------------------------------------------------------------------------

class TestLogDecisionEvaluatedAt:
    def _make_candidate(self, symbol: str = "BTC") -> MagicMock:
        c = MagicMock()
        c.symbol = symbol
        c.direction = "LONG"
        c.ma_period = 50
        c.confidence = 0.72
        c.ts = datetime(2025, 5, 30, 8, 0, 0, tzinfo=timezone.utc)
        c.regime = MagicMock()
        c.regime.value = "TREND"
        return c

    def _make_decision(self) -> MagicMock:
        from src.agents.arbiter import ArbiterVerdict
        d = MagicMock()
        d.verdict = MagicMock()
        d.verdict.value = "GO"
        d.kill_codes_fired = []
        d.ts = datetime(2025, 5, 30, 8, 0, 1, tzinfo=timezone.utc)
        return d

    def test_evaluated_at_is_written(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.jsonl"
        before = datetime.now(tz=timezone.utc)
        log_decision(self._make_candidate(), self._make_decision(), memory_path=path)
        after = datetime.now(tz=timezone.utc)

        entries = path.read_text().splitlines()
        assert len(entries) == 1
        record = MemoryEntry.model_validate_json(entries[0])
        assert record.evaluated_at is not None

        from src.pipeline.decision_memory import _parse_dt
        evaluated = _parse_dt(record.evaluated_at)
        assert before <= evaluated <= after

    def test_candidate_ts_is_written(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.jsonl"
        log_decision(self._make_candidate(), self._make_decision(), memory_path=path)

        record = MemoryEntry.model_validate_json(path.read_text().strip())
        assert record.candidate_ts == "2025-05-30T08:00:00+00:00"

    def test_fresh_after_log(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.jsonl"
        log_decision(self._make_candidate(), self._make_decision(), memory_path=path)

        keys = get_cached_keys(path, cache_ttl_hours=24.0)
        assert ("BTC", "2025-05-30T08:00:00+00:00") in keys
