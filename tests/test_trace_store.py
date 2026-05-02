import re

import pytest

from main.core.trace_store import TraceStore, default_trace_store


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class TestTraceStorePutGet:
    def test_put_returns_16_hex_id(self):
        store = TraceStore(ttl_seconds=60)
        tid = store.put({"foo": "bar"})
        assert re.fullmatch(r"[0-9a-f]{16}", tid)

    def test_get_returns_stored_payload(self):
        store = TraceStore(ttl_seconds=60)
        payload = {"schemaVersion": 1, "query": {"raw": "hi"}}
        tid = store.put(payload)
        assert store.get(tid) == payload

    def test_get_unknown_id_returns_none(self):
        store = TraceStore(ttl_seconds=60)
        assert store.get("0000000000000000") is None

    def test_distinct_puts_get_distinct_ids(self):
        store = TraceStore(ttl_seconds=60)
        ids = {store.put({"i": i}) for i in range(50)}
        assert len(ids) == 50


class TestTraceStoreTTL:
    def test_expired_entry_returns_none_and_is_evicted(self):
        clock = FakeClock(t=1000.0)
        store = TraceStore(ttl_seconds=60, clock=clock)
        tid = store.put({"x": 1})
        clock.t = 1061.0
        assert store.get(tid) is None
        assert len(store._traces) == 0

    def test_entry_alive_just_before_expiry(self):
        clock = FakeClock(t=1000.0)
        store = TraceStore(ttl_seconds=60, clock=clock)
        tid = store.put({"x": 1})
        clock.t = 1059.999
        assert store.get(tid) == {"x": 1}

    def test_expiry_at_exact_boundary_is_dead(self):
        clock = FakeClock(t=1000.0)
        store = TraceStore(ttl_seconds=60, clock=clock)
        tid = store.put({"x": 1})
        clock.t = 1060.0
        assert store.get(tid) is None

    def test_put_evicts_expired_entries(self):
        clock = FakeClock(t=0.0)
        store = TraceStore(ttl_seconds=10, clock=clock)
        store.put({"a": 1})
        store.put({"b": 2})
        clock.t = 11.0
        store.put({"c": 3})
        assert len(store._traces) == 1


class TestTraceStoreMaxEntries:
    def test_overflow_evicts_soonest_to_expire(self):
        clock = FakeClock(t=0.0)
        store = TraceStore(ttl_seconds=100, max_entries=2, clock=clock)
        clock.t = 0.0
        first = store.put({"i": 0})  # expires at 100
        clock.t = 1.0
        second = store.put({"i": 1})  # expires at 101
        clock.t = 2.0
        store.put({"i": 2})  # expires at 102, triggers overflow
        assert store.get(first) is None
        assert store.get(second) == {"i": 1}
        assert len(store._traces) == 2


class TestTraceStoreEnv:
    def test_ttl_env_override(self, monkeypatch):
        monkeypatch.setenv("HUGINN_TRACE_TTL_SECONDS", "42")
        store = TraceStore()
        assert store.ttl_seconds == 42

    def test_ttl_env_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HUGINN_TRACE_TTL_SECONDS", "not-a-number")
        store = TraceStore()
        assert store.ttl_seconds == 300

    def test_ttl_env_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HUGINN_TRACE_TTL_SECONDS", "-5")
        store = TraceStore()
        assert store.ttl_seconds == 300


class TestDefaultStore:
    def test_default_is_singleton(self):
        assert default_trace_store() is default_trace_store()
