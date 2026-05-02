"""In-memory TTL store for search traces.

Pointer pattern: when an MCP adapter would otherwise embed the full trace JSON
in the tool result text, it instead stores the trace here, gets a short ID
back, and emits only `huginn-trace-url: <url>` in the tool result. The
orchestrator (Muninn) fetches the full trace via `GET /api/trace/<id>` after
the tool call returns. Keeps tool-result text under MCP-stdio output-size
limits.

IDs are 16 hex chars (8 bytes from `secrets.token_hex`) — collision-free within
a TTL window without the bulk of full UUIDs.
"""

import os
import secrets
import threading
import time

from main.utils.env import env_bool

TRACE_POINTER_ENV = "HUGINN_TRACE_POINTER"
TRACE_DEFAULT_ENV = "HUGINN_TRACE_DEFAULT"

DEFAULT_MAX_ENTRIES = 10_000


def pointer_mode_enabled():
    return env_bool(TRACE_POINTER_ENV)


def any_trace_enabled():
    """True if either trace env flag is set; pointer mode implies tracing on."""
    return env_bool(TRACE_POINTER_ENV) or env_bool(TRACE_DEFAULT_ENV)


def _ttl_from_env(default=300):
    raw = os.environ.get("HUGINN_TRACE_TTL_SECONDS")
    if raw is None:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


class TraceStore:
    def __init__(self, ttl_seconds=None, max_entries=DEFAULT_MAX_ENTRIES, clock=None):
        self._ttl = ttl_seconds if ttl_seconds is not None else _ttl_from_env()
        self._max_entries = max_entries
        self._clock = clock or time.monotonic
        self._traces = {}
        self._lock = threading.Lock()

    @property
    def ttl_seconds(self):
        return self._ttl

    def put(self, trace_dict):
        trace_id = secrets.token_hex(8)
        expires_at = self._clock() + self._ttl
        with self._lock:
            self._gc_locked()
            # Belt-and-suspenders: if puts arrive faster than fetches/expiry drain
            # them, evict the soonest-to-expire entry so memory cannot grow without
            # bound. Hits the put-only path; never affects normal pointer-fetch flow.
            if len(self._traces) >= self._max_entries:
                oldest = min(self._traces, key=lambda k: self._traces[k][1])
                del self._traces[oldest]
            self._traces[trace_id] = (trace_dict, expires_at)
        return trace_id

    def get(self, trace_id):
        now = self._clock()
        with self._lock:
            entry = self._traces.get(trace_id)
            if entry is None:
                return None
            trace_dict, expires_at = entry
            if expires_at <= now:
                del self._traces[trace_id]
                return None
            return trace_dict

    def _gc_locked(self):
        now = self._clock()
        expired = [k for k, (_, exp) in self._traces.items() if exp <= now]
        for k in expired:
            del self._traces[k]


_default_store = TraceStore()


def default_trace_store():
    return _default_store
