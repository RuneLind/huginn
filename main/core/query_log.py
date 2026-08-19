"""Persistent per-query search log (JSONL, append-only).

Written once per request at the ``run_search_request`` seam, so every
transport — the HTTP route and both MCP stdio adapters — produces the same
record. Exists to answer "which collections / queries / documents actually get
retrieved": per-collection usage was previously unobservable (the trace store
is an in-memory TTL, and uvicorn access lines die with the tty the server
runs on).

Config is env-only and read live per request — the ``HUGINN_TRACE_*``
precedent. ``ServerConfig`` deliberately doesn't own per-request observability
flags, and the MCP stdio adapters have no ``ServerConfig`` at all:

    HUGINN_QUERY_LOG            unset → ``logs/query-log.jsonl`` under the repo root
    HUGINN_QUERY_LOG=off        disable ("off" / "0" / "false")
    HUGINN_QUERY_LOG=<path>     custom log file (parent dirs created)
    HUGINN_QUERY_LOG_MAX_BYTES  rotation threshold, default 10 MiB; ``0`` disables
                                rotation; blank, invalid, or negative falls back
                                to the default (never to "unbounded")

At/over the threshold the log is renamed to ``<path>.1`` before the append, so
the file is bounded at two generations — the previous ``.1`` is overwritten.
Retained history therefore oscillates between 1× the bound (right after a
rotation, when the live file is empty) and 2× (just before the next one).
Measured 2026-08-19 over the live log — 1 534 995 B / 3 531 records across 43
days — ~434 B/query at ~82 queries/day, so 10 MiB ≈ 24k queries ≈ 10 months at
the observed rate.

Rotation is serialized on a sibling ``<path>.lock`` because three processes
share this file (HTTP server + two MCP stdio adapters): unsynchronized, two
writers acting on the same over-threshold reading both rename, and the second
overwrites ``.1`` with the fresh near-empty log — losing the retained
generation outright. Same lock-before-stat ordering as the run ledger
(``main/runtime/indexing_run_ledger.py``).

Logging must never break a search: every failure is swallowed.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY_LOG_PATH = _REPO_ROOT / "logs" / "query-log.jsonl"
_DISABLED_VALUES = {"off", "0", "false"}
_MAX_QUERY_CHARS = 300
_DEFAULT_MAX_LOG_BYTES = 10 * 1024 * 1024


def _resolve_path() -> Path | None:
    """The log path, or ``None`` when logging is disabled."""
    value = os.environ.get("HUGINN_QUERY_LOG", "").strip()
    if value.lower() in _DISABLED_VALUES:
        return None
    if value:
        return Path(value)
    return DEFAULT_QUERY_LOG_PATH


def _resolve_max_bytes() -> int:
    """Rotation threshold in bytes; ``0`` means never rotate."""
    value = os.environ.get("HUGINN_QUERY_LOG_MAX_BYTES", "").strip()
    try:
        parsed = int(value)
    except ValueError:
        return _DEFAULT_MAX_LOG_BYTES  # blank or garbage: keep the bound, don't disable it
    return parsed if parsed >= 0 else _DEFAULT_MAX_LOG_BYTES


@contextmanager
def _rotation_lock(path: Path):
    """``LOCK_EX`` on ``<path>.lock``, taken before the size is read."""
    fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _rotate_if_oversized(path: Path) -> None:
    """Rename an at-threshold log to ``<path>.1``. Call with the lock held."""
    max_bytes = _resolve_max_bytes()
    if max_bytes == 0:
        return
    try:
        size = os.stat(path).st_size
    except OSError:  # no log yet, or unreadable: nothing to rotate
        return
    if size >= max_bytes:
        os.replace(path, path.with_name(path.name + ".1"))


def _append(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def log_search_request(*, collections, query, response) -> None:
    """Append one JSONL record for a completed search request. Never raises."""
    try:
        path = _resolve_path()
        if path is None:
            return
        results = response.get("results") or []
        top = results[0] if results else {}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collections": list(collections),
            "query": str(query)[:_MAX_QUERY_CHARS],
            "resultCount": len(results),
            "bestScore": top.get("relevance"),
            "topDoc": top.get("id") or top.get("path") or top.get("url"),
            "lowConfidence": bool(response.get("lowConfidence")),
            # run_corrective_search only adds a ``corrective`` dict on rescue
            "rescued": "corrective" in response,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        appended = False
        try:
            with _rotation_lock(path):
                _rotate_if_oversized(path)
                _append(path, line)
                appended = True
        except Exception:
            # An unlockable or unrotatable log still takes appends; losing the
            # record too would trade a size bound for data loss.
            pass
        if not appended:
            _append(path, line)
    except Exception:
        # Observability must not break search; drop the record.
        pass
