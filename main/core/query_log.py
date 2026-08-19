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
                                rotation; blank/invalid falls back to the default

At/over the threshold the log is renamed to ``<path>.1`` before the append, so
the file is bounded at two generations — the previous ``.1`` is overwritten.
Measured 2026-08-19: ~434 B/query, so 10 MiB ≈ 24k queries ≈ 7 years at the
observed rate.

Logging must never break a search: every failure is swallowed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY_LOG_PATH = _REPO_ROOT / "logs" / "query-log.jsonl"
_DISABLED_VALUES = {"off", "0", "false"}
_MAX_QUERY_CHARS = 300
_MAX_LOG_BYTES = 10 * 1024 * 1024


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
        return _MAX_LOG_BYTES  # blank or garbage: keep the bound, don't disable it
    return parsed if parsed >= 0 else _MAX_LOG_BYTES


def _rotate_if_oversized(path: Path) -> None:
    """Rename an at-threshold log to ``<path>.1``, keeping one generation."""
    max_bytes = _resolve_max_bytes()
    if max_bytes == 0:
        return
    if path.exists() and path.stat().st_size >= max_bytes:
        os.replace(path, path.with_suffix(path.suffix + ".1"))


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
        try:
            _rotate_if_oversized(path)
        except Exception:
            # An unrotatable log still takes appends; losing the record too
            # would trade a size bound for a data loss.
            pass
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Observability must not break search; drop the record.
        pass
