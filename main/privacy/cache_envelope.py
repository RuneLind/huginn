"""The on-disk envelope every privacy-adjacent LLM cache is written in.

    {"<metadata key>": …, "entries": {"<doc id>": {…}}}

Two callers today — ``scripts/knowledge_graph/extract_entities_llm.py`` and
``main/privacy/sensitivity_sweep.py`` — and they had two copies of the same
three decisions:

* **an envelope, not a flat dict.** A document id may itself start with ``_``,
  so a ``_policy_version`` sibling key beside the entries is not safely
  distinguishable from an entry. The metadata lives one level up instead.
* **the write is atomic.** Both caches are rewritten every N documents during a
  run that takes minutes; a crash mid-``write_text`` used to leave a truncated
  JSON file, which the next run reads as a cold cache and re-asks the model
  about the whole collection. Temp file in the same directory plus
  ``os.replace``.
* **the lock is taken BEFORE the file is opened.** The same rule
  ``main/runtime/indexing_run_ledger.py`` documents at length: ``os.replace``
  swaps the inode, so a handle opened before the lock is a handle on an
  unlinked file. Here it also means a reader never sees a half-written cache
  even on a filesystem where ``os.replace`` is not the only writer.

The metadata is NOT interpreted here. Each caller decides what a mismatch
means — the extractor keeps a legacy flat cache for an out-of-scope collection,
the sweep discards wholesale — and a module that made that decision for them
would have to know about privacy scope to be right.
"""
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

ENTRIES_KEY = "entries"


@contextmanager
def _locked(path: Path):
    """``flock`` on a sidecar lockfile, held across the whole read or write.

    A sidecar rather than the data file itself: the data file is replaced by
    inode swap, so a lock on it protects nothing after the first write.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_envelope(path) -> tuple[dict, dict]:
    """``(metadata, entries)`` from a cache file.

    ``({}, {})`` for a missing, unreadable or non-object file — a cold cache is
    always a valid answer here, and raising would turn a truncated cache into a
    crashed nightly job.

    A **legacy flat** file (entries at the top level, no ``entries`` key) comes
    back as ``({}, <the file>)``: empty metadata, so a caller that requires a
    metadata match rejects it, and a caller that accepts pre-envelope caches
    (the graph extractor, for its ~30 out-of-scope collections) can take it.
    """
    path = Path(path)
    if not path.exists():
        return {}, {}
    with _locked(path):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, {}
    if not isinstance(raw, dict):
        return {}, {}
    entries = raw.get(ENTRIES_KEY)
    if isinstance(entries, dict):
        return {k: v for k, v in raw.items() if k != ENTRIES_KEY}, entries
    return {}, raw


def write_envelope(path, metadata: dict, entries: dict) -> None:
    """Write ``{**metadata, "entries": entries}`` atomically, under the lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({**metadata, ENTRIES_KEY: entries}, ensure_ascii=False, separators=(',', ':'))
    with _locked(path):
        temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
