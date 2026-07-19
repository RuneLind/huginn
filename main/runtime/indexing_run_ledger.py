"""Durable per-collection ledger of indexing runs.

One JSONL file per collection under ``data/state/runs/<collection>.jsonl``, plus a
sidecar ``<collection>.lock``. Records describe one reindex/tagging run: when it
started, how long it took, which phases ran and whether they succeeded.

Why a bespoke store rather than the ``DiskPersister`` the rest of the server uses:
that persister is constructed with ``base_path="./data/collections"``
(knowledge_store.py) so ``data/state/`` is only reachable through ``../``
traversal, and it has no append primitive at all.

Concurrency contract (load-bearing, see the module tests):

* Every append AND every compaction takes ``fcntl.flock`` LOCK_EX on the sidecar
  lockfile. flock is advisory — it excludes only other lock takers — so locking
  compaction alone would protect nothing.
* The lock is taken BEFORE the JSONL is opened, and the data fd is never cached
  across calls. Compaction swaps the JSONL inode with ``os.replace``; an appender
  that opened its fd first would write into the unlinked old inode and its record
  would vanish with no error. The natural ordering (open, then lock) is the wrong
  one.
* Compaction is atomic: temp file in the same directory, ``os.replace``, fsync the
  directory — the pattern ``DiskPersister.__atomic_write`` already uses. Never
  truncate in place; a crash mid-rewrite would take the whole ledger with it.

Records sharing a ``runId`` are folded at READ time. PR 2 adds a second writer
(the shell scripts) that appends its own partial record for the same run, so a
run is generally spread over several lines.
"""
import fcntl
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = "./data/state/runs"

# Folded runs kept per collection. The cap counts FOLDED RUNS, never raw lines,
# and compaction only ever trims at runId-group boundaries — splitting a group
# would leave an orphan record reporting a short reindex with its tagging phase
# missing, which is the exact failure mode this ledger exists to make visible.
MAX_RUNS_PER_COLLECTION = 500

# Compact once the raw line count exceeds this. Slack above the cap keeps the
# common append-only path from rewriting the file on every read.
COMPACT_LINE_THRESHOLD = MAX_RUNS_PER_COLLECTION * 3

# Per-record size guard. Oversized phase detail payloads are truncated (and
# flagged on the phase); a record is never dropped.
MAX_RECORD_BYTES = 64 * 1024

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9._-]+$")

VALID_TRIGGERS = ("scheduled", "manual", "cli", "unknown")
VALID_VARIANTS = ("incremental", "rebuild")
VALID_STATUSES = ("succeeded", "degraded", "failed", "unknown", "running",
                  "incomplete", "skipped")

# A run whose opening record never got a matching closing record from the same
# writer is `running` until this age, then `incomplete`. Well clear of the
# slowest observed job (mimir, ~76 min) and of POLL_TIMEOUT (3600s), and well
# inside the daily cadence, so a genuinely in-flight run is never mislabelled.
INCOMPLETE_AFTER_SECONDS = 6 * 3600


class InvalidCollectionName(ValueError):
    """Raised for a collection name that must not be turned into a path."""


def validate_collection(collection):
    """Validate a collection name BEFORE it is used to build any path.

    With PR 2 this value arrives in a POST body, so a name like
    ``../../collections/mimir/manifest`` must be rejected outright rather than
    sanitised — same posture as the realpath check in ``main/routes/collections.py``
    for user-supplied document paths.
    """
    if not isinstance(collection, str) or not _COLLECTION_RE.match(collection):
        raise InvalidCollectionName(f"Invalid collection name: {collection!r}")
    if collection in (".", ".."):
        raise InvalidCollectionName(f"Invalid collection name: {collection!r}")
    return collection


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mint_run_id(collection, started_at=None):
    return f"{collection}-{started_at or now_iso()}"


def _parse_ts(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def rollup_status(phases):
    """Run-level status from phase statuses.

    ``failed`` if any phase marked ``fatal`` failed, ``degraded`` if any
    non-fatal phase failed (or any phase is itself degraded or unknown),
    ``skipped`` if every phase was skipped, else ``succeeded``.

    ``skipped`` is deliberately NOT a degradation. A reindex skipped on 409
    means another process is already doing that exact work — expected several
    times a day for an hourly job — and alarming on it would train the reader to
    ignore `degraded`. It just must not read as `succeeded`, which would assert
    an index freshness the run did not deliver.
    """
    if not phases:
        return "unknown"
    degraded = False
    skipped = 0
    for phase in phases:
        status = phase.get("status")
        if status == "failed":
            if phase.get("fatal"):
                return "failed"
            degraded = True
        elif status in ("degraded", "unknown"):
            degraded = True
        elif status == "skipped":
            skipped += 1
        elif status != "succeeded":
            # A phase with no status, or one this version does not know, is not
            # evidence of success. Say so rather than rounding it up.
            degraded = True
    if degraded:
        return "degraded"
    return "skipped" if skipped == len(phases) else "succeeded"


class IndexingRunLedger:
    """Append-only run ledger. Constructible without a loaded KnowledgeStore."""

    def __init__(self, runs_dir=None):
        # Resolved per-instance rather than at import so HUGINN_RUNS_DIR can be
        # set late (the test suite points it at a tmp dir so a reindex test does
        # not append to the real ledger — same posture as HUGINN_QUERY_LOG).
        self.runs_dir = runs_dir or os.environ.get("HUGINN_RUNS_DIR") or DEFAULT_RUNS_DIR

    # ---------------------------------------------------------------- paths

    def _ensure_dir(self):
        os.makedirs(self.runs_dir, exist_ok=True)

    def path_for(self, collection):
        validate_collection(collection)
        return os.path.join(self.runs_dir, f"{collection}.jsonl")

    def _lock_path(self, collection):
        validate_collection(collection)
        return os.path.join(self.runs_dir, f"{collection}.lock")

    def collections(self):
        """Collections that have a ledger file, from the file names on disk."""
        try:
            names = os.listdir(self.runs_dir)
        except OSError:
            return []
        found = []
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            base = name[: -len(".jsonl")]
            try:
                validate_collection(base)
            except InvalidCollectionName:
                continue
            found.append(base)
        return sorted(found)

    # ---------------------------------------------------------------- write

    def append(self, record):
        """Append one run record. Returns the normalized record that was written."""
        record = self._normalize(record)
        collection = record["collection"]
        self._ensure_dir()
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if len(line) > MAX_RECORD_BYTES:
            record = self._truncate_details(record)
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")

        with self._locked(collection):
            # Open only after the lock is held — see the module docstring.
            fd = os.open(
                self.path_for(collection),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o644,
            )
            try:
                written = 0
                while written < len(line):
                    # os.write is allowed to write short; loop rather than
                    # discarding the tail. Buffered text-mode open() can split a
                    # large write across the append boundary, so it is not used.
                    written += os.write(fd, line[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
        return record

    def _normalize(self, record):
        record = dict(record or {})
        collection = record.get("collection")
        validate_collection(collection)

        record.setdefault("runId", mint_run_id(collection, record.get("startedAt")))
        record.setdefault("job", None)
        trigger = record.get("trigger") or "unknown"
        record["trigger"] = trigger if trigger in VALID_TRIGGERS else "unknown"
        variant = record.get("variant") or "incremental"
        record["variant"] = variant if variant in VALID_VARIANTS else "incremental"
        record.setdefault("startedAt", None)
        record.setdefault("finishedAt", None)
        record.setdefault("error", None)

        phases = record.get("phases") or []
        record["phases"] = [dict(p) for p in phases if isinstance(p, dict)]

        status = record.get("status")
        if status not in VALID_STATUSES:
            status = rollup_status(record["phases"])
        record["status"] = status

        if record.get("durationSeconds") is None:
            record["durationSeconds"] = _duration(
                record.get("startedAt"), record.get("finishedAt")
            )

        # documentCount/chunkCount are meaningless for a failed run: the failure
        # path never rewrote the manifest, and collection creation removes the
        # folder outright when it reads zero documents.
        if record["status"] == "failed":
            record["documentCount"] = None
            record["chunkCount"] = None
        else:
            record.setdefault("documentCount", None)
            record.setdefault("chunkCount", None)
        return record

    @staticmethod
    def _truncate_details(record):
        """Drop oversized phase detail payloads rather than dropping the record."""
        record = dict(record)
        phases = []
        for phase in record.get("phases") or []:
            phase = dict(phase)
            detail = phase.get("detail")
            if detail is not None and len(json.dumps(detail, ensure_ascii=False)) > 2048:
                phase["detail"] = None
                phase["detailTruncated"] = True
            phases.append(phase)
        record["phases"] = phases
        if len(json.dumps(record, ensure_ascii=False)) > MAX_RECORD_BYTES:
            record["error"] = (record.get("error") or "")[:2048] or None
        return record

    # ----------------------------------------------------------------- read

    def recent(self, collection, limit=50):
        """Folded runs for a collection, oldest→newest, at most ``limit``."""
        validate_collection(collection)
        runs = self._read_folded(collection)
        if limit is not None and limit >= 0:
            runs = runs[-limit:]
        return runs

    def all_recent(self, limit_per_collection=50):
        return {c: self.recent(c, limit_per_collection) for c in self.collections()}

    def _read_folded(self, collection):
        raw, line_count = self._read_raw(collection)
        runs = fold_records(raw)
        if line_count > COMPACT_LINE_THRESHOLD or len(runs) > MAX_RUNS_PER_COLLECTION:
            runs = runs[-MAX_RUNS_PER_COLLECTION:]
            try:
                self._compact(collection, runs)
            except OSError:
                logger.warning("Compaction failed for ledger %s", collection, exc_info=True)
        return runs

    def _read_raw(self, collection):
        path = self.path_for(collection)
        records = []
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    count += 1
                    try:
                        parsed = json.loads(line)
                    except ValueError:
                        # Partial/corrupt line (torn write, manual edit) — skip it
                        # and keep reading; one bad line must not hide the ledger.
                        continue
                    if isinstance(parsed, dict) and parsed.get("collection") == collection:
                        records.append(parsed)
        except FileNotFoundError:
            return [], 0
        return records, count

    # ----------------------------------------------------------- compaction

    def _before_replace_hook(self):
        """Test seam: raised-from here simulates a crash between temp write and
        replace, which must leave the previous ledger fully intact."""

    def _compact(self, collection, runs):
        self._ensure_dir()
        path = self.path_for(collection)
        with self._locked(collection):
            directory = os.path.dirname(os.path.abspath(path))
            fd, temp = tempfile.mkstemp(dir=directory, prefix=".tmp_runs_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for run in runs:
                        handle.write(json.dumps(run, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._before_replace_hook()
                os.replace(temp, path)
            except BaseException:
                if os.path.exists(temp):
                    os.remove(temp)
                raise
            _fsync_dir(directory)

    # ---------------------------------------------------------------- locks

    class _Lock:
        def __init__(self, path):
            self.path = path
            self.fd = None

        def __enter__(self):
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
            return False

    def _locked(self, collection):
        self._ensure_dir()
        return self._Lock(self._lock_path(collection))


def _fsync_dir(directory):
    if not directory:
        return
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        # not all platforms allow fsync on a directory handle
        pass
    finally:
        os.close(dir_fd)


def _duration(started_at, finished_at):
    start, finish = _parse_ts(started_at), _parse_ts(finished_at)
    if start is None or finish is None:
        return None
    return max(0, int((finish - start).total_seconds()))


def fold_records(records, now=None):
    """Fold records sharing a ``runId`` into one run each, oldest group first.

    Scalar precedence when two partials collide: job/trigger/variant come from the
    script-side record (PR 2 knows the launchd job and how it was triggered),
    documentCount/chunkCount from huginn's record (only huginn reads the
    manifest), errors are concatenated, and status is always recomputed from the
    folded phase list rather than taken from either side.
    """
    now = now or datetime.now(timezone.utc)
    order = []
    groups = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        run_id = record.get("runId") or f"__anon_{index}"
        if run_id not in groups:
            groups[run_id] = []
            order.append(run_id)
        groups[run_id].append(record)
    return [_mark_unclosed(_fold_group(run_id, groups[run_id]), groups[run_id], now)
            for run_id in order]


def _mark_unclosed(folded, records, now):
    """Downgrade a run whose writer opened it but never closed it.

    Each writer that appends an opening partial (``stage: "begin"``) is expected
    to append a matching ``stage: "end"``. If the script is killed after the
    reindex finished but before ``run_end``, the group still holds huginn's
    complete reindex record — and folding it naively would report a tidy
    15-minute success for a job that actually blocked for 76 and then died. So
    the missing closer wins over the surviving partial, and the duration is
    dropped rather than published as if it were the whole run.
    """
    begun = {r.get("source") for r in records if r.get("stage") == "begin"}
    # Compaction rewrites folded runs, which drops the per-record stage markers.
    # Carrying the unclosed set on the folded record keeps this re-derivable, so
    # a run compacted while still open is re-evaluated on later reads (and
    # properly closed by a late-arriving `end`) rather than frozen as `running`.
    for record in records:
        begun.update(record.get("unclosedSources") or [])
    ended = {r.get("source") for r in records if r.get("stage") == "end"}
    unclosed = sorted(s for s in begun - ended if s)
    if not unclosed:
        return folded
    folded["unclosedSources"] = unclosed
    started = _parse_ts(folded.get("startedAt"))
    age = (now - started).total_seconds() if started else None
    folded["status"] = "running" if age is not None and age < INCOMPLETE_AFTER_SECONDS \
        else "incomplete"
    folded["durationSeconds"] = None
    return folded


def _fold_group(run_id, records):
    if len(records) == 1:
        folded = dict(records[0])
        folded["runId"] = run_id
        return folded

    folded = {"runId": run_id, "collection": records[0].get("collection")}

    # Script-side wins: it is the process that knows the job label and trigger.
    for field in ("job", "trigger"):
        value = None
        for record in records:
            candidate = record.get(field)
            if candidate is None:
                continue
            value = candidate
            if record.get("source") == "script":
                break
        folded[field] = value
    folded.setdefault("trigger", None)

    # `variant` is script-side too, but its OPENING record only carries a guess.
    # x-feed cannot know whether it is doing an incremental update or a full
    # rebuild until cleanup has run and reported what it deleted, so the closing
    # record is the reclassified one and must outrank the partial that opened
    # the run — otherwise every rebuild medians in with the incrementals it is
    # an order of magnitude slower than.
    folded["variant"] = _pick_variant(records)

    # huginn-side wins: only huginn reads the collection manifest.
    for field in ("documentCount", "chunkCount"):
        value = None
        for record in records:
            candidate = record.get(field)
            if candidate is None:
                continue
            value = candidate
            if record.get("source") != "script":
                break
        folded[field] = value

    starts = [t for t in (_parse_ts(r.get("startedAt")) for r in records) if t]
    finishes = [t for t in (_parse_ts(r.get("finishedAt")) for r in records) if t]
    folded["startedAt"] = _iso(min(starts)) if starts else None
    folded["finishedAt"] = _iso(max(finishes)) if finishes else None
    folded["durationSeconds"] = _duration(folded["startedAt"], folded["finishedAt"])

    phases = _merge_phases(records)
    folded["phases"] = phases
    folded["status"] = rollup_status(phases) if phases else _worst_status(records)

    errors = [r.get("error") for r in records if r.get("error")]
    folded["error"] = "; ".join(errors) if errors else None

    if any(r.get("backfilled") for r in records):
        folded["backfilled"] = True
    return folded


def _pick_variant(records):
    """Best `variant` in the group: a script's closing word beats its opening
    guess, and either beats huginn's (which only ever infers)."""
    def rank(record):
        if record.get("source") != "script":
            return 0
        return 1 if record.get("stage") == "begin" else 2

    best, best_rank = None, -1
    for record in records:
        value = record.get("variant")
        if value is None:
            continue
        if rank(record) >= best_rank:
            best, best_rank = value, rank(record)
    return best


def _merge_phases(records):
    """Union the phase lists, but only ONE phase per name.

    Both sides legitimately report a `reindex` phase for the same work: huginn
    times the rebuild, the script times trigger-plus-poll around it. Appending
    both leaves a phase list whose durations sum to more than the run itself (a
    real 26s run folded to 13s + 16s of "reindex"). huginn's copy wins where it
    exists — it measures the rebuild rather than the wait for it — and the
    script's survives on the API-down path where huginn never wrote a record.
    """
    merged = {}
    order = []
    for record in records:
        from_huginn = record.get("source") != "script"
        for phase in record.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            name = phase.get("name")
            if name is None:
                order.append(len(merged))
                merged[len(merged)] = (phase, from_huginn)
                continue
            if name not in merged:
                order.append(name)
                merged[name] = (phase, from_huginn)
                continue
            existing, existing_from_huginn = merged[name]
            prefer_new = from_huginn and not existing_from_huginn
            winner, loser = (phase, existing) if prefer_new else (existing, phase)
            # Duration and detail come from the winner, but the STATUS is the
            # worse of the two. huginn reporting a clean rebuild must not erase
            # the script's report that its own wait around that rebuild failed —
            # in a ledger built to surface failures, a disagreement resolves
            # pessimistically.
            winner = dict(winner)
            winner["status"] = _worse_phase_status(existing.get("status"),
                                                   phase.get("status"))
            # Preferring huginn's copy must not COST information. The CLI
            # adapter writes a reindex phase carrying no duration, so on the
            # API-down path a blind preference dropped the script's measured
            # duration and left the phase timeless inside a timed run.
            for field in ("durationSeconds", "detail"):
                if winner.get(field) is None and loser.get(field) is not None:
                    winner[field] = loser[field]
            merged[name] = (winner, existing_from_huginn or from_huginn)
    return [merged[key][0] for key in order]


# Worst first. `skipped` sits just above `succeeded`: it did less work, so it
# wins a disagreement against a copy claiming success, but it is not a fault.
_PHASE_STATUS_ORDER = ("failed", "degraded", "unknown", "skipped", "succeeded")


def _worse_phase_status(*statuses):
    for candidate in _PHASE_STATUS_ORDER:
        if candidate in statuses:
            return candidate
    return next((s for s in statuses if s), None)


def _worst_status(records):
    statuses = [r.get("status") for r in records]
    for candidate in ("failed", "degraded", "unknown"):
        if candidate in statuses:
            return candidate
    return "succeeded" if statuses else "unknown"


def _iso(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    """CLI: ``python -m main.runtime.indexing_run_ledger append --file -``."""
    import argparse

    parser = argparse.ArgumentParser(description="Indexing run ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    appender = sub.add_parser("append", help="Append a JSON run record")
    appender.add_argument("--file", default="-", help="JSON file, or '-' for stdin")
    # Default None, not DEFAULT_RUNS_DIR: an explicit value would shadow
    # HUGINN_RUNS_DIR, and this CLI is how the shell fallback writes.
    appender.add_argument("--runs-dir", default=None)
    args = parser.parse_args(argv)

    payload = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    record = json.loads(payload)
    ledger = IndexingRunLedger(runs_dir=args.runs_dir)
    try:
        written = ledger.append(record)
    except InvalidCollectionName as e:
        print(str(e), file=sys.stderr)
        return 2
    print(written["runId"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
