"""Tests for the durable indexing run ledger.

The concurrency tests here use real subprocesses rather than threads: the
correctness property under test is about ``fcntl.flock`` and file descriptors,
and flock is per-open-file-description, so threads in one process would share
locks and prove nothing.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from main.runtime.indexing_run_ledger import (
    IndexingRunLedger,
    InvalidCollectionName,
    MAX_RUNS_PER_COLLECTION,
    fold_records,
    rollup_status,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _record(collection="c", run_id=None, **extra):
    record = {
        "collection": collection,
        "runId": run_id or f"{collection}-run",
        "startedAt": "2026-07-18T09:00:00Z",
        "finishedAt": "2026-07-18T09:10:00Z",
        "status": "succeeded",
        "phases": [{"name": "reindex", "status": "succeeded", "fatal": True}],
    }
    record.update(extra)
    return record


@pytest.fixture
def ledger(tmp_path):
    return IndexingRunLedger(runs_dir=str(tmp_path / "runs"))


class TestAppendAndRead:
    def test_append_then_read_roundtrips(self, ledger):
        ledger.append(_record(run_id="r1"))
        runs = ledger.recent("c", limit=10)
        assert len(runs) == 1
        assert runs[0]["runId"] == "r1"
        assert runs[0]["durationSeconds"] == 600

    def test_reads_are_oldest_first_and_limited_to_the_newest(self, ledger):
        for i in range(5):
            ledger.append(_record(run_id=f"r{i}"))
        runs = ledger.recent("c", limit=2)
        assert [r["runId"] for r in runs] == ["r3", "r4"]

    def test_unknown_collection_reads_empty(self, ledger):
        assert ledger.recent("never-written", limit=10) == []

    def test_failed_runs_carry_no_document_counts(self, ledger):
        written = ledger.append(
            _record(run_id="r1", status="failed", documentCount=99, chunkCount=99)
        )
        # A failed run never rewrote the manifest, so any count is stale by
        # construction and must not be published.
        assert written["documentCount"] is None
        assert written["chunkCount"] is None

    def test_missing_manifest_counts_tolerated_as_null(self, ledger):
        written = ledger.append(_record(run_id="r1"))
        assert written["documentCount"] is None

    def test_collections_lists_ledger_files(self, ledger):
        ledger.append(_record(collection="a"))
        ledger.append(_record(collection="b"))
        assert ledger.collections() == ["a", "b"]

    def test_all_recent_covers_every_collection(self, ledger):
        ledger.append(_record(collection="a", run_id="a1"))
        ledger.append(_record(collection="b", run_id="b1"))
        assert set(ledger.all_recent(10)) == {"a", "b"}


class TestCorruptLines:
    def test_corrupt_lines_are_skipped_not_fatal(self, ledger):
        ledger.append(_record(run_id="r1"))
        with open(ledger.path_for("c"), "a", encoding="utf-8") as handle:
            handle.write('{"collection": "c", "runId": "torn"\n')  # truncated write
            handle.write("not json at all\n")
        ledger.append(_record(run_id="r2"))
        assert [r["runId"] for r in ledger.recent("c", limit=10)] == ["r1", "r2"]

    def test_records_for_another_collection_are_ignored(self, ledger):
        ledger.append(_record(run_id="r1"))
        with open(ledger.path_for("c"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"collection": "other", "runId": "x"}) + "\n")
        assert [r["runId"] for r in ledger.recent("c", limit=10)] == ["r1"]


class TestPathTraversal:
    @pytest.mark.parametrize("name", [
        "../../x",
        "../../collections/mimir/manifest",
        "..",
        ".",
        "a/b",
        "a\x00b",
        "",
        None,
    ])
    def test_traversal_names_are_rejected(self, ledger, name):
        with pytest.raises(InvalidCollectionName):
            ledger.append(_record(collection=name))

    def test_nothing_is_written_outside_the_runs_dir(self, tmp_path):
        runs_dir = tmp_path / "runs"
        ledger = IndexingRunLedger(runs_dir=str(runs_dir))
        with pytest.raises(InvalidCollectionName):
            ledger.append(_record(collection="../escaped"))
        assert not (tmp_path / "escaped.jsonl").exists()
        assert list(runs_dir.glob("*")) == [] or not runs_dir.exists()

    def test_path_and_lock_helpers_validate_too(self, ledger):
        with pytest.raises(InvalidCollectionName):
            ledger.path_for("../x")


class TestFolding:
    def test_two_partials_sharing_a_run_id_fold_into_one_run(self, ledger):
        ledger.append({
            "collection": "c", "runId": "shared", "source": "script",
            "job": "com.huginn.mimir-index", "trigger": "scheduled",
            "startedAt": "2026-07-18T09:28:37Z", "finishedAt": "2026-07-18T10:19:10Z",
            "phases": [{"name": "tag", "status": "degraded", "durationSeconds": 3033}],
        })
        ledger.append({
            "collection": "c", "runId": "shared", "source": "huginn",
            "trigger": "manual", "documentCount": 224, "chunkCount": 1893,
            "startedAt": "2026-07-18T10:19:13Z", "finishedAt": "2026-07-18T10:44:14Z",
            "phases": [{"name": "reindex", "status": "succeeded", "fatal": True,
                        "durationSeconds": 1501}],
        })
        runs = ledger.recent("c", limit=10)
        assert len(runs) == 1
        run = runs[0]
        assert {p["name"] for p in run["phases"]} == {"tag", "reindex"}
        assert run["startedAt"] == "2026-07-18T09:28:37Z"
        assert run["finishedAt"] == "2026-07-18T10:44:14Z"
        assert run["durationSeconds"] == 4537
        # Script-side wins job/trigger; huginn-side wins the manifest counts.
        assert run["job"] == "com.huginn.mimir-index"
        assert run["trigger"] == "scheduled"
        assert run["documentCount"] == 224
        # A non-fatal phase failure degrades the run rather than failing it.
        assert run["status"] == "degraded"

    def test_fatal_phase_failure_fails_the_run(self):
        assert rollup_status([
            {"name": "tag", "status": "succeeded"},
            {"name": "reindex", "status": "failed", "fatal": True},
        ]) == "failed"

    def test_non_fatal_phase_failure_only_degrades(self):
        assert rollup_status([
            {"name": "tag", "status": "failed"},
            {"name": "reindex", "status": "succeeded", "fatal": True},
        ]) == "degraded"

    def test_all_succeeded_rolls_up_to_succeeded(self):
        assert rollup_status([{"name": "reindex", "status": "succeeded"}]) == "succeeded"

    def test_errors_from_both_sides_are_concatenated(self):
        folded = fold_records([
            {"collection": "c", "runId": "x", "error": "tag blew up"},
            {"collection": "c", "runId": "x", "error": "reindex blew up"},
        ])
        assert folded[0]["error"] == "tag blew up; reindex blew up"

    def test_records_without_a_run_id_stay_separate(self):
        folded = fold_records([
            {"collection": "c", "status": "succeeded"},
            {"collection": "c", "status": "succeeded"},
        ])
        assert len(folded) == 2


class TestCompaction:
    def test_retention_trims_to_the_cap_and_keeps_the_newest(self, ledger):
        total = MAX_RUNS_PER_COLLECTION + 25
        for i in range(total):
            ledger.append(_record(run_id=f"r{i:05d}"))
        runs = ledger.recent("c", limit=None)
        assert len(runs) == MAX_RUNS_PER_COLLECTION
        assert runs[-1]["runId"] == f"r{total - 1:05d}"

    def test_compaction_rewrites_the_file_not_just_the_view(self, ledger):
        for i in range(MAX_RUNS_PER_COLLECTION + 25):
            ledger.append(_record(run_id=f"r{i:05d}"))
        ledger.recent("c", limit=None)
        with open(ledger.path_for("c"), encoding="utf-8") as handle:
            assert sum(1 for _ in handle) == MAX_RUNS_PER_COLLECTION

    def test_the_cap_never_splits_a_fold_group(self, ledger):
        """A run's phases must survive the retention edge together.

        Trimming inside a runId group would leave an orphan reporting a 15-minute
        reindex with no tagging phase — the exact wrong answer this ledger exists
        to prevent.
        """
        total = MAX_RUNS_PER_COLLECTION + 40
        for i in range(total):
            run_id = f"r{i:05d}"
            ledger.append({"collection": "c", "runId": run_id, "source": "script",
                           "phases": [{"name": "tag", "status": "succeeded"}]})
            ledger.append({"collection": "c", "runId": run_id, "source": "huginn",
                           "phases": [{"name": "reindex", "status": "succeeded",
                                       "fatal": True}]})
        runs = ledger.recent("c", limit=None)
        assert len(runs) == MAX_RUNS_PER_COLLECTION
        for run in runs:
            assert {p["name"] for p in run["phases"]} == {"tag", "reindex"}, run["runId"]

    def test_interrupted_compaction_leaves_the_previous_ledger_intact(self, ledger):
        for i in range(MAX_RUNS_PER_COLLECTION + 25):
            ledger.append(_record(run_id=f"r{i:05d}"))
        before = open(ledger.path_for("c"), encoding="utf-8").read()

        def boom():
            raise OSError("crash between temp write and replace")

        ledger._before_replace_hook = boom
        # The read still succeeds: compaction is best-effort, never load-bearing.
        runs = ledger.recent("c", limit=None)
        assert len(runs) == MAX_RUNS_PER_COLLECTION

        after = open(ledger.path_for("c"), encoding="utf-8").read()
        assert after == before, "a crashed compaction must not touch the live ledger"
        # And no temp file is left behind to be mistaken for ledger content.
        leftovers = [p for p in os.listdir(ledger.runs_dir) if p.startswith(".tmp_runs_")]
        assert leftovers == []


def _spawn(script_body, runs_dir, count, tag, **env):
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script_body)],
        cwd=REPO_ROOT,
        env={**os.environ, "RUNS_DIR": runs_dir, "COUNT": str(count), "TAG": tag,
             "PYTHONPATH": REPO_ROOT, **{k: str(v) for k, v in env.items()}},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


APPENDER = """
    import os
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    ledger = IndexingRunLedger(runs_dir=os.environ["RUNS_DIR"])
    tag = os.environ["TAG"]
    for i in range(int(os.environ["COUNT"])):
        ledger.append({
            "collection": "c",
            "runId": f"{tag}-{i:04d}",
            "startedAt": "2026-07-18T09:00:00Z",
            "finishedAt": "2026-07-18T09:01:00Z",
            "status": "succeeded",
            "phases": [{"name": "reindex", "status": "succeeded", "fatal": True}],
            "padding": "x" * 400,
        })
"""

COMPACTOR = """
    import os
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    ledger = IndexingRunLedger(runs_dir=os.environ["RUNS_DIR"])
    for _ in range(int(os.environ["COUNT"])):
        ledger.recent("c", limit=None)
"""

# A compactor that holds the lock across the inode swap for a beat, so appenders
# are guaranteed to be parked on the lock at the moment os.replace() runs. Without
# this the race window is microseconds wide and the test passes against broken
# code by luck — verified by inverting the ordering in the module.
SLOW_COMPACTOR = """
    import os, time
    from main.runtime.indexing_run_ledger import IndexingRunLedger

    class SlowLedger(IndexingRunLedger):
        def _before_replace_hook(self):
            time.sleep(float(os.environ["HOLD_SECONDS"]))

    SlowLedger(runs_dir=os.environ["RUNS_DIR"]).recent("c", limit=None)
"""

# Appenders that wait for the compactor to be mid-swap before they start.
DELAYED_APPENDER = """
    import os, time
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    time.sleep(float(os.environ["DELAY_SECONDS"]))
    ledger = IndexingRunLedger(runs_dir=os.environ["RUNS_DIR"])
    tag = os.environ["TAG"]
    for i in range(int(os.environ["COUNT"])):
        ledger.append({
            "collection": "c",
            "runId": f"{tag}-{i:04d}",
            "startedAt": "2026-07-18T09:00:00Z",
            "finishedAt": "2026-07-18T09:01:00Z",
            "status": "succeeded",
            "phases": [{"name": "reindex", "status": "succeeded", "fatal": True}],
        })
"""


class TestConcurrency:
    """Real multi-process tests — flock is per open file description, so threads
    within one process would share the lock and prove nothing."""

    def test_concurrent_appenders_lose_no_records(self, tmp_path):
        runs_dir = str(tmp_path / "runs")
        per_writer, writers = 60, 6
        procs = [_spawn(APPENDER, runs_dir, per_writer, f"w{n}") for n in range(writers)]
        for proc in procs:
            _, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err.decode()

        runs = IndexingRunLedger(runs_dir=runs_dir).recent("c", limit=None)
        assert len(runs) == per_writer * writers
        assert len({r["runId"] for r in runs}) == per_writer * writers

    def test_appenders_racing_a_compactor_lose_no_records(self, tmp_path):
        """The fd-ordering trap: neither the plain concurrent-append test nor the
        interrupted-compaction test catches it.

        Compaction replaces the JSONL inode via os.replace. An appender that
        opened its fd BEFORE taking the lock would write into the now-unlinked old
        inode — the write succeeds, returns a byte count, and the record is gone
        with no error anywhere. Only appends racing an actual compaction expose it.
        """
        runs_dir = str(tmp_path / "runs")
        ledger = IndexingRunLedger(runs_dir=runs_dir)
        # Seed past the compaction threshold so the reader really does compact.
        for i in range(MAX_RUNS_PER_COLLECTION * 3 + 50):
            ledger.append(_record(run_id=f"seed{i:05d}"))

        hold, delay = 2.0, 0.6
        compactor = _spawn(SLOW_COMPACTOR, runs_dir, 1, "compact", HOLD_SECONDS=hold)
        writers = 3
        appenders = [
            _spawn(DELAYED_APPENDER, runs_dir, 20, f"w{n}", DELAY_SECONDS=delay)
            for n in range(writers)
        ]
        for proc in [compactor, *appenders]:
            _, err = proc.communicate(timeout=180)
            assert proc.returncode == 0, err.decode()

        runs = IndexingRunLedger(runs_dir=runs_dir).recent("c", limit=None)
        found = {r["runId"] for r in runs}
        expected = {f"w{n}-{i:04d}" for n in range(writers) for i in range(20)}
        missing = expected - found
        assert not missing, (
            f"{len(missing)} of {len(expected)} records written during a compaction "
            "were lost — the appender opened its data fd before taking the lock and "
            "wrote into the replaced inode"
        )
