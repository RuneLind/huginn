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

from datetime import datetime, timedelta, timezone

from main.runtime.indexing_run_ledger import (
    INCOMPLETE_AFTER_SECONDS,
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


class TestUnclosedRuns:
    """A writer that opens a run and never closes it must not fold into a tidy
    short success — that is the "15 min for a job that blocked for 76" error
    returning as a crash mode."""

    def _open(self, source="script", **extra):
        record = {
            "collection": "c", "runId": "shared", "source": source,
            "stage": "begin", "startedAt": "2026-07-18T09:28:37Z",
            "job": "com.huginn.mimir-index", "trigger": "scheduled",
        }
        record.update(extra)
        return record

    def _huginn_reindex(self):
        return {
            "collection": "c", "runId": "shared", "source": "huginn",
            "stage": "end", "status": "succeeded",
            "startedAt": "2026-07-18T10:19:13Z", "finishedAt": "2026-07-18T10:44:14Z",
            "phases": [{"name": "reindex", "status": "succeeded", "fatal": True,
                        "durationSeconds": 1501}],
        }

    def test_script_dying_after_the_reindex_does_not_fold_to_a_short_success(self):
        now = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
        folded = fold_records([self._open(), self._huginn_reindex()], now=now)[0]
        assert folded["status"] == "incomplete"
        # The reindex's own 25 minutes must not be published as the run duration.
        assert folded["durationSeconds"] is None

    def test_an_open_run_is_running_before_the_threshold(self):
        started = datetime(2026, 7, 18, 9, 28, 37, tzinfo=timezone.utc)
        now = started + timedelta(seconds=INCOMPLETE_AFTER_SECONDS - 60)
        folded = fold_records([self._open()], now=now)[0]
        assert folded["status"] == "running"

    def test_an_open_run_becomes_incomplete_after_the_threshold(self):
        started = datetime(2026, 7, 18, 9, 28, 37, tzinfo=timezone.utc)
        now = started + timedelta(seconds=INCOMPLETE_AFTER_SECONDS + 60)
        folded = fold_records([self._open()], now=now)[0]
        assert folded["status"] == "incomplete"

    def test_a_matching_end_closes_the_run(self):
        now = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
        closing = {
            "collection": "c", "runId": "shared", "source": "script", "stage": "end",
            "finishedAt": "2026-07-18T10:44:20Z",
            "phases": [{"name": "tag", "status": "succeeded", "durationSeconds": 3033}],
        }
        folded = fold_records([self._open(), self._huginn_reindex(), closing], now=now)[0]
        assert folded["status"] == "succeeded"
        assert folded["durationSeconds"] == 4543

    def test_a_server_restart_mid_reindex_leaves_an_incomplete_run(self):
        """huginn's own opening partial with no __finish_update behind it."""
        now = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
        folded = fold_records([self._open(source="huginn")], now=now)[0]
        assert folded["status"] == "incomplete"

    def test_openness_survives_compaction(self, ledger):
        """Compaction rewrites folded runs and drops the stage markers, so the
        unclosed set is carried on the folded record and re-derived on read."""
        ledger.append(self._open())
        ledger.append(self._huginn_reindex())
        folded = ledger.recent("c", limit=10)[0]
        assert folded["unclosedSources"] == ["script"]

        # Simulate the compacted form being all that is left on disk.
        refolded = fold_records([folded],
                                now=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc))[0]
        assert refolded["status"] == "incomplete"

        # ...and a late-arriving closer still closes it.
        closed = fold_records([folded, {
            "collection": "c", "runId": "shared", "source": "script", "stage": "end",
            "finishedAt": "2026-07-18T10:45:00Z",
            "phases": [{"name": "tag", "status": "succeeded"}],
        }], now=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc))[0]
        assert closed["status"] == "succeeded"


class TestCadenceAwareIncompleteThreshold:
    """The age past which an unclosed run folds `incomplete` is a caller-supplied
    per-collection value (the jobs endpoint derives it from launchd cadence), so a
    dead hourly run stops reading as `running` across six later runs the way a flat
    6h let it. The ledger never imports the schedule module — that would invert the
    layering and break the CLI path — so the value threads in as a parameter."""

    def _open(self, started="2026-07-18T09:28:37Z"):
        return {"collection": "c", "runId": "shared", "source": "script",
                "stage": "begin", "startedAt": started}

    def test_default_still_uses_the_flat_constant(self):
        started = datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc)
        now = started + timedelta(seconds=INCOMPLETE_AFTER_SECONDS - 60)
        folded = fold_records([self._open("2026-07-18T09:00:00Z")], now=now)[0]
        assert folded["status"] == "running"

    def test_a_shorter_threshold_flips_running_to_incomplete_sooner(self):
        started = datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc)
        # 3h old: still running under the flat 6h default, but incomplete once
        # the caller passes a 2×hourly (7200s) threshold.
        now = started + timedelta(seconds=3 * 3600)
        assert fold_records([self._open("2026-07-18T09:00:00Z")], now=now)[0][
            "status"] == "running"
        assert fold_records([self._open("2026-07-18T09:00:00Z")], now=now,
                            incomplete_after=7200)[0]["status"] == "incomplete"

    def test_recent_threads_the_threshold_through(self, ledger):
        from main.runtime.indexing_run_ledger import now_iso
        ledger.append(self._open(now_iso()))  # opened "just now"
        # The default keeps a fresh open running (well inside 6h).
        assert ledger.recent("c", limit=5)[0]["status"] == "running"
        # A zero threshold flips the same fresh open to incomplete on read.
        assert ledger.recent("c", limit=5, incomplete_after=0)[0]["status"] == "incomplete"


class TestSingleRecordStatusRecompute:
    """A one-record group re-derives its status from the phases, matching the
    multi-record path — otherwise a stored `running`/`incomplete` (both now
    accepted values) reads as permanently running. Recompute only when phases are
    present: a backfilled record carries a status with no phases and must keep it."""

    def test_a_stored_running_with_succeeded_phases_recomputes(self):
        record = {"collection": "c", "runId": "r", "status": "running",
                  "startedAt": "2026-07-18T09:00:00Z", "finishedAt": "2026-07-18T09:10:00Z",
                  "phases": [{"name": "reindex", "status": "succeeded", "fatal": True}]}
        assert fold_records([record])[0]["status"] == "succeeded"

    def test_a_stored_status_with_a_failed_phase_recomputes_to_failed(self):
        record = {"collection": "c", "runId": "r", "status": "running",
                  "phases": [{"name": "reindex", "status": "failed", "fatal": True}]}
        assert fold_records([record])[0]["status"] == "failed"

    def test_a_backfilled_record_with_no_phases_keeps_its_stored_status(self):
        """Rolling up an empty phase list would turn a known `succeeded` into
        `unknown` — backfilled rows carry a status and no phases on purpose."""
        record = {"collection": "c", "runId": "r", "status": "succeeded",
                  "backfilled": True, "phases": []}
        assert fold_records([record])[0]["status"] == "succeeded"

    def test_a_record_with_no_phases_key_keeps_its_stored_status(self):
        record = {"collection": "c", "runId": "r", "status": "degraded"}
        assert fold_records([record])[0]["status"] == "degraded"


class TestPhaseMerging:
    """Both sides report a `reindex` phase for the same work — huginn times the
    rebuild, the script times trigger-plus-poll around it. Observed live on the
    2026-07-19 09:15 mimir run: a 26-second run folded to 13s + 16s of reindex."""

    def _script(self, **extra):
        record = {
            "collection": "c", "runId": "shared", "source": "script",
            "startedAt": "2026-07-19T07:15:05Z", "finishedAt": "2026-07-19T07:15:31Z",
            "phases": [
                {"name": "tag", "status": "succeeded", "durationSeconds": 10},
                {"name": "reindex", "status": "succeeded", "durationSeconds": 16,
                 "fatal": True},
            ],
        }
        record.update(extra)
        return record

    def _huginn(self):
        return {
            "collection": "c", "runId": "shared", "source": "huginn",
            "startedAt": "2026-07-19T07:15:15Z", "finishedAt": "2026-07-19T07:15:29Z",
            "phases": [{"name": "reindex", "status": "succeeded",
                        "durationSeconds": 13, "fatal": True}],
        }

    def test_a_phase_reported_by_both_sides_appears_once(self):
        folded = fold_records([self._script(), self._huginn()])[0]
        names = [p["name"] for p in folded["phases"]]
        assert names.count("reindex") == 1
        # Phase durations must not sum past the run's own duration.
        assert sum(p["durationSeconds"] for p in folded["phases"]) <= \
            folded["durationSeconds"]

    def test_huginns_copy_wins_because_it_times_the_rebuild(self):
        folded = fold_records([self._script(), self._huginn()])[0]
        reindex = next(p for p in folded["phases"] if p["name"] == "reindex")
        assert reindex["durationSeconds"] == 13

    def test_arrival_order_does_not_change_the_winner(self):
        folded = fold_records([self._huginn(), self._script()])[0]
        reindex = next(p for p in folded["phases"] if p["name"] == "reindex")
        assert reindex["durationSeconds"] == 13

    def test_the_scripts_copy_survives_when_huginn_wrote_nothing(self):
        """The API-down path: the CLI does the rebuild and huginn's store never
        sees it, so the script's phase is the only record of the reindex."""
        folded = fold_records([self._script()])[0]
        reindex = next(p for p in folded["phases"] if p["name"] == "reindex")
        assert reindex["durationSeconds"] == 16

    def test_winning_copy_borrows_a_duration_it_does_not_have(self):
        """The CLI adapter writes a reindex phase with no duration at all. Seen
        live on the API-down dry-run of daily_capra_wiki_update.sh: preferring
        huginn's copy left the phase timeless inside a timed 9-second run.
        Preference decides WHICH copy wins, never whether data is dropped."""
        huginn = self._huginn()
        del huginn["phases"][0]["durationSeconds"]
        folded = fold_records([self._script(), huginn])[0]
        reindex = next(p for p in folded["phases"] if p["name"] == "reindex")
        assert reindex["durationSeconds"] == 16

    def test_a_failed_script_phase_is_not_masked_by_huginns_copy(self):
        """huginn's copy wins on duration, but a disagreement on STATUS resolves
        pessimistically — a clean rebuild must not erase the script's report that
        its wait around that rebuild failed."""
        script = self._script()
        script["phases"][1] = {"name": "reindex", "status": "failed",
                               "durationSeconds": 16, "fatal": True}
        folded = fold_records([script, self._huginn()])[0]
        reindex = next(p for p in folded["phases"] if p["name"] == "reindex")
        assert reindex["durationSeconds"] == 13
        assert reindex["status"] == "failed"
        assert folded["status"] == "failed"


class TestPhaseOrdering:
    """The fold unions phases in record-arrival order, and huginn's `reindex`
    record is appended when the rebuild STARTS while the script's closing record
    (carrying its earlier `fetch`) lands later — so without a per-phase
    `startedAt` the fold renders `reindex` before the `fetch` that preceded it.
    `startedAt` gives a real order; legacy phases that predate it must not move."""

    def test_new_phases_sort_by_started_at_not_arrival(self):
        # Records arrive reindex-first, but fetch started earlier in wall time.
        script = {
            "collection": "c", "runId": "shared", "source": "script",
            "startedAt": "2026-07-19T07:15:00Z", "finishedAt": "2026-07-19T07:16:00Z",
            "phases": [
                {"name": "fetch", "status": "succeeded", "durationSeconds": 5,
                 "startedAt": "2026-07-19T07:15:00Z"},
                {"name": "reindex", "status": "succeeded", "durationSeconds": 16,
                 "startedAt": "2026-07-19T07:15:10Z", "fatal": True},
            ],
        }
        huginn = {
            "collection": "c", "runId": "shared", "source": "huginn",
            "startedAt": "2026-07-19T07:15:10Z", "finishedAt": "2026-07-19T07:15:26Z",
            "phases": [{"name": "reindex", "status": "succeeded",
                        "durationSeconds": 13, "startedAt": "2026-07-19T07:15:10Z",
                        "fatal": True}],
        }
        # huginn arrives FIRST, so its reindex would otherwise pin to position 0.
        folded = fold_records([huginn, script])[0]
        assert [p["name"] for p in folded["phases"]] == ["fetch", "reindex"]

    def test_legacy_phases_without_started_at_keep_arrival_order(self):
        """Every phase written before this field existed has no startedAt, and
        those read correctly by arrival accident today. A naive sort keying on a
        default would drag them all to the front and reorder history — so a run
        with no timestamped phase at all must come back byte-for-byte in order."""
        script = {
            "collection": "c", "runId": "old", "source": "script",
            "startedAt": "2026-07-01T09:00:00Z", "finishedAt": "2026-07-01T09:05:00Z",
            "phases": [
                {"name": "cleanup", "status": "succeeded", "durationSeconds": 2},
                {"name": "fetch", "status": "succeeded", "durationSeconds": 3},
                {"name": "reindex", "status": "succeeded", "durationSeconds": 4,
                 "fatal": True},
            ],
        }
        folded = fold_records([script])[0]
        assert [p["name"] for p in folded["phases"]] == ["cleanup", "fetch", "reindex"]

    def test_mixed_old_and_new_phases_only_the_timestamped_ones_move(self):
        """A folded run straddling the change: the merged phase list has some
        phases carrying startedAt and some not. Only the timestamped phases sort —
        into the positions they already hold — and the field-less ones stay pinned
        to their arrival index. (Two records so the merge path, which is what
        sorts, actually runs — a single writer emits its phases chronologically.)"""
        huginn = {
            "collection": "c", "runId": "mixed", "source": "huginn",
            "startedAt": "2026-07-19T07:05:00Z", "finishedAt": "2026-07-19T07:05:04Z",
            "phases": [{"name": "reindex", "status": "succeeded", "durationSeconds": 4,
                        "startedAt": "2026-07-19T07:05:00Z", "fatal": True}],
        }
        script = {
            "collection": "c", "runId": "mixed", "source": "script",
            "startedAt": "2026-07-19T07:00:00Z", "finishedAt": "2026-07-19T07:10:00Z",
            "phases": [
                {"name": "legacy", "status": "succeeded", "durationSeconds": 1},
                {"name": "fetch", "status": "succeeded", "durationSeconds": 3,
                 "startedAt": "2026-07-19T07:00:00Z"},
            ],
        }
        # Arrival order of merged phases: reindex(07:05), legacy(none), fetch(07:00).
        folded = fold_records([huginn, script])[0]
        # fetch (07:00) and reindex (07:05) swap into their two timestamped slots
        # (indices 0 and 2); `legacy` holds index 1 untouched.
        assert [p["name"] for p in folded["phases"]] == ["fetch", "legacy", "reindex"]

    def test_api_down_fold_inherits_and_sorts_the_scripts_started_at(self):
        """The x-feed API-down path: the CLI adapter writes a huginn-source
        `reindex` phase with NO startedAt, and the wrapping script writes `fetch`
        + `reindex` with startedAt. huginn's copy wins the reindex merge but must
        inherit the script's startedAt — otherwise the merged phase has nothing to
        sort on and pins to arrival order, landing `reindex` before `fetch`."""
        script = {
            "collection": "c", "runId": "apidown", "source": "script",
            "startedAt": "2026-07-19T07:15:00Z", "finishedAt": "2026-07-19T07:16:00Z",
            "phases": [
                {"name": "fetch", "status": "succeeded", "durationSeconds": 5,
                 "startedAt": "2026-07-19T07:15:00Z"},
                {"name": "reindex", "status": "succeeded", "durationSeconds": 16,
                 "startedAt": "2026-07-19T07:15:10Z", "fatal": True},
            ],
        }
        cli_huginn = {  # CLI-adapter-style: huginn source, reindex, no startedAt
            "collection": "c", "runId": "apidown", "source": "huginn",
            "startedAt": "2026-07-19T07:15:10Z", "finishedAt": "2026-07-19T07:15:26Z",
            "phases": [{"name": "reindex", "status": "succeeded", "fatal": True}],
        }
        folded = fold_records([cli_huginn, script])[0]
        reindex = next(p for p in folded["phases"] if p["name"] == "reindex")
        assert reindex["startedAt"] == "2026-07-19T07:15:10Z"
        assert [p["name"] for p in folded["phases"]] == ["fetch", "reindex"]


class TestSkippedRollup:
    """`skipped` distinguishes "did not run" from "ran fine". A reindex skipped
    on 409 exits 0, and rolling that up as `succeeded` asserts a freshness the
    run never delivered."""

    def test_all_phases_skipped_is_a_skipped_run(self):
        assert rollup_status([{"name": "reindex", "status": "skipped", "fatal": True}]) \
            == "skipped"

    def test_skipping_beside_real_work_is_not_a_degradation(self):
        phases = [{"name": "fetch", "status": "succeeded"},
                  {"name": "reindex", "status": "skipped", "fatal": True}]
        assert rollup_status(phases) == "succeeded"

    def test_a_real_failure_still_outranks_a_skip(self):
        phases = [{"name": "fetch", "status": "failed", "fatal": True},
                  {"name": "reindex", "status": "skipped", "fatal": True}]
        assert rollup_status(phases) == "failed"

    def test_a_statusless_phase_is_not_rounded_up_to_success(self):
        """A phase with no outcome is not evidence of one. Previously it fell
        through the elif chain and the run reported `succeeded`."""
        assert rollup_status([{"name": "reindex"}]) == "degraded"

    def test_a_skip_wins_a_disagreement_against_a_claimed_success(self):
        """Pessimistic merge: the copy that did less work wins, but a skip is
        not escalated to a failure."""
        script = {"collection": "c", "runId": "r", "source": "script",
                  "phases": [{"name": "reindex", "status": "skipped"}]}
        huginn = {"collection": "c", "runId": "r", "source": "huginn",
                  "phases": [{"name": "reindex", "status": "succeeded"}]}
        folded = fold_records([script, huginn])[0]
        assert folded["phases"][0]["status"] == "skipped"

    def test_a_skipped_run_is_excluded_from_the_median(self):
        """Its duration covers a job that never reindexed, so medianing it in
        would understate how long a real incremental takes."""
        from main.routes.collections import _median_by_variant
        runs = [{"durationSeconds": 10, "status": "succeeded", "variant": "incremental"},
                {"durationSeconds": 2, "status": "skipped", "variant": "incremental"}]
        assert _median_by_variant(runs) == {"incremental": 10}
