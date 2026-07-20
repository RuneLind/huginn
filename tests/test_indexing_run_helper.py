"""Contract tests for scripts/lib/indexing_run.sh.

The helper is sourced by six unattended launchd jobs that run `set -euo
pipefail` and fail silently at 08:00-10:15 with no alerting. A regression here
does not cost observability, it stops indexing — strictly worse than the problem
the helper solves. These assert the three shell hazards the plan calls out, in
code rather than in prose.
"""
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(REPO_ROOT, "scripts", "lib", "indexing_run.sh")

EXPORTED = ("run_begin", "run_variant", "phase_begin", "phase_end", "run_end")


@pytest.fixture(scope="module")
def source():
    with open(HELPER, encoding="utf-8") as handle:
        return handle.read()


def test_it_is_syntactically_valid(source):
    assert subprocess.run(["bash", "-n", HELPER]).returncode == 0


def test_the_last_command_is_return_zero(source):
    """`.` exits with the status of the LAST COMMAND in the sourced file, so a
    helper that loaded perfectly but ended on a non-zero command sends every
    call site down the no-op stub branch. Neither the `&&`/`||` nor the
    `if/else` sourcing form fixes this — only an explicit trailing `return 0`.
    """
    lines = [ln.strip() for ln in source.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    assert lines[-1] == "return 0"


def test_every_exported_helper_ends_with_return_zero(source):
    """Under `set -e` a helper that returns non-zero aborts the job at that
    point — run_end curls an API these jobs routinely find down, and run_begin
    would abort before any indexing happens at all."""
    for name in EXPORTED:
        statements = _function_body(source, name)
        assert statements, f"{name} is not defined in {HELPER}"
        assert statements[-1] == "return 0", f"{name} does not end with `return 0`"


def _function_body(source, name):
    """Statements of a shell function, comments and blank lines stripped.

    Deliberately not a `.*?^\\}` regex: the helpers embed python snippets whose
    dict literals put a bare `}` in column 0, which such a regex stops at. Slice
    to the next top-level definition instead, then take the last `^}`.
    """
    defs = [(m.start(), m.group(1)) for m in
            re.finditer(r"^([a-z_][a-z0-9_]*)\(\) \{$", source, re.M)]
    for index, (start, found) in enumerate(defs):
        if found != name:
            continue
        end = defs[index + 1][0] if index + 1 < len(defs) else len(source)
        block = source[start:end].splitlines()
        close = max(i for i, ln in enumerate(block) if ln == "}")
        body = [ln.strip() for ln in block[1:close]]
        return [ln for ln in body if ln and not ln.startswith("#")]
    return []


def test_the_helper_exports_nothing_functional(source):
    """Every symbol the helper exports must be safely replaceable by a `:`
    no-op. poll_update_status is functional, not observational: stubbed it would
    make a script treat every reindex as instantly complete, and unstubbed a
    missing helper aborts the job after the reindex was triggered but before it
    is awaited. It therefore stays duplicated in each script."""
    defined = set(re.findall(r"^([a-z_][a-z0-9_]*)\(\) \{", source, re.M))
    public = {name for name in defined if not name.startswith("_")}
    assert public == set(EXPORTED), f"unexpected public helper(s): {public - set(EXPORTED)}"


def _run(script, runs_dir, helper=HELPER, api_url="http://127.0.0.1:59999"):
    """Run a snippet with the guarded source, a dead API, and a scratch ledger."""
    prelude = f"""
set -euo pipefail
PROJECT_DIR={REPO_ROOT!r}
API_URL={api_url!r}
HELPER={helper!r}
if [ -f "${{HELPER:-}}" ] && . "$HELPER"; then :; else
    run_begin(){{ :; }}; run_variant(){{ :; }}; phase_begin(){{ :; }}
    phase_end(){{ :; }}; run_end(){{ :; }}; RUN_ID=""
fi
"""
    env = dict(os.environ, HUGINN_RUNS_DIR=str(runs_dir))
    return subprocess.run(["bash", "-c", prelude + script], env=env,
                          capture_output=True, text=True)


def test_a_missing_helper_does_not_abort_the_job(tmp_path):
    """Trading "no observability" for "no indexing" is the wrong failure mode."""
    result = _run(
        'run_begin c j scheduled || true\n'
        'phase_begin tag 0; rc=0; false || rc=$?; phase_end "$rc" || true\n'
        'echo "run-id=[${RUN_ID:-}]"\n'
        'run_end "" || true\n'
        'echo REACHED_END\n',
        tmp_path, helper="/nonexistent/indexing_run.sh",
    )
    assert result.returncode == 0, result.stderr
    assert "REACHED_END" in result.stdout
    # set -u: RUN_ID is defaulted in the stub block, so ${RUN_ID:-} is safe.
    assert "run-id=[]" in result.stdout


def test_a_dead_api_does_not_abort_the_job_and_still_records(tmp_path):
    """The fallback path IS the API-down path, so it is the one that must not
    abort — and it writes through the ledger module, never a `>>` redirect
    (macOS has no flock(1), so a shell redirect cannot take LOCK_EX)."""
    result = _run(
        'run_begin c com.huginn.test scheduled || true\n'
        'phase_begin tag 0; rc=0; false || rc=$?; phase_end "$rc" || true\n'
        'phase_begin reindex 1; rc=0; true || rc=$?; phase_end "$rc" || true\n'
        'run_end "" || true\n'
        'echo REACHED_END\n',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "REACHED_END" in result.stdout

    sys.path.insert(0, REPO_ROOT)
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    runs = IndexingRunLedger(runs_dir=str(tmp_path)).recent("c", limit=5)
    assert len(runs) == 1
    run = runs[0]
    assert {p["name"] for p in run["phases"]} == {"tag", "reindex"}
    # A failed non-fatal phase degrades the run; it does not fail it.
    assert run["status"] == "degraded"


def test_a_skipped_phase_is_not_a_succeeded_one(tmp_path):
    """A reindex skipped on 409 exits 0. Recording that as `succeeded` asserts an
    index freshness the run never delivered — and at hourly cadence 409 is the
    single most likely outcome, so the dashboard would repeat that lie all day."""
    result = _run(
        'run_begin c j scheduled || true\n'
        'phase_begin reindex 1; rc=0\n'
        'phase_end skipped || true\n'
        'run_end "" || true\n',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    sys.path.insert(0, REPO_ROOT)
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    run = IndexingRunLedger(runs_dir=str(tmp_path)).recent("c", limit=5)[0]
    assert [p["status"] for p in run["phases"]] == ["skipped"]
    # Every phase skipped ⇒ the run did nothing, and says so.
    assert run["status"] == "skipped"


def test_a_skipped_phase_does_not_alarm_alongside_real_work(tmp_path):
    """Skipping is not a degradation: another process is already doing that
    work. Alarming on it would train the reader to ignore `degraded`."""
    result = _run(
        'run_begin c j scheduled || true\n'
        'phase_begin fetch 1; rc=0; true || rc=$?; phase_end "$rc" || true\n'
        'phase_begin reindex 1; rc=0\n'
        'phase_end skipped || true\n'
        'run_end "" || true\n',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    sys.path.insert(0, REPO_ROOT)
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    run = IndexingRunLedger(runs_dir=str(tmp_path)).recent("c", limit=5)[0]
    assert run["status"] == "succeeded"
    phases = {p["name"]: p["status"] for p in run["phases"]}
    assert phases == {"fetch": "succeeded", "reindex": "skipped"}


def test_a_run_can_be_reclassified_after_it_starts(tmp_path):
    """x-feed only learns which kind of run it is doing partway through: cleanup
    runs first, and only if it deleted something does the script drop the
    collection and rebuild rather than update incrementally. The two differ by an
    order of magnitude, so the record has to say which one this was."""
    result = _run(
        'run_begin c j scheduled || true\n'
        'phase_begin cleanup 0; rc=0; true || rc=$?; phase_end "$rc" || true\n'
        'run_variant rebuild || true\n'
        'run_end "" || true\n',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    sys.path.insert(0, REPO_ROOT)
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    run = IndexingRunLedger(runs_dir=str(tmp_path)).recent("c", limit=5)[0]
    assert run["variant"] == "rebuild"


def test_reclassifying_rejects_a_variant_the_dashboard_cannot_median(tmp_path):
    """The dashboard medians `incremental` and `rebuild` separately. A typo'd
    third value would silently become its own bucket of one."""
    result = _run(
        'run_begin c j scheduled || true\n'
        'run_variant nonsense || true\n'
        'run_end "" || true\n',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    sys.path.insert(0, REPO_ROOT)
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    run = IndexingRunLedger(runs_dir=str(tmp_path)).recent("c", limit=5)[0]
    assert run["variant"] == "incremental"


def test_a_failed_phase_does_not_poison_every_later_phase(tmp_path):
    """phase_begin resets `rc` internally as well as at the call site. Without
    the reset `rc` is assigned only on failure and never cleared, so one failed
    tagging phase would mark the reindex failed too and roll the run up to
    `failed` — destroying the succeeded/degraded/failed fidelity in six
    unattended scripts."""
    result = _run(
        'run_begin c j scheduled || true\n'
        'phase_begin tag 0; false || rc=$?; phase_end "$rc" || true\n'
        # Deliberately omit the call-site `rc=0` to prove phase_begin resets it.
        'phase_begin reindex 1; true || rc=$?; phase_end "$rc" || true\n'
        'run_end "" || true\n',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    sys.path.insert(0, REPO_ROOT)
    from main.runtime.indexing_run_ledger import IndexingRunLedger
    run = IndexingRunLedger(runs_dir=str(tmp_path)).recent("c", limit=5)[0]
    phases = {p["name"]: p["status"] for p in run["phases"]}
    assert phases == {"tag": "degraded", "reindex": "succeeded"}
    assert run["status"] == "degraded"
