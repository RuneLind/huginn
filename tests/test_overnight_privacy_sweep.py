"""Contract tests for scripts/overnight_privacy_sweep.sh.

The script exists to spend a whole night asking a local model about every
document in every in-scope collection, unattended. Two properties decide whether
that night is worth anything, and both are easy to break silently:

  - a collection that reports someone (exit 2) must DEGRADE the run without
    stopping the collections queued behind it — the finding is the point of the
    night, not a reason to abandon it;
  - the targets come from the privacy scope, so a collection whose name is not
    public is swept without that name being written into this repo.

The sweep itself is stubbed here. What is under test is the runner's control
flow and what it records, not the model.
"""
import json
import os
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "overnight_privacy_sweep.sh")

# A dead port, so the ledger helper falls through to its module writer, which is
# what honours HUGINN_RUNS_DIR. Pointing the tests at a live server would write
# their rows into the real ledger.
DEAD_API = "http://127.0.0.1:9"

STUB_UV = """#!/usr/bin/env bash
# Stands in for `uv run scripts/audit/sensitivity_sweep.py …`. Records the
# arguments it was called with and exits with STUB_RC_<collection, _-separated>.
collection=""
previous=""
for argument in "$@"; do
    [ "$previous" = "--collection" ] && collection="$argument"
    previous="$argument"
done
echo "$*" >> "$STUB_CALLS"
variable="STUB_RC_${collection//-/_}"
exit "${!variable:-0}"
"""


@pytest.fixture
def run(tmp_path):
    """Run the script with a stubbed sweep, a temp ledger and a temp log dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(STUB_UV, encoding="utf-8")
    stub.chmod(0o755)
    calls = tmp_path / "calls.txt"
    runs = tmp_path / "runs"
    runs.mkdir()

    def _run(*args, exit_codes=None):
        # Truncated per invocation: a test that runs the script twice reads the
        # SECOND run's arguments, not a file still holding the first run's.
        calls.write_text("", encoding="utf-8")
        environment = dict(os.environ)
        environment.update({
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "STUB_CALLS": str(calls),
            "HUGINN_RUNS_DIR": str(runs),
            "API_URL": DEAD_API,
            "LOG_DIR": str(tmp_path / "logs"),
            "LEDGER_KEY": "privacy-baseline-test",
            "JOB_LABEL": "test_overnight",
            "TRIGGER": "manual",
        })
        for collection, code in (exit_codes or {}).items():
            environment[f"STUB_RC_{collection.replace('-', '_')}"] = str(code)
        completed = subprocess.run(["bash", SCRIPT, *args], env=environment,
                                   capture_output=True, text=True)
        called = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        ledger = runs / "privacy-baseline-test.jsonl"
        records = [json.loads(line) for line in
                   ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
        return completed, called, records

    return _run


def test_it_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0


def test_every_ledger_call_site_tolerates_failure():
    """The same rule the helper's own tests assert, at this call site: under
    `set -euo pipefail` an unguarded helper failure aborts the sweep, trading the
    night's work for a missing ledger row."""
    source = open(SCRIPT, encoding="utf-8").read()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("run_begin ", "run_end ", "phase_end ")):
            assert stripped.endswith("|| true"), stripped


def test_a_collection_that_reports_someone_does_not_stop_the_night(run):
    """Exit 2 is an unknown person. The two collections queued behind it are the
    ones a stop would silently cost, and their absence would look like a clean
    night rather than an abandoned one."""
    completed, called, records = run(
        "--collection", "alpha", "--collection", "beta", "--collection", "gamma",
        exit_codes={"alpha": 2})

    assert [call.split()[3] for call in called] == ["alpha", "beta", "gamma"]
    closing = [record for record in records if record.get("stage") == "end"][-1]
    statuses = {phase["name"]: phase["status"] for phase in closing["phases"]}
    assert statuses == {"sweep:alpha": "degraded",
                        "sweep:beta": "succeeded",
                        "sweep:gamma": "succeeded"}
    assert closing["status"] == "degraded"
    assert completed.returncode == 0


def test_an_unreachable_model_degrades_rather_than_fails(run):
    """Exit 1 is no collection, no map, or no Ollama. None of those is this
    script's fault and none should read as a failed job — but nor may they read
    as a clean sweep of a collection nobody looked at."""
    _, _, records = run("--collection", "alpha", exit_codes={"alpha": 1})
    closing = [record for record in records if record.get("stage") == "end"][-1]
    phase = closing["phases"][0]
    assert phase["status"] == "degraded"
    assert phase.get("fatal", False) is False
    assert phase["detail"] == {"collection": "alpha", "mode": "baseline", "exit": 1}


def test_the_opening_partial_is_written_before_any_sweep(run):
    """A night killed halfway must fold to `incomplete`, not to nothing. That is
    what the opening record buys, and it is only worth anything if it is written
    before the hours of work rather than after."""
    _, _, records = run("--collection", "alpha")
    assert records[0]["stage"] == "begin"
    assert records[0]["variant"] == "rebuild"


def test_baseline_is_the_default_and_incremental_is_opt_in(run):
    """A run that quietly swept incrementally would answer a different question
    than the one it was scheduled for, and its report would still look complete."""
    _, baseline_calls, _ = run("--collection", "alpha")
    assert "--baseline" in baseline_calls[0]

    _, incremental_calls, records = run("--collection", "alpha", "--incremental")
    assert "--baseline" not in incremental_calls[0]
    closing = [record for record in records if record.get("stage") == "end"][-1]
    assert closing["variant"] == "incremental"


def test_the_job_label_reaches_the_sweeps_own_ledger_row(run):
    """The sweep writes its own run under `sensitivity-audit`; without these the
    row cannot say which job asked for it."""
    _, called, _ = run("--collection", "alpha")
    assert "--job test_overnight" in called[0]
    assert "--trigger manual" in called[0]


@pytest.mark.parametrize("argument", ["--limit 0", "--limit x", "--nope"])
def test_a_bad_argument_is_refused_before_any_work(run, argument):
    completed, called, _ = run(*argument.split())
    assert completed.returncode == 1
    assert called == []


def test_targets_are_discovered_from_the_privacy_scope_not_a_list(run):
    """No collection name is written into this public repo: the script asks
    `load_scope()`, which reads the public scope file plus any private one. The
    order is smallest first, so the cheap verdicts are on disk early."""
    source = open(SCRIPT, encoding="utf-8").read()
    assert "load_scope" in source

    _, called, _ = run()
    swept = [call.split()[3] for call in called]
    assert swept, "expected at least one in-scope collection with a built index"

    sizes = []
    for name in swept:
        manifest = os.path.join(REPO_ROOT, "data", "collections", name, "manifest.json")
        with open(manifest, encoding="utf-8") as handle:
            sizes.append(json.load(handle).get("numberOfDocuments") or 0)
    assert sizes == sorted(sizes)
