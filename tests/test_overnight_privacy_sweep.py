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
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "overnight_privacy_sweep.sh")

# launchd runs `/bin/bash`, which on macOS is 3.2 — a decade behind the bash on
# PATH. Testing under the newer one would let bash 4 syntax (`mapfile`,
# `${x,,}`, associative arrays) pass here and kill the 01:00 job at that line.
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"

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
variable="STUB_RC_$(printf '%s' "$collection" | tr -c 'A-Za-z0-9_' '_')"
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
    (bin_dir / "launchctl").write_text(
        '#!/usr/bin/env bash\necho "$*" >> "$STUB_LAUNCHCTL"\n', encoding="utf-8")
    (bin_dir / "launchctl").chmod(0o755)
    calls = tmp_path / "calls.txt"
    launchctl_calls = tmp_path / "launchctl.txt"
    runs = tmp_path / "runs"
    runs.mkdir()

    def _run(*args, exit_codes=None, script=SCRIPT, home=None, label=None):
        # Truncated per invocation: a test that runs the script twice reads the
        # SECOND run's arguments, not a file still holding the first run's.
        calls.write_text("", encoding="utf-8")
        launchctl_calls.write_text("", encoding="utf-8")
        environment = dict(os.environ)
        environment.update({
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "STUB_CALLS": str(calls),
            "STUB_LAUNCHCTL": str(launchctl_calls),
            "HUGINN_RUNS_DIR": str(runs),
            "API_URL": DEAD_API,
            "LOG_DIR": str(tmp_path / "logs"),
            "LEDGER_KEY": "privacy-baseline-test",
            "JOB_LABEL": "test_overnight",
            "TRIGGER": "manual",
        })
        if home:
            environment["HOME"] = home
        if label:
            environment["LAUNCHD_LABEL"] = label
        for collection, code in (exit_codes or {}).items():
            key = re.sub(r"[^A-Za-z0-9_]", "_", collection)
            environment[f"STUB_RC_{key}"] = str(code)
        completed = subprocess.run([BASH, script, *args], env=environment,
                                   capture_output=True, text=True)
        called = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        ledger = runs / "privacy-baseline-test.jsonl"
        records = [json.loads(line) for line in
                   ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
        completed.launchctl = (launchctl_calls.read_text(encoding="utf-8").splitlines()
                               if launchctl_calls.exists() else [])
        return completed, called, records

    return _run


def test_it_is_syntactically_valid():
    assert subprocess.run([BASH, "-n", SCRIPT]).returncode == 0


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


def test_the_launchd_template_is_readable_by_the_ledgers_own_parser():
    """`plutil -lint` is NOT the gate that matters. The ledger reports this job's
    schedule by reading the installed plist with `plistlib`, which is stricter:
    a double hyphen inside an XML comment (writing out the `--once` flag, say)
    parses for launchd and raises here, and the job then silently has no
    schedule — which costs it the cadence-aware `incomplete` threshold and gives
    it the flat 6 h one, under a sweep whose measured worst case is 5.1 h."""
    import plistlib
    template = os.path.join(REPO_ROOT, "scripts", "com.huginn.privacy-sweep.plist.example")
    with open(template, "rb") as handle:
        data = plistlib.load(handle)
    assert data["Label"] == "com.huginn.privacy-sweep"
    # The script must still be findable by basename among the wrapper's arguments.
    scripts = [os.path.basename(a) for a in data["ProgramArguments"] if a.endswith(".sh")]
    assert scripts == ["overnight_privacy_sweep.sh"]
    assert "--once" in data["ProgramArguments"], "the installed job must disarm itself"

    routing = os.path.join(REPO_ROOT, "scripts", "schedule_routing.json")
    with open(routing, encoding="utf-8") as handle:
        entries = json.load(handle)["scriptCollections"]
    assert entries[scripts[0]], "the job needs a routing entry or the ledger cannot date it"


def test_no_absolute_home_path_reaches_the_public_template():
    """The same shape index_scan.py check 10 blocks as a distributor fingerprint
    and alias_registry rewrites: an operator's account name in a public repo."""
    template = os.path.join(REPO_ROOT, "scripts", "com.huginn.privacy-sweep.plist.example")
    assert "/Users/" not in open(template, encoding="utf-8").read()


def test_a_night_that_discovers_nothing_still_leaves_a_row(run, tmp_path, monkeypatch):
    """The failure this exists for: a missing private sub-repo, an absent .venv or
    a raising `load_scope()` sweeps nothing — and a job that records nothing is
    indistinguishable on the dashboard from a job that was never scheduled."""
    empty = tmp_path / "empty-scope"
    (empty / "data" / "collections").mkdir(parents=True)
    for path in ("scripts", "main", ".venv"):
        os.symlink(os.path.join(REPO_ROOT, path), empty / path)

    completed, called, records = run(script=str(empty / "scripts" / "overnight_privacy_sweep.sh"))
    assert called == []
    assert completed.returncode == 0
    assert records[0]["stage"] == "begin"
    closing = [record for record in records if record.get("stage") == "end"][-1]
    assert closing["status"] == "degraded"
    assert [phase["name"] for phase in closing["phases"]] == ["discover", "sweep"]


def test_an_armed_but_unstamped_collection_is_skipped_with_its_reason(run):
    """A pre-alias backup shares an in-scope basePath and reports its own people
    by construction. Sweeping it would spend hours to rediscover that; dropping it
    silently would hide a real sibling that lost its stamp. It is named in the log
    with the reason instead."""
    completed, _, _ = run()
    skipped = [line for line in completed.stdout.splitlines() if "Not swept:" in line]
    assert skipped, "expected at least one armed-but-unstamped collection on this machine"
    assert all("no privacy stamp" in line or "unreadable manifest" in line for line in skipped)


def test_a_quote_in_a_collection_name_still_produces_structured_detail(run):
    """The detail is built by python3, not concatenated: invalid JSON is stored by
    the helper as an opaque `note` string, dropping the structured fields a
    dashboard reads — without failing anything."""
    _, _, records = run("--collection", 'we"ird')
    closing = [record for record in records if record.get("stage") == "end"][-1]
    assert closing["phases"][0]["detail"] == {
        "collection": 'we"ird', "mode": "baseline", "exit": 0}


def test_once_disarms_the_installed_job_after_the_sweep(run, tmp_path):
    """`--once` is what makes the schedule genuinely one-shot. Without it, an
    unattended five-hour local-model job quietly becomes nightly."""
    agents = tmp_path / "home" / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    plist = agents / "com.huginn.test-sweep.plist"
    plist.write_text("<plist/>", encoding="utf-8")

    completed, _, _ = run("--collection", "alpha", "--once",
                          home=str(tmp_path / "home"), label="com.huginn.test-sweep")
    assert completed.launchctl == [f"unload {plist}"]

    without, _, _ = run("--collection", "alpha",
                        home=str(tmp_path / "home"), label="com.huginn.test-sweep")
    assert without.launchctl == []


@pytest.mark.parametrize("argument", ["--collection", "--limit"])
def test_a_flag_with_no_value_says_so(run, argument):
    """`shift 2` with nothing to shift aborts under `set -e` before the
    unknown-option arm prints: a console typo would look like a job that ran and
    found nothing."""
    completed, called, _ = run(argument)
    assert completed.returncode == 1
    assert "Missing value" in completed.stderr
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
