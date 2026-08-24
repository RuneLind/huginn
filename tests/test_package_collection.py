"""Guards on scripts/audit/package_collection.py — the hand-off gate.

The point of the packager is that it *produces* the artifact: a scan someone has
to remember to run before copying a directory is advisory, and the one time it
is skipped is the time it mattered. So the tests that matter are the ones where
a failing scan must leave NOTHING behind, and where a collection that was never
aliased must not be packaged at all.

Every name is invented ("Kari Ukjent", "Ada Example"); the fixtures come from
tests/test_scan_index.py.
"""
import getpass
import json
import subprocess
import sys
import tarfile
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import package_collection as pkg  # noqa: E402
from main.privacy import index_scan  # noqa: E402
import scan_index as cli  # noqa: E402
from main.privacy import sensitivity_sweep as sweep  # noqa: E402
from main.utils.ollama_cli import DEFAULT_MODEL  # noqa: E402
from tests.test_scan_index import (  # noqa: E402
    MAP_VERSION, _clean_document, _document, _map, _write_collection,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A tmp repo root whose scope, map and collections are all discoverable.

    `alias_registry.REPO_ROOT` is what makes `resolve_registry` see the tmp scope
    file: the private scope and map globs resolve against the repo root, not the
    process CWD, so that a guard built on them arms wherever it is called from.
    `chdir` stays so a CWD-relative read would still show up as a failure.
    """
    from main.privacy import alias_registry
    privacy = tmp_path / "huginn-x" / "privacy"
    privacy.mkdir(parents=True)
    (privacy / "aliases.json").write_text(json.dumps(_map()), encoding="utf-8")
    (privacy / "scope.json").write_text(
        json.dumps({"collections": ["demo-aliased"], "basePaths": []}), encoding="utf-8")
    (tmp_path / "collections").mkdir()
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(alias_registry, "REPO_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pkg, "REPO_ROOT", tmp_path)
    # The sweep gate reads its reports from here. Repointed so the packager's
    # verdict depends on what a test wrote, never on what happens to be in the
    # operator's private sub-repo.
    monkeypatch.setattr(sweep, "report_dirs", lambda: [tmp_path / "sweeps"])
    return tmp_path


SWEPT_AT = "2026-08-20T13:02:01.631639"


def _sweep_report(workspace, collection, *, generated_at, unknown=0, documents=1,
                  expected=1, limit=None, last_modified=SWEPT_AT, directory=None):
    """A report of the shape the gate reads: coverage, inputs, collection stamp.

    Every field matters to a different branch of `sweep_gate`, so a helper that
    filled in only `unknownCount` would have every test below exercising the
    "this report cannot prove anything" path instead of the one it names.
    """
    directory = directory or workspace / "sweeps"
    directory.mkdir(parents=True, exist_ok=True)
    mode = "baseline" if limit is None else f"baseline-limit{limit}"
    (directory / f"sweep_{collection}_2026-08-23_{mode}.json").write_text(
        json.dumps({"collection": collection, "generatedAt": generated_at,
                    "unknownCount": unknown, "documents": documents,
                    "documentsExpected": expected, "limit": limit,
                    "mapVersion": MAP_VERSION, "policyVersion": index_scan.POLICY_VERSION,
                    "model": DEFAULT_MODEL,
                    "collectionLastModifiedDocumentTime": last_modified,
                    "findings": []}), encoding="utf-8")


def _with_last_modified(collection, stamp=SWEPT_AT):
    manifest = json.loads((collection / "manifest.json").read_text(encoding="utf-8"))
    manifest["lastModifiedDocumentTime"] = stamp
    (collection / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _package(workspace, monkeypatch, name="demo-aliased", extra=()):
    monkeypatch.setattr(sys, "argv", [
        "package_collection.py", "--collection", name,
        "--collections-dir", str(workspace / "collections"),
        "--out", str(workspace / "out"),
        "--map", str(workspace / "huginn-x" / "privacy" / "aliases.json"),
        *extra,
    ])
    try:
        pkg.main()
    except SystemExit as e:      # every refusal path is a sys.exit("REFUSED: …")
        return e.code
    return 0


def _combined(capsys, code) -> str:
    """Everything the run said.

    `sys.exit("message")` only prints when the interpreter unwinds, and pytest
    catches the SystemExit — so the refusal text lives on `.code`, not in the
    captured streams. The scan report before it is a real `print`.
    """
    captured = capsys.readouterr()
    return captured.out + captured.err + str(code)


def _tarballs(workspace):
    return sorted(p.name for p in (workspace / "out").glob("*.tar.gz"))


def test_a_clean_collection_is_packaged(workspace, monkeypatch, capsys):
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    monkeypatch.setattr(sys, "argv", [
        "package_collection.py", "--collection", "demo-aliased",
        "--collections-dir", str(workspace / "collections"),
        "--out", str(workspace / "out"),
        "--map", str(workspace / "huginn-x" / "privacy" / "aliases.json"),
    ])
    pkg.main()
    assert "PASS" in capsys.readouterr().out
    tarballs = list((workspace / "out").glob("*.tar.gz"))
    assert len(tarballs) == 1

    with tarfile.open(tarballs[0]) as tar:
        names = tar.getnames()
        stamp = json.loads(tar.extractfile("PACKAGE-STAMP.json").read().decode())
    assert "PACKAGE-STAMP.json" in names
    # Everything else lives at the path it unpacks to, and nothing outside it.
    assert all(n == "PACKAGE-STAMP.json" or n.startswith("data/collections/demo-aliased/")
               for n in names), names
    assert not any("prealias" in n or n.endswith("aliases.json") for n in names)
    assert stamp["collection"] == "demo-aliased"
    assert stamp["map_version"] == 7 \
        and stamp["policy_version"] == index_scan.POLICY_VERSION
    assert stamp["numberOfDocuments"] == 1
    assert stamp["scanChecks"]["bigram_candidates"] == {"passed": True, "count": 0,
                                                        "ran": True}


def test_the_untarred_package_rescans_clean(workspace, monkeypatch, capsys, tmp_path):
    """The recipient's half of the hand-off, and the one path check 12 must not
    break: the tarball unpacks to `data/collections/<name>/`, so the directory
    name it lands under is the one the mapping's `documentPath` prefixes name.
    A check keyed off anything else fails every package the moment it is opened.
    """
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0), _clean_document(1)])
    _package(workspace, monkeypatch)
    capsys.readouterr()

    unpacked = tmp_path / "recipient"
    with tarfile.open(next((workspace / "out").glob("*.tar.gz"))) as tar:
        tar.extractall(unpacked, filter="data")

    report = index_scan.scan_collection(
        unpacked / "data" / "collections" / "demo-aliased",
        workspace / "huginn-x" / "privacy" / "aliases.json")
    assert report.check("document_paths").passed is True
    assert report.passed is True


@pytest.mark.parametrize("text,failing_check", [
    ("Notatet ble skrevet av Kari Ukjent.", "bigram_candidates"),
    ("bygget fra /Users/someone/source/huginn", "fingerprints"),
    ("gjenopprett fra manifest.json.bak først", "fingerprints"),
    ("kontonummer 1234.56.78903 for utbetaling", "sensitive_tokens"),
    ("Ada Example00 skrev dette.", "person_forms"),
])
def test_a_failing_scan_writes_nothing(workspace, monkeypatch, capsys, text, failing_check):
    """Each seeded leak: non-zero exit, the report printed, and NO file on disk.

    A packager that writes the tarball and then complains has already produced
    the thing someone will copy.
    """
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_document(0, text)])
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code != 0
    assert failing_check in out and "REFUSED" in out
    assert _tarballs(workspace) == []


def test_a_collection_without_a_privacy_stamp_is_refused(workspace, monkeypatch, capsys):
    """No stamp means part of this index may predate aliasing, and no scan can
    tell which part."""
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)], stamp=None)
    code = _package(workspace, monkeypatch)
    assert code != 0
    assert "no privacy stamp" in _combined(capsys, code)
    assert _tarballs(workspace) == []


def test_an_out_of_scope_collection_is_refused(workspace, monkeypatch, capsys):
    """Nothing aliased it, so a clean scan against a map that does not apply to
    it would certify nothing at all."""
    _write_collection(workspace / "collections", "other-collection",
                      documents=[_clean_document(0)])
    code = _package(workspace, monkeypatch, name="other-collection")
    assert code != 0
    assert "not in privacy scope" in _combined(capsys, code)
    assert _tarballs(workspace) == []


def test_a_missing_collection_is_refused(workspace, monkeypatch, capsys):
    code = _package(workspace, monkeypatch, name="nope")
    assert code != 0
    assert "no such collection" in _combined(capsys, code)


def test_a_symlink_inside_the_collection_is_refused(workspace, monkeypatch):
    """A symlink would be archived as a link resolved against the RECIPIENT's
    filesystem — either a dangling path or someone else's file."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    (workspace / "elsewhere.txt").write_text("x", encoding="utf-8")
    (collection / "documents" / "link.json").symlink_to(workspace / "elsewhere.txt")
    members = [{"path": "documents/link.json", "size": 1, "mtime": 0}]
    with pytest.raises(pkg.Refused, match="non-regular file"):
        list(pkg.archive_members(collection, "demo-aliased", members))


def test_the_package_stamp_carries_no_literals(workspace, monkeypatch):
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    monkeypatch.setattr(sys, "argv", [
        "package_collection.py", "--collection", "demo-aliased",
        "--collections-dir", str(workspace / "collections"),
        "--out", str(workspace / "out"),
        "--map", str(workspace / "huginn-x" / "privacy" / "aliases.json"),
    ])
    pkg.main()
    tarball = next((workspace / "out").glob("*.tar.gz"))
    with tarfile.open(tarball) as tar:
        stamp = tar.extractfile("PACKAGE-STAMP.json").read().decode()
    # Counts, versions and check names — nothing read out of the corpus.
    assert "Ada" not in stamp and "Zylphia" not in stamp
    # And the temporary stamp file is not left lying next to the tarball.
    assert [p.name for p in (workspace / "out").iterdir()] == [tarball.name]


# --- the tarball is produced atomically --------------------------------------

def test_a_failure_midway_through_the_tar_leaves_nothing_behind(
        workspace, monkeypatch, capsys):
    """A half-written `.tar.gz` is the worst possible artifact: it has the name
    of a certified package and the contents of an interrupted one. The tar is
    built under a temp name in the same directory and `os.replace`d into place,
    so the final path either does not exist or is complete."""
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    real_add = tarfile.TarFile.add
    calls = []

    def exploding_add(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 3:
            raise OSError("disk full")
        return real_add(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", exploding_add)
    with pytest.raises(OSError, match="disk full"):
        _package(workspace, monkeypatch)
    assert len(calls) == 3
    assert list((workspace / "out").iterdir()) == []


# --- the tar contains exactly what the scan read ------------------------------

def test_a_file_that_appeared_after_the_scan_refuses_the_package(
        workspace, monkeypatch, capsys):
    """Between the scan and the tar the collection is unlocked. A nightly
    reindex finishing in that window would put a document nobody scanned inside
    a tarball stamped as scanned, so the packager tars the member list the scan
    returned and refuses if the directory no longer matches it."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    real_scan = pkg.scan_collection

    def scan_then_meddle(*args, **kwargs):
        report = real_scan(*args, **kwargs)
        (collection / "documents" / "sneaked.json").write_text(
            json.dumps(_document(9, "Kari Ukjentsen skrev dette.")), encoding="utf-8")
        return report

    monkeypatch.setattr(pkg, "scan_collection", scan_then_meddle)
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code != 0
    assert "REFUSED" in out and "sneaked.json" in out
    assert _tarballs(workspace) == []


def test_a_file_that_changed_after_the_scan_refuses_the_package(
        workspace, monkeypatch, capsys):
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    real_scan = pkg.scan_collection

    def scan_then_meddle(*args, **kwargs):
        report = real_scan(*args, **kwargs)
        (collection / "documents" / "doc0.json").write_text(
            json.dumps(_document(0, "Kari Ukjentsen skrev dette i stedet.")),
            encoding="utf-8")
        return report

    monkeypatch.setattr(pkg, "scan_collection", scan_then_meddle)
    code = _package(workspace, monkeypatch)
    assert code != 0 and "doc0.json" in _combined(capsys, code)
    assert _tarballs(workspace) == []


# --- the tar carries no identity ---------------------------------------------

def test_the_tarball_carries_no_owner_names(workspace, monkeypatch):
    """`tar` records the BUILDER's uid, gid and login name on every member. That
    is the same class of distributor fingerprint as an absolute `/Users/` path
    (check 10), just in the header rather than the content."""
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    _package(workspace, monkeypatch)
    tarball = next((workspace / "out").glob("*.tar.gz"))
    with tarfile.open(tarball) as tar:
        members = tar.getmembers()
    assert members
    for member in members:
        assert member.uname == "" and member.gname == ""
        assert member.uid == 0 and member.gid == 0
    listing = subprocess.run(["tar", "-tvf", str(tarball)],
                             capture_output=True, text=True, check=True).stdout
    assert getpass.getuser() not in listing


def test_the_gzip_header_carries_no_temp_name_and_no_build_time(workspace, monkeypatch):
    """`tarfile.open(path, "w:gz")` puts the file it is WRITING into the gzip
    FNAME field — and the packager writes under `.<name>.tar.gz.tmp-<pid>`.

    So the certified artifact shipped the builder's process id in its header,
    which is the same class of fingerprint as the uid/uname above and is not
    visible in `tar -tvf` at all. `gzip -l` and `gzip -N` show it. The build
    time (MTIME) goes with it: it is a second fingerprint and it makes two
    byte-identical packages differ.
    """
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    _package(workspace, monkeypatch)
    tarball = next((workspace / "out").glob("*.tar.gz"))
    header = tarball.read_bytes()[:10]
    assert header[:2] == b"\x1f\x8b"
    flags = header[3]
    assert not flags & 0x08, f"gzip FNAME flag is set (FLG={flags:#04x})"
    assert header[4:8] == b"\x00\x00\x00\x00", "gzip MTIME is not zeroed"
    assert b".tmp-" not in tarball.read_bytes()[:512]


# --- a refusal from inside the scan is a refusal, not a traceback -------------

def test_a_gazetteer_below_the_floor_is_a_refusal_not_a_traceback(
        workspace, monkeypatch, capsys):
    """`scan_collection` raises ValueError when the given-name gazetteer is too
    small to make check 9 mean anything. That came out of `main()` as a
    traceback, which reads as a broken tool rather than as the gate refusing."""
    from main.privacy import index_scan
    monkeypatch.setattr(index_scan, "load_public_given_names", lambda *a, **k: set())
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code != 0 and "REFUSED" in out and "gazetteer" in out
    assert "Traceback" not in out
    assert _tarballs(workspace) == []


def test_an_invalid_explicit_map_is_a_refusal_not_a_traceback(
        workspace, monkeypatch, capsys):
    """`--map` is not the map `refusal()` resolves the scope with, so an
    unusable one first shows up inside `scan_collection` as PrivacyMapInvalid."""
    broken = workspace / "broken-map.json"
    broken.write_text(json.dumps({"version": 7, "entries": []}), encoding="utf-8")
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    monkeypatch.setattr(sys, "argv", [
        "package_collection.py", "--collection", "demo-aliased",
        "--collections-dir", str(workspace / "collections"),
        "--out", str(workspace / "out"), "--map", str(broken),
    ])
    try:
        pkg.main()
        code = 0
    except SystemExit as e:
        code = e.code
    out = _combined(capsys, code)
    assert code != 0 and "REFUSED" in out and "Traceback" not in out
    assert _tarballs(workspace) == []


# --- the same check set as the CLI -------------------------------------------

def test_the_packager_runs_the_compare_checks_when_asked(workspace, monkeypatch, capsys):
    """The CLI and the packager must not certify different things. Without
    `--compare` the exemption invariant and the ident-exception twin never run,
    and a stamp that simply omits them reads exactly like one where they passed.
    """
    _write_collection(workspace / "collections", "demo", documents=[_clean_document(0)])
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    _package(workspace, monkeypatch, extra=("--compare", "demo"))
    tarball = next((workspace / "out").glob("*.tar.gz"))
    with tarfile.open(tarball) as tar:
        stamp = json.loads(tar.extractfile("PACKAGE-STAMP.json").read().decode())
    checks = stamp["scanChecks"]
    assert set(checks) == set(pkg.CHECK_NAMES)
    assert checks["exempt_labels_unmoved"] == {"passed": True, "count": 0, "ran": True}
    assert checks["person_forms"]["ran"] is True
    assert stamp["allowlistSha256"] is not None
    assert stamp["gazetteerSha256"] is not None


def test_a_check_that_did_not_run_is_stamped_as_such(workspace, monkeypatch):
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    _package(workspace, monkeypatch)
    tarball = next((workspace / "out").glob("*.tar.gz"))
    with tarfile.open(tarball) as tar:
        stamp = json.loads(tar.extractfile("PACKAGE-STAMP.json").read().decode())
    assert stamp["scanChecks"]["exempt_labels_unmoved"]["ran"] is False
    assert stamp["scanChecks"]["person_forms"]["ran"] is True


def test_count_drift_against_the_twin_refuses_unless_allowed(workspace, monkeypatch, capsys):
    _write_collection(workspace / "collections", "demo", documents=[_clean_document(0)])
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0), _clean_document(1)])
    code = _package(workspace, monkeypatch, extra=("--compare", "demo"))
    assert code != 0 and _tarballs(workspace) == []
    _package(workspace, monkeypatch,
             extra=("--compare", "demo", "--allow-count-drift"))
    assert len(_tarballs(workspace)) == 1


# --- the filename says what is inside -----------------------------------------

def test_the_filename_carries_the_map_and_policy_version(workspace, monkeypatch):
    """`nav-wiki-2026-08-23.tar.gz` says nothing about which map certified it, so
    two tarballs built the same day from different maps are indistinguishable."""
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    _package(workspace, monkeypatch)
    today = date.today().isoformat()
    assert _tarballs(workspace) == [
        f"demo-aliased-{today}-map7-policy{index_scan.POLICY_VERSION}.tar.gz"]


def test_a_second_package_the_same_day_refuses_without_force(workspace, monkeypatch, capsys):
    """Same collection, same day, same map: the name collides, and silently
    overwriting means the tarball someone already copied out no longer matches
    the one on disk."""
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    _package(workspace, monkeypatch)
    code = _package(workspace, monkeypatch)
    assert code != 0
    assert "already exists" in _combined(capsys, code) and "--force" in _combined(capsys, code)
    _package(workspace, monkeypatch, extra=("--force",))
    assert len(_tarballs(workspace)) == 1


# --- a non-regular file is a refusal, not a traceback -------------------------

def test_a_symlink_in_the_tree_prints_a_refusal(workspace, monkeypatch, capsys):
    """`archive_members` raising ValueError out of `main()` printed a traceback,
    which reads as a broken tool rather than as the gate doing its job."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    (workspace / "elsewhere.txt").write_text("x", encoding="utf-8")
    (collection / "documents" / "link.json").symlink_to(workspace / "elsewhere.txt")
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code != 0 and "REFUSED" in out
    assert "Traceback" not in out
    assert _tarballs(workspace) == []


# --- the local sensitivity sweep gate -----------------------------------------
#
# The deterministic scan certifies that no LISTED name survived. The sweep is the
# second opinion about the people the map never listed, and the packager is where
# its verdict has to bite — a report nobody reads is the advisory scanner this
# whole script exists to replace.

def test_no_sweep_report_warns_but_still_packages(workspace, monkeypatch, capsys):
    """A second opinion, not a prerequisite. Making the hand-off depend on a local
    GPU being up is how a gate gets routed around."""
    _write_collection(workspace / "collections", "demo-aliased",
                      documents=[_clean_document(0)])
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code == 0 and "WARN" in out and "no local sensitivity sweep" in out
    assert len(_tarballs(workspace)) == 1


def test_a_sweep_that_found_an_unknown_person_refuses(workspace, monkeypatch, capsys):
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    _with_last_modified(collection)
    _sweep_report(workspace, "demo-aliased", generated_at="2099-01-01T00:00:00Z", unknown=3)
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code != 0 and "REFUSED" in out and "3 unknown" in out
    # The strings stay in the gitignored report; the refusal is a count.
    assert _tarballs(workspace) == []


def test_a_sweep_of_older_documents_warns(workspace, monkeypatch, capsys):
    """A clean verdict about text that has since changed certifies nothing — but
    it is a warning, not a refusal: the deterministic scan just passed on the
    documents that are actually there."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    _with_last_modified(collection, "2026-08-22T09:00:00")
    _sweep_report(workspace, "demo-aliased", generated_at="2026-08-19T00:00:00Z")
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code == 0 and "WARN" in out and "has since changed" in out
    assert len(_tarballs(workspace)) == 1


def test_a_limited_sweep_does_not_certify(workspace, monkeypatch, capsys):
    """A 1-of-2 sample is a spot check. It warns with its coverage in the line, so
    the operator sees WHAT was read rather than "no second opinion"."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0), _clean_document(1)])
    _with_last_modified(collection)
    _sweep_report(workspace, "demo-aliased", generated_at="2026-08-22T00:00:00Z",
                  documents=1, expected=2, limit=1)
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code == 0 and "WARN" in out and "1/2 documents" in out


def test_a_clean_full_sweep_is_stamped_into_the_package(workspace, monkeypatch, capsys):
    """"Certified without a second opinion" has to be legible in the artifact, not
    only in the console the builder happened to be looking at."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    _with_last_modified(collection)
    _sweep_report(workspace, "demo-aliased", generated_at="2026-08-22T00:00:00Z")
    code = _package(workspace, monkeypatch)
    assert code == 0, _combined(capsys, code)
    with tarfile.open(next((workspace / "out").glob("*.tar.gz"))) as tar:
        stamp = json.loads(tar.extractfile("PACKAGE-STAMP.json").read().decode())
    assert stamp["sensitivitySweep"]["status"] == "pass"


def test_sweep_reports_can_be_pointed_somewhere_else(workspace, monkeypatch, capsys):
    """`--sweep-reports` exists for certifying an unpacked copy on a machine that
    is not the one that swept it — the report travels beside the tarball, not in
    a private sub-repo that machine does not have."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    _with_last_modified(collection)
    elsewhere = workspace / "handover"
    _sweep_report(workspace, "demo-aliased", generated_at="2026-08-22T00:00:00Z",
                  directory=elsewhere)
    code = _package(workspace, monkeypatch, extra=["--sweep-reports", str(elsewhere)])
    assert code == 0, _combined(capsys, code)
    assert "clean (1/1 documents)" in _combined(capsys, code)


def test_a_map_version_the_sweep_never_saw_warns(workspace, monkeypatch, capsys):
    """The map decides which findings are dropped as already-aliased. A verdict
    produced under another one is about a different filter."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    _with_last_modified(collection)
    directory = workspace / "sweeps"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sweep_demo-aliased_2026-08-23_baseline.json").write_text(
        json.dumps({"collection": "demo-aliased", "generatedAt": "2026-08-22T00:00:00Z",
                    "unknownCount": 0, "documents": 1, "documentsExpected": 1,
                    "limit": None, "mapVersion": MAP_VERSION - 1,
                    "policyVersion": index_scan.POLICY_VERSION, "model": DEFAULT_MODEL,
                    "collectionLastModifiedDocumentTime": SWEPT_AT, "findings": []}),
        encoding="utf-8")
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code == 0 and "WARN" in out and "different inputs" in out


def test_drift_is_refused_before_the_sweep_gate_is_consulted(workspace, monkeypatch, capsys):
    """The cheap refusal first: when the collection moved under us, what some
    report says about the previous contents cannot matter."""
    collection = _write_collection(workspace / "collections", "demo-aliased",
                                   documents=[_clean_document(0)])
    _with_last_modified(collection)
    _sweep_report(workspace, "demo-aliased", generated_at="2026-08-22T00:00:00Z", unknown=9)

    real_verify = pkg.verify_members

    def drifting(collection_dir, members):
        (collection_dir / "documents" / "late.json").write_text("{}", encoding="utf-8")
        return real_verify(collection_dir, members)

    monkeypatch.setattr(pkg, "verify_members", drifting)
    code = _package(workspace, monkeypatch)
    out = _combined(capsys, code)
    assert code != 0 and "changed between the scan and the tar" in out
    # The dirty sweep report would also have refused — the point is which line
    # the operator is shown, and it is the one about the directory moving.
    assert "unknown person reference" not in out
