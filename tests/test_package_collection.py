"""Guards on scripts/audit/package_collection.py — the hand-off gate.

The point of the packager is that it *produces* the artifact: a scan someone has
to remember to run before copying a directory is advisory, and the one time it
is skipped is the time it mattered. So the tests that matter are the ones where
a failing scan must leave NOTHING behind, and where a collection that was never
aliased must not be packaged at all.

Every name is invented ("Kari Ukjent", "Ada Example"); the fixtures come from
tests/test_scan_index.py.
"""
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import package_collection as pkg  # noqa: E402
import scan_index as cli  # noqa: E402
from tests.test_scan_index import _clean_document, _document, _map, _write_collection  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A tmp repo root whose scope, map and collections are all discoverable.

    `monkeypatch.chdir` is what makes `resolve_registry` see the tmp scope file:
    the private scope and map globs are resolved against the process CWD, the
    same way every other privacy path in this repo is.
    """
    privacy = tmp_path / "huginn-x" / "privacy"
    privacy.mkdir(parents=True)
    (privacy / "aliases.json").write_text(json.dumps(_map()), encoding="utf-8")
    (privacy / "scope.json").write_text(
        json.dumps({"collections": ["demo-aliased"], "basePaths": []}), encoding="utf-8")
    (tmp_path / "collections").mkdir()
    (tmp_path / "out").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    return tmp_path


def _package(workspace, monkeypatch, name="demo-aliased", extra=()):
    monkeypatch.setattr(sys, "argv", [
        "package_collection.py", "--collection", name,
        "--collections-dir", str(workspace / "collections"),
        "--out", str(workspace / "out"),
        "--map", str(workspace / "huginn-x" / "privacy" / "aliases.json"),
        *extra,
    ])
    with pytest.raises(SystemExit) as excinfo:
        pkg.main()
        return 0
    return excinfo.value.code


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
    assert stamp["map_version"] == 7 and stamp["policy_version"] == 1
    assert stamp["numberOfDocuments"] == 1
    assert stamp["scanChecks"]["bigram_candidates"] == 0


@pytest.mark.parametrize("text,failing_check", [
    ("Notatet ble skrevet av Kari Ukjent.", "bigram_candidates"),
    ("bygget fra /Users/someone/source/huginn", "fingerprints"),
    ("gjenopprett fra manifest.json.bak først", "fingerprints"),
    ("Arbeidsgiver med orgnr 987654325.", "sensitive_tokens"),
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
    with pytest.raises(ValueError, match="non-regular file"):
        list(pkg.archive_members(collection, "demo-aliased"))


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
