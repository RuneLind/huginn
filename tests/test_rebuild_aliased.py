"""Guards on scripts/audit/rebuild_aliased.py.

The script's whole job is to produce a collection that is safe to swap over a
live one, so every way it can produce a *silently different* collection has to
be a hard stop rather than a warning. Names here are invented.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit"))

import rebuild_aliased  # noqa: E402

STAMP = {"policy_version": 1, "map_version": 7, "aliasedAt": "2026-01-02T00:00:00+00:00"}


class TestStampCheck:
    def test_matching_stamp_passes(self):
        assert rebuild_aliased.stamp_mismatch({"privacy": STAMP}, 1, 7) is None

    def test_missing_stamp_is_a_mismatch(self):
        # The one failure that matters: a build that silently ran unaliased
        # (scope file edited, map moved) looks exactly like a good one on disk.
        assert rebuild_aliased.stamp_mismatch({}, 1, 7)

    @pytest.mark.parametrize("stamp", [
        {**STAMP, "map_version": 6},
        {**STAMP, "policy_version": 0},
    ])
    def test_stale_stamp_is_a_mismatch(self, stamp):
        assert rebuild_aliased.stamp_mismatch({"privacy": stamp}, 1, 7)


class TestReaderCheck:
    def test_identical_reader_passes(self):
        reader = {"type": "localFiles", "basePath": "./data/sources/demo",
                  "includePatterns": [".*"], "excludePatterns": ["^x/.*"]}
        assert rebuild_aliased.reader_mismatch(reader, dict(reader)) is None

    def test_base_path_normalization_is_allowed(self):
        source = {"type": "localFiles", "basePath": str(REPO_ROOT / "data" / "sources" / "demo")}
        built = {"type": "localFiles", "basePath": "./data/sources/demo"}
        assert rebuild_aliased.reader_mismatch(source, built) is None

    @pytest.mark.parametrize("built", [
        {"type": "localFiles", "basePath": "./data/sources/demo", "excludePatterns": []},
        {"type": "localFiles", "basePath": "./data/sources/other"},
    ])
    def test_dropped_or_moved_reader_fields_are_a_mismatch(self, built):
        # A dropped excludePattern silently indexes documents the live
        # collection never had — including the ones excluded for privacy.
        source = {"type": "localFiles", "basePath": "./data/sources/demo",
                  "excludePatterns": ["^x/.*"]}
        assert rebuild_aliased.reader_mismatch(source, built)


class TestBuildCommand:
    def _manifest(self, **reader):
        return {"reader": {"type": "localFiles", "basePath": "./data/sources/demo", **reader},
                "indexers": [{"name": "indexer_BM25"}]}

    def test_empty_include_patterns_fall_back_to_the_default(self):
        cmd = rebuild_aliased.build_command(self._manifest(includePatterns=[]), "c", "c-aliased", 1)
        assert "-includePatterns" in cmd
        assert cmd[cmd.index("-includePatterns") + 1] == ".*"

    def test_empty_exclude_patterns_omit_the_flag(self):
        cmd = rebuild_aliased.build_command(self._manifest(excludePatterns=[]), "c", "c-aliased", 1)
        assert "-excludePatterns" not in cmd

    def test_empty_indexer_list_omits_the_flag_rather_than_passing_nothing(self):
        # `-indexers` with no values is an argparse error that kills the build
        # after the reader has already been resolved.
        manifest = {"reader": {"type": "localFiles", "basePath": "./x"}, "indexers": []}
        assert "-indexers" not in rebuild_aliased.build_command(manifest, "c", "c-aliased", 1)

    def test_contextual_cache_points_at_the_real_collection(self):
        cmd = rebuild_aliased.build_command(self._manifest(), "c", "c-aliased", 1)
        assert cmd[cmd.index("--contextual-cache") + 1] == "./data/contextual_caches/c.json"


def test_parking_path_is_outside_the_collections_dir():
    """A parked pre-alias copy inside data/collections/ is served by any server
    started with a glob, and scanned by the audit sweep as a live collection."""
    parked = rebuild_aliased.parking_path("demo")
    assert rebuild_aliased.COLLECTIONS_DIR not in parked.parents
    assert parked.parent == REPO_ROOT / "data" / "prealias"
    assert parked.name.startswith("demo-")


class TestGuardsAreActuallyCalled:
    """`stamp_mismatch` and `reader_mismatch` are pure functions, so testing them
    directly says nothing about whether build() and swap() still consult them.

    Deleting either call site leaves every other test in this file green while
    turning the script into "move whatever happened to be there over the live
    collection". These two drive the real entry points with a helper forced to
    report a mismatch and require a non-zero exit.
    """

    def _collections(self, tmp_path, monkeypatch, *, built_manifest=None):
        collections = tmp_path / "collections"
        for name, manifest in (("demo", {"reader": {"type": "localFiles",
                                                    "basePath": "./data/sources/demo"}}),
                               ("demo-aliased", built_manifest or {
                                   "reader": {"type": "localFiles",
                                              "basePath": "./data/sources/demo"},
                                   "numberOfDocuments": 1, "numberOfChunks": 1,
                                   "privacy": STAMP})):
            (collections / name).mkdir(parents=True)
            (collections / name / "manifest.json").write_text(json.dumps(manifest),
                                                              encoding="utf-8")
        monkeypatch.setattr(rebuild_aliased, "COLLECTIONS_DIR", collections)
        monkeypatch.setattr(rebuild_aliased, "PREALIAS_DIR", tmp_path / "prealias")
        monkeypatch.setattr(rebuild_aliased, "current_map_version", lambda: 7)
        return collections

    def test_build_refuses_when_the_reader_block_moved(self, tmp_path, monkeypatch):
        self._collections(tmp_path, monkeypatch)
        monkeypatch.setattr(rebuild_aliased.subprocess, "run", lambda *a, **k: None)
        monkeypatch.setattr(rebuild_aliased, "stamp_mismatch", lambda *a: None)
        monkeypatch.setattr(rebuild_aliased, "reader_mismatch",
                            lambda *a: "reader block differs on ['excludePatterns']")
        with pytest.raises(SystemExit) as excinfo:
            rebuild_aliased.build("demo", "demo-aliased", 1)
        assert excinfo.value.code

    def test_build_completes_when_both_guards_are_satisfied(self, tmp_path, monkeypatch):
        # Control for the two refusal tests: without it they would also pass if
        # build() had started exiting for some unrelated reason.
        collections = self._collections(tmp_path, monkeypatch)
        monkeypatch.setattr(rebuild_aliased.subprocess, "run", lambda *a, **k: None)
        rebuild_aliased.build("demo", "demo-aliased", 1)
        written = json.loads((collections / "demo-aliased" / "manifest.json")
                             .read_text(encoding="utf-8"))
        assert written["collectionName"] == "demo"

    def test_swap_refuses_when_the_privacy_stamp_is_stale(self, tmp_path, monkeypatch):
        collections = self._collections(tmp_path, monkeypatch)
        monkeypatch.setattr(rebuild_aliased, "stamp_mismatch", lambda *a: "map_version 6 != 7")
        with pytest.raises(SystemExit) as excinfo:
            rebuild_aliased.swap("demo", "demo-aliased", "http://127.0.0.1:8321")
        assert excinfo.value.code
        # Nothing moved: the live collection is still where it was.
        assert (collections / "demo" / "manifest.json").exists()
        assert not (tmp_path / "prealias").exists()
