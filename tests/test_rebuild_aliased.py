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


def _mapping(prefix, ids=("a.md", "b.md")):
    return {str(n): {"documentId": doc_id,
                     "documentUrl": f"file://./src/{doc_id}",
                     "documentPath": f"{prefix}/documents/{doc_id}.json",
                     "chunkNumber": 0}
            for n, doc_id in enumerate(ids)}


def _write_index_artifacts(collection_dir: Path, prefix: str):
    indexes = collection_dir / "indexes"
    indexes.mkdir(parents=True, exist_ok=True)
    (indexes / "index_document_mapping.json").write_text(
        json.dumps(_mapping(prefix), indent=2), encoding="utf-8")
    (indexes / "reverse_index_document_mapping.json").write_text(
        json.dumps({"a.md": [0], "b.md": [1]}), encoding="utf-8")
    (indexes / "index_info.json").write_text(json.dumps({"lastIndexItemId": 1}),
                                             encoding="utf-8")
    (indexes / "indexer_BM25").mkdir(exist_ok=True)
    # A binary artifact next to the JSON: the sweep must not try to decode it.
    (indexes / "indexer_BM25" / "indexer").write_bytes(b"\x80\x04\x95\xff\xfe")
    return indexes


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
        # A real build always leaves these; build() rewrites the mapping's
        # documentPath prefixes and refuses if any survive.
        _write_index_artifacts(collections / "demo-aliased", "demo-aliased")
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


class TestDocumentPathRewrite:
    """The bug this exists for: `swap()` renames the directory, but every
    `documentPath` in the mapping still said `<name>-aliased/documents/…`, so
    after the swap the searcher read no chunk text at all — retrieval worked and
    every result was scored on an empty string."""

    def test_every_document_path_is_repointed_at_the_real_name(self, tmp_path):
        built = tmp_path / "demo-aliased"
        indexes = _write_index_artifacts(built, "demo-aliased")
        rewritten = rebuild_aliased.rewrite_document_paths(built, "demo-aliased", "demo")
        mapping = json.loads((indexes / "index_document_mapping.json").read_text(encoding="utf-8"))
        assert rewritten == 2
        assert [e["documentPath"] for e in mapping.values()] == [
            "demo/documents/a.md.json", "demo/documents/b.md.json"]
        # Only the prefix moves; ids and urls are the join keys to the source.
        assert mapping["0"]["documentId"] == "a.md"
        assert mapping["0"]["documentUrl"] == "file://./src/a.md"

    def test_the_rewrite_leaves_no_partial_file_behind(self, tmp_path):
        built = tmp_path / "demo-aliased"
        indexes = _write_index_artifacts(built, "demo-aliased")
        rebuild_aliased.rewrite_document_paths(built, "demo-aliased", "demo")
        assert sorted(p.name for p in indexes.iterdir()) == [
            "index_document_mapping.json", "index_info.json", "indexer_BM25",
            "reverse_index_document_mapping.json"]

    def test_an_already_rewritten_mapping_is_untouched(self, tmp_path):
        built = tmp_path / "demo-aliased"
        indexes = _write_index_artifacts(built, "demo")
        before = (indexes / "index_document_mapping.json").read_text(encoding="utf-8")
        assert rebuild_aliased.rewrite_document_paths(built, "demo-aliased", "demo") == 0
        assert (indexes / "index_document_mapping.json").read_text(encoding="utf-8") == before

    def test_stale_sweep_is_clean_after_the_rewrite(self, tmp_path):
        built = tmp_path / "demo-aliased"
        _write_index_artifacts(built, "demo-aliased")
        assert rebuild_aliased.stale_temp_paths(built, "demo-aliased")
        rebuild_aliased.rewrite_document_paths(built, "demo-aliased", "demo")
        assert rebuild_aliased.stale_temp_paths(built, "demo-aliased") == []

    def test_stale_sweep_names_any_indexes_json_not_just_the_mapping(self, tmp_path):
        """The sweep re-derives what holds the temp name instead of trusting the
        one-file list the rewrite works from."""
        built = tmp_path / "demo-aliased"
        indexes = _write_index_artifacts(built, "demo")
        (indexes / "something_else.json").write_text(
            json.dumps({"path": "demo-aliased/documents/a.md.json"}), encoding="utf-8")
        assert rebuild_aliased.stale_temp_paths(built, "demo-aliased") == [
            "indexes/something_else.json"]


class TestSwapRefusesAStalePrefix:

    def _built(self, tmp_path, monkeypatch, prefix):
        collections = tmp_path / "collections"
        for name in ("demo", "demo-aliased"):
            (collections / name).mkdir(parents=True)
            (collections / name / "manifest.json").write_text(
                json.dumps({"reader": {"type": "localFiles", "basePath": "./data/sources/demo"},
                            "numberOfDocuments": 2, "numberOfChunks": 2, "privacy": STAMP}),
                encoding="utf-8")
        _write_index_artifacts(collections / "demo-aliased", prefix)
        monkeypatch.setattr(rebuild_aliased, "COLLECTIONS_DIR", collections)
        monkeypatch.setattr(rebuild_aliased, "PREALIAS_DIR", tmp_path / "prealias")
        monkeypatch.setattr(rebuild_aliased, "current_map_version", lambda: 7)
        return collections

    def test_swap_refuses_a_mapping_that_still_carries_the_temp_prefix(
            self, tmp_path, monkeypatch):
        collections = self._built(tmp_path, monkeypatch, "demo-aliased")
        with pytest.raises(SystemExit) as excinfo:
            rebuild_aliased.swap("demo", "demo-aliased", "http://127.0.0.1:8321")
        assert excinfo.value.code
        # Nothing moved: a swap that would have served empty chunk texts.
        assert (collections / "demo" / "manifest.json").exists()
        assert not (tmp_path / "prealias").exists()

    def test_swap_proceeds_once_the_prefix_is_the_real_name(self, tmp_path, monkeypatch):
        collections = self._built(tmp_path, monkeypatch, "demo")
        reloaded = []
        monkeypatch.setattr(rebuild_aliased.urllib.request, "urlopen",
                            lambda request, timeout=None: reloaded.append(request) or _NullResponse())
        rebuild_aliased.swap("demo", "demo-aliased", "http://127.0.0.1:8321")
        assert not (collections / "demo-aliased").exists()
        assert (collections / "demo" / "indexes" / "index_document_mapping.json").exists()
        assert reloaded


class _NullResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b'{"status":"reloaded"}'


def test_build_rewrites_the_document_paths_it_just_built(tmp_path, monkeypatch):
    collections = tmp_path / "collections"
    (collections / "demo").mkdir(parents=True)
    (collections / "demo" / "manifest.json").write_text(
        json.dumps({"reader": {"type": "localFiles", "basePath": "./data/sources/demo"}}),
        encoding="utf-8")
    (collections / "demo-aliased").mkdir(parents=True)
    (collections / "demo-aliased" / "manifest.json").write_text(
        json.dumps({"reader": {"type": "localFiles", "basePath": "./data/sources/demo"},
                    "numberOfDocuments": 2, "numberOfChunks": 2, "privacy": STAMP}),
        encoding="utf-8")
    _write_index_artifacts(collections / "demo-aliased", "demo-aliased")
    monkeypatch.setattr(rebuild_aliased, "COLLECTIONS_DIR", collections)
    monkeypatch.setattr(rebuild_aliased, "current_map_version", lambda: 7)
    monkeypatch.setattr(rebuild_aliased.subprocess, "run", lambda *a, **k: None)

    rebuild_aliased.build("demo", "demo-aliased", 1)

    assert rebuild_aliased.stale_temp_paths(collections / "demo-aliased", "demo-aliased") == []
