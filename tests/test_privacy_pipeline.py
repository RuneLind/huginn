"""Build-pipeline behaviour of build-time aliasing.

Three things this file exists to pin:

1. An out-of-scope collection is byte-identical to what the converter produced
   before this feature existed — asserted against a snapshot committed alongside
   the fixture sources, over the three converter shapes (plain, tags/epic
   metadata, session frontmatter) plus a truncated deep path.
2. An in-scope collection with no alias map FAILS, on both construction sites,
   before anything is written.
3. The manifest carries the privacy stamp on both the create and the update
   branch, and the contextual-prefix cache is invalidated per changed document.

Invented names only; the real map is never read here.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from main.core.contextual_prefix.cache import ContextualCache
from main.core.documents_collection_creator import DocumentCollectionCreator, OPERATION_TYPE
from main.privacy.alias_registry import AliasRegistry, PrivacyMapMissing
from main.sources.files.files_document_converter import FilesDocumentConverter
from main.sources.files.files_document_reader import FilesDocumentReader

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "privacy"
SOURCE_DOCS = FIXTURES / "docs"
SNAPSHOT = FIXTURES / "converter_snapshot.json"

# A public scope entry, so resolve_registry() arms on the name alone.
IN_SCOPE_COLLECTION = "jira-issues"

FIXTURE_MAP = {
    "version": 7,
    "entries": [
        {"alias": "dev-01", "name": "Ada Example", "role": "dev",
         "variants": ["Ada Example [X]", "Example, Ada", "Ada Example",
                      "ada.example", "ada_example", "example.ada", "example_ada"],
         "require_full_name": False, "idents": [], "departed": False,
         "confirmed": True, "extra_variants": []},
        {"alias": "fag-01", "name": "Bo Tester", "role": "fag",
         "variants": ["Bo Tester [X]", "Tester, Bo", "Bo Tester",
                      "bo.tester", "bo_tester", "tester.bo", "tester_bo"],
         "require_full_name": False, "idents": [], "departed": False,
         "confirmed": True, "extra_variants": []},
    ],
    "ident_policy": "redact",
    # "Bo Tester-rutinen" contains a mapped person variant on purpose: the
    # exemption assertions below fail if exempt labels stop winning.
    "non_person_labels": ["utsendt arbeidstaker", "Bo Tester-rutinen"],
    "unmapped_people": ["Kari Ukjent"],
    "unmapped_people_variants": {
        "Kari Ukjent": ["Kari Ukjent [X]", "Ukjent, Kari", "Kari Ukjent",
                        "kari.ukjent", "ukjent.kari"],
    },
    "person_redaction_token": "[~ukjent-person]",
    "retired_aliases": ["dev-99"],
}


def convert_all(alias_registry=None):
    reader = FilesDocumentReader(base_path=str(SOURCE_DOCS))
    converter = FilesDocumentConverter(alias_registry=alias_registry)
    out = {}
    for document in reader.read_all_documents():
        for converted in converter.convert(document):
            out[converted["id"]] = converted
    return out


def normalized(converted):
    """Drop the two machine-dependent fields the snapshot cannot pin."""
    document = dict(converted)
    document.pop("modifiedTime", None)
    if document["url"].startswith("file://"):
        document["url"] = "file://<BASE>/" + document["id"]
    return document


# --- 1. out-of-scope regression --------------------------------------------

def test_converter_without_registry_is_byte_identical_to_the_snapshot():
    """The 28 collections outside privacy scope must not move by one byte.

    The snapshot was taken from the converter as it stood before the alias hook
    existed. Covers all three shapes: plain markdown (code-block/image/S3
    stripping), frontmatter with tags + epic_summary (the injected context
    lines), and a session_id document (the is_session splitter path, which keeps
    code blocks).
    """
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = {doc_id: normalized(doc) for doc_id, doc in convert_all(alias_registry=None).items()}
    assert actual == expected


def test_snapshot_covers_all_three_converter_shapes():
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert "session_id" in json.dumps(expected["session.md"]["chunks"][0]["metadata"])
    assert "epic:" in expected["tagged.md"]["chunks"][0]["indexedData"]
    assert "tags:" in expected["tagged.md"]["chunks"][0]["indexedData"]
    assert expected["plain.md"]["text"].count("[file]") == 1        # S3 url stripped
    assert "```" not in expected["plain.md"]["text"]                 # code block stripped
    assert "```bash" in expected["session.md"]["chunks"][0]["indexedData"]  # …but kept for sessions
    assert expected["Team/Sub/Deeper/nested-deep-page.md"]["text"].startswith(
        "[Team > Sub > Deeper > nested-deep-page]")


def test_registry_changes_every_shape_including_the_session_path():
    registry = AliasRegistry(FIXTURE_MAP)
    documents = convert_all(alias_registry=registry)
    blob = json.dumps(documents, ensure_ascii=False)
    assert "Ada Example" not in blob
    assert "Tester, Bo" not in blob
    assert "bo.tester" not in blob
    assert "Q000124" not in blob
    assert "dev-01" in documents["session.md"]["chunks"][0]["indexedData"]
    assert "[~person]" in documents["session.md"]["chunks"][0]["indexedData"]
    # Exempt label survives, non-vacuously: it contains "Bo Tester".
    assert "Bo Tester-rutinen" in documents["clean.md"]["text"]
    # Unmapped person redacted rather than aliased.
    assert documents["unmapped.md"]["text"].count("[~ukjent-person]") == 3
    # ids and urls are untouched even though the fixture path is otherwise aliased text
    assert documents["plain.md"]["id"] == "plain.md"


def test_alias_changed_marker_tracks_whether_the_document_actually_moved():
    """The marker drives contextual-prefix invalidation, so a false positive
    re-prefixes a document through the LLM for nothing, and a false negative
    replays a pre-alias prefix."""
    documents = convert_all(alias_registry=AliasRegistry(FIXTURE_MAP))
    assert documents["plain.md"].get("_aliasChanged") is True
    # clean.md's only person-shaped text sits inside an exempt label.
    assert "_aliasChanged" not in documents["clean.md"]


# --- 2. fail-closed on both construction sites ------------------------------

@pytest.fixture
def mapless_cwd(tmp_path, monkeypatch):
    """A working directory where the huginn-*/privacy/aliases.json glob finds nothing."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def mapped_cwd(tmp_path, monkeypatch):
    privacy_dir = tmp_path / "huginn-fixture" / "privacy"
    privacy_dir.mkdir(parents=True)
    (privacy_dir / "aliases.json").write_text(json.dumps(FIXTURE_MAP), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_update_factory_raises_when_an_in_scope_collection_has_no_map(mapless_cwd):
    from main.factories.update_collection_factory import _build_local_files
    manifest = {
        "collectionName": IN_SCOPE_COLLECTION,
        "lastModifiedDocumentTime": "2026-01-01T00:00:00",
        "reader": {"type": "localFiles", "basePath": str(SOURCE_DOCS)},
    }
    with pytest.raises(PrivacyMapMissing):
        _build_local_files(manifest)


def test_update_factory_arms_the_converter_when_the_map_is_present(mapped_cwd):
    """The nightly path: the converter the update factory builds must alias."""
    from main.factories.update_collection_factory import _build_local_files
    manifest = {
        "collectionName": IN_SCOPE_COLLECTION,
        "lastModifiedDocumentTime": "2026-01-01T00:00:00",
        "reader": {"type": "localFiles", "basePath": str(SOURCE_DOCS)},
    }
    _, converter = _build_local_files(manifest)
    assert converter.alias_registry is not None

    reader = FilesDocumentReader(base_path=str(SOURCE_DOCS))
    document = next(d for d in reader.read_all_documents() if d["fileRelativePath"] == "plain.md")
    converted = converter.convert(document)[0]
    assert "Ada Example" not in json.dumps(converted, ensure_ascii=False)


def test_update_factory_re_arms_from_the_manifest_stamp_alone(mapped_cwd):
    """The stamp outranks the scope files: a collection built aliased stays
    aliased even after its name or basePath drops out of scope, because
    un-aliasing half a collection on the next nightly run is the worse failure."""
    from main.factories.update_collection_factory import _build_local_files
    manifest = {
        "collectionName": "some-unrelated-collection",
        "lastModifiedDocumentTime": "2026-01-01T00:00:00",
        "reader": {"type": "localFiles", "basePath": str(SOURCE_DOCS)},
        "privacy": {"policy_version": 1, "map_version": 7, "aliasedAt": "2026-01-01T00:00:00+00:00"},
    }
    _, converter = _build_local_files(manifest)
    assert converter.alias_registry is not None


def test_update_factory_leaves_an_out_of_scope_collection_alone(mapped_cwd):
    from main.factories.update_collection_factory import _build_local_files
    manifest = {
        "collectionName": "some-unrelated-collection",
        "lastModifiedDocumentTime": "2026-01-01T00:00:00",
        "reader": {"type": "localFiles", "basePath": str(SOURCE_DOCS)},
    }
    _, converter = _build_local_files(manifest)
    assert converter.alias_registry is None


def test_cli_adapter_refuses_to_build_an_in_scope_collection_without_a_map(tmp_path):
    """The create path removes the collection folder first, so it must fail EARLIER."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    docs = tmp_path / "src"
    shutil.copytree(SOURCE_DOCS, docs)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "files_collection_create_cmd_adapter.py"),
         "-collection", IN_SCOPE_COLLECTION, "-basePath", str(docs)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode != 0
    assert "PrivacyMapMissing" in result.stderr
    assert not (tmp_path / "data" / "collections" / IN_SCOPE_COLLECTION).exists()


def test_update_adapter_records_a_failed_run_when_the_map_is_missing(tmp_path):
    """PrivacyMapMissing is raised while the updater is being BUILT.

    With construction outside the try, a nightly job that fails closed left no
    ledger record at all — the dashboard would show the last successful run and
    nothing to say the collection had stopped updating.
    """
    collections = tmp_path / "data" / "collections" / IN_SCOPE_COLLECTION
    collections.mkdir(parents=True)
    (collections / "manifest.json").write_text(json.dumps({
        "collectionName": IN_SCOPE_COLLECTION,
        "lastModifiedDocumentTime": "2026-01-01T00:00:00",
        "reader": {"type": "localFiles", "basePath": str(SOURCE_DOCS)},
        "indexers": [{"name": "indexer_BM25"}],
    }), encoding="utf-8")
    runs = tmp_path / "runs"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "collection_update_cmd_adapter.py"),
         "-collection", IN_SCOPE_COLLECTION],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "HUGINN_RUNS_DIR": str(runs)},
    )
    assert result.returncode != 0
    assert "PrivacyMapMissing" in result.stderr

    records = [json.loads(line) for line
               in (runs / f"{IN_SCOPE_COLLECTION}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in records] == ["failed"]


# --- 3. manifest stamp + cache invalidation ---------------------------------

class _FakeIndexer:
    def get_name(self):
        return "indexer_BM25"

    def get_size(self):
        return 3


class _FakeReader:
    def get_reader_details(self):
        return {"type": "localFiles", "basePath": "./data/sources/demo"}


class _FakePersister:
    def __init__(self, documents):
        self._documents = documents

    def read_folder_files(self, path):
        return self._documents


def _manifest_content(converter, existing_manifest=None):
    creator = DocumentCollectionCreator(
        collection_name="col",
        document_reader=_FakeReader(),
        document_converter=converter,
        document_indexers=[_FakeIndexer()],
        persister=_FakePersister(["a.json"]),
        operation_type=OPERATION_TYPE.UPDATE if existing_manifest else OPERATION_TYPE.CREATE,
    )
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    return creator._DocumentCollectionCreator__create_manifest_content(
        now, now, 3, existing_manifest=existing_manifest)


def test_manifest_stamp_on_the_create_branch():
    converter = FilesDocumentConverter(alias_registry=AliasRegistry(FIXTURE_MAP))
    manifest = _manifest_content(converter)
    assert manifest["privacy"] == {"policy_version": 1, "map_version": 7,
                                   "aliasedAt": "2026-01-02T03:04:05+00:00"}


def _existing_manifest(**extra):
    return {"collectionName": "col", "reader": {"type": "localFiles"},
            "indexers": [{"name": "indexer_BM25"}], **extra}


def test_update_branch_never_adds_a_privacy_stamp():
    """A nightly update re-converts only documents newer than the last run.

    Stamping there would assert a collection whose other 99% was built before
    aliasing existed is clean — the stamp has to mean "this whole index was
    built aliased", which only the create branch can promise.
    """
    converter = FilesDocumentConverter(alias_registry=AliasRegistry(FIXTURE_MAP))
    manifest = _manifest_content(converter, existing_manifest=_existing_manifest())
    assert "privacy" not in manifest
    assert manifest["reader"] == {"type": "localFiles"}      # merge preserved


def test_update_branch_preserves_an_existing_privacy_stamp():
    """…and never rewrites one: the stamp is what re-arms aliasing next time."""
    stamp = {"policy_version": 1, "map_version": 6, "aliasedAt": "2026-01-01T00:00:00+00:00"}
    converter = FilesDocumentConverter(alias_registry=AliasRegistry(FIXTURE_MAP))
    manifest = _manifest_content(converter, existing_manifest=_existing_manifest(privacy=stamp))
    assert manifest["privacy"] == stamp


def test_no_privacy_stamp_without_a_registry():
    manifest = _manifest_content(FilesDocumentConverter())
    assert "privacy" not in manifest


def test_contextual_cache_invalidate_doc_is_scoped_to_one_document(tmp_path):
    cache = ContextualCache(str(tmp_path / "cache.json"))
    cache.put("a.md", "chunk one", "model-x", "prefix 1")
    cache.put("a.md", "chunk two", "model-x", "prefix 2")
    cache.put("b.md", "chunk one", "model-x", "prefix 3")

    assert cache.invalidate_doc("a.md") == 2
    assert len(cache) == 1
    assert cache.get("b.md", "chunk one", "model-x") == "prefix 3"
    assert cache.get("a.md", "chunk one", "model-x") is None
    assert cache.invalidate_doc("missing.md") == 0


def test_contextual_cache_invalidate_doc_matches_the_whole_document_id(tmp_path):
    """`::` is the key separator, so a prefix test also eats sibling documents.

    Session collections have ids like `sess-1::part-2`; invalidating `sess-1`
    must not wipe them.
    """
    cache = ContextualCache(str(tmp_path / "cache.json"))
    cache.put("a", "chunk one", "model-x", "prefix 1")
    cache.put("a::b", "chunk one", "model-x", "prefix 2")

    assert cache.invalidate_doc("a") == 1
    assert cache.get("a::b", "chunk one", "model-x") == "prefix 2"


class _RecordingCache:
    def __init__(self):
        self.invalidated = []

    def invalidate_doc(self, doc_id):
        self.invalidated.append(doc_id)
        return 1


class _RecordingPrefixer:
    def __init__(self, cache):
        self.cache = cache
        self.prefixed = []

    def prefix_document(self, converted_document):
        # Any prefix generated here would be derived from the document text as it
        # is NOW — which is why invalidation has to happen before this runs.
        self.prefixed.append(converted_document["id"])


def test_pipeline_invalidates_prefixes_only_for_documents_the_aliasing_changed(tmp_path):
    cache = _RecordingCache()
    prefixer = _RecordingPrefixer(cache)
    creator = DocumentCollectionCreator(
        collection_name="col",
        document_reader=None,
        document_converter=None,
        document_indexers=[_FakeIndexer()],
        persister=_FakePersister([]),
        chunk_prefixer=prefixer,
    )
    invalidate = creator._DocumentCollectionCreator__invalidate_prefix_cache_if_aliased

    changed = {"id": "changed.md", "_aliasChanged": True}
    unchanged = {"id": "unchanged.md"}
    invalidate(changed)
    invalidate(unchanged)

    assert cache.invalidated == ["changed.md"]
    # the marker must never reach the persisted document JSON
    assert "_aliasChanged" not in changed
