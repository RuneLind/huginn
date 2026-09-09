"""Tests for ``GET /api/document/{collection}/{doc_id}?raw=1``.

The stored document JSON carries the CLEANED text (fenced code dropped, images
rewritten, a breadcrumb prepended), so it cannot round-trip a source file. The
raw form serves the bytes on disk instead, for callers that re-ingest a document
they first read back — muninn's capture re-run splits a summary at its
``## Transcript`` heading and posts the transcript back, and a lossy read would
shrink the appendix a little more on every run.

Fixture shape (a real BM25 localFiles collection under an isolated CWD) mirrors
``tests/test_document_delete.py`` — the raw route reuses that route's basePath,
containment and index-membership helpers, so it inherits the same rejections.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from knowledge_api_server import app
from main.core.documents_collection_creator import DocumentCollectionCreator, OPERATION_TYPE
from main.indexes.indexers.bm25_indexer import BM25Indexer
from main.persisters.disk_persister import DiskPersister
from main.runtime.knowledge_store import KnowledgeStore, get_store
from main.sources.files.files_document_converter import FilesDocumentConverter
from main.sources.files.files_document_reader import FilesDocumentReader


COLLECTION = "fixture-collection"
SOURCE_REL = "./data/sources/fixture-collection"
COLLECTIONS_REL = "./data/collections"

#: The document the whole point rests on: a fenced code block and a
#: ``## Transcript`` appendix, neither of which survives into the stored
#: ``text``. Trailing whitespace and a CRLF line are deliberate — "byte
#: identical" has to mean bytes, not "equal after normalization".
SUMMARY_DOC = (
    "# A captured talk\n"
    "\n"
    "The summary body, with trailing spaces here:   \n"
    "\r\n"
    "```python\n"
    "print(\"fenced-code-canary\")\n"
    "```\n"
    "\n"
    "## Transcript\n"
    "\n"
    "transcript-canary: the appendix a re-run has to post back unchanged.\n"
)


def _write_sources(docs: dict[str, str], source_rel: str = SOURCE_REL) -> None:
    source_dir = os.path.abspath(source_rel)
    for rel, body in docs.items():
        path = os.path.join(source_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(body)


def _build_fixture_collection(
    docs: dict[str, str],
    name: str = COLLECTION,
    source_rel: str = SOURCE_REL,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Create a real localFiles collection from ``docs`` (relative CWD paths).

    BM25-only: the one indexer that needs no embedding model, so the full create
    path runs in-process in well under a second.
    """
    _write_sources(docs, source_rel)

    DocumentCollectionCreator(
        collection_name=name,
        # A RELATIVE basePath, like the real x-articles/anthropic-docs manifests:
        # this is what pins the endpoint's "resolve against the server CWD" rule.
        document_reader=FilesDocumentReader(
            base_path=source_rel,
            include_patterns=[".*"],
            exclude_patterns=exclude_patterns or [],
        ),
        document_converter=FilesDocumentConverter(),
        document_indexers=[BM25Indexer("indexer_BM25")],
        persister=DiskPersister(base_path=COLLECTIONS_REL),
        operation_type=OPERATION_TYPE.CREATE,
    ).run()


def _indexed_document_ids(name: str = COLLECTION) -> set[str]:
    with open(
        os.path.join(COLLECTIONS_REL, name, "indexes", "index_document_mapping.json"),
        encoding="utf-8",
    ) as f:
        return {entry["documentId"] for entry in json.load(f).values()}


def _source_bytes(rel: str, source_rel: str = SOURCE_REL) -> bytes:
    with open(os.path.join(source_rel, rel), "rb") as f:
        return f.read()


class _RawCase:
    """Shared TestClient wiring + a store that serves ``COLLECTION``."""

    def _store(self, extra_collections=()) -> KnowledgeStore:
        store = KnowledgeStore()
        store.disk_persister = DiskPersister(base_path=COLLECTIONS_REL)
        store.searchers[COLLECTION] = object()  # makes has_collection() true
        for name in extra_collections:
            store.searchers[name] = object()
        store._build_aux_indexes = False
        return store

    def _client(self, store=None) -> TestClient:
        app.dependency_overrides[get_store] = lambda: store or self._store()
        return TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.pop(get_store, None)


@pytest.fixture
def fixture_collection(tmp_path, monkeypatch):
    """A real BM25 localFiles collection in an isolated CWD.

    ``monkeypatch.chdir`` keeps every relative path in play (the manifest's
    basePath and the update factory's hardcoded ``./data/collections``) pointed
    at tmp_path — no real ``data/`` directory is touched.
    """
    monkeypatch.chdir(tmp_path)
    _build_fixture_collection({
        "talk.md": SUMMARY_DOC,
        "nested/deep-talk.md": SUMMARY_DOC,
        "småprat.md": SUMMARY_DOC,  # non-ASCII id: header encoding
    })
    return tmp_path


class TestRawReadServesTheSourceFile(_RawCase):

    def test_raw_returns_the_exact_bytes_the_json_text_has_lost(
        self, fixture_collection
    ):
        client = self._client()
        on_disk = _source_bytes("talk.md")

        raw = client.get(f"/api/document/{COLLECTION}/talk.md?raw=1")
        cleaned = client.get(f"/api/document/{COLLECTION}/talk.md").json()

        assert raw.status_code == 200
        assert raw.content == on_disk
        assert raw.headers["content-type"] == "text/markdown; charset=utf-8"
        assert raw.headers["x-huginn-source-path"] == "talk.md"

        # The two halves of the point: the raw body carries the fence and the
        # appendix, and the stored text is the lossy copy that does not.
        assert b"fenced-code-canary" in raw.content
        assert b"## Transcript" in raw.content
        assert "fenced-code-canary" not in cleaned["text"]
        assert "transcript-canary" in cleaned["text"]

    def test_raw_read_touches_nothing(self, fixture_collection):
        before = _indexed_document_ids()
        mapping = os.path.join(
            COLLECTIONS_REL, COLLECTION, "indexes", "index_document_mapping.json"
        )
        stat_before = os.stat(mapping)
        derived = os.path.join(COLLECTIONS_REL, COLLECTION, "documents", "talk.md.json")
        derived_before = _source_bytes(derived, ".")

        store = self._store()
        assert self._client(store).get(
            f"/api/document/{COLLECTION}/talk.md?raw=1"
        ).status_code == 200

        assert _indexed_document_ids() == before
        assert os.stat(mapping).st_mtime_ns == stat_before.st_mtime_ns
        assert _source_bytes(derived, ".") == derived_before
        assert _source_bytes("talk.md") == SUMMARY_DOC.encode("utf-8")
        # No reindex was queued — a read must not move the collection at all.
        assert store.get_update_status(COLLECTION)["status"] == "idle"

    def test_raw_nested_document_id_reports_its_base_path_relative_path(
        self, fixture_collection
    ):
        resp = self._client().get(f"/api/document/{COLLECTION}/nested/deep-talk.md?raw=1")

        assert resp.status_code == 200
        assert resp.content == _source_bytes("nested/deep-talk.md")
        assert resp.headers["x-huginn-source-path"] == "nested/deep-talk.md"

    def test_raw_non_ascii_document_id(self, fixture_collection):
        # A header value is latin-1 on the wire, so an unencoded 'å' in the
        # source path raises at response-render time — a 500 on a perfectly
        # ordinary Norwegian wiki page.
        resp = self._client().get(f"/api/document/{COLLECTION}/småprat.md?raw=1")

        assert resp.status_code == 200
        assert resp.content == _source_bytes("småprat.md")
        assert resp.headers["x-huginn-source-path"] == "sm%C3%A5prat.md"

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True"])
    def test_raw_accepts_one_and_true_case_insensitively(
        self, fixture_collection, value
    ):
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw={value}")

        assert resp.status_code == 200
        assert resp.content == _source_bytes("talk.md")

    @pytest.mark.parametrize("query", ["", "?raw=0", "?raw=false", "?raw=yes", "?raw="])
    def test_absent_or_unrecognized_raw_is_the_unchanged_json_form(
        self, fixture_collection, query
    ):
        # The JSON form is a cross-repo contract (muninn reads it): anything but
        # an explicit 1/true must be byte-identical to no parameter at all.
        client = self._client()
        baseline = client.get(f"/api/document/{COLLECTION}/talk.md")

        resp = client.get(f"/api/document/{COLLECTION}/talk.md{query}")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.content == baseline.content
        assert "fenced-code-canary" not in resp.json()["text"]


class TestRawReadRejections(_RawCase):
    """The raw path inherits the delete route's guards, and must prove it."""

    def test_unknown_collection_404(self, fixture_collection):
        resp = self._client().get("/api/document/nope/talk.md?raw=1")
        assert resp.status_code == 404

    def test_file_on_disk_but_not_indexed_404(self, fixture_collection):
        # basePath is not the collection: a file can sit under it without being
        # a document of it (arrived after the last update, or excluded).
        _write_sources({"arrived-later.md": "# Later\n\nNot in the index yet.\n"})

        resp = self._client().get(f"/api/document/{COLLECTION}/arrived-later.md?raw=1")

        assert resp.status_code == 404
        assert "not indexed" in resp.json()["detail"]

    def test_indexed_but_missing_on_disk_404(self, fixture_collection):
        # A stale index: the source is gone, the index entry has not been pruned
        # yet. The derived JSON is still there, so the JSON form still answers.
        os.remove(os.path.join(SOURCE_REL, "talk.md"))

        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")

        assert resp.status_code == 404
        assert "talk.md" in resp.json()["detail"]

    def test_non_localfiles_collection_400(self, fixture_collection):
        manifest_path = os.path.join(COLLECTIONS_REL, COLLECTION, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["reader"] = {"type": "jira", "baseUrl": "https://example.invalid"}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")

        assert resp.status_code == 400
        assert "localFiles" in resp.json()["detail"]

    def test_traversal_document_id_400(self, fixture_collection):
        outside = fixture_collection / "outside.md"
        outside.write_text("secret-outside-canary", encoding="utf-8")

        # Percent-encoded: an HTTP client collapses a literal ``../`` in the URL
        # before it is sent, so only the encoded form reaches the handler.
        resp = self._client().get(
            f"/api/document/{COLLECTION}/%2E%2E%2F%2E%2E%2F%2E%2E%2Foutside.md?raw=1"
        )

        assert resp.status_code == 400
        assert b"secret-outside-canary" not in resp.content

    def test_symlink_escaping_base_path_400(self, fixture_collection):
        outside = fixture_collection / "outside.md"
        outside.write_text("secret-outside-canary", encoding="utf-8")
        os.symlink(outside, os.path.join(SOURCE_REL, "escape.md"))

        resp = self._client().get(f"/api/document/{COLLECTION}/escape.md?raw=1")

        assert resp.status_code == 400
        assert b"secret-outside-canary" not in resp.content

    def test_symlinked_document_id_pointing_inside_base_path_400(
        self, fixture_collection
    ):
        # Refused even though the target is inside basePath: realpath would
        # serve talk.md under an id the collection does not own.
        os.symlink(
            os.path.abspath(os.path.join(SOURCE_REL, "talk.md")),
            os.path.join(SOURCE_REL, "alias.md"),
        )

        resp = self._client().get(f"/api/document/{COLLECTION}/alias.md?raw=1")

        assert resp.status_code == 400
        assert b"fenced-code-canary" not in resp.content

    def test_nul_byte_document_id_400(self, fixture_collection):
        resp = self._client().get(f"/api/document/{COLLECTION}/a%00b?raw=1")

        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid document ID"
