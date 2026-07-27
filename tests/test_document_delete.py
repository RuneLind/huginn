"""Tests for DELETE /api/document/{collection}/{doc_id}.

The endpoint soft-deletes by MOVING the source file out of the reader's
``basePath`` and then triggering the normal incremental update, whose orphan
pruning drops the index entries and the derived document JSON. The headline test
here is the index-integrity one: it builds a real (BM25-only) localFiles fixture
collection, deletes through the route, and asserts the whole chain — source
moved outside basePath, index entry gone, derived JSON gone.
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


def _build_fixture_collection(docs: dict[str, str]) -> None:
    """Create a real localFiles collection from ``docs`` (relative CWD paths).

    BM25-only on purpose: it is the one indexer that needs no embedding model, so
    the full create/update path runs in-process in well under a second.
    """
    source_dir = os.path.abspath(SOURCE_REL)
    for rel, body in docs.items():
        path = os.path.join(source_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)

    DocumentCollectionCreator(
        collection_name=COLLECTION,
        # A RELATIVE basePath, like the real x-articles/anthropic-docs manifests:
        # this is what pins the endpoint's "resolve against the server CWD" rule.
        document_reader=FilesDocumentReader(base_path=SOURCE_REL, include_patterns=[".*"]),
        document_converter=FilesDocumentConverter(),
        document_indexers=[BM25Indexer("indexer_BM25")],
        persister=DiskPersister(base_path=COLLECTIONS_REL),
        operation_type=OPERATION_TYPE.CREATE,
    ).run()


def _indexed_document_ids() -> set[str]:
    with open(
        os.path.join(COLLECTIONS_REL, COLLECTION, "indexes", "index_document_mapping.json"),
        encoding="utf-8",
    ) as f:
        return {entry["documentId"] for entry in json.load(f).values()}


class _DeleteCase:
    """Shared TestClient wiring + a store that serves ``COLLECTION``."""

    def _store(self, monkeypatch=None) -> KnowledgeStore:
        store = KnowledgeStore()
        store.disk_persister = DiskPersister(base_path=COLLECTIONS_REL)
        store.searchers[COLLECTION] = object()  # makes has_collection() true
        store._build_aux_indexes = False
        if monkeypatch is not None:
            # The fixture collection has no FAISS index, so the post-update
            # searcher swap cannot run. Deletion durability lives on disk (index
            # mapping + document JSONs), which is what these tests assert.
            monkeypatch.setattr(store, "reload_collection", lambda name: None)
        return store

    def _client(self, store) -> TestClient:
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.pop(get_store, None)


@pytest.fixture
def fixture_collection(tmp_path, monkeypatch):
    """A real BM25 localFiles collection in an isolated CWD, plus a trash dir.

    ``monkeypatch.chdir`` is what keeps every relative path in play (the
    manifest's basePath, the update factory's hardcoded ``./data/collections``,
    and the endpoint's default deleted-dir) pointed at tmp_path — no real
    ``data/`` directory is touched.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HUGINN_DELETED_DIR", str(tmp_path / "trash"))
    _build_fixture_collection({
        "keep.md": "# Keep\n\nA real article worth keeping in the index.\n",
        "junk.md": "# Junk\n\nSmoke-test residue that must be deletable.\n",
        "nested/also-junk.md": "# Nested junk\n\nMore residue, one folder deep.\n",
    })
    return tmp_path


class TestDeleteDocumentIndexIntegrity(_DeleteCase):
    """The whole chain: move the source, reindex, index + derived JSON follow."""

    def test_delete_moves_source_and_prunes_index_and_document_json(
        self, fixture_collection, monkeypatch
    ):
        tmp_path = fixture_collection
        assert _indexed_document_ids() == {"keep.md", "junk.md", "nested/also-junk.md"}
        derived = os.path.join(COLLECTIONS_REL, COLLECTION, "documents", "junk.md.json")
        assert os.path.isfile(derived)

        store = self._store(monkeypatch)
        # TestClient runs background tasks synchronously once the response is
        # returned, so the reindex has finished by the time this call returns.
        resp = self._client(store).delete(f"/api/document/{COLLECTION}/junk.md")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["collection"] == COLLECTION
        assert body["doc_id"] == "junk.md"
        assert body["reindex"] == "started"
        assert body["pollUrl"] == f"/api/collections/{COLLECTION}/update-status"

        # 1. The source left basePath and landed in the trash (the undo story).
        assert not os.path.exists(os.path.join(SOURCE_REL, "junk.md"))
        moved_to = os.path.join(os.getcwd(), body["movedTo"])
        assert os.path.isfile(moved_to)
        assert os.path.realpath(moved_to) == str(tmp_path / "trash" / COLLECTION / "junk.md")
        assert not os.path.realpath(moved_to).startswith(os.path.abspath(SOURCE_REL) + os.sep)

        # 2. The index no longer lists it; the surviving documents are untouched.
        assert _indexed_document_ids() == {"keep.md", "nested/also-junk.md"}

        # 3. The derived JSON is gone (orphan pruning, not manual surgery).
        assert not os.path.exists(derived)
        assert os.path.isfile(
            os.path.join(COLLECTIONS_REL, COLLECTION, "documents", "keep.md.json")
        )
        assert store.get_update_status(COLLECTION)["status"] == "succeeded"

    def test_delete_nested_document_id(self, fixture_collection, monkeypatch):
        store = self._store(monkeypatch)
        resp = self._client(store).delete(f"/api/document/{COLLECTION}/nested/also-junk.md")

        assert resp.status_code == 200
        # The trash mirrors the source-relative path, so the id round-trips.
        assert os.path.isfile(
            str(fixture_collection / "trash" / COLLECTION / "nested" / "also-junk.md")
        )
        assert _indexed_document_ids() == {"keep.md", "junk.md"}

    def test_second_delete_of_same_name_does_not_overwrite_the_first(
        self, fixture_collection, monkeypatch
    ):
        store = self._store(monkeypatch)
        client = self._client(store)
        assert client.delete(f"/api/document/{COLLECTION}/junk.md").status_code == 200

        # Re-ingest a document under the same id, then delete it again.
        with open(os.path.join(SOURCE_REL, "junk.md"), "w", encoding="utf-8") as f:
            f.write("# Junk again\n\nA second round of residue.\n")
        body = client.delete(f"/api/document/{COLLECTION}/junk.md").json()

        trash = fixture_collection / "trash" / COLLECTION
        assert os.path.isfile(trash / "junk.md")       # first delete survives
        assert body["movedTo"].endswith("junk.1.md")   # second is disambiguated
        assert os.path.isfile(trash / "junk.1.md")

    def test_reindex_skipped_when_an_update_is_already_running(
        self, fixture_collection, monkeypatch
    ):
        store = self._store(monkeypatch)
        store.try_begin_update(COLLECTION)  # simulate an in-flight rebuild

        body = self._client(store).delete(f"/api/document/{COLLECTION}/junk.md").json()

        # The move is unconditional and already durable; only the reindex is
        # deferred to the caller, who can POST /update once the running one ends.
        assert body["status"] == "deleted"
        assert body["reindex"] == "skipped_already_running"
        assert not os.path.exists(os.path.join(SOURCE_REL, "junk.md"))
        assert os.path.isfile(fixture_collection / "trash" / COLLECTION / "junk.md")
        # Nothing reindexed, so the index still carries the (now sourceless) entry.
        assert "junk.md" in _indexed_document_ids()


class TestDeleteDocumentRejections(_DeleteCase):
    """Every "this request cannot be honoured" path, before anything is moved."""

    def test_unknown_collection_404(self, fixture_collection, monkeypatch):
        resp = self._client(self._store(monkeypatch)).delete("/api/document/nope/junk.md")
        assert resp.status_code == 404

    def test_missing_source_file_404(self, fixture_collection, monkeypatch):
        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/never-existed.md"
        )
        assert resp.status_code == 404
        assert "never-existed.md" in resp.json()["detail"]

    def test_traversal_document_id_400_and_moves_nothing(
        self, fixture_collection, monkeypatch
    ):
        outside = fixture_collection / "outside.md"
        outside.write_text("must not be touched", encoding="utf-8")

        # Percent-encoded, because an HTTP client collapses a literal ``../`` in
        # the URL before it is ever sent — the encoded form is the one that
        # actually reaches the handler as a traversing path parameter.
        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/%2E%2E%2F%2E%2E%2F%2E%2E%2Foutside.md"
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid document ID"
        assert outside.is_file()

    def test_symlink_escaping_base_path_400(self, fixture_collection, monkeypatch):
        outside = fixture_collection / "outside.md"
        outside.write_text("must not be touched", encoding="utf-8")
        os.symlink(outside, os.path.join(SOURCE_REL, "escape.md"))

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/escape.md"
        )

        assert resp.status_code == 400
        assert outside.is_file()

    def test_non_localfiles_collection_400(self, fixture_collection, monkeypatch):
        manifest_path = os.path.join(COLLECTIONS_REL, COLLECTION, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["reader"] = {"type": "jira", "baseUrl": "https://example.invalid"}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/junk.md"
        )

        assert resp.status_code == 400
        assert "localFiles" in resp.json()["detail"]
        # A query-based reader cannot enumerate its ids, so orphan pruning would
        # never fire — half-deleting it is exactly what the 400 prevents.
        assert os.path.isfile(os.path.join(SOURCE_REL, "junk.md"))

    def test_unresolvable_base_path_400(self, fixture_collection, monkeypatch):
        manifest_path = os.path.join(COLLECTIONS_REL, COLLECTION, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["reader"]["basePath"] = "./data/sources/gone-missing"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/junk.md"
        )

        assert resp.status_code == 400
        assert "does not resolve to an existing directory" in resp.json()["detail"]

    def test_deleted_dir_inside_base_path_400(self, fixture_collection, monkeypatch):
        # A trash dir nested under basePath would leave the file indexed under a
        # new id (includePatterns is ".*"), which is not a deletion at all.
        monkeypatch.setenv("HUGINN_DELETED_DIR", os.path.join(SOURCE_REL, "trash"))

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/junk.md"
        )

        assert resp.status_code == 400
        assert "re-indexed" in resp.json()["detail"]
        assert os.path.isfile(os.path.join(SOURCE_REL, "junk.md"))


class TestResolveSourceFileContainment:
    """The containment guard on its own — no HTTP client in the way.

    An HTTP client normalizes ``../`` out of a URL before sending it, so the
    route test can only reach the guard through percent-encoding. These assert
    the guard directly, so it stays correct whatever a client does to the path.
    """

    def _resolve(self, base_dir, doc_id):
        from main.routes.collections import _resolve_source_file
        return _resolve_source_file(base_dir, doc_id)

    def test_plain_id_resolves_inside_base(self, tmp_path):
        base = str(tmp_path.resolve())
        assert self._resolve(base, "a/b.md") == os.path.join(base, "a", "b.md")

    @pytest.mark.parametrize("doc_id", [
        "",                     # empty
        "/etc/passwd",          # absolute
        "../x",                 # one level out
        "a/../../x",            # out via a valid-looking prefix
        "..",                   # basePath itself, not a file inside it
    ])
    def test_escaping_ids_rejected(self, tmp_path, doc_id):
        from fastapi import HTTPException
        base = str(tmp_path.resolve())
        with pytest.raises(HTTPException) as exc:
            self._resolve(base, doc_id)
        assert exc.value.status_code == 400

    def test_sibling_prefix_directory_is_not_inside_base(self, tmp_path):
        # ``/data/sources/x-articles-old`` must not pass a containment check
        # against ``/data/sources/x-articles`` — string prefix, different tree.
        from fastapi import HTTPException
        base = tmp_path / "coll"
        base.mkdir()
        (tmp_path / "coll-old").mkdir()
        with pytest.raises(HTTPException) as exc:
            self._resolve(str(base.resolve()), "../coll-old/x.md")
        assert exc.value.status_code == 400
