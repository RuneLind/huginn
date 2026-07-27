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
from main.runtime.knowledge_store import KnowledgeStore, get_store, run_collection_update
from main.sources.files.files_document_converter import FilesDocumentConverter
from main.sources.files.files_document_reader import FilesDocumentReader


COLLECTION = "fixture-collection"
SOURCE_REL = "./data/sources/fixture-collection"
COLLECTIONS_REL = "./data/collections"


def _write_sources(docs: dict[str, str], source_rel: str = SOURCE_REL) -> None:
    source_dir = os.path.abspath(source_rel)
    for rel, body in docs.items():
        path = os.path.join(source_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)


def _build_fixture_collection(
    docs: dict[str, str],
    name: str = COLLECTION,
    source_rel: str = SOURCE_REL,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Create a real localFiles collection from ``docs`` (relative CWD paths).

    BM25-only on purpose: it is the one indexer that needs no embedding model, so
    the full create/update path runs in-process in well under a second.
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


class _DeleteCase:
    """Shared TestClient wiring + a store that serves ``COLLECTION``."""

    def _store(self, monkeypatch=None, extra_collections=()) -> KnowledgeStore:
        store = KnowledgeStore()
        store.disk_persister = DiskPersister(base_path=COLLECTIONS_REL)
        store.searchers[COLLECTION] = object()  # makes has_collection() true
        for name in extra_collections:
            store.searchers[name] = object()
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
        assert body["reindex"] == {COLLECTION: "started"}
        assert body["pollUrls"] == {
            COLLECTION: f"/api/collections/{COLLECTION}/update-status"
        }

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

    def test_trailing_slash_in_document_id_is_normalized(
        self, fixture_collection, monkeypatch
    ):
        store = self._store(monkeypatch)
        body = self._client(store).delete(f"/api/document/{COLLECTION}/junk.md/").json()

        # The echoed id has to be one the caller could use again, not "junk.md/".
        assert body["doc_id"] == "junk.md"
        assert os.path.isfile(fixture_collection / "trash" / COLLECTION / "junk.md")
        assert _indexed_document_ids() == {"keep.md", "nested/also-junk.md"}

    def test_sibling_collections_over_the_same_base_path_are_reindexed_too(
        self, fixture_collection, monkeypatch
    ):
        # wiki + wiki-life, the nav-wiki* family, capra-notion + capra-notion-v9:
        # one basePath, several collections. Reindexing only the named one leaves
        # the siblings serving the deleted doc from a dangling index entry.
        sibling = "fixture-sibling"
        _build_fixture_collection({}, name=sibling)
        assert "junk.md" in _indexed_document_ids(sibling)

        store = self._store(monkeypatch, extra_collections=[sibling])
        body = self._client(store).delete(f"/api/document/{COLLECTION}/junk.md").json()

        assert body["reindex"] == {COLLECTION: "started", sibling: "started"}
        assert list(body["reindex"]) == [COLLECTION, sibling]  # named one first
        assert set(body["pollUrls"]) == {COLLECTION, sibling}
        assert "junk.md" not in _indexed_document_ids()
        assert "junk.md" not in _indexed_document_ids(sibling)

    def test_trash_collision_where_a_file_blocks_a_needed_directory(
        self, fixture_collection, monkeypatch
    ):
        # A previous delete of an extensionless doc ``nested`` parked a FILE at
        # the trash path where this delete needs a DIRECTORY. Naively that is a
        # permanent 500; the prior soft-delete must also survive untouched.
        trash = fixture_collection / "trash" / COLLECTION
        trash.mkdir(parents=True)
        (trash / "nested").write_text("an earlier extensionless doc", encoding="utf-8")

        store = self._store(monkeypatch)
        resp = self._client(store).delete(f"/api/document/{COLLECTION}/nested/also-junk.md")

        assert resp.status_code == 200
        body = resp.json()
        assert (trash / "nested").read_text(encoding="utf-8") == "an earlier extensionless doc"
        moved_to = os.path.join(os.getcwd(), body["movedTo"])
        assert os.path.isfile(moved_to)
        with open(moved_to, encoding="utf-8") as f:
            assert "More residue" in f.read()
        assert not os.path.exists(os.path.join(SOURCE_REL, "nested", "also-junk.md"))
        assert _indexed_document_ids() == {"keep.md", "junk.md"}

    def test_second_delete_of_same_name_does_not_overwrite_the_first(
        self, fixture_collection, monkeypatch
    ):
        store = self._store(monkeypatch)
        client = self._client(store)
        first_body = "# Junk\n\nSmoke-test residue that must be deletable.\n"
        assert client.delete(f"/api/document/{COLLECTION}/junk.md").status_code == 200

        # Re-ingest a document under the same id (index it, so it is a real
        # member of the collection again), then delete it a second time.
        with open(os.path.join(SOURCE_REL, "junk.md"), "w", encoding="utf-8") as f:
            f.write("# Junk again\n\nA second round of residue.\n")
        store.try_begin_update(COLLECTION)
        run_collection_update(COLLECTION, store)
        body = client.delete(f"/api/document/{COLLECTION}/junk.md").json()

        trash = fixture_collection / "trash" / COLLECTION
        assert os.path.isfile(trash / "junk.md")       # first delete survives
        assert body["movedTo"].endswith("junk.1.md")   # second is disambiguated
        assert os.path.isfile(trash / "junk.1.md")
        # The trash is the only undo story, so "survives" has to mean the first
        # file's CONTENT is untouched — not merely that the name still exists.
        assert (trash / "junk.md").read_text(encoding="utf-8") == first_body
        assert "A second round" in (trash / "junk.1.md").read_text(encoding="utf-8")

    def test_reindex_skipped_when_an_update_is_already_running(
        self, fixture_collection, monkeypatch
    ):
        store = self._store(monkeypatch)
        store.try_begin_update(COLLECTION)  # simulate an in-flight rebuild

        body = self._client(store).delete(f"/api/document/{COLLECTION}/junk.md").json()

        # The move is unconditional and already durable; only the reindex is
        # deferred to the caller, who can POST /update once the running one ends.
        assert body["status"] == "deleted"
        assert body["reindex"] == {COLLECTION: "skipped_already_running"}
        # No pollUrl for a skipped collection: it would point at the PRE-EXISTING
        # run, which reports "succeeded" without ever having seen this delete.
        assert body["pollUrls"] == {}
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

    def test_nul_byte_document_id_400(self, fixture_collection, monkeypatch):
        # ``os.path.realpath``/``lstat`` raise ValueError on an embedded NUL, so
        # without a guard this is a 500 on a plainly invalid id.
        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/a%00b"
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid document ID"

    def test_symlinked_document_id_pointing_inside_base_path_400(
        self, fixture_collection, monkeypatch
    ):
        # realpath would resolve ``alias.md`` to ``keep.md`` and move THAT —
        # deleting a different document and leaving a dangling symlink behind.
        os.symlink(
            os.path.abspath(os.path.join(SOURCE_REL, "keep.md")),
            os.path.join(SOURCE_REL, "alias.md"),
        )

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/alias.md"
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid document ID"
        assert os.path.isfile(os.path.join(SOURCE_REL, "keep.md"))
        assert os.path.islink(os.path.join(SOURCE_REL, "alias.md"))

    def test_file_on_disk_but_not_indexed_404(self, fixture_collection, monkeypatch):
        # basePath is not the collection: a file can sit under it without ever
        # having been indexed (arrived after the last update, or never matched).
        _write_sources({"arrived-later.md": "# Later\n\nNot in the index yet.\n"})

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/arrived-later.md"
        )

        assert resp.status_code == 404
        assert "not indexed" in resp.json()["detail"]
        assert os.path.isfile(os.path.join(SOURCE_REL, "arrived-later.md"))

    def test_git_internals_are_not_deletable_404(self, fixture_collection, monkeypatch):
        # Several wikis' basePath IS a live git repo root. The reader's walk skips
        # ``.git`` entirely, so nothing in it is ever a document of the collection.
        _write_sources({".git/config": "[core]\n\trepositoryformatversion = 0\n"})

        resp = self._client(self._store(monkeypatch)).delete(
            f"/api/document/{COLLECTION}/.git/config"
        )

        assert resp.status_code == 404
        assert os.path.isfile(os.path.join(SOURCE_REL, ".git", "config"))

    def test_excluded_pattern_file_404(self, fixture_collection, monkeypatch):
        # ``CLAUDE.md`` is excluded by mimir's and the jarvis wiki's readers, yet
        # it lives right under basePath — moving it out would edit a real repo.
        excluded_collection = "fixture-excluded"
        excluded_source = "./data/sources/fixture-excluded"
        _build_fixture_collection(
            {
                "real.md": "# Real\n\nAn indexed page.\n",
                "CLAUDE.md": "# Instructions\n\nNot part of the collection.\n",
            },
            name=excluded_collection,
            source_rel=excluded_source,
            exclude_patterns=[r"^CLAUDE\.md$"],
        )
        assert _indexed_document_ids(excluded_collection) == {"real.md"}

        store = self._store(monkeypatch, extra_collections=[excluded_collection])
        resp = self._client(store).delete(
            f"/api/document/{excluded_collection}/CLAUDE.md"
        )

        assert resp.status_code == 404
        assert "not indexed" in resp.json()["detail"]
        assert os.path.isfile(os.path.join(excluded_source, "CLAUDE.md"))

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
        "a\x00b",               # embedded NUL — realpath raises, not returns
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
