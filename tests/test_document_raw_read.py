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


def _unavailable(doc_id: str, collection: str = COLLECTION) -> str:
    """The ONE 404 detail the raw form gives for every id it will not serve.

    Missing, present-but-unindexed and excluded all answer with this exact
    string: any wording that separated them would report, to an unauthenticated
    caller, whether a path it does not own exists under ``reader.basePath``.
    """
    return f"Document '{doc_id}' is not available in collection '{collection}'"


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

    @pytest.mark.parametrize(
        "query", ["", "?raw=0", "?raw=false", "?raw=FALSE", "?raw=", "?raw=0&raw=false"]
    )
    def test_absent_or_false_raw_is_the_unchanged_json_form(
        self, fixture_collection, query
    ):
        # The JSON form is a cross-repo contract (muninn reads it): absent, and
        # every spelling of false, must be byte-identical to no parameter at all.
        client = self._client()
        baseline = client.get(f"/api/document/{COLLECTION}/talk.md")

        resp = client.get(f"/api/document/{COLLECTION}/talk.md{query}")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.content == baseline.content
        assert "fenced-code-canary" not in resp.json()["text"]

    def test_trailing_slash_in_document_id_is_normalized(self, fixture_collection):
        # ``GET .../talk.md/?raw=1`` reaches the handler with the slash intact.
        # The delete route normalizes it, so an id that deletes must also read.
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md/?raw=1")

        assert resp.status_code == 200
        assert resp.content == _source_bytes("talk.md")
        assert resp.headers["x-huginn-source-path"] == "talk.md"

    def test_raw_response_forbids_content_type_sniffing(self, fixture_collection):
        # The body is caller-supplied file content served under this API's
        # origin; a browser must not be free to re-type it as something active.
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")

        assert resp.status_code == 200
        assert resp.headers["x-content-type-options"] == "nosniff"


class TestRawParameterIsExplicit(_RawCase):
    """An unreadable ``?raw=`` value is a 400, never a silent JSON fall-through.

    Degrading to the JSON form is precisely the failure this endpoint exists to
    prevent: a caller that asked for the source and got the cleaned copy
    re-ingests a lossy document and never sees an error.
    """

    @pytest.mark.parametrize("value", ["yes", "on", "2", "y", "TrUthy", " 1"])
    def test_unrecognized_raw_value_400(self, fixture_collection, value):
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw={value}")

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # The 400 has to name what IS accepted, or the caller has to read source.
        assert "raw" in detail
        assert "true" in detail and "false" in detail

    def test_repeated_raw_values_that_disagree_400(self, fixture_collection):
        # Last-wins would make ``?raw=1&raw=0`` silently serve the JSON form to a
        # caller whose first parameter asked for the source.
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1&raw=0")

        assert resp.status_code == 400

    def test_repeated_raw_values_that_agree_are_that_value(self, fixture_collection):
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1&raw=true")

        assert resp.status_code == 200
        assert resp.content == _source_bytes("talk.md")

    def test_unrecognized_raw_value_is_rejected_before_the_store_is_read(
        self, fixture_collection
    ):
        # A malformed request is answered the same way whether or not the
        # collection exists — the parameter is a request-shape problem.
        resp = self._client().get("/api/document/nope/talk.md?raw=yes")

        assert resp.status_code == 400


class TestSourcePathHeaderEncoding:
    """``X-Huginn-Source-Path`` is percent-encoded, and why.

    Driven through the helper rather than the route: the reader's walk drops a
    filename containing CR/LF outright (measured — such a file is never indexed),
    so a CRLF-bearing id cannot reach the header over HTTP at all. The encoder is
    still the thing that has to hold, since the reader is not this route's guard.
    """

    def test_crlf_in_a_path_cannot_inject_a_header(self):
        from main.routes.collections import _source_path_header_value as encode

        value = encode("a\r\nX-Injected: 1.md")

        assert "\r" not in value and "\n" not in value
        assert value == "a%0D%0AX-Injected%3A%201.md"

    def test_non_latin1_path_survives_the_latin1_header_encoding(self):
        from main.routes.collections import _source_path_header_value as encode
        from starlette.responses import Response

        # 'ł' is NOT latin-1: an unencoded value raises when the response
        # renders. ('å'/'æ'/'ø' are latin-1 and would have gone out fine.)
        with pytest.raises(UnicodeEncodeError):
            Response(content=b"x", headers={"X-Probe": "łódź.md"})

        value = encode("łódź.md")

        assert Response(content=b"x", headers={"X-Probe": value}).headers["X-Probe"] == value


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
        assert resp.json()["detail"] == _unavailable("arrived-later.md")

    def test_indexed_but_missing_on_disk_404(self, fixture_collection):
        # A stale index: the source is gone, the index entry has not been pruned
        # yet. The derived JSON is still there, so the JSON form still answers.
        os.remove(os.path.join(SOURCE_REL, "talk.md"))

        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")

        assert resp.status_code == 404
        assert resp.json()["detail"] == _unavailable("talk.md")

    def test_present_but_unindexed_is_indistinguishable_from_absent(
        self, fixture_collection
    ):
        # The oracle this collapse closes: an unauthenticated GET must not tell
        # a caller whether a path it does not own EXISTS under basePath. Same id,
        # two states on disk, one answer.
        client = self._client()
        _write_sources({"probe.md": "# Probe\n\nUnder basePath, not a document.\n"})

        present = client.get(f"/api/document/{COLLECTION}/probe.md?raw=1")
        os.remove(os.path.join(SOURCE_REL, "probe.md"))
        absent = client.get(f"/api/document/{COLLECTION}/probe.md?raw=1")

        assert present.status_code == absent.status_code == 404
        assert present.json()["detail"] == absent.json()["detail"]
        assert present.content == absent.content

    def test_raw_404_does_not_say_the_read_refuses_to_move_anything(
        self, fixture_collection
    ):
        # The delete route's wording travelled with the shared helper. A GET
        # moves nothing, and saying so on a read is simply false. The id has to
        # be one that REACHES the membership check — present under basePath,
        # absent from the index — since that is the branch that said it.
        _write_sources({"arrived-later.md": "# Later\n\nNot in the index yet.\n"})

        resp = self._client().get(f"/api/document/{COLLECTION}/arrived-later.md?raw=1")

        assert resp.status_code == 404
        assert "move" not in resp.json()["detail"]

    def test_git_internals_404_with_the_collapsed_detail(self, fixture_collection):
        # Several wikis' basePath IS a live git repo root. The reader's walk skips
        # ``.git`` entirely, so nothing in it is ever a document of the collection
        # — and the answer must not confirm that ``.git/config`` is there.
        _write_sources({".git/config": "[core]\n\trepositoryformatversion = 0\n"})

        resp = self._client().get(f"/api/document/{COLLECTION}/.git/config?raw=1")

        assert resp.status_code == 404
        assert resp.json()["detail"] == _unavailable(".git/config")
        assert b"repositoryformatversion" not in resp.content

    def test_excluded_pattern_file_404(self, fixture_collection):
        # ``CLAUDE.md`` is excluded by mimir's and the jarvis wiki's readers, yet
        # it lives right under basePath — serving it would leak a non-document.
        excluded_collection = "fixture-excluded"
        excluded_source = "./data/sources/fixture-excluded"
        _build_fixture_collection(
            {
                "real.md": "# Real\n\nAn indexed page.\n",
                "CLAUDE.md": "# Instructions\n\nexcluded-canary\n",
            },
            name=excluded_collection,
            source_rel=excluded_source,
            exclude_patterns=[r"^CLAUDE\.md$"],
        )
        assert _indexed_document_ids(excluded_collection) == {"real.md"}

        store = self._store(extra_collections=[excluded_collection])
        resp = self._client(store).get(
            f"/api/document/{excluded_collection}/CLAUDE.md?raw=1"
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == _unavailable("CLAUDE.md", excluded_collection)
        assert b"excluded-canary" not in resp.content

    @pytest.mark.skipif(
        os.name != "posix" or os.geteuid() == 0,
        reason="needs POSIX permissions and a non-root user to make a file unreadable",
    )
    def test_unreadable_source_file_500_does_not_leak_the_server_path(
        self, fixture_collection
    ):
        source = os.path.join(SOURCE_REL, "talk.md")
        base_dir = os.path.realpath(SOURCE_REL)
        os.chmod(source, 0o000)
        try:
            resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")
        finally:
            os.chmod(source, 0o644)

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert base_dir not in detail
        assert str(fixture_collection) not in detail
        # An errno string carries the path too ("[Errno 13] … /private/var/…").
        assert "Errno" not in detail

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
