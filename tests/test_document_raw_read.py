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
import socket
import threading

import pytest
import uvicorn
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


def _assert_path_free(detail: str, tmp_root) -> None:
    """No server-filesystem path in an error the raw route hands an outsider.

    Both shared helpers used to interpolate the caught exception, whose string
    carries the file it failed on ("[Errno 13] … /private/var/…"), and the
    basePath 400 spelled out its resolved absolute path.
    """
    assert str(tmp_root) not in detail
    assert "Errno" not in detail
    assert COLLECTIONS_REL.lstrip("./") not in detail
    assert SOURCE_REL.lstrip("./") not in detail


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

    @pytest.mark.parametrize("query", ["?raw=1", "?raw=0", ""])
    def test_a_trailing_slash_reads_the_same_document_on_every_form(
        self, fixture_collection, query
    ):
        # ``GET .../talk.md/`` reaches the handler with the slash intact. The
        # delete route normalizes it; so must BOTH forms of the read, or one id
        # deletes, reads raw, and 404s as JSON.
        client = self._client()
        baseline = client.get(f"/api/document/{COLLECTION}/talk.md{query}")

        resp = client.get(f"/api/document/{COLLECTION}/talk.md/{query}")

        assert baseline.status_code == 200
        assert resp.status_code == 200
        assert resp.content == baseline.content
        assert resp.headers["content-type"] == baseline.headers["content-type"]

    def test_the_trailing_slash_json_form_is_the_document_not_an_empty_shell(
        self, fixture_collection
    ):
        # The widening this normalization brings: on main ``talk.md/`` was a 404
        # for the JSON form. It must now be the SAME document, not some other
        # row the store happened to answer with.
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md/")

        assert resp.status_code == 200
        assert resp.json()["id"] == "talk.md"
        assert "transcript-canary" in resp.json()["text"]

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
        from main.routes.collections import RAW_VALUES_HELP

        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw={value}")

        assert resp.status_code == 400
        # Byte for byte, against the constant the route builds it from: the 400
        # has to name the value it rejected AND what is accepted, or the caller
        # has to go read the source to find out.
        assert resp.json()["detail"] == (
            f"Invalid 'raw' value '{value}'; accepted values are {RAW_VALUES_HELP}"
        )
        assert "'1' or 'true'" in RAW_VALUES_HELP and "'0', 'false'" in RAW_VALUES_HELP

    def test_repeated_raw_values_that_disagree_400(self, fixture_collection):
        # Last-wins would make ``?raw=1&raw=0`` silently serve the JSON form to a
        # caller whose first parameter asked for the source.
        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1&raw=0")

        assert resp.status_code == 400
        assert resp.json()["detail"] == (
            "Conflicting 'raw' values in the query string; pass the parameter "
            "at most once"
        )

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


#: A source path shaped like a response-splitting attempt. Used by the encoder
#: probes below; never reachable through the route (see the reader probe).
CRLF_PATH = "a\r\nX-Injected: 1.md"


def _http_get_over_a_real_server(app_under_test, path: str) -> bytes:
    """Everything a real uvicorn writes for ``GET path``, or ``b""``.

    The whole point of driving a REAL server rather than ``TestClient``: the
    ASGI transport TestClient uses never runs h11, and h11 is the layer that
    decides what a CR/LF header value does on the wire.
    """
    config = uvicorn.Config(
        app_under_test, host="127.0.0.1", port=0, log_level="critical"
    )
    server = uvicorn.Server(config)
    sock = config.bind_socket()
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            threading.Event().wait(0.02)
        assert server.started, "uvicorn did not start"

        conn = socket.create_connection(("127.0.0.1", port), timeout=10)
        conn.settimeout(10)
        try:
            conn.sendall(
                f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                f"Connection: close\r\n\r\n".encode()
            )
            chunks = []
            while True:
                block = conn.recv(65536)
                if not block:
                    break
                chunks.append(block)
        finally:
            conn.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    return b"".join(chunks)


class TestSourcePathHeaderEncoding:
    """``X-Huginn-Source-Path`` is percent-encoded, and what that actually buys.

    Two layers, measured here rather than asserted about: Starlette does NOT
    reject CR/LF in a header value — its ``Response`` constructs and carries the
    bytes verbatim — while h11, the protocol layer uvicorn writes through,
    refuses the value outright and the caller gets a dropped connection instead
    of a response. So the encoding is not what stops header injection at the
    wire; it is what keeps a CR/LF-bearing name from turning the read into a
    dropped connection, and what keeps a NON-latin-1 name (``łódź.md``,
    ``日本語.md``) from raising at ``Response`` construction.

    Driven through the helper rather than the route: the reader never indexes a
    CR/LF-bearing filename (measured below), and the membership check now runs
    before anything else, so such an id cannot reach the header over HTTP.
    """

    def test_crlf_in_a_path_is_percent_encoded(self):
        from main.routes.collections import _source_path_header_value as encode

        value = encode(CRLF_PATH)

        assert "\r" not in value and "\n" not in value
        assert value == "a%0D%0AX-Injected%3A%201.md"

    def test_starlette_does_not_reject_crlf_in_a_header_value(self):
        from starlette.responses import Response

        # The measurement that corrects this PR's first-round claim: no
        # exception, and the raw bytes go into the header list unchanged.
        response = Response(content=b"x", headers={"X-Probe": CRLF_PATH})

        assert (b"x-probe", CRLF_PATH.encode("latin-1")) in response.raw_headers

    def test_a_real_server_drops_the_crlf_value_and_serves_the_encoded_one(self):
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Route

        from main.routes.collections import _source_path_header_value as encode

        async def unencoded(request):
            return Response(content=b"body\n", headers={"X-Probe": CRLF_PATH})

        async def encoded(request):
            return Response(content=b"body\n", headers={"X-Probe": encode(CRLF_PATH)})

        probe_app = Starlette(routes=[
            Route("/unencoded", unencoded),
            Route("/encoded", encoded),
        ])

        # h11 raises ``LocalProtocolError: Illegal header value`` and uvicorn
        # closes the connection: the client gets nothing at all — no injected
        # header, and no response either.
        assert _http_get_over_a_real_server(probe_app, "/unencoded") == b""

        served = _http_get_over_a_real_server(probe_app, "/encoded")
        assert served.startswith(b"HTTP/1.1 200 OK\r\n")
        assert b"\r\nx-probe: a%0D%0AX-Injected%3A%201.md\r\n" in served
        assert b"X-Injected" not in served.split(b"\r\n\r\n", 1)[0].replace(
            b"a%0D%0AX-Injected%3A%201.md", b""
        )

    def test_non_latin1_path_raises_at_response_construction_unless_encoded(self):
        from main.routes.collections import _source_path_header_value as encode
        from starlette.responses import Response

        # 'ł' is NOT latin-1: an unencoded value raises before any server sees
        # it. ('å'/'æ'/'ø' are latin-1 and would have gone out fine.)
        with pytest.raises(UnicodeEncodeError):
            Response(content=b"x", headers={"X-Probe": "łódź.md"})

        value = encode("łódź.md")

        assert Response(content=b"x", headers={"X-Probe": value}).headers["X-Probe"] == value

    def test_the_reader_never_indexes_a_crlf_filename(self, tmp_path, monkeypatch):
        # Why the encoder is tested through the helper and not the route: such a
        # file can exist on disk, and the reader still does not make it a
        # document, so no request can ever put CR/LF into the header.
        monkeypatch.chdir(tmp_path)
        _build_fixture_collection({"talk.md": SUMMARY_DOC, CRLF_PATH: SUMMARY_DOC})

        assert CRLF_PATH in os.listdir(os.path.abspath(SOURCE_REL))
        assert _indexed_document_ids() == {"talk.md"}


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

    @pytest.mark.skipif(
        os.name != "posix" or os.geteuid() == 0,
        reason="needs POSIX permissions and a non-root user to make a file unreadable",
    )
    def test_unreadable_manifest_500_does_not_leak_the_server_path(
        self, fixture_collection
    ):
        manifest_path = os.path.join(COLLECTIONS_REL, COLLECTION, "manifest.json")
        os.chmod(manifest_path, 0o000)
        try:
            resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")
        finally:
            os.chmod(manifest_path, 0o644)

        assert resp.status_code == 500
        _assert_path_free(resp.json()["detail"], fixture_collection)

    def test_unresolvable_base_path_400_does_not_leak_the_server_path(
        self, fixture_collection
    ):
        # The resolved absolute basePath is a server-filesystem path, and this
        # 400 is reachable by any caller of either route.
        manifest_path = os.path.join(COLLECTIONS_REL, COLLECTION, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["reader"]["basePath"] = "./data/sources/gone-missing"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "does not resolve to an existing directory" in detail
        assert "gone-missing" not in detail
        _assert_path_free(detail, fixture_collection)

    @pytest.mark.skipif(
        os.name != "posix" or os.geteuid() == 0,
        reason="needs POSIX permissions and a non-root user to make a file unreadable",
    )
    def test_unreadable_index_mapping_500_does_not_leak_the_server_path(
        self, fixture_collection
    ):
        mapping_path = os.path.join(
            COLLECTIONS_REL, COLLECTION, "indexes",
            "reverse_index_document_mapping.json",
        )
        os.chmod(mapping_path, 0o000)
        try:
            resp = self._client().get(f"/api/document/{COLLECTION}/talk.md?raw=1")
        finally:
            os.chmod(mapping_path, 0o644)

        assert resp.status_code == 500
        _assert_path_free(resp.json()["detail"], fixture_collection)

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

    def test_traversal_document_id_404_like_every_other_unindexed_id(
        self, fixture_collection
    ):
        # A traversing id is not a document of the collection, so it is answered
        # by the membership check and never reaches the resolver: the delete
        # route's "Invalid document ID" 400 would tell a caller which shape of
        # rejection it hit, which is a bit of the oracle back.
        outside = fixture_collection / "outside.md"
        outside.write_text("secret-outside-canary", encoding="utf-8")

        # Percent-encoded: an HTTP client collapses a literal ``../`` in the URL
        # before it is sent, so only the encoded form reaches the handler.
        traversal = "%2E%2E%2F%2E%2E%2F%2E%2E%2Foutside.md"
        resp = self._client().get(f"/api/document/{COLLECTION}/{traversal}?raw=1")

        assert resp.status_code == 404
        assert resp.json()["detail"] == _unavailable("../../../outside.md")
        assert b"secret-outside-canary" not in resp.content

    def test_symlink_escaping_base_path_404(self, fixture_collection):
        outside = fixture_collection / "outside.md"
        outside.write_text("secret-outside-canary", encoding="utf-8")
        os.symlink(outside, os.path.join(SOURCE_REL, "escape.md"))

        resp = self._client().get(f"/api/document/{COLLECTION}/escape.md?raw=1")

        assert resp.status_code == 404
        assert resp.json()["detail"] == _unavailable("escape.md")
        assert b"secret-outside-canary" not in resp.content

    def test_unindexed_symlink_is_indistinguishable_from_a_missing_id(
        self, fixture_collection
    ):
        # The residual oracle this reorder closes: the resolver refuses a
        # symlinked id with a 400, so "there is a symlink here" used to be
        # distinguishable from "there is nothing here" — for a path under
        # basePath that the collection does not own.
        client = self._client()
        os.symlink(
            os.path.abspath(os.path.join(SOURCE_REL, "talk.md")),
            os.path.join(SOURCE_REL, "postlink.md"),
        )

        symlinked = client.get(f"/api/document/{COLLECTION}/postlink.md?raw=1")
        missing = client.get(f"/api/document/{COLLECTION}/never-indexed.md?raw=1")

        assert symlinked.status_code == missing.status_code == 404
        assert symlinked.json()["detail"] == _unavailable("postlink.md")
        assert missing.json()["detail"] == _unavailable("never-indexed.md")
        assert b"fenced-code-canary" not in symlinked.content

    def test_indexed_symlink_document_400(self, tmp_path, monkeypatch):
        # Landed as-is: a symlink that was present when the collection was built
        # IS a document of it (measured — the reader indexes it), so it passes
        # the membership check and then hits the resolver's refusal. That 400
        # reveals nothing the caller did not already know: the id is indexed.
        monkeypatch.chdir(tmp_path)
        _write_sources({"talk.md": SUMMARY_DOC})
        os.symlink(
            os.path.abspath(os.path.join(SOURCE_REL, "talk.md")),
            os.path.join(os.path.abspath(SOURCE_REL), "alias.md"),
        )
        _build_fixture_collection({})
        assert "alias.md" in _indexed_document_ids()

        resp = self._client().get(f"/api/document/{COLLECTION}/alias.md?raw=1")

        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid document ID"
        assert b"fenced-code-canary" not in resp.content

    def test_nul_byte_document_id_404(self, fixture_collection):
        # Also answered by the membership check now: an id carrying a NUL is not
        # in the mapping, so it collapses with everything else unserved.
        resp = self._client().get(f"/api/document/{COLLECTION}/a%00b?raw=1")

        assert resp.status_code == 404
        assert resp.json()["detail"] == _unavailable("a\x00b")

    def test_membership_check_runs_before_the_resolver_and_the_stat(
        self, fixture_collection, monkeypatch
    ):
        """The collapse is an ORDER, and only a call-order assertion pins it.

        Over HTTP the two orders are indistinguishable for an ordinary id — both
        answer the same 404 — so a response-only test passes against a build
        that resolves and stats a path the collection does not own first, which
        is where the symlink and traversal 400s came back from.
        """
        import main.routes.collections as mod

        client = self._client()
        _write_sources({"probe.md": "# Probe\n\nUnder basePath, not a document.\n"})

        resolved: list[str] = []
        stated: list[str] = []
        real_resolve = mod._resolve_source_file
        real_isfile = os.path.isfile

        def spy_resolve(base_dir, doc_id):
            resolved.append(doc_id)
            return real_resolve(base_dir, doc_id)

        def spy_isfile(path):
            stated.append(path)
            return real_isfile(path)

        monkeypatch.setattr(mod, "_resolve_source_file", spy_resolve)
        monkeypatch.setattr(os.path, "isfile", spy_isfile)

        unindexed = client.get(f"/api/document/{COLLECTION}/probe.md?raw=1")
        assert unindexed.status_code == 404
        assert unindexed.json()["detail"] == _unavailable("probe.md")
        assert resolved == []
        assert [p for p in stated if p.endswith("probe.md")] == []

        # Positive control, so neither emptiness above can be a dead spy: an
        # INDEXED id goes through both.
        resolved.clear()
        stated.clear()
        served = client.get(f"/api/document/{COLLECTION}/talk.md?raw=1")
        assert served.status_code == 200
        assert resolved == ["talk.md"]
        assert [p for p in stated if p.endswith("talk.md")] != []


class TestRawFormIsDeclaredInTheSchema:
    """``text/markdown`` is part of the endpoint's published contract.

    The route declares it in ``responses=``; nothing else in the app would fail
    if that declaration were dropped, so the generated schema is pinned here.
    Consumers (muninn's client, anyone reading ``/docs``) learn the raw form
    exists from this and nothing else.
    """

    def test_openapi_declares_both_response_content_types(self):
        content = (
            app.openapi()["paths"]["/api/document/{collection}/{doc_id}"]["get"]
            ["responses"]["200"]["content"]
        )

        assert set(content) == {"application/json", "text/markdown"}

    def test_openapi_declares_raw_as_a_repeatable_string_parameter(self):
        params = (
            app.openapi()["paths"]["/api/document/{collection}/{doc_id}"]["get"]
            ["parameters"]
        )
        raw = next(p for p in params if p["name"] == "raw")

        # An array, because the route reads every occurrence rather than the
        # last one — a single-valued declaration would tell a generated client
        # the opposite of what the route does with ``?raw=1&raw=0``.
        assert {"type": "array", "items": {"type": "string"}} in raw["schema"]["anyOf"]
        assert raw["in"] == "query"
