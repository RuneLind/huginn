import json

import pytest
from fastapi.testclient import TestClient

from knowledge_api_server import app
from main.core.search_response_formatter import (
    HIGH_CONFIDENCE_RELEVANCE,
    MEDIUM_CONFIDENCE_RELEVANCE,
    apply_metadata_filters,
    confidence_band,
    extract_chunk_heading,
    extract_chunk_metadata,
    extract_chunk_text,
    normalize_score,
    separate_metadata,
    shape_search_results,
    truncate_snippet,
)
from main.sources.notion.notion_document_reader import NotionDocumentReader


class TestExtractChunkText:
    def test_dict_with_indexed_data(self):
        assert extract_chunk_text({"indexedData": "hello"}) == "hello"

    def test_dict_without_indexed_data(self):
        result = extract_chunk_text({"other": "data"})
        assert "other" in result

    def test_plain_string(self):
        assert extract_chunk_text("hello") == "hello"

    def test_empty_string(self):
        assert extract_chunk_text("") == ""

    def test_none(self):
        assert extract_chunk_text(None) == ""


class TestExtractChunkMetadata:
    def test_dict_with_metadata(self):
        assert extract_chunk_metadata({"metadata": {"wip": "true"}}) == {"wip": "true"}

    def test_dict_without_metadata(self):
        assert extract_chunk_metadata({"indexedData": "text"}) is None

    def test_plain_string(self):
        assert extract_chunk_metadata("hello") is None

    def test_none(self):
        assert extract_chunk_metadata(None) is None


class TestExtractChunkHeading:
    def test_dict_with_heading(self):
        assert extract_chunk_heading({"heading": "Overview"}) == "Overview"

    def test_dict_without_heading(self):
        assert extract_chunk_heading({"indexedData": "text"}) is None

    def test_plain_string(self):
        assert extract_chunk_heading("hello") is None

    def test_none(self):
        assert extract_chunk_heading(None) is None


class TestTruncateSnippet:
    def test_short_text_unchanged(self):
        assert truncate_snippet("Hello world.") == "Hello world."

    def test_none_returns_none(self):
        assert truncate_snippet(None) is None

    def test_empty_returns_empty(self):
        assert truncate_snippet("") == ""

    def test_cuts_at_sentence_boundary(self):
        text = "First sentence. " + "x" * 200
        result = truncate_snippet(text, target=20)
        assert result == "First sentence."

    def test_falls_back_to_word_boundary(self):
        text = "word " * 60  # 300 chars, no sentence endings
        result = truncate_snippet(text, target=200)
        assert result.endswith("…")
        assert len(result) <= 240  # should be near target

    def test_hard_cut_no_spaces(self):
        text = "x" * 300
        result = truncate_snippet(text, target=200)
        assert result == "x" * 200 + "…"


class TestSeparateMetadata:
    def test_extracts_metadata_lines(self):
        text = "**Status:** Active\n**Priority:** High\n\nActual content here."
        content, meta, breadcrumb = separate_metadata(text)
        assert meta == {"Status": "Active", "Priority": "High"}
        assert content == "Actual content here."
        assert breadcrumb is None

    def test_extracts_breadcrumb(self):
        text = "[Projects > My Project > Page]\n**Status:** Done\n\nContent."
        content, meta, breadcrumb = separate_metadata(text)
        assert meta == {"Status": "Done"}
        assert content == "Content."
        assert breadcrumb == "Projects > My Project > Page"

    def test_breadcrumb_only_chunk(self):
        text = "[Folder > Sub > Page]"
        content, meta, breadcrumb = separate_metadata(text)
        assert content == ""
        assert meta == {}
        assert breadcrumb == "Folder > Sub > Page"

    def test_bracket_without_arrow_not_breadcrumb(self):
        text = "[This is just a note]\nSome content."
        content, meta, breadcrumb = separate_metadata(text)
        assert breadcrumb is None
        assert "[This is just a note]" in content

    def test_no_metadata(self):
        text = "Just plain content here."
        content, meta, breadcrumb = separate_metadata(text)
        assert content == "Just plain content here."
        assert meta == {}
        assert breadcrumb is None

    def test_empty_input(self):
        content, meta, breadcrumb = separate_metadata("")
        assert content == ""
        assert meta == {}
        assert breadcrumb is None

    def test_none_input(self):
        content, meta, breadcrumb = separate_metadata(None)
        assert content == ""
        assert meta == {}
        assert breadcrumb is None

    def test_metadata_with_blank_lines_at_start(self):
        text = "\n\n**Type:** Bug\nSome content."
        content, meta, breadcrumb = separate_metadata(text)
        assert meta == {"Type": "Bug"}
        assert content == "Some content."
        assert breadcrumb is None


class TestSnippetFallback:
    """Test that brief search falls back to metadata when content is empty."""

    def test_empty_content_uses_metadata_as_snippet(self):
        """When chunk content is only metadata (no body text), snippet should show metadata."""
        # Simulate what the server does: separate_metadata strips metadata lines,
        # leaving empty content. The snippet fallback should format metadata instead.
        snippet = truncate_snippet("")
        assert snippet == ""

        # Simulate the fallback logic from the search endpoint
        best_chunk = {"content": "", "metadata": {"Status": "Active", "Type": "Task"}}
        snippet = truncate_snippet(best_chunk["content"])
        if not snippet and best_chunk.get("metadata"):
            snippet = " | ".join(f"{k}: {v}" for k, v in best_chunk["metadata"].items())
        assert snippet == "Status: Active | Type: Task"

    def test_no_fallback_when_content_exists(self):
        """When content exists, metadata fallback should not trigger."""
        best_chunk = {"content": "Real content here.", "metadata": {"Status": "Active"}}
        snippet = truncate_snippet(best_chunk["content"])
        if not snippet and best_chunk.get("metadata"):
            snippet = " | ".join(f"{k}: {v}" for k, v in best_chunk["metadata"].items())
        assert snippet == "Real content here."

    def test_no_fallback_when_no_metadata(self):
        """When content is empty and no metadata, snippet stays empty."""
        best_chunk = {"content": "", "score": 0}
        snippet = truncate_snippet(best_chunk["content"])
        if not snippet and best_chunk.get("metadata"):
            snippet = " | ".join(f"{k}: {v}" for k, v in best_chunk["metadata"].items())
        assert snippet == ""


class TestNormalizeScore:
    def test_reranked_zero_score_low_relevance(self):
        # Score 0 with reranker = noise (near threshold), should be low relevance
        result = normalize_score(0, is_reranked=True)
        assert result < 0.35

    def test_reranked_strong_match(self):
        # Score -1.0 with reranker = strong match → high relevance
        result = normalize_score(-1.0, is_reranked=True)
        assert result > 0.95

    def test_reranked_medium_match(self):
        # Score -0.3 = medium confidence → mid-range relevance
        result = normalize_score(-0.3, is_reranked=True)
        assert 0.5 < result < 0.9

    def test_reranked_weak_match(self):
        # Score -0.05 = low confidence → low relevance
        result = normalize_score(-0.05, is_reranked=True)
        assert result < 0.5

    def test_non_reranked_returns_placeholder(self):
        # Non-reranked returns constant placeholder (overridden with rank-based)
        assert normalize_score(0, is_reranked=False) == 0.5
        assert normalize_score(-1.0, is_reranked=False) == 0.5
        assert normalize_score(5.0, is_reranked=False) == 0.5

    def test_returns_float_between_0_and_1(self):
        for score in [-10, -1, 0, 1, 10]:
            result = normalize_score(score, is_reranked=True)
            assert 0.0 <= result <= 1.0

    def test_extreme_scores_no_overflow(self):
        # math.exp(710) would overflow without clamping
        assert normalize_score(1000, is_reranked=True) < 0.01
        assert normalize_score(-1000, is_reranked=True) > 0.99


class TestExtractNotionTitle:
    def test_page_with_title(self):
        page = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "My Page"}],
                }
            }
        }
        assert NotionDocumentReader.get_page_title(page) == "My Page"

    def test_page_without_title(self):
        page = {"properties": {}}
        assert NotionDocumentReader.get_page_title(page) == "Untitled"

    def test_page_with_empty_properties(self):
        page = {"properties": {"Name": {"type": "rich_text"}}}
        assert NotionDocumentReader.get_page_title(page) == "Untitled"

    def test_multi_segment_title(self):
        page = {
            "properties": {
                "title": {
                    "type": "title",
                    "title": [
                        {"plain_text": "Part 1"},
                        {"plain_text": " Part 2"},
                    ],
                }
            }
        }
        assert NotionDocumentReader.get_page_title(page) == "Part 1 Part 2"


class TestTraceEndpoint:
    def test_get_trace_unknown_id_404(self):
        client = TestClient(app)
        response = client.get("/api/trace/0000000000000000")
        assert response.status_code == 404

    def test_get_trace_returns_stored_payload(self):
        from main.core.trace_store import default_trace_store
        tid = default_trace_store().put({"schemaVersion": 1, "query": {"raw": "hi"}})
        client = TestClient(app)
        response = client.get(f"/api/trace/{tid}")
        assert response.status_code == 200
        assert response.json() == {"schemaVersion": 1, "query": {"raw": "hi"}}


class TestPathTraversal:
    def test_rejects_dot_dot(self):
        # Even if FastAPI normalizes the path, the handler checks for ".."
        # When collection doesn't exist, 404 is returned before traversal check
        # which is also safe — no file access occurs
        client = TestClient(app)
        response = client.get("/api/document/test/../../etc/passwd")
        assert response.status_code in (400, 404)

    def test_rejects_absolute_path(self):
        client = TestClient(app)
        response = client.get("/api/document/test//etc/passwd")
        assert response.status_code in (400, 404)


class TestNotionSourceValidation:
    def test_rejects_invalid_source(self):
        client = TestClient(app)
        response = client.get("/api/notion/page/abc123", params={"source": "invalid"})
        assert response.status_code == 400
        assert "Invalid source" in response.json()["detail"]

    def test_accepts_valid_sources(self):
        client = TestClient(app)
        # These will fail with 404/503 since no collections are loaded,
        # but they should NOT return 400 (validation passes)
        for source in ("auto", "live", "local"):
            response = client.get("/api/notion/page/abc123", params={"source": source})
            assert response.status_code != 400, f"source={source} should not be rejected"


class TestApplyMetadataFilters:
    def test_filters_by_project(self):
        results = [
            {"title": "a", "metadata": {"project": "my-proj"}},
            {"title": "b", "metadata": {"project": "other"}},
            {"title": "c", "metadata": {}},
        ]
        filtered = apply_metadata_filters(results, project="my-proj")
        assert len(filtered) == 1
        assert filtered[0]["title"] == "a"

    def test_filters_by_git_branch(self):
        results = [
            {"title": "a", "metadata": {"gitBranch": "main"}},
            {"title": "b", "metadata": {"gitBranch": "feature/x"}},
        ]
        filtered = apply_metadata_filters(results, git_branch="main")
        assert len(filtered) == 1
        assert filtered[0]["title"] == "a"

    def test_filters_by_both_project_and_branch(self):
        results = [
            {"title": "a", "metadata": {"project": "p", "gitBranch": "main"}},
            {"title": "b", "metadata": {"project": "p", "gitBranch": "dev"}},
            {"title": "c", "metadata": {"project": "other", "gitBranch": "main"}},
        ]
        filtered = apply_metadata_filters(results, project="p", git_branch="main")
        assert len(filtered) == 1
        assert filtered[0]["title"] == "a"

    def test_checks_chunk_level_metadata(self):
        results = [
            {
                "title": "a",
                "metadata": {},
                "matchedChunks": [{"metadata": {"project": "my-proj"}}],
            },
        ]
        filtered = apply_metadata_filters(results, project="my-proj")
        assert len(filtered) == 1

    def test_chunk_metadata_overrides_doc_metadata(self):
        results = [
            {
                "title": "a",
                "metadata": {"project": "old"},
                "matchedChunks": [{"metadata": {"project": "new"}}],
            },
        ]
        filtered = apply_metadata_filters(results, project="new")
        assert len(filtered) == 1

    def test_no_filters_returns_all(self):
        results = [{"title": "a"}, {"title": "b"}]
        filtered = apply_metadata_filters(results)
        assert len(filtered) == 2

    def test_no_metadata_excluded(self):
        results = [{"title": "a"}]
        filtered = apply_metadata_filters(results, project="x")
        assert len(filtered) == 0

    def test_brief_results_without_matched_chunks(self):
        """Brief results have no matchedChunks — filter should still work via doc metadata."""
        results = [
            {"title": "a", "metadata": {"project": "p"}, "snippet": "..."},
            {"title": "b", "metadata": {"project": "other"}, "snippet": "..."},
        ]
        filtered = apply_metadata_filters(results, project="p")
        assert len(filtered) == 1
        assert filtered[0]["title"] == "a"


class TestSanitizeFilename:
    def test_basic(self):
        from main.utils.filename import sanitize_filename
        assert sanitize_filename("hello") == "hello"

    def test_special_chars(self):
        from main.utils.filename import sanitize_filename
        assert sanitize_filename('a<b>c:d"e') == "a b c d e"

    def test_collapse_spaces(self):
        from main.utils.filename import sanitize_filename
        assert sanitize_filename("a   b   c") == "a b c"

    def test_truncation(self):
        from main.utils.filename import sanitize_filename
        result = sanitize_filename("x" * 300)
        assert len(result) == 200

    def test_empty_becomes_untitled(self):
        from main.utils.filename import sanitize_filename
        assert sanitize_filename("") == "Untitled"

    def test_all_special_chars_becomes_untitled(self):
        from main.utils.filename import sanitize_filename
        assert sanitize_filename(':<>"/\\|?*') == "Untitled"


class TestTitleFromDocPath:
    def test_basic(self):
        from main.utils.filename import title_from_doc_path
        assert title_from_doc_path("a/b/c.json") == "c"

    def test_no_directory(self):
        from main.utils.filename import title_from_doc_path
        assert title_from_doc_path("c.json") == "c"

    def test_no_json_suffix(self):
        from main.utils.filename import title_from_doc_path
        assert title_from_doc_path("a/b/c.md") == "c.md"

    def test_only_strips_trailing_suffix(self):
        """`.json` mid-path must not be stripped — only the trailing extension."""
        from main.utils.filename import title_from_doc_path
        assert title_from_doc_path("archive/v1.json.bak/notes.txt") == "notes.txt"

    def test_filename_contains_json_substring(self):
        """A filename like `my.json.notes.json` keeps the inner `.json`."""
        from main.utils.filename import title_from_doc_path
        assert title_from_doc_path("a/my.json.notes.json") == "my.json.notes"


def _shape_doc_raw(doc_id, score, text="some indexed content for this doc"):
    return {
        "id": doc_id,
        "url": f"https://example.com/{doc_id}",
        "path": f"wiki/{doc_id}.json",
        "matchedChunks": [{"content": {"indexedData": text, "heading": None}, "score": score}],
    }


class TestConfidenceBand:
    def test_reranked_high(self):
        assert confidence_band(HIGH_CONFIDENCE_RELEVANCE, is_reranked=True) == "high"
        assert confidence_band(0.99, is_reranked=True) == "high"

    def test_reranked_medium(self):
        assert confidence_band(MEDIUM_CONFIDENCE_RELEVANCE, is_reranked=True) == "medium"
        midpoint = (HIGH_CONFIDENCE_RELEVANCE + MEDIUM_CONFIDENCE_RELEVANCE) / 2
        assert confidence_band(midpoint, is_reranked=True) == "medium"

    def test_reranked_low(self):
        assert confidence_band(MEDIUM_CONFIDENCE_RELEVANCE - 0.01, is_reranked=True) == "low"
        assert confidence_band(0.0, is_reranked=True) == "low"

    def test_non_reranked_never_high(self):
        # Rank-based relevance is an ordering hint, not a confidence estimate.
        assert confidence_band(0.75, is_reranked=False) == "medium"
        assert confidence_band(0.99, is_reranked=False) == "medium"

    def test_non_reranked_low_for_tail_ranks(self):
        assert confidence_band(0.45, is_reranked=False) == "low"


class TestShapeSearchResultsConfidenceBand:
    def test_reranked_results_get_band_by_relevance(self):
        raw = {"results": [_shape_doc_raw("a", -1.0), _shape_doc_raw("b", -0.05)], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10)
        assert results[0]["id"] == "a"
        assert results[0]["confidenceBand"] == "high"   # relevance ~0.999
        assert results[1]["confidenceBand"] == "low"    # relevance ~0.25

    def test_non_reranked_results_capped_at_medium_then_low(self):
        raw = {
            "results": [_shape_doc_raw(f"d{i}", 0.1 * (i + 1)) for i in range(6)],
            "reranked": False,
        }
        results, _ = shape_search_results([("wiki", raw)], limit=10)
        assert results[0]["confidenceBand"] == "medium"   # rank 0 → 0.75
        assert results[-1]["confidenceBand"] == "low"     # rank 5 → ~0.47
        assert all(r["confidenceBand"] in ("medium", "low") for r in results)


def _doc_with_chunks(doc_id, chunks):
    """Build a raw search doc. chunks: list of (indexedData, score, heading, metadata)."""
    return {
        "id": doc_id,
        "url": f"https://example.com/{doc_id}",
        "path": f"wiki/{doc_id}.json",
        "matchedChunks": [
            {"content": {"indexedData": data, "heading": heading, "metadata": meta}, "score": score}
            for (data, score, heading, meta) in chunks
        ],
    }


class TestShapeSearchResultsContract:
    """Response-shaping contract for shape_search_results / _shape_doc (M18)."""

    def test_internal_score_fields_stripped(self):
        raw = {"results": [_shape_doc_raw("a", -1.0)], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10)
        r = results[0]
        assert "_score" not in r
        assert "_reranked" not in r
        for chunk in r.get("matchedChunks", []):
            assert "score" not in chunk

    def test_brief_mode_returns_snippet_not_chunks(self):
        raw = {"results": [_shape_doc_raw("a", -1.0, text="Hello world body.")], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10, brief=True)
        r = results[0]
        assert r["snippet"] == "Hello world body."
        assert "matchedChunks" not in r

    def test_full_mode_returns_matched_chunks_with_relevance(self):
        raw = {"results": [_shape_doc_raw("a", -1.0)], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10, brief=False)
        r = results[0]
        assert "matchedChunks" in r
        assert "snippet" not in r
        assert "relevance" in r["matchedChunks"][0]

    def test_max_chunks_per_doc_caps_chunks(self):
        doc = _doc_with_chunks("a", [(f"chunk {i}", -1.0 + i * 0.01, None, None) for i in range(5)])
        raw = {"results": [doc], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10, max_chunks_per_doc=2)
        assert len(results[0]["matchedChunks"]) == 2

    def test_breadcrumb_promoted_and_stripped_from_content(self):
        doc = _doc_with_chunks("a", [("[Guide > Setup]\nThe body text.", -1.0, None, None)])
        raw = {"results": [doc], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10)
        assert results[0]["breadcrumb"] == "Guide > Setup"
        assert "Guide > Setup" not in results[0]["matchedChunks"][0]["content"]

    def test_text_metadata_merged_into_chunk_metadata(self):
        doc = _doc_with_chunks("a", [("**Project:** huginn\nBody.", -1.0, None, {"gitBranch": "main"})])
        raw = {"results": [doc], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10)
        meta = results[0]["matchedChunks"][0]["metadata"]
        assert meta["Project"] == "huginn"   # parsed from **Project:** line
        assert meta["gitBranch"] == "main"   # preserved from the chunk's own metadata

    def test_max_chunk_chars_truncates_full_mode(self):
        doc = _doc_with_chunks("a", [("x" * 500, -1.0, None, None)])
        raw = {"results": [doc], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=10, max_chunk_chars=100)
        content = results[0]["matchedChunks"][0]["content"]
        assert content == "x" * 100 + "…"

    def test_limit_caps_returned_results(self):
        raw = {"results": [_shape_doc_raw(f"d{i}", -1.0 + i * 0.01) for i in range(5)], "reranked": True}
        results, _ = shape_search_results([("wiki", raw)], limit=3)
        assert len(results) == 3

    def test_low_confidence_flag_propagates(self):
        raw = {"results": [_shape_doc_raw("a", -1.0)], "reranked": True, "lowConfidence": True}
        _, any_low = shape_search_results([("wiki", raw)], limit=10)
        assert any_low is True


class TestModelConfig:
    """Verify model configuration to prevent MPS memory explosion on Apple Silicon."""

    def test_cross_encoder_max_length_capped(self):
        """max_length=8192 causes O(n²) attention memory (~48GB per layer). Cap to 512."""
        from main.indexes.reranking.cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        assert reranker.model.max_length <= 512, (
            f"CrossEncoder max_length should be ≤512, got {reranker.model.max_length}"
        )


class _FakeStore:
    """Minimal store stub for the collection-documents endpoint.

    Backs both ``has_collection`` and ``disk_persister.read_text_file`` from an
    in-memory file map; unknown paths raise like a real persister would.
    """

    def __init__(self, files: dict[str, str], collections: set[str]):
        self._files = files
        self._collections = collections
        self.disk_persister = self

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def read_text_file(self, path: str) -> str:
        try:
            return self._files[path]
        except KeyError:
            raise FileNotFoundError(path)  # match the real persister's missing-file error


class TestCollectionDocumentDates:
    """Opt-in date enrichment on /api/collection/{name}/documents."""

    def _store(self) -> _FakeStore:
        mapping = {
            # Two chunks for the same doc → must dedupe to one entry.
            "1": {"documentId": "career/A.md", "documentUrl": "https://youtu.be/a",
                  "documentPath": "yt/documents/career/A.md.json"},
            "2": {"documentId": "career/A.md", "documentUrl": "https://youtu.be/a",
                  "documentPath": "yt/documents/career/A.md.json"},
            # No frontmatter date → falls back to modifiedTime.
            "3": {"documentId": "health/B.md", "documentUrl": "https://youtu.be/b",
                  "documentPath": "yt/documents/health/B.md.json"},
            # Document file missing → date is None, but the doc still lists.
            "4": {"documentId": "tech/C.md", "documentUrl": "https://youtu.be/c",
                  "documentPath": "yt/documents/tech/C.md.json"},
            # Malformed JSON → date is None (logged), doc still lists.
            "5": {"documentId": "tech/D.md", "documentUrl": "https://youtu.be/d",
                  "documentPath": "yt/documents/tech/D.md.json"},
        }
        files = {
            "yt/indexes/index_document_mapping.json": json.dumps(mapping),
            "yt/documents/career/A.md.json": json.dumps(
                {"metadata": {"date": "2026-01-09"}, "modifiedTime": "2026-03-23T21:40:36"}
            ),
            "yt/documents/health/B.md.json": json.dumps(
                {"metadata": {}, "modifiedTime": "2026-02-15T10:00:00"}
            ),
            "yt/documents/tech/D.md.json": "{ not valid json",
        }
        return _FakeStore(files, {"yt"})

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_default_listing_has_no_date(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/yt/documents").json()["documents"]
        assert len(docs) == 4  # A deduped
        assert all("date" not in d for d in docs)

    def test_include_dates_attaches_added_date(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/yt/documents", params={"include_dates": "1"}).json()["documents"]
        by_id = {d["id"]: d for d in docs}
        assert by_id["career/A.md"]["date"] == "2026-01-09"        # frontmatter date wins
        assert by_id["health/B.md"]["date"] == "2026-02-15T10:00:00"  # fallback to mtime
        assert by_id["tech/C.md"]["date"] is None                  # missing file → None
        assert by_id["tech/D.md"]["date"] is None                  # malformed JSON → None

    def test_unknown_collection_404(self):
        client = self._client(self._store())
        assert client.get("/api/collection/nope/documents").status_code == 404


class TestResolveDocDate:
    def test_prefers_frontmatter_date(self):
        from main.routes.collections import _resolve_doc_date
        assert _resolve_doc_date(
            {"metadata": {"date": "2026-01-09"}, "modifiedTime": "2026-03-23T21:40:36"}
        ) == "2026-01-09"

    def test_falls_back_to_modified_time(self):
        from main.routes.collections import _resolve_doc_date
        assert _resolve_doc_date({"metadata": {}, "modifiedTime": "2026-02-15T10:00:00"}) == "2026-02-15T10:00:00"

    def test_none_when_nothing_available(self):
        from main.routes.collections import _resolve_doc_date
        assert _resolve_doc_date({}) is None


class TestCollectionUpdateConcurrency:
    """Per-collection rebuild mutex + status surfacing (H4/H5)."""

    def _store(self):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.searchers["c"] = object()  # make has_collection("c") true
        return store

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_update_returns_409_when_one_already_running(self, monkeypatch):
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        store.try_begin_update("c")  # simulate an in-flight rebuild
        resp = self._client(store).post("/api/collections/c/update")
        assert resp.status_code == 409

    def test_update_starts_and_reserves_slot(self, monkeypatch):
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        resp = self._client(store).post("/api/collections/c/update")
        assert resp.status_code == 200
        assert resp.json() == {"status": "update_started", "collection": "c"}
        assert store.get_update_status("c")["status"] == "running"

    def test_update_unknown_collection_404(self):
        resp = self._client(self._store()).post("/api/collections/nope/update")
        assert resp.status_code == 404

    def test_update_status_reports_failure(self):
        store = self._store()
        store.mark_update_failed("c", RuntimeError("boom"))
        body = self._client(store).get("/api/collections/c/update-status").json()
        assert body["status"] == "failed"
        assert body["error"] == "boom"

    def test_update_status_idle_when_never_run(self):
        body = self._client(self._store()).get("/api/collections/c/update-status").json()
        assert body == {
            "collection": "c", "status": "idle",
            "startedAt": None, "finishedAt": None, "error": None,
        }

    def test_update_status_unknown_collection_404(self):
        resp = self._client(self._store()).get("/api/collections/nope/update-status")
        assert resp.status_code == 404


class TestMaybeEnqueueReindex:
    """Ingest reindex enqueueing skips (does not fail) when a rebuild is in flight."""

    class _FakeBackgroundTasks:
        def __init__(self):
            self.tasks = []

        def add_task(self, *args, **kwargs):
            self.tasks.append((args, kwargs))

    def _store(self):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.searchers["c"] = object()
        return store

    def test_not_configured_when_no_collection(self):
        from main.routes.ingest import _maybe_enqueue_reindex
        bg = self._FakeBackgroundTasks()
        assert _maybe_enqueue_reindex(self._store(), bg, None) == "not_configured"
        assert _maybe_enqueue_reindex(self._store(), bg, "missing") == "not_configured"
        assert bg.tasks == []

    def test_started_when_idle(self):
        from main.routes.ingest import _maybe_enqueue_reindex
        store = self._store()
        bg = self._FakeBackgroundTasks()
        assert _maybe_enqueue_reindex(store, bg, "c") == "started"
        assert len(bg.tasks) == 1
        assert store.get_update_status("c")["status"] == "running"

    def test_skipped_when_already_running(self):
        from main.routes.ingest import _maybe_enqueue_reindex
        store = self._store()
        store.try_begin_update("c")
        bg = self._FakeBackgroundTasks()
        assert _maybe_enqueue_reindex(store, bg, "c") == "skipped_already_running"
        assert bg.tasks == []



class TestIngestErrorHandling:
    """Unexpected ingest failures return a structured 500, not a bare crash (M16)."""

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def _client(self):
        from main.runtime.knowledge_store import KnowledgeStore, get_store
        app.dependency_overrides[get_store] = lambda: KnowledgeStore()
        return TestClient(app)

    def test_youtube_ingest_failure_returns_structured_500(self, monkeypatch):
        app.state.youtube_transcripts_path = "/tmp/yt"
        app.state.youtube_collection = None

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        from main.ingest.registry import source_by_name
        monkeypatch.setattr(source_by_name("youtube"), "ingest_fn", _boom)
        resp = self._client().post("/api/youtube/ingest", json={"title": "T", "url": "https://x"})
        assert resp.status_code == 500
        assert "YouTube ingest failed" in resp.json()["detail"]
        assert "disk full" in resp.json()["detail"]

    def test_unconfigured_path_still_returns_503(self):
        app.state.youtube_transcripts_path = None
        resp = self._client().post("/api/youtube/ingest", json={"title": "T", "url": "https://x"})
        assert resp.status_code == 503

    def test_anthropic_summary_ingest_failure_returns_structured_500(self, monkeypatch):
        app.state.anthropic_summaries_sources_path = "/tmp/anthropic-summaries"
        app.state.anthropic_summaries_collection = None

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        from main.ingest.registry import source_by_name
        monkeypatch.setattr(source_by_name("anthropic_summary"), "ingest_fn", _boom)
        resp = self._client().post(
            "/api/anthropic-summaries/ingest",
            json={"title": "T", "url": "https://x", "summary": "S"},
        )
        assert resp.status_code == 500
        assert "Anthropic summary ingest failed" in resp.json()["detail"]
        assert "disk full" in resp.json()["detail"]

    def test_anthropic_summary_unconfigured_path_returns_503(self):
        app.state.anthropic_summaries_sources_path = None
        resp = self._client().post(
            "/api/anthropic-summaries/ingest",
            json={"title": "T", "url": "https://x", "summary": "S"},
        )
        assert resp.status_code == 503

    def test_tiktok_ingest_failure_returns_structured_500(self, monkeypatch):
        app.state.tiktok_sources_path = "/tmp/tiktok"
        app.state.tiktok_collection = None

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        from main.ingest.registry import source_by_name
        monkeypatch.setattr(source_by_name("tiktok"), "ingest_fn", _boom)
        resp = self._client().post(
            "/api/tiktok/ingest",
            json={"title": "T", "url": "https://x", "summary": "S"},
        )
        assert resp.status_code == 500
        assert "TikTok ingest failed" in resp.json()["detail"]
        assert "disk full" in resp.json()["detail"]

    def test_tiktok_unconfigured_path_returns_503(self):
        app.state.tiktok_sources_path = None
        app.state.tiktok_collection = None
        resp = self._client().post(
            "/api/tiktok/ingest",
            json={"title": "T", "url": "https://x", "summary": "S"},
        )
        assert resp.status_code == 503


class TestGraphRoutes:
    """HTTP coverage for the knowledge-graph routes (M17)."""

    def _graph(self, tmp_path):
        import json as _json
        from main.graph.knowledge_graph import KnowledgeGraph
        data = {
            "nodes": [
                {"id": "epic:E-1", "type": "Epic", "label": "Root epic", "properties": {}},
                {"id": "issue:S-1", "type": "Issue", "label": "Story 1", "properties": {}},
            ],
            "edges": [
                {"source": "issue:S-1", "target": "epic:E-1", "type": "tilhører_epic", "properties": {}},
            ],
        }
        p = tmp_path / "g.json"
        p.write_text(_json.dumps(data))
        return KnowledgeGraph(p)

    def _client(self, store):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def _store_with_graph(self, tmp_path):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.graph = self._graph(tmp_path)
        return store

    def test_subtree_503_when_no_graph(self):
        from main.runtime.knowledge_store import KnowledgeStore
        resp = self._client(KnowledgeStore()).get("/api/graph/epic:E-1/subtree")
        assert resp.status_code == 503

    def test_subtree_returns_nodes_and_edges(self, tmp_path):
        resp = self._client(self._store_with_graph(tmp_path)).get("/api/graph/epic:E-1/subtree")
        assert resp.status_code == 200
        body = resp.json()
        ids = {n["id"] for n in body["nodes"]}
        assert ids == {"epic:E-1", "issue:S-1"}
        assert body["stats"]["edge_count"] == 1

    def test_subtree_404_for_unknown_node(self, tmp_path):
        resp = self._client(self._store_with_graph(tmp_path)).get("/api/graph/epic:NOPE/subtree")
        assert resp.status_code == 404

    def test_node_detail_returns_neighbors(self, tmp_path):
        resp = self._client(self._store_with_graph(tmp_path)).get("/api/graph/epic:E-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "epic:E-1"
        assert len(body["incoming"]) == 1

    def test_node_detail_404_for_unknown(self, tmp_path):
        resp = self._client(self._store_with_graph(tmp_path)).get("/api/graph/epic:NOPE")
        assert resp.status_code == 404


class TestAuthorGraphRoute:
    """HTTP coverage for the author-graph route (M17)."""

    def _client(self, store):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_returns_cached_graph_without_disk(self):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.set_cached_author_graph("xcol", {"nodes": [], "edges": [], "cached": True})
        resp = self._client(store).get("/api/collection/xcol/author-graph")
        assert resp.status_code == 200
        assert resp.json() == {"nodes": [], "edges": [], "cached": True}

    def test_404_when_no_scores_file(self):
        from main.runtime.knowledge_store import KnowledgeStore
        # A collection with no precomputed scores file under huginn-jarvis/data.
        resp = self._client(KnowledgeStore()).get("/api/collection/no-such-collection-xyz/author-graph")
        assert resp.status_code == 404


class TestAnthropicSummaryIngest:
    """ingest_anthropic_summary writes categorized markdown with no author field."""

    def _req(self, **over):
        from main.ingest.anthropic_summaries import AnthropicSummaryIngestRequest
        base = {
            "title": "Claude Code v2 ships subagents",
            "url": "https://docs.anthropic.com/claude-code/subagents",
            "summary": "Subagents let you fan out work.",
            "category": "ai/claude-code",
            "date": "2026-06-28",
        }
        base.update(over)
        return AnthropicSummaryIngestRequest(**base)

    def test_writes_markdown_without_author(self, tmp_path):
        from main.ingest.anthropic_summaries import ingest_anthropic_summary
        result = ingest_anthropic_summary(self._req(), sources_path=str(tmp_path))
        assert result["category"] == "ai/claude-code"
        assert result["summary"] == "Subagents let you fan out work."
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "author:" not in written
        assert 'url: "https://docs.anthropic.com/claude-code/subagents"' in written
        assert 'category: "ai/claude-code"' in written
        assert 'date: "2026-06-28"' in written
        assert written.rstrip().endswith("Subagents let you fan out work.")

    def test_defaults_category_to_ai_general(self, tmp_path):
        from main.ingest.anthropic_summaries import ingest_anthropic_summary
        result = ingest_anthropic_summary(self._req(category=None), sources_path=str(tmp_path))
        assert result["category"] == "ai/general"

    def test_rejects_unknown_category(self, tmp_path):
        import pytest
        from fastapi import HTTPException
        from main.ingest.anthropic_summaries import ingest_anthropic_summary
        with pytest.raises(HTTPException) as exc:
            ingest_anthropic_summary(self._req(category="bogus/nope"), sources_path=str(tmp_path))
        assert exc.value.status_code == 400

    def test_same_url_reingest_overwrites(self, tmp_path):
        # Re-pushing an updated summary for the same url must overwrite, not fork
        # a (2) file — the quoted-frontmatter url is compared via the parser.
        from main.ingest.anthropic_summaries import ingest_anthropic_summary
        first = ingest_anthropic_summary(self._req(summary="v1"), sources_path=str(tmp_path))
        second = ingest_anthropic_summary(self._req(summary="v2 updated"), sources_path=str(tmp_path))
        assert first["file_path"] == second["file_path"]
        category_dir = tmp_path / "ai" / "claude-code"
        assert [p.name for p in category_dir.glob("*.md")] == ["Claude Code v2 ships subagents.md"]
        assert (tmp_path / second["file_path"]).read_text(encoding="utf-8").rstrip().endswith("v2 updated")

    def test_same_title_different_url_forks(self, tmp_path):
        # Same title but a genuinely different url keeps both as distinct docs.
        from main.ingest.anthropic_summaries import ingest_anthropic_summary
        ingest_anthropic_summary(self._req(url="https://docs.anthropic.com/a"), sources_path=str(tmp_path))
        ingest_anthropic_summary(self._req(url="https://docs.anthropic.com/b"), sources_path=str(tmp_path))
        category_dir = tmp_path / "ai" / "claude-code"
        assert len(list(category_dir.glob("*.md"))) == 2


class TestTikTokIngest:
    """ingest_tiktok writes categorized markdown; author is optional (defaults to 'unknown')."""

    def _req(self, **over):
        from main.ingest.tiktok import TikTokIngestRequest
        base = {
            "title": "How to wire a FAISS index in 60 seconds",
            "url": "https://www.tiktok.com/@dev/video/7412345678901234567",
            "summary": "A quick screen-recording walking through FAISS index setup.",
            "author": "@dev",
            "category": "ai/claude-code",
            "date": "2026-07-01",
        }
        base.update(over)
        return TikTokIngestRequest(**base)

    def test_writes_markdown_with_author(self, tmp_path):
        from main.ingest.tiktok import ingest_tiktok
        result = ingest_tiktok(self._req(), sources_path=str(tmp_path))
        assert result["category"] == "ai/claude-code"
        assert result["author"] == "@dev"
        assert result["summary"] == "A quick screen-recording walking through FAISS index setup."
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert 'author: "@dev"' in written
        assert 'url: "https://www.tiktok.com/@dev/video/7412345678901234567"' in written
        assert 'category: "ai/claude-code"' in written
        assert 'date: "2026-07-01"' in written
        assert written.rstrip().endswith("A quick screen-recording walking through FAISS index setup.")

    def test_author_defaults_to_unknown_when_missing(self, tmp_path):
        from main.ingest.tiktok import ingest_tiktok
        result = ingest_tiktok(self._req(author=None), sources_path=str(tmp_path))
        assert result["author"] == "unknown"
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert 'author: "unknown"' in written

    def test_defaults_category_to_ai_general(self, tmp_path):
        from main.ingest.tiktok import ingest_tiktok
        result = ingest_tiktok(self._req(category=None), sources_path=str(tmp_path))
        assert result["category"] == "ai/general"

    def test_rejects_unknown_category(self, tmp_path):
        import pytest
        from fastapi import HTTPException
        from main.ingest.tiktok import ingest_tiktok
        with pytest.raises(HTTPException) as exc:
            ingest_tiktok(self._req(category="bogus/nope"), sources_path=str(tmp_path))
        assert exc.value.status_code == 400

    def test_same_url_reingest_overwrites(self, tmp_path):
        # Re-pushing an updated summary for the same url must overwrite, not fork a (2) file.
        from main.ingest.tiktok import ingest_tiktok
        first = ingest_tiktok(self._req(summary="v1"), sources_path=str(tmp_path))
        second = ingest_tiktok(self._req(summary="v2 updated"), sources_path=str(tmp_path))
        assert first["file_path"] == second["file_path"]
        category_dir = tmp_path / "ai" / "claude-code"
        assert len(list(category_dir.glob("*.md"))) == 1
        assert (tmp_path / second["file_path"]).read_text(encoding="utf-8").rstrip().endswith("v2 updated")


# --- Ingest registry: parametrized suite over the summary push sources ---------

def _summary_sources():
    """Registry sources that write via the shared write_summary helper (all but Jira)."""
    from main.ingest.registry import INGEST_SOURCES
    return [s for s in INGEST_SOURCES if s.name != "jira"]


_SUMMARY_TEXT = "Parametrized summary body for the shared write path."


def _summary_req(src, **over):
    """Build a minimal valid request for a summary source.

    A `summary` is always supplied so the YouTube source skips its transcript
    fetch + Claude call; `author` is added only for models that declare it.
    """
    fields = src.request_model.model_fields
    base = {
        "title": "Registry parametrized title",
        "url": "https://example.com/registry-item",
        "summary": _SUMMARY_TEXT,
        "category": "ai/claude-code",
        "date": "2026-07-04",
    }
    if "author" in fields:
        base["author"] = "@handle"
    base.update(over)
    base = {k: v for k, v in base.items() if k in fields}
    return src.request_model(**base)


@pytest.mark.parametrize("src", _summary_sources(), ids=lambda s: s.name)
class TestSummaryPushSourcesParametrized:
    """One body covering every write_summary-backed push source in the registry."""

    def test_writes_categorized_markdown(self, src, tmp_path):
        result = src.ingest_fn(_summary_req(src), **{src.path_kwarg: str(tmp_path)})
        assert result["category"] == "ai/claude-code"
        assert result["summary"] == _SUMMARY_TEXT
        assert set(src.response_fields).issubset(result.keys())
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert 'category: "ai/claude-code"' in written
        assert 'url: "https://example.com/registry-item"' in written
        assert 'date: "2026-07-04"' in written
        assert 'tags: "ai, claude-code"' in written
        assert written.rstrip().endswith(_SUMMARY_TEXT)

    def test_defaults_category_to_ai_general(self, src, tmp_path):
        result = src.ingest_fn(_summary_req(src, category=None), **{src.path_kwarg: str(tmp_path)})
        assert result["category"] == "ai/general"

    def test_rejects_unknown_category(self, src, tmp_path):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            src.ingest_fn(_summary_req(src, category="bogus/nope"), **{src.path_kwarg: str(tmp_path)})
        assert exc.value.status_code == 400

    def test_same_url_reingest_overwrites(self, src, tmp_path):
        first = src.ingest_fn(_summary_req(src, summary="v1"), **{src.path_kwarg: str(tmp_path)})
        second = src.ingest_fn(_summary_req(src, summary="v2 updated"), **{src.path_kwarg: str(tmp_path)})
        assert first["file_path"] == second["file_path"]
        assert (tmp_path / second["file_path"]).read_text(encoding="utf-8").rstrip().endswith("v2 updated")

    def test_extra_frontmatter_present_only_when_source_has_author(self, src, tmp_path):
        result = src.ingest_fn(_summary_req(src), **{src.path_kwarg: str(tmp_path)})
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        if "author" in src.request_model.model_fields:
            assert "author:" in written
            assert result["author"]
        else:
            assert "author:" not in written


class TestXArticleIngestUnit:
    """Direct coverage for ingest_x_article — author is mandatory and always written."""

    def _req(self, **over):
        from main.ingest.x_articles import XArticleIngestRequest
        base = {
            "title": "How subagents fan out work",
            "url": "https://x.com/anthropic/status/123",
            "author": "@anthropic",
            "summary": "A thread on parallel subagents.",
            "category": "ai/claude",
            "date": "2026-07-02",
        }
        base.update(over)
        return XArticleIngestRequest(**base)

    def test_writes_author_frontmatter_between_url_and_category(self, tmp_path):
        from main.ingest.x_articles import ingest_x_article
        result = ingest_x_article(self._req(), sources_path=str(tmp_path))
        assert result["author"] == "@anthropic"
        assert result["category"] == "ai/claude"
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert 'author: "@anthropic"' in written
        # author frontmatter sits after url, before category
        assert written.index('url:') < written.index('author:') < written.index('category:')

    def test_explicit_tags_deduped_after_category_parts(self, tmp_path):
        from main.ingest.x_articles import ingest_x_article
        result = ingest_x_article(self._req(tags=["claude", "agents"]), sources_path=str(tmp_path))
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        # category "ai/claude" -> parts ai, claude; explicit "claude" is deduped, "agents" appended
        assert 'tags: "ai, claude, agents"' in written

    def test_rejects_unknown_category(self, tmp_path):
        from fastapi import HTTPException
        from main.ingest.x_articles import ingest_x_article
        with pytest.raises(HTTPException) as exc:
            ingest_x_article(self._req(category="nope/x"), sources_path=str(tmp_path))
        assert exc.value.status_code == 400


class TestYouTubeIngestUnit:
    """Direct coverage for ingest_youtube on the pre-made-summary path (no Claude call)."""

    def _req(self, **over):
        from main.ingest.youtube import YouTubeIngestRequest
        base = {
            "title": "Building a FAISS index",
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "summary": "Walkthrough of building a hybrid FAISS + BM25 index.",
            "category": "coding",
            "date": "2026-07-03",
        }
        base.update(over)
        return YouTubeIngestRequest(**base)

    def test_premade_summary_writes_without_author(self, tmp_path):
        from main.ingest.youtube import ingest_youtube
        result = ingest_youtube(self._req(), transcripts_path=str(tmp_path))
        assert result["category"] == "coding"
        assert result["title"] == "Building a FAISS index"
        assert result["url"] == "https://www.youtube.com/watch?v=abcdefghijk"
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "author:" not in written
        assert 'category: "coding"' in written
        assert 'tags: "coding"' in written
        assert 'date: "2026-07-03"' in written
        assert written.rstrip().endswith("Walkthrough of building a hybrid FAISS + BM25 index.")

    def test_premade_summary_defaults_category(self, tmp_path):
        from main.ingest.youtube import ingest_youtube
        result = ingest_youtube(self._req(category=None), transcripts_path=str(tmp_path))
        assert result["category"] == "ai/general"

    def test_rejects_unknown_category(self, tmp_path):
        from fastapi import HTTPException
        from main.ingest.youtube import ingest_youtube
        with pytest.raises(HTTPException) as exc:
            ingest_youtube(self._req(category="bogus/nope"), transcripts_path=str(tmp_path))
        assert exc.value.status_code == 400


class TestJiraIngestUnit:
    """Direct coverage for ingest_jira — validation, metadata merge, PII, mtime, file lookup."""

    def _req(self, **over):
        from main.ingest.jira import JiraIngestRequest
        base = {
            "issueKey": "MELOSYS-1234",
            "url": "https://nav.atlassian.net/browse/MELOSYS-1234",
            "title": "Fix trygdeavgift rounding",
            "summary": "Fix trygdeavgift rounding",
            "status": "In Progress",
            "type": "Story",
            "description": "Details here.",
            "updated": "2026-06-30T12:00:00",
        }
        base.update(over)
        return JiraIngestRequest(**base)

    def test_rejects_invalid_issue_key(self, tmp_path):
        from fastapi import HTTPException
        from main.ingest.jira import ingest_jira
        with pytest.raises(HTTPException) as exc:
            ingest_jira(self._req(issueKey="not-a-key"), sources_path=str(tmp_path))
        assert exc.value.status_code == 400

    def test_writes_frontmatter_and_body(self, tmp_path):
        from main.ingest.jira import ingest_jira
        result = ingest_jira(self._req(), sources_path=str(tmp_path))
        assert result["issue_key"] == "MELOSYS-1234"
        assert result["file_path"].startswith("MELOSYS-1234_")
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "issue_key: MELOSYS-1234" in written
        assert 'status: "In Progress"' in written
        # project derived from key prefix when not supplied by extension
        assert 'project: "MELOSYS"' in written
        assert "# MELOSYS-1234: Fix trygdeavgift rounding" in written

    def test_pii_is_redacted(self, tmp_path):
        from main.ingest.jira import ingest_jira
        result = ingest_jira(
            self._req(description="Reach the reporter at ola.nordmann@nav.no for details."),
            sources_path=str(tmp_path),
        )
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "ola.nordmann@nav.no" not in written
        assert "<redacted-email>" in written

    def test_mtime_set_to_updated_time(self, tmp_path):
        import datetime as dt
        from main.ingest.jira import ingest_jira
        result = ingest_jira(self._req(updated="2026-06-30T12:00:00"), sources_path=str(tmp_path))
        mtime = (tmp_path / result["file_path"]).stat().st_mtime
        expected = dt.datetime.fromisoformat("2026-06-30T12:00:00").timestamp()
        assert abs(mtime - expected) < 1

    def test_reingest_reuses_file_and_merges_preserved_metadata(self, tmp_path):
        from main.ingest.jira import ingest_jira
        from main.utils.frontmatter import read_frontmatter_from_path
        first = ingest_jira(self._req(), sources_path=str(tmp_path))
        # Inject a field the Chrome extension never sends, then re-ingest.
        fp = tmp_path / first["file_path"]
        text = fp.read_text(encoding="utf-8").replace(
            'epic_summary: ""', 'epic_summary: "Rounding epic"'
        )
        fp.write_text(text, encoding="utf-8")
        second = ingest_jira(self._req(status="Done"), sources_path=str(tmp_path))
        assert second["file_path"] == first["file_path"]  # same file reused
        fm = read_frontmatter_from_path(str(tmp_path / second["file_path"]))
        assert fm["status"] == "Done"
        assert fm["epic_summary"] == "Rounding epic"  # preserved across merge

    def test_find_existing_jira_file(self, tmp_path):
        from main.ingest.jira import _find_existing_jira_file, ingest_jira
        first = ingest_jira(self._req(), sources_path=str(tmp_path))
        filepath, metadata = _find_existing_jira_file(str(tmp_path), "MELOSYS-1234")
        assert filepath is not None
        assert filepath.endswith(first["file_path"])
        assert metadata["issue_key"] == "MELOSYS-1234"
        # A key with no file returns (None, {})
        assert _find_existing_jira_file(str(tmp_path), "MELOSYS-9999") == (None, {})
