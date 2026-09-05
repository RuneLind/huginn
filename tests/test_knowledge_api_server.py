import json
import os
import re
import types

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
from main.ingest.registry import INGEST_SOURCES
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


class _CollectionDocumentsCase:
    """Shared TestClient wiring for /api/collection/{name}/documents suites."""

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)


class TestCollectionDocumentDates(_CollectionDocumentsCase):
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
        # Full-precision timestamp rides along for intra-day tie-breaking.
        assert by_id["career/A.md"]["modifiedTime"] == "2026-03-23T21:40:36"
        assert by_id["health/B.md"]["modifiedTime"] == "2026-02-15T10:00:00"
        assert "modifiedTime" not in by_id["tech/C.md"]            # unreadable → omitted

    def test_unknown_collection_404(self):
        client = self._client(self._store())
        assert client.get("/api/collection/nope/documents").status_code == 404


class TestCollectionDocumentThumbnails(_CollectionDocumentsCase):
    """Opt-in ``include_thumbnails`` — the shelf card's picture, off the same
    one-read-per-document pass the dates use."""

    def _store(self) -> _FakeStore:
        mapping = {
            "1": {"documentId": "ai/A.md", "documentUrl": "https://vimeo.com/1",
                  "documentPath": "vm/documents/ai/A.md.json"},
            "2": {"documentId": "ai/B.md", "documentUrl": "https://vimeo.com/2",
                  "documentPath": "vm/documents/ai/B.md.json"},
            "3": {"documentId": "ai/C.md", "documentUrl": "https://vimeo.com/3",
                  "documentPath": "vm/documents/ai/C.md.json"},
            "4": {"documentId": "ai/D.md", "documentUrl": "https://vimeo.com/4",
                  "documentPath": "vm/documents/ai/D.md.json"},
            "5": {"documentId": "ai/E.md", "documentUrl": "https://vimeo.com/5",
                  "documentPath": "vm/documents/ai/E.md.json"},
        }
        files = {
            "vm/indexes/index_document_mapping.json": json.dumps(mapping),
            "vm/documents/ai/A.md.json": json.dumps(
                {"metadata": {"date": "2026-09-05", "thumbnail_url": "https://i.vimeocdn.com/video/a.jpg"}}
            ),
            # No thumbnail in the frontmatter → the key is OMITTED, not null.
            "vm/documents/ai/B.md.json": json.dumps({"metadata": {"date": "2026-09-05"}}),
            # A non-string value is not a thumbnail.
            "vm/documents/ai/C.md.json": json.dumps({"metadata": {"thumbnail_url": 7}}),
            # An EMPTY string is not one either — omitted, never served as "".
            "vm/documents/ai/D.md.json": json.dumps({"metadata": {"thumbnail_url": ""}}),
            # A document whose metadata is not a dict at all collapses to the
            # no-op, like the dates branch's non-dict document — not a 500.
            "vm/documents/ai/E.md.json": json.dumps({"metadata": "a string"}),
        }
        return _FakeStore(files, {"vm"})

    def test_default_listing_has_no_thumbnail(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/vm/documents").json()["documents"]
        assert all("thumbnail_url" not in d for d in docs)

    def test_include_thumbnails_attaches_the_url_when_present(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/vm/documents", params={"include_thumbnails": "1"}).json()["documents"]
        by_id = {d["id"]: d for d in docs}
        assert by_id["ai/A.md"]["thumbnail_url"] == "https://i.vimeocdn.com/video/a.jpg"
        assert "thumbnail_url" not in by_id["ai/B.md"]
        assert "thumbnail_url" not in by_id["ai/C.md"]
        assert "thumbnail_url" not in by_id["ai/D.md"]
        assert "thumbnail_url" not in by_id["ai/E.md"]
        assert len(docs) == 5

    def test_the_listing_reads_metadata_through_the_one_accessor(self):
        # The mechanical half of `_doc_metadata`'s docstring: a resolver that
        # reads `metadata` any other way re-opens the string-metadata 500, and
        # the flag-enumeration test below cannot see a flag it does not know.
        import inspect, re
        from main.routes import collections
        code = "\n".join(
            line for line in inspect.getsource(collections).splitlines()
            if not line.lstrip().startswith("#")
        )
        # Every spelling of a metadata read — .get('metadata'), .get("metadata"),
        # ["metadata"], ['metadata'] — over the code lines; exactly one, the accessor.
        reads = re.findall(r"""(?:\.get\(\s*['"]metadata['"]\s*\)|\[\s*['"]metadata['"]\s*\])""", code)
        assert len(reads) == 1, reads
        assert code.count("_doc_metadata(") >= 4  # the def + three resolvers

    def test_a_non_dict_metadata_document_never_500s_the_listing_whatever_is_asked_for(self):
        # The three resolvers on the one-read pass share ONE metadata accessor;
        # this is the enumeration of the flags that reach it.
        client = self._client(self._store())
        for params in (
            {"include_dates": "1"},
            {"include_scores": "1"},
            {"include_thumbnails": "1"},
            {"include_dates": "1", "include_scores": "1", "include_thumbnails": "1"},
        ):
            res = client.get("/api/collection/vm/documents", params=params)
            assert res.status_code == 200, params
            assert {d["id"] for d in res.json()["documents"]} >= {"ai/E.md"}, params

    def test_thumbnails_and_dates_share_one_read(self):
        store = self._store()
        reads = []
        real = store.read_text_file
        def counting(path):
            reads.append(path)
            return real(path)
        store.read_text_file = counting
        client = self._client(store)
        docs = client.get(
            "/api/collection/vm/documents",
            params={"include_dates": "1", "include_thumbnails": "1"},
        ).json()["documents"]
        by_id = {d["id"]: d for d in docs}
        assert by_id["ai/A.md"]["date"] == "2026-09-05"
        assert by_id["ai/A.md"]["thumbnail_url"] == "https://i.vimeocdn.com/video/a.jpg"
        # One mapping read + one read per document, never two per document.
        assert reads.count("vm/documents/ai/A.md.json") == 1


class TestCollectionDocumentScores(_CollectionDocumentsCase):
    """Opt-in score enrichment on /api/collection/{name}/documents."""

    def _store(self) -> _FakeStore:
        mapping = {
            # Two chunks for the same doc → must dedupe to one entry.
            "1": {"documentId": "a.md", "documentUrl": "https://x.com/a",
                  "documentPath": "xf/documents/a.md.json"},
            "2": {"documentId": "a.md", "documentUrl": "https://x.com/a",
                  "documentPath": "xf/documents/a.md.json"},
            # Only combined_score present.
            "3": {"documentId": "b.md", "documentUrl": "https://x.com/b",
                  "documentPath": "xf/documents/b.md.json"},
            # Unparseable / non-finite scores → omitted, doc still lists.
            "4": {"documentId": "c.md", "documentUrl": "https://x.com/c",
                  "documentPath": "xf/documents/c.md.json"},
            # Document file missing → no score keys, doc still lists.
            "5": {"documentId": "d.md", "documentUrl": "https://x.com/d",
                  "documentPath": "xf/documents/d.md.json"},
            # Malformed JSON → no score keys, doc still lists.
            "6": {"documentId": "e.md", "documentUrl": "https://x.com/e",
                  "documentPath": "xf/documents/e.md.json"},
            # Valid JSON that is NOT an object (a list) → must not blow up the listing.
            "7": {"documentId": "f.md", "documentUrl": "https://x.com/f",
                  "documentPath": "xf/documents/f.md.json"},
            # Valid JSON that is a bare string → same.
            "8": {"documentId": "g.md", "documentUrl": "https://x.com/g",
                  "documentPath": "xf/documents/g.md.json"},
        }
        files = {
            "xf/indexes/index_document_mapping.json": json.dumps(mapping),
            # Frontmatter numerics arrive as STRINGS — the whole point of the coercion.
            "xf/documents/a.md.json": json.dumps(
                {"metadata": {"date": "2026-07-24", "combined_score": "0.604",
                              "relevance_score": "0.993", "engagement_score": "18.8778"},
                 "modifiedTime": "2026-07-24T21:40:36"}
            ),
            "xf/documents/b.md.json": json.dumps({"metadata": {"combined_score": 0.5}}),
            "xf/documents/c.md.json": json.dumps(
                {"metadata": {"combined_score": "not-a-number", "relevance_score": "NaN",
                              "engagement_score": None}}
            ),
            "xf/documents/e.md.json": "{ not valid json",
            "xf/documents/f.md.json": json.dumps([{"metadata": {"combined_score": "0.9"}}]),
            "xf/documents/g.md.json": json.dumps("just a string"),
        }
        return _FakeStore(files, {"xf"})

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_default_listing_has_no_scores(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/xf/documents").json()["documents"]
        assert len(docs) == 7  # a deduped
        assert all("combined_score" not in d for d in docs)

    def test_include_scores_attaches_floats(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/xf/documents", params={"include_scores": "1"}).json()["documents"]
        by_id = {d["id"]: d for d in docs}
        # String frontmatter is coerced to a real float, not passed through.
        assert by_id["a.md"]["combined_score"] == 0.604
        assert isinstance(by_id["a.md"]["combined_score"], float)
        assert by_id["a.md"]["relevance_score"] == 0.993
        assert by_id["a.md"]["engagement_score"] == 18.8778
        # Already-numeric frontmatter passes through unharmed.
        assert by_id["b.md"]["combined_score"] == 0.5
        assert "relevance_score" not in by_id["b.md"]

    def test_include_scores_omits_unparseable_and_missing(self):
        client = self._client(self._store())
        docs = client.get("/api/collection/xf/documents", params={"include_scores": "1"}).json()["documents"]
        by_id = {d["id"]: d for d in docs}
        for doc_id in ("c.md", "d.md", "e.md"):
            assert not any(k in by_id[doc_id] for k in
                           ("combined_score", "relevance_score", "engagement_score")), doc_id
        # Dates are not attached unless asked for.
        assert all("date" not in d for d in docs)

    def test_both_flags_attach_dates_and_scores(self):
        client = self._client(self._store())
        docs = client.get(
            "/api/collection/xf/documents",
            params={"include_dates": "1", "include_scores": "1"},
        ).json()["documents"]
        by_id = {d["id"]: d for d in docs}
        assert by_id["a.md"]["date"] == "2026-07-24"
        assert by_id["a.md"]["modifiedTime"] == "2026-07-24T21:40:36"
        assert by_id["a.md"]["combined_score"] == 0.604
        assert by_id["d.md"]["date"] is None

    def test_non_dict_document_json_does_not_break_listing(self):
        """A document file that parses to a list/string is treated as unreadable.

        Without the isinstance guard the resolvers would call ``.get`` on a list
        and 500 the *whole* listing, not just that one document.
        """
        client = self._client(self._store())
        resp = client.get(
            "/api/collection/xf/documents",
            params={"include_dates": "1", "include_scores": "1"},
        )
        assert resp.status_code == 200
        by_id = {d["id"]: d for d in resp.json()["documents"]}
        for doc_id in ("f.md", "g.md"):
            assert by_id[doc_id]["date"] is None, doc_id
            assert "modifiedTime" not in by_id[doc_id], doc_id
            assert not any(k in by_id[doc_id] for k in
                           ("combined_score", "relevance_score", "engagement_score")), doc_id


class TestResolveDocScores:
    def test_coerces_string_frontmatter(self):
        from main.routes.collections import _resolve_doc_scores
        assert _resolve_doc_scores(
            {"metadata": {"combined_score": "0.604", "relevance_score": "0.993",
                          "engagement_score": "18.8778"}}
        ) == {"combined_score": 0.604, "relevance_score": 0.993, "engagement_score": 18.8778}

    def test_omits_absent_unparseable_and_nonfinite(self):
        from main.routes.collections import _resolve_doc_scores
        assert _resolve_doc_scores({"metadata": {"combined_score": "abc"}}) == {}
        assert _resolve_doc_scores({"metadata": {"combined_score": "inf"}}) == {}
        assert _resolve_doc_scores({"metadata": {"combined_score": "nan"}}) == {}
        assert _resolve_doc_scores({"metadata": {"combined_score": None}}) == {}
        assert _resolve_doc_scores({"metadata": {}}) == {}
        assert _resolve_doc_scores({}) == {}

    def test_ignores_booleans(self):
        from main.routes.collections import _resolve_doc_scores
        # bool is an int subclass — float(True) == 1.0 would be a silent lie.
        assert _resolve_doc_scores({"metadata": {"combined_score": True}}) == {}


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


class TestCollectionReload:
    """POST /api/collections/{name}/reload — swap the in-memory searcher for the
    on-disk one after an out-of-band rebuild, without loading unserved collections."""

    def _store(self):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.searchers["c"] = object()  # make has_collection("c") true
        store._build_aux_indexes = False  # no disk-backed aux indexes in the test
        return store

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_reload_swaps_searcher(self, monkeypatch):
        store = self._store()
        sentinel = types.SimpleNamespace(indexer=types.SimpleNamespace(get_size=lambda: 7))
        monkeypatch.setattr(store, "_build_searcher", lambda name: sentinel)
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda **k: None)
        old = store.searchers["c"]
        resp = self._client(store).post("/api/collections/c/reload")
        assert resp.status_code == 200
        assert resp.json() == {"reloaded": "c"}
        assert store.searchers["c"] is sentinel
        assert store.searchers["c"] is not old

    def test_reload_unknown_collection_404(self):
        resp = self._client(self._store()).post("/api/collections/nope/reload")
        assert resp.status_code == 404

    def test_reload_failure_returns_clean_500(self, monkeypatch):
        store = self._store()

        def boom(name):
            raise RuntimeError("on-disk index gone")

        monkeypatch.setattr(store, "reload_collection", boom)
        resp = self._client(store).post("/api/collections/c/reload")
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "c" in detail
        assert "previous index still serving" in detail


class TestUpdateCorrelation:
    """The optional {runId, job, trigger} body on POST /update.

    The correlation fields must stay OUT of _update_states: __finish_update
    replaces that dict wholesale (so they would be dropped before the ledger
    record is built) and get_update_status splats it into the public response
    (so they would leak onto an endpoint with an exact-shape contract).
    """

    def _store(self):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.searchers["c"] = object()
        return store

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_update_without_body_still_works(self, monkeypatch):
        """Existing callers send no body and no Content-Type at all."""
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        resp = self._client(store).post("/api/collections/c/update")
        assert resp.status_code == 200
        assert resp.json() == {"status": "update_started", "collection": "c"}

    def test_no_body_mints_a_run_id_and_trigger_manual(self, monkeypatch):
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        self._client(store).post("/api/collections/c/update")
        correlation = store._update_correlation["c"]
        assert correlation["runId"]
        assert correlation["trigger"] == "manual"

    def test_body_correlation_is_stored(self, monkeypatch):
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        resp = self._client(store).post(
            "/api/collections/c/update",
            json={"runId": "mimir-2026-07-18T09:28:37Z", "job": "com.huginn.mimir-index",
                  "trigger": "scheduled"},
        )
        assert resp.status_code == 200
        assert store._update_correlation["c"] == {
            "runId": "mimir-2026-07-18T09:28:37Z",
            "job": "com.huginn.mimir-index",
            "trigger": "scheduled",
            "variant": "incremental",
        }

    def test_garbage_body_is_ignored_not_rejected(self, monkeypatch):
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        resp = self._client(store).post(
            "/api/collections/c/update",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert store._update_correlation["c"]["trigger"] == "manual"

    def test_invalid_trigger_falls_back_to_manual(self, monkeypatch):
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        self._client(store).post("/api/collections/c/update", json={"trigger": "bogus"})
        assert store._update_correlation["c"]["trigger"] == "manual"

    def test_correlation_does_not_leak_into_update_status(self, monkeypatch):
        """GET /update-status response shape must be UNCHANGED by this feature."""
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        client = self._client(store)
        client.post("/api/collections/c/update",
                    json={"runId": "r1", "job": "j", "trigger": "scheduled"})
        body = client.get("/api/collections/c/update-status").json()
        assert set(body) == {"collection", "status", "startedAt", "finishedAt", "error"}
        assert body["status"] == "running"

    def test_idle_update_status_shape_is_unchanged(self):
        body = self._client(self._store()).get("/api/collections/c/update-status").json()
        assert body == {
            "collection": "c", "status": "idle",
            "startedAt": None, "finishedAt": None, "error": None,
        }

    def test_finish_writes_a_ledger_record_with_the_correlation(self, monkeypatch, tmp_path):
        import main.runtime.knowledge_store as ks
        from main.runtime.indexing_run_ledger import IndexingRunLedger

        runs_dir = str(tmp_path / "runs")
        monkeypatch.setattr(ks, "IndexingRunLedger",
                            lambda *a, **k: IndexingRunLedger(runs_dir=runs_dir))
        monkeypatch.setattr("main.routes.collections.run_collection_update", lambda *a, **k: None)
        store = self._store()
        self._client(store).post(
            "/api/collections/c/update",
            json={"runId": "shared-run", "job": "com.huginn.c", "trigger": "scheduled"},
        )
        store.mark_update_succeeded("c")

        runs = IndexingRunLedger(runs_dir=runs_dir).recent("c", limit=10)
        assert len(runs) == 1
        assert runs[0]["runId"] == "shared-run"
        assert runs[0]["job"] == "com.huginn.c"
        assert runs[0]["trigger"] == "scheduled"
        assert [p["name"] for p in runs[0]["phases"]] == ["reindex"]
        # The reindex phase carries startedAt so the fold can order phases by time
        # rather than record-arrival (huginn's record lands before a wrapping
        # script's closing record, which would otherwise mis-sort reindex early).
        reindex = runs[0]["phases"][0]
        assert reindex["startedAt"], "reindex phase carries no startedAt"
        assert reindex["startedAt"] == runs[0]["startedAt"]

    def test_failed_run_is_recorded_as_failed(self, monkeypatch, tmp_path):
        import main.runtime.knowledge_store as ks
        from main.runtime.indexing_run_ledger import IndexingRunLedger

        runs_dir = str(tmp_path / "runs")
        monkeypatch.setattr(ks, "IndexingRunLedger",
                            lambda *a, **k: IndexingRunLedger(runs_dir=runs_dir))
        store = self._store()
        store.try_begin_update("c")
        store.mark_update_failed("c", RuntimeError("boom"))

        runs = IndexingRunLedger(runs_dir=runs_dir).recent("c", limit=10)
        assert runs[0]["status"] == "failed"
        assert runs[0]["error"] == "boom"
        assert runs[0]["documentCount"] is None

    def test_a_ledger_failure_never_breaks_the_update(self, monkeypatch):
        """The ledger is observability — it must not be able to fail a reindex."""
        import main.runtime.knowledge_store as ks

        def explode(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(ks, "IndexingRunLedger", explode)
        store = self._store()
        store.try_begin_update("c")
        store.mark_update_succeeded("c")
        assert store.get_update_status("c")["status"] == "succeeded"


class TestIndexingJobsEndpoint:
    def _store(self, names=("c",)):
        from main.runtime.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        for name in names:
            store.searchers[name] = object()
        return store

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def _patch(self, monkeypatch, tmp_path, schedules=None):
        from main.runtime.indexing_run_ledger import IndexingRunLedger
        runs_dir = str(tmp_path / "runs")
        monkeypatch.setattr("main.routes.collections.IndexingRunLedger",
                            lambda *a, **k: IndexingRunLedger(runs_dir=runs_dir))
        monkeypatch.setattr("main.routes.collections.load_schedules",
                            lambda *a, **k: schedules or {})
        return IndexingRunLedger(runs_dir=runs_dir)

    def test_returns_a_row_per_collection(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        body = self._client(self._store()).get("/api/indexing/jobs").json()
        assert [j["collection"] for j in body["jobs"]] == ["c"]

    def test_enumerates_the_union_of_ledger_files_and_loaded_collections(
            self, monkeypatch, tmp_path):
        """A collection with history but not served must still appear, flagged
        loaded:false — otherwise the whole Jira/Confluence/Notion backfill is
        invisible. And a served collection with no history must appear too."""
        ledger = self._patch(monkeypatch, tmp_path)
        ledger.append({"collection": "jira-issues", "runId": "j1",
                       "startedAt": "2026-07-18T09:00:00Z",
                       "finishedAt": "2026-07-18T09:05:00Z", "status": "succeeded"})
        body = self._client(self._store(names=("c",))).get("/api/indexing/jobs").json()
        rows = {j["collection"]: j for j in body["jobs"]}
        assert set(rows) == {"c", "jira-issues"}
        assert rows["jira-issues"]["loaded"] is False
        assert rows["jira-issues"]["lastRun"]["runId"] == "j1"
        assert rows["c"]["loaded"] is True
        assert rows["c"]["lastRun"] is None

    def test_history_and_last_run(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path)
        for i in range(3):
            ledger.append({"collection": "c", "runId": f"r{i}",
                           "startedAt": f"2026-07-1{i}T09:00:00Z",
                           "finishedAt": f"2026-07-1{i}T09:0{i + 1}:00Z",
                           "status": "succeeded"})
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["lastRun"]["runId"] == "r2"
        assert [h["runId"] for h in row["history"]] == ["r0", "r1", "r2"]
        assert row["history"][0]["durationSeconds"] == 60

    def test_median_duration_is_split_by_variant(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path)
        for i, (variant, seconds) in enumerate(
                [("incremental", 10), ("incremental", 30), ("rebuild", 600)]):
            ledger.append({"collection": "c", "runId": f"r{i}", "variant": variant,
                           "durationSeconds": seconds, "status": "succeeded"})
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["medianDurationSeconds"] == {"incremental": 20, "rebuild": 600}

    def test_current_reports_a_running_update(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        store = self._store()
        store.try_begin_update("c")
        row = self._client(store).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["current"]["status"] == "running"
        # try_begin_update also appends the ledger's opening partial, so an
        # API-triggered reindex is genuinely visible on both channels.
        assert row["current"]["source"] == "both"
        assert row["current"]["elapsedSeconds"] >= 0

    def test_current_reports_a_script_side_run(self, monkeypatch, tmp_path):
        """A script-phase run in flight (fetch/tag before the reindex triggers, or
        any run on an unserved collection) must surface through `current` too —
        not force every consumer to also inspect lastRun.status."""
        from datetime import datetime, timezone
        ledger = self._patch(monkeypatch, tmp_path)
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ledger.append({"collection": "jira-issues", "runId": "s1",
                       "source": "script", "stage": "begin", "startedAt": started})
        body = self._client(self._store(names=("c",))).get("/api/indexing/jobs").json()
        row = {j["collection"]: j for j in body["jobs"]}["jira-issues"]
        assert row["current"] == {
            "status": "running", "source": "script",
            "startedAt": started, "elapsedSeconds": row["current"]["elapsedSeconds"],
        }
        assert row["current"]["elapsedSeconds"] >= 0

    def test_current_merges_both_channels_on_the_earlier_start(
            self, monkeypatch, tmp_path):
        """During a wrapped reindex both channels report running. The script
        wraps the reindex, so its earlier start is the whole-run start."""
        from datetime import datetime, timedelta, timezone
        ledger = self._patch(monkeypatch, tmp_path)
        script_start = (datetime.now(timezone.utc) - timedelta(minutes=10)) \
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        ledger.append({"collection": "c", "runId": "s1",
                       "source": "script", "stage": "begin", "startedAt": script_start})
        store = self._store()
        store.try_begin_update("c", run_id="s1")
        row = self._client(store).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["current"]["source"] == "both"
        assert row["current"]["startedAt"] == script_start
        assert row["current"]["elapsedSeconds"] >= 590

    def test_last_run_is_a_fixed_projection(self, monkeypatch, tmp_path):
        """The folded record carries whatever keys any writer appended (the open
        POST accepts extras by design). lastRun is the contract, so it exposes
        exactly LAST_RUN_FIELDS — always all of them, and nothing else."""
        from main.routes.collections import LAST_RUN_FIELDS
        ledger = self._patch(monkeypatch, tmp_path)
        ledger.append({"collection": "c", "runId": "r1",
                       "startedAt": "2026-07-18T09:00:00Z",
                       "finishedAt": "2026-07-18T09:01:00Z", "status": "succeeded",
                       "source": "script", "padding": "x" * 100,
                       "sourceLog": "/tmp/x.log"})
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert set(row["lastRun"]) == set(LAST_RUN_FIELDS)
        assert row["lastRun"]["runId"] == "r1"
        assert row["lastRun"]["durationSeconds"] == 60

    def test_median_is_independent_of_the_history_param(self, monkeypatch, tmp_path):
        """?history=N must change how much history is returned, never what the
        median claims — the median window is fixed at MEDIAN_WINDOW_RUNS."""
        ledger = self._patch(monkeypatch, tmp_path)
        for i in range(10):
            ledger.append({"collection": "c", "runId": f"old{i}",
                           "durationSeconds": 1000, "status": "succeeded"})
        for i in range(50):
            ledger.append({"collection": "c", "runId": f"new{i}",
                           "durationSeconds": 10, "status": "succeeded"})
        client = self._client(self._store())
        deep = client.get("/api/indexing/jobs?history=500").json()["jobs"][0]
        shallow = client.get("/api/indexing/jobs?history=5").json()["jobs"][0]
        assert deep["medianDurationSeconds"] == {"incremental": 10}
        assert shallow["medianDurationSeconds"] == {"incremental": 10}
        assert len(deep["history"]) == 60
        assert len(shallow["history"]) == 5

    def test_schedule_is_tagged_local_without_mutating_the_cache(
            self, monkeypatch, tmp_path):
        """launchd hours are machine-local wall-clock beside UTC timestamps — the
        response says so. And load_schedules returns its shared cached dict, so
        the tag must go on a copy, never the cached entry."""
        cached_schedule = {"kind": "calendar", "hour": 9, "minute": 15}
        self._patch(monkeypatch, tmp_path, schedules={
            "c": {"job": "com.huginn.c-index", "schedule": cached_schedule},
        })
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["schedule"]["timezone"] == "local"
        assert row["nextRunAt"] is not None
        assert "timezone" not in cached_schedule

    def test_current_is_null_when_idle(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["current"] is None

    def test_schedule_is_attached_when_a_plist_exists(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path, schedules={
            "c": {"job": "com.huginn.c-index",
                  "schedule": {"kind": "calendar", "hour": 9, "minute": 15}},
        })
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["job"] == "com.huginn.c-index"
        assert row["schedule"]["hour"] == 9

    def test_schedule_is_null_when_no_plist(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["schedule"] is None

    def test_unreadable_launch_agents_do_not_fail_the_endpoint(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr("main.routes.collections.load_schedules",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
        # A broken schedule source costs the "schedule" field, not the endpoint.
        resp = self._client(self._store()).get("/api/indexing/jobs")
        assert resp.status_code == 200
        assert resp.json()["jobs"][0]["schedule"] is None

    def test_unreadable_ledger_file_does_not_fail_the_endpoint(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path)
        ledger.append({"collection": "c", "runId": "r1", "status": "succeeded"})
        os.chmod(ledger.path_for("c"), 0o000)
        try:
            resp = self._client(self._store()).get("/api/indexing/jobs")
            assert resp.status_code == 200
        finally:
            os.chmod(ledger.path_for("c"), 0o644)


class TestNextRunAt:
    """Server-side "next fire" so consumers never mix launchd's machine-local
    wall-clock schedule fields with the endpoint's UTC timestamps."""

    def _fn(self):
        from main.routes.collections import _next_run_at
        return _next_run_at

    def _local(self, iso_z):
        from datetime import datetime, timezone
        return datetime.strptime(iso_z, "%Y-%m-%dT%H:%M:%SZ") \
            .replace(tzinfo=timezone.utc).astimezone()

    def test_hourly_fires_at_the_next_minute_mark_within_an_hour(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        result = self._fn()({"kind": "hourly", "minute": 35}, None, now=now)
        local = self._local(result)
        assert (local.minute, local.second) == (35, 0)
        assert timedelta(0) < local - now.astimezone() <= timedelta(hours=1)

    def test_calendar_fires_at_the_local_wall_clock_time(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        result = self._fn()({"kind": "calendar", "hour": 9, "minute": 15}, None, now=now)
        local = self._local(result)
        assert (local.hour, local.minute) == (9, 15)
        assert timedelta(0) < local - now.astimezone() <= timedelta(days=1)

    def test_weekday_calendar_lands_on_that_weekday(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        # launchd weekday 0 = Sunday = Python weekday 6.
        result = self._fn()(
            {"kind": "calendar", "hour": 6, "minute": 0, "weekday": 0}, None, now=now)
        local = self._local(result)
        assert local.weekday() == 6
        assert (local.hour, local.minute) == (6, 0)
        assert timedelta(0) < local - now.astimezone() <= timedelta(days=7)

    def test_interval_derives_from_the_last_finish(self):
        result = self._fn()({"kind": "interval", "seconds": 600},
                            {"finishedAt": "2026-07-20T10:00:00Z"})
        assert result == "2026-07-20T10:10:00Z"

    def test_interval_without_a_finished_run_is_none(self):
        assert self._fn()({"kind": "interval", "seconds": 600}, None) is None

    def test_absent_or_malformed_schedule_is_none(self):
        assert self._fn()(None, None) is None
        assert self._fn()({"kind": "hourly", "minute": None}, None) is None
        assert self._fn()({"kind": "calendar", "hour": None}, None) is None


class TestIncompleteAfterForSchedule:
    """The jobs endpoint derives the per-collection `incomplete_after` from launchd
    cadence — max(2×cadence, floor) — so a dead hourly run stops reading `running`
    across six later runs the way the flat 6h let it."""

    def _fn(self):
        from main.routes.collections import _incomplete_after_for_schedule
        return _incomplete_after_for_schedule

    def test_hourly_is_two_hours(self):
        assert self._fn()({"kind": "hourly", "minute": 35}) == 2 * 3600

    def test_interval_is_twice_its_seconds_above_the_floor(self):
        assert self._fn()({"kind": "interval", "seconds": 4 * 3600}) == 8 * 3600

    def test_a_short_interval_is_clamped_to_the_floor(self):
        assert self._fn()({"kind": "interval", "seconds": 900}) == 2 * 3600

    def test_a_plain_calendar_entry_is_daily(self):
        assert self._fn()({"kind": "calendar", "hour": 9, "minute": 0}) == 2 * 86400

    def test_a_weekday_calendar_entry_is_weekly(self):
        assert self._fn()({"kind": "calendar", "hour": 9, "minute": 0,
                           "weekday": 1}) == 2 * 604800

    def test_no_schedule_keeps_the_flat_constant(self):
        from main.runtime.indexing_run_ledger import INCOMPLETE_AFTER_SECONDS
        assert self._fn()(None) == INCOMPLETE_AFTER_SECONDS
        assert self._fn()({"kind": "mystery"}) == INCOMPLETE_AFTER_SECONDS


class TestJobsEndpointCadenceThreshold(TestIndexingJobsEndpoint):
    """End to end: the SAME 3h-old unclosed run reads `incomplete` for an hourly
    collection but still `running` for a daily one — the whole point of finding 3."""

    def _open_3h_ago(self, ledger):
        from datetime import datetime, timedelta, timezone
        started = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        ledger.append({"collection": "c", "runId": "open", "source": "script",
                       "stage": "begin", "startedAt": started})

    def test_hourly_cadence_marks_the_stale_run_incomplete(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path, schedules={
            "c": {"job": "com.huginn.c", "schedule": {"kind": "hourly", "minute": 35}}})
        self._open_3h_ago(ledger)
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["lastRun"]["status"] == "incomplete"

    def test_daily_cadence_keeps_the_same_run_running(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path, schedules={
            "c": {"job": "com.huginn.c",
                  "schedule": {"kind": "calendar", "hour": 9, "minute": 0}}})
        self._open_3h_ago(ledger)
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["lastRun"]["status"] == "running"

    def test_no_schedule_keeps_the_run_running_under_the_flat_default(
            self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path)
        self._open_3h_ago(ledger)
        row = self._client(self._store()).get("/api/indexing/jobs").json()["jobs"][0]
        assert row["lastRun"]["status"] == "running"


class TestIndexingRunsEndpoint:
    """POST /api/indexing/runs — the channel the shell helper reports phases on."""

    def _patch(self, monkeypatch, tmp_path):
        from main.runtime.indexing_run_ledger import IndexingRunLedger
        runs_dir = str(tmp_path / "runs")
        monkeypatch.setattr("main.routes.collections.IndexingRunLedger",
                            lambda *a, **k: IndexingRunLedger(runs_dir=runs_dir))
        return IndexingRunLedger(runs_dir=runs_dir)

    def test_a_script_record_lands_in_the_ledger(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path)
        resp = TestClient(app).post("/api/indexing/runs", json={
            "collection": "c", "runId": "shared", "source": "script", "stage": "end",
            "job": "com.huginn.mimir-index", "trigger": "scheduled",
            "startedAt": "2026-07-18T09:28:37Z", "finishedAt": "2026-07-18T10:44:14Z",
            "phases": [{"name": "tag", "status": "degraded", "durationSeconds": 3033}],
        })
        assert resp.status_code == 200
        assert resp.json()["runId"] == "shared"
        assert ledger.recent("c", limit=5)[0]["phases"][0]["name"] == "tag"

    def test_it_accepts_runs_for_collections_this_server_does_not_serve(
            self, monkeypatch, tmp_path):
        """The CLI-fallback and rebuild paths report for unloaded collections;
        rejecting those would restore the blind spot the ledger removes."""
        ledger = self._patch(monkeypatch, tmp_path)
        resp = TestClient(app).post("/api/indexing/runs",
                                    json={"collection": "not-loaded", "runId": "r"})
        assert resp.status_code == 200
        assert len(ledger.recent("not-loaded", limit=5)) == 1

    def test_a_traversing_collection_name_is_rejected(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        resp = TestClient(app).post("/api/indexing/runs",
                                    json={"collection": "../../escape", "runId": "r"})
        assert resp.status_code == 400
        assert not (tmp_path / "escape.jsonl").exists()

    def test_a_non_object_body_is_rejected(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        assert TestClient(app).post("/api/indexing/runs", json=[1, 2]).status_code == 400
        assert TestClient(app).post(
            "/api/indexing/runs", content=b"not json").status_code == 400

    def test_an_oversize_body_is_rejected_by_content_length(self, monkeypatch, tmp_path):
        """The unauthenticated endpoint must not buffer an arbitrarily large body
        into memory. A Content-Length over the ceiling is refused with 413 before
        the body is read."""
        from main.routes.collections import MAX_REQUEST_BODY_BYTES
        ledger = self._patch(monkeypatch, tmp_path)
        oversize = b'{"collection":"c","runId":"r","x":"' + \
            b"A" * (MAX_REQUEST_BODY_BYTES + 1) + b'"}'
        resp = TestClient(app).post("/api/indexing/runs", content=oversize)
        assert resp.status_code == 413
        # Nothing was written to the ledger.
        assert ledger.recent("c", limit=5) == []

    def test_an_oversize_chunked_body_is_rejected_by_the_streamed_read(
            self, monkeypatch, tmp_path):
        """A chunked body carries no Content-Length, so the header check cannot
        catch it — the streamed read is the real guard."""
        from main.routes.collections import MAX_REQUEST_BODY_BYTES
        self._patch(monkeypatch, tmp_path)

        def chunks():
            yield b'{"collection":"c","runId":"r","x":"'
            for _ in range((MAX_REQUEST_BODY_BYTES // 1024) + 2):
                yield b"A" * 1024

        # A generator body makes httpx send Transfer-Encoding: chunked (no
        # Content-Length), exercising the len(body) guard rather than the header.
        resp = TestClient(app).post("/api/indexing/runs", content=chunks())
        assert resp.status_code == 413

    def test_a_normal_record_is_still_accepted(self, monkeypatch, tmp_path):
        ledger = self._patch(monkeypatch, tmp_path)
        resp = TestClient(app).post("/api/indexing/runs", json={
            "collection": "c", "runId": "ok", "status": "succeeded"})
        assert resp.status_code == 200
        assert len(ledger.recent("c", limit=5)) == 1


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



def _set_ingest(name, path, collection=None):
    """Point a registered ingest source at ``path``/``collection`` on the module app config."""
    from main.runtime.server_config import IngestSourceConfig
    app.state.config.ingest_sources[name] = IngestSourceConfig(path=path, collection=collection)


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
        _set_ingest("youtube", "/tmp/yt", None)

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        from main.ingest.registry import source_by_name
        monkeypatch.setattr(source_by_name("youtube"), "ingest_fn", _boom)
        resp = self._client().post("/api/youtube/ingest", json={"title": "T", "url": "https://x"})
        assert resp.status_code == 500
        assert "YouTube ingest failed" in resp.json()["detail"]
        assert "disk full" in resp.json()["detail"]

    def test_unconfigured_path_still_returns_503(self):
        _set_ingest("youtube", None)
        resp = self._client().post("/api/youtube/ingest", json={"title": "T", "url": "https://x"})
        assert resp.status_code == 503

    def test_anthropic_summary_ingest_failure_returns_structured_500(self, monkeypatch):
        _set_ingest("anthropic_summary", "/tmp/anthropic-summaries", None)

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
        _set_ingest("anthropic_summary", None)
        resp = self._client().post(
            "/api/anthropic-summaries/ingest",
            json={"title": "T", "url": "https://x", "summary": "S"},
        )
        assert resp.status_code == 503

    def test_tiktok_ingest_failure_returns_structured_500(self, monkeypatch):
        _set_ingest("tiktok", "/tmp/tiktok", None)

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
        _set_ingest("tiktok", None, None)
        resp = self._client().post(
            "/api/tiktok/ingest",
            json={"title": "T", "url": "https://x", "summary": "S"},
        )
        assert resp.status_code == 503


#: URL posted by the contract test. It embeds `_CONTRACT_ISSUE_KEY` so the
#: self-hit below is excluded by BOTH exclude_match flavours in the registry:
#: url-equality (every summary source) and issue-key-substring (Jira).
_CONTRACT_ISSUE_KEY = "CONTRACT-1"
_CONTRACT_SELF_URL = f"https://example.com/{_CONTRACT_ISSUE_KEY}/self-doc"


class _IngestFakeSearcher:
    """Records each similarity query and returns two canned hits: the just-posted
    document itself (which exclude_match must drop) and an unrelated one."""

    def __init__(self):
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append(query)
        return {"results": [
            {
                "path": "documents/self-doc.json",
                "url": _CONTRACT_SELF_URL,
                "matchedChunks": [{"content": {"indexedData": "self chunk text"}}],
            },
            {
                "path": "documents/other-doc.json",
                "url": "https://example.com/other-doc",
                "matchedChunks": [{"content": {"indexedData": "canned chunk text"}}],
            },
        ]}


class _IngestFakeStore:
    """Serves any collection name with the same searcher, so the real
    _similar_for_collection / _find_similar_documents shaping still runs — but
    records the names asked for, so the configured collection is pinned too."""

    def __init__(self, searcher):
        self.searcher = searcher
        self.asked = []           # names passed to has_collection
        self.searchers_for = []   # names passed to get_searchers

    def has_collection(self, name):
        self.asked.append(name)
        return True

    def get_searchers(self, names):
        self.searchers_for.extend(names)
        return {n: self.searcher for n in names}

    def get_searchers_and_registries(self, names):
        # Nothing in scope here: the ingest similarity query is de-aliased only
        # for a collection served from a privacy-aliased index.
        return self.get_searchers(names), {n: None for n in names}


class TestIngestContract:
    """Happy-path response contract over every registered ingest source.

    The parametrization is the registry itself, so a source added to
    INGEST_SOURCES without honouring the documented response shape fails here.

    Pinned: the response key set and the relative order of `response_fields`;
    `status`; each response field copied verbatim from the ingest result; the
    similarity query being exactly `src.similar_query(req, result)`; the
    similarity search running against the *configured* collection; `exclude_match`
    actually dropping the self-hit; and the reindex enqueue — its return value
    surfaced as `reindex`, called once with the injected store, a real
    BackgroundTasks and the configured collection, and not called at all for
    do_reindex=False sources.
    """

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def _client(self, store):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    @pytest.mark.parametrize("src", INGEST_SOURCES, ids=lambda s: s.name)
    def test_ingest_happy_path_contract(self, src, monkeypatch):
        from fastapi import BackgroundTasks

        # setitem rather than _set_ingest: monkeypatch restores the entry after.
        from main.runtime.server_config import IngestSourceConfig
        monkeypatch.setitem(
            app.state.config.ingest_sources, src.name,
            IngestSourceConfig(path="/tmp/ingest-contract", collection="contract-collection"),
        )

        result = {key: f"canned-{key}" for key in src.response_fields}
        # The handler dereferences src.ingest_fn at call time, so patching the
        # (non-frozen) registry entry is the seam — same as the 500-path tests.
        monkeypatch.setattr(src, "ingest_fn", lambda req, **kw: dict(result))

        reindex_calls = []

        def _fake_reindex(store, background_tasks, collection):
            reindex_calls.append((store, background_tasks, collection))
            return "stub-reindex"

        monkeypatch.setattr("main.routes.ingest._maybe_enqueue_reindex", _fake_reindex)

        searcher = _IngestFakeSearcher()
        store = _IngestFakeStore(searcher)
        # The same model instance the posted body is built from, so the expected
        # similarity query below is the registry's own lambda, not a copy of it.
        req = _summary_req(src, url=_CONTRACT_SELF_URL)
        resp = self._client(store).post(src.route_path, json=req.model_dump(exclude_none=True))

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Exact key set, plus the relative order of the declared response_fields
        # (what the registry documents); reindex only for sources that reindex.
        assert set(body) == {
            "status", *src.response_fields, "similar",
            *(["reindex"] if src.do_reindex else []),
        }
        assert [k for k in body if k in src.response_fields] == list(src.response_fields)
        assert body["status"] == "ingested"
        for key in src.response_fields:
            assert body[key] == result[key]
        # The self-hit is dropped by exclude_match; only the unrelated one survives.
        assert body["similar"] == [{
            "title": "other-doc",
            "url": "https://example.com/other-doc",
            "snippet": "canned chunk text",
        }]
        assert searcher.queries == [src.similar_query(req, result)]
        assert store.asked == ["contract-collection"]
        assert store.searchers_for == ["contract-collection"]

        if src.do_reindex:
            (called_store, called_tasks, called_collection), = reindex_calls
            assert called_store is store
            assert isinstance(called_tasks, BackgroundTasks)
            assert called_collection == "contract-collection"
            assert body["reindex"] == "stub-reindex"
        else:
            assert reindex_calls == []

    def test_route_table_matches_registry(self):
        # Pairs, not a dict: a duplicate route_path stays visible.
        registered = [
            (getattr(r, "path", ""), getattr(r, "methods", None)) for r in app.routes
            if getattr(r, "path", "").startswith("/api/")
            and getattr(r, "path", "").endswith("/ingest")
        ]
        assert sorted(p for p, _ in registered) == sorted(s.route_path for s in INGEST_SOURCES)
        for path, methods in registered:
            assert methods == {"POST"}, (path, methods)


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
        # A collection with no precomputed scores file under any discovered dir.
        resp = self._client(KnowledgeStore()).get("/api/collection/no-such-collection-xyz/author-graph")
        assert resp.status_code == 404

    @staticmethod
    def _scores():
        # Shape matches the producer's {handle: info} dict. Edges come from the
        # indexed documents (none here), so a discovered file yields an empty
        # graph with 200 -- the 404 is what proves discovery failed.
        return {"alice": {"author_score": 0.9, "tweet_count": 10, "community": 0}}

    def test_scores_discovered_from_any_private_subrepo(self, tmp_path, monkeypatch):
        # No sub-repo name is hardcoded: a hypothetical huginn-zzz/data/ serves it.
        import json
        from main.runtime.knowledge_store import KnowledgeStore
        scores_dir = tmp_path / "huginn-zzz" / "data"
        scores_dir.mkdir(parents=True)
        (scores_dir / "xcol-author-scores.json").write_text(json.dumps(self._scores()))
        monkeypatch.setattr(app.state, "huginn_root", tmp_path)
        resp = self._client(KnowledgeStore()).get("/api/collection/xcol/author-graph")
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    def test_scores_fall_back_to_public_data_dir(self, tmp_path, monkeypatch):
        import json
        from main.runtime.knowledge_store import KnowledgeStore
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "xcol-author-scores.json").write_text(json.dumps(self._scores()))
        monkeypatch.setattr(app.state, "huginn_root", tmp_path)
        resp = self._client(KnowledgeStore()).get("/api/collection/xcol/author-graph")
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []


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


class TestVimeoIngest:
    """ingest_vimeo writes the summary plus an optional ## Transcript section."""

    #: The real shape Muninn sends: one `### [HH:MM:SS]` HEADING per window, not
    #: a bare cue line. See `ingest_vimeo`'s docstring — a heading is what makes
    #: the splitter label a mid-window chunk with the window it came from, and
    #: `TestVimeoDocumentThroughTheConverter` pins that end of it.
    _TRANSCRIPT = (
        "### [00:00:00]\n\nHello and welcome.\n\n### [00:02:00]\n\nOn to the demo."
    )

    def _req(self, **over):
        from main.ingest.vimeo import VimeoIngestRequest
        base = {
            "title": "Trust but verify",
            "url": "https://vimeo.com/1223358361",
            "summary": "A conference talk about verifying model output.",
            "category": "ai/claude-code",
            "date": "2026-09-01",
            "transcript_markdown": self._TRANSCRIPT,
            "caption_lang": "en-x-autogen",
            "caption_kind": "auto",
            "duration_sec": 3180,
            "summary_kind": "standard",
            "summary_lang": "en",
            "author": "JavaZone",
            "upload_date": "2026-09-03 12:11:41",
            "speaker": "Totto - Kari Nordmann",
            "thumbnail_url": "https://i.vimeocdn.com/video/x-1280x720.jpg",
        }
        base.update(over)
        return VimeoIngestRequest(**base)

    def test_writes_frontmatter_and_transcript_section(self, tmp_path):
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(self._req(), sources_path=str(tmp_path))
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert 'url: "https://vimeo.com/1223358361"' in written
        assert 'vimeo_video_id: "1223358361"' in written
        assert 'caption_lang: "en-x-autogen"' in written
        assert 'caption_kind: "auto"' in written
        # The summary's own provenance (Muninn's v2 kind + language picker),
        # beside the caption's: a Norwegian summary of an English talk is a
        # legitimate document, and the reader must be able to tell which half
        # is in which language without re-deriving it.
        assert 'summary_kind: "standard"' in written
        assert 'summary_lang: "en"' in written
        # v2 PR 2 — what oEmbed knew and the route used to throw away.
        assert 'author: "JavaZone"' in written
        assert 'upload_date: "2026-09-03 12:11:41"' in written
        assert 'speaker: "Totto - Kari Nordmann"' in written
        assert 'thumbnail_url: "https://i.vimeocdn.com/video/x-1280x720.jpg"' in written
        # Bare, not quoted: the reader serves it as a number (see
        # TestVimeoDocumentThroughTheConverter::test_duration_is_a_number_not_a_string).
        assert "duration_sec: 3180" in written
        assert 'duration_sec: "3180"' not in written
        # The summary comes first; the transcript is appended under its heading.
        assert written.index("A conference talk") < written.index("## Transcript")
        assert "[00:02:00]" in written
        assert written.index("## Transcript") < written.index("Hello and welcome")

    def test_transcript_is_not_in_the_response_summary(self, tmp_path):
        # response_fields ships `summary` back over HTTP and the registry builds
        # the similarity query from it — 50 minutes of captions belongs in
        # neither.
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(self._req(), sources_path=str(tmp_path))
        assert result["summary"] == "A conference talk about verifying model output."
        assert "Hello and welcome" not in result["summary"]

    def test_no_transcript_section_without_transcript(self, tmp_path):
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(
            self._req(transcript_markdown=None), sources_path=str(tmp_path)
        )
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "## Transcript" not in written
        assert written.rstrip().endswith("A conference talk about verifying model output.")

    def test_blank_transcript_writes_no_heading(self, tmp_path):
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(
            self._req(transcript_markdown="   \n\n  "), sources_path=str(tmp_path)
        )
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "## Transcript" not in written

    def test_the_response_field_set_names_author(self):
        # The HTTP half is `TestVimeoTranscriptCap::test_the_http_response_carries_author`
        # (a real POST); this pins the registry tuple the route projects from.
        from main.ingest.registry import INGEST_SOURCES
        vimeo = next(s for s in INGEST_SOURCES if s.name == "vimeo")
        assert "author" in vimeo.response_fields

    #: Spelled out, not imported (the transcript cap's rule): a test that builds
    #: its boundary from the constant cannot notice the constant moving, and the
    #: number matters — four of these at the cap must stay well inside the
    #: 8192-CHARACTER frontmatter head the readers parse.
    _FIELD_CAP = 512

    def test_the_field_cap_is_the_number_the_frontmatter_head_allows(self):
        from main.ingest.vimeo import VIMEO_FIELD_MAX_BYTES
        assert VIMEO_FIELD_MAX_BYTES == self._FIELD_CAP

    def test_oembed_fields_are_capped_like_the_transcript(self):
        from pydantic import ValidationError
        for key in ("author", "upload_date", "speaker", "thumbnail_url"):
            with pytest.raises(ValidationError):
                self._req(**{key: "x" * (self._FIELD_CAP + 1)})
            self._req(**{key: "x" * self._FIELD_CAP})  # at the cap: fine

    def test_every_frontmatter_bound_string_on_the_request_is_capped(self):
        # The enumeration, not a sample: every Optional[str] the request writes
        # into the frontmatter, and each tag. `title` is the FILENAME (its own
        # 200-char truncation) and `summary`/`transcript_markdown` are body,
        # so they are not in this set.
        from pydantic import ValidationError
        for key in ("date", "caption_lang", "caption_kind", "summary_kind", "summary_lang",
                    "author", "upload_date", "speaker", "thumbnail_url"):
            with pytest.raises(ValidationError, match="exceeds"):
                self._req(**{key: "x" * (self._FIELD_CAP + 1)})
        with pytest.raises(ValidationError, match="exceeds"):
            self._req(tags=["ok", "x" * (self._FIELD_CAP + 1)])
        self._req(tags=["x" * self._FIELD_CAP])

    def test_the_cap_counts_bytes_not_characters(self):
        from pydantic import ValidationError
        # "é" is two bytes: 256 of them fit, 257 do not, though both are far
        # under the cap counted as characters.
        self._req(speaker="é" * (self._FIELD_CAP // 2))
        with pytest.raises(ValidationError):
            self._req(speaker="é" * (self._FIELD_CAP // 2 + 1))

    def test_optional_provenance_fields_are_omitted_when_absent(self, tmp_path):
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(
            self._req(
                caption_lang=None, caption_kind=None, duration_sec=None,
                summary_kind=None, summary_lang=None,
                author=None, upload_date=None, speaker=None, thumbnail_url=None,
            ),
            sources_path=str(tmp_path),
        )
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "caption_lang:" not in written
        assert "caption_kind:" not in written
        assert "duration_sec:" not in written
        assert "summary_kind:" not in written
        assert "summary_lang:" not in written
        assert "author:" not in written
        assert "upload_date:" not in written
        assert "speaker:" not in written
        assert "thumbnail_url:" not in written
        assert 'vimeo_video_id: "1223358361"' in written

    def test_the_id_key_is_namespaced_against_the_other_video_sources(self, tmp_path):
        # `video_id` is NOT this vertical's word: the YouTube channel fetcher
        # writes a bare `video_id: <11-char YouTube id>` into every file of the
        # `emma-hubbard-transcripts` collection, and the converter's allowlist is
        # global. An unqualified key would have served a YouTube id under the
        # name a Vimeo consumer reads.
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(self._req(), sources_path=str(tmp_path))
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert 'vimeo_video_id: "1223358361"' in written
        assert not re.search(r"^video_id:", written, re.MULTILINE)

    def test_zero_duration_still_written(self, tmp_path):
        # 0 means "the player never said"; `is not None` keeps it, a falsy check
        # would silently drop it.
        from main.ingest.vimeo import ingest_vimeo
        result = ingest_vimeo(self._req(duration_sec=0), sources_path=str(tmp_path))
        written = (tmp_path / result["file_path"]).read_text(encoding="utf-8")
        assert "duration_sec: 0" in written

    def test_same_url_reingest_overwrites(self, tmp_path):
        from main.ingest.vimeo import ingest_vimeo
        first = ingest_vimeo(self._req(summary="v1"), sources_path=str(tmp_path))
        second = ingest_vimeo(self._req(summary="v2 updated"), sources_path=str(tmp_path))
        assert first["file_path"] == second["file_path"]
        assert len(list((tmp_path / "ai" / "claude-code").glob("*.md"))) == 1

    def test_rejects_unknown_category(self, tmp_path):
        from fastapi import HTTPException
        from main.ingest.vimeo import ingest_vimeo
        with pytest.raises(HTTPException) as exc:
            ingest_vimeo(self._req(category="bogus/nope"), sources_path=str(tmp_path))
        assert exc.value.status_code == 400


class TestVimeoVideoIdFromUrl:
    """The document's video_id is derived from the url, never sent alongside it.

    Same accepted forms as Muninn's ``src/vimeo/url.ts``: a real Vimeo host, and
    the id is the video-naming segment of a KNOWN path shape — never simply the
    first run of digits in the string.
    """

    @pytest.mark.parametrize("url,expected", [
        ("https://vimeo.com/1223358361", "1223358361"),
        ("https://vimeo.com/1223358361/abc123", "1223358361"),
        ("https://vimeo.com/1223358361?h=abc123", "1223358361"),
        ("https://player.vimeo.com/video/1223358361", "1223358361"),
        ("https://vimeo.com/notanid", None),
        # A digit inside a word is not an id: the match is anchored on a path
        # separator at the front and a segment boundary at the back, so a
        # vanity/on-demand slug cannot donate its digits.
        ("https://vimeo.com/ondemand/film2024", None),
        ("https://vimeo.com/abc123", None),
        ("https://vimeo.com/1223358361abc", None),
        ("", None),
        (None, None),
    ])
    def test_extracts_numeric_segment(self, url, expected):
        from main.ingest.vimeo import vimeo_video_id_from_url
        assert vimeo_video_id_from_url(url) == expected

    @pytest.mark.parametrize("url,expected", [
        # The container id comes FIRST in these two shapes: a first-numeric-run
        # match keys the document on the showcase / group instead of the video.
        ("https://vimeo.com/showcase/7654321/video/1223358361", "1223358361"),
        ("https://vimeo.com/groups/12345/videos/1223358361", "1223358361"),
        ("https://vimeo.com/channels/staffpicks/1223358361", "1223358361"),
        ("https://www.vimeo.com/1223358361", "1223358361"),
        ("https://player.vimeo.com/video/1223358361/deadbeef99", "1223358361"),
    ])
    def test_container_shapes_yield_the_video_id(self, url, expected):
        from main.ingest.vimeo import vimeo_video_id_from_url
        assert vimeo_video_id_from_url(url) == expected

    @pytest.mark.parametrize("url", [
        # No host gate at all is the security half: a url a third party controls
        # can spell "vimeo.com" anywhere in its path and donate an id to a
        # document keyed on it.
        "https://evil.example/vimeo.com/999/redirect",
        "https://evilvimeo.com/1223358361",
        "https://vimeo.com.evil.example/1223358361",
        "ftp://vimeo.com/1223358361",
        # A leading zero is a malformed id, not a second video — Muninn refuses
        # it, so huginn must not derive one either or the two disagree about
        # what the same document is called.
        "https://vimeo.com/0001223358361",
        # A group / showcase landing page names no video.
        "https://vimeo.com/groups/12345",
        "https://vimeo.com/showcase/7654321",
    ])
    def test_rejected(self, url):
        from main.ingest.vimeo import vimeo_video_id_from_url
        assert vimeo_video_id_from_url(url) is None


class TestVimeoPathShapesArePairedWithTheirHost:
    """A path shape belongs to ONE host, as it does in Muninn's ``url.ts``.

    Muninn matches ``PLAYER_PATH_RE`` against ``player.vimeo.com`` and
    ``WATCH_PATH_RE`` against ``vimeo.com`` (``src/vimeo/url.ts``). Matching
    every shape against both hosts invents ids for urls Muninn calls no video —
    and the two sides then disagree about what a document is called.
    """

    @pytest.mark.parametrize("url", [
        # Watch-host spelling of the embed path: Muninn's WATCH_PATH_RE wants
        # `/<id>`, and "video" is not an id.
        "https://vimeo.com/video/1223358361",
        # Embed host without the `/video/` prefix: Muninn's PLAYER_PATH_RE
        # requires it, and the watch shapes are not tried on this host.
        "https://player.vimeo.com/1223358361",
        # The same pairing for the container shapes.
        "https://player.vimeo.com/channels/staffpicks/1223358361",
        "https://player.vimeo.com/showcase/7654321/video/1223358361",
    ])
    def test_a_shape_from_the_other_host_names_no_video(self, url):
        from main.ingest.vimeo import vimeo_video_id_from_url
        assert vimeo_video_id_from_url(url) is None

    @pytest.mark.parametrize("url,expected", [
        # A trailing space is not part of the url: Muninn parses through
        # `new URL`, which strips leading and trailing C0-control-or-space, so
        # `.../123 ` is the same video there. (`urlsplit` lstrips only.)
        ("https://vimeo.com/1223358361 ", "1223358361"),
        ("  https://vimeo.com/1223358361  ", "1223358361"),
        # NOT stripped, on either side: a non-breaking space is an ordinary
        # character to the WHATWG parser, which percent-encodes it into the path.
        ("https://vimeo.com/1223358361\u00a0", None),
    ])
    def test_surrounding_whitespace_matches_muninns_url_parser(self, url, expected):
        from main.ingest.vimeo import vimeo_video_id_from_url
        assert vimeo_video_id_from_url(url) == expected


class TestVimeoSelfLinkExclusion:
    """The ingest route's `similar` list must not link the document to itself.

    The stored url and the incoming one address the same video through different
    spellings (``www.``, an unlisted ``?h=`` hash, the player host), so an exact
    string compare lets the just-written document come back as its own
    "related reading".
    """

    _REQ_URL = "https://vimeo.com/1223358361"

    def _exclude(self, doc_url):
        from main.ingest.registry import source_by_name
        from main.ingest.vimeo import VimeoIngestRequest
        req = VimeoIngestRequest(title="T", url=self._REQ_URL, summary="S")
        return source_by_name("vimeo").exclude_match(req, {"url": doc_url})

    @pytest.mark.parametrize("doc_url", [
        "https://vimeo.com/1223358361",
        "https://www.vimeo.com/1223358361",
        "https://vimeo.com/1223358361?h=deadbeef99",
        "https://vimeo.com/1223358361/deadbeef99",
        "https://player.vimeo.com/video/1223358361",
    ])
    def test_same_video_is_excluded(self, doc_url):
        assert self._exclude(doc_url) is True

    @pytest.mark.parametrize("doc_url", [
        "https://vimeo.com/999999999",
        "https://example.com/some-article",
        "",
    ])
    def test_other_documents_are_kept(self, doc_url):
        assert self._exclude(doc_url) is False

    def test_falls_back_to_the_string_when_neither_side_parses(self):
        # Nothing forces a stored doc's url to be a Vimeo one (a re-categorized
        # paste, a hand-written page): with no id on either side the old exact
        # compare is still the answer.
        from main.ingest.registry import source_by_name
        from main.ingest.vimeo import VimeoIngestRequest
        req = VimeoIngestRequest(title="T", url="https://example.com/x", summary="S")
        src = source_by_name("vimeo")
        assert src.exclude_match(req, {"url": "https://example.com/x"}) is True
        assert src.exclude_match(req, {"url": "https://example.com/y"}) is False


class TestVimeoDocumentThroughTheConverter:
    """What the INDEX ends up holding for a Vimeo capture.

    The two halves the ingest tests above cannot see on their own: a search hit
    can only cite a minute of a 50-minute talk if the chunk the searcher matched
    carries the window cue as its heading, and a reader can only tell an ``auto``
    transcript from a ``manual`` one if the provenance frontmatter survives the
    converter's metadata allowlist.
    """

    _WINDOWS = ("[00:00:00]", "[00:02:00]", "[00:04:00]")
    #: `w2s07` = window 2, sentence 7. Every sentence is uniquely attributable,
    #: so a chunk's text alone says which window it was cut from.
    _TOKEN_RE = re.compile(r"w(\d)s\d\d")

    def _window_body(self, index):
        return " ".join(
            f"w{index}s{i:02d} the speaker keeps talking about verifying model output."
            for i in range(1, 26)
        )

    def _transcript(self):
        # ~1.7 kB per window, i.e. every window is longer than the splitter's
        # 1000-char chunk_size and MUST be cut mid-section — which is the case
        # the single `## Transcript` heading lost the cue on.
        return "\n\n".join(
            f"### {cue}\n\n{self._window_body(i + 1)}"
            for i, cue in enumerate(self._WINDOWS)
        )

    def _convert(self, tmp_path, **over):
        from main.ingest.vimeo import VimeoIngestRequest, ingest_vimeo
        from main.sources.files.files_document_converter import FilesDocumentConverter
        base = {
            "title": "Trust but verify",
            "url": "https://vimeo.com/1223358361",
            "summary": "A conference talk about verifying model output.",
            "category": "ai/claude-code",
            "date": "2026-09-01",
            "transcript_markdown": self._transcript(),
            "caption_lang": "en-x-autogen",
            "caption_kind": "auto",
            "duration_sec": 3220,
            "summary_kind": "talk-notes",
            "summary_lang": "nb",
            "author": "JavaZone",
            "upload_date": "2026-09-03 12:11:41",
            "speaker": "Kari Nordmann",
            "thumbnail_url": "https://i.vimeocdn.com/video/x-1280x720.jpg",
        }
        base.update(over)
        result = ingest_vimeo(VimeoIngestRequest(**base), sources_path=str(tmp_path))
        full_path = tmp_path / result["file_path"]
        document = {
            "fileRelativePath": result["file_path"],
            "fileFullPath": str(full_path),
            "modifiedTime": "2026-09-01T00:00:00Z",
            "content": [{"text": full_path.read_text(encoding="utf-8")}],
        }
        return FilesDocumentConverter().convert(document)[0]

    def test_every_transcript_chunk_names_its_window(self, tmp_path):
        converted = self._convert(tmp_path)
        seen = {}
        for chunk in converted["chunks"]:
            windows = {int(m) for m in self._TOKEN_RE.findall(chunk["indexedData"])}
            if not windows:
                continue
            assert len(windows) == 1, "a chunk spans two windows; the cue is ambiguous"
            cue = self._WINDOWS[windows.pop() - 1]
            assert chunk.get("heading") == cue
            seen[cue] = seen.get(cue, 0) + 1
        # Every window is represented, and each was cut into more than one chunk
        # — so this pins the MID-SECTION chunk, not just the one that starts on
        # the heading line.
        assert set(seen) == set(self._WINDOWS)
        assert all(count >= 2 for count in seen.values()), seen

    def test_provenance_frontmatter_reaches_chunk_metadata(self, tmp_path):
        converted = self._convert(tmp_path)
        metadata = converted["metadata"]
        assert metadata["vimeo_video_id"] == "1223358361"
        assert metadata["caption_lang"] == "en-x-autogen"
        assert metadata["caption_kind"] == "auto"
        # The two v2 keys are ALLOWLISTED, not just written: a key the writer
        # emits and `_FRONTMATTER_METADATA_FIELDS` does not name is invisible
        # over the API, which is the silent half of this contract.
        assert metadata["summary_kind"] == "talk-notes"
        assert metadata["summary_lang"] == "nb"
        assert metadata["author"] == "JavaZone"
        assert metadata["upload_date"] == "2026-09-03 12:11:41"
        assert metadata["speaker"] == "Kari Nordmann"
        assert metadata["thumbnail_url"] == "https://i.vimeocdn.com/video/x-1280x720.jpg"
        for chunk in converted["chunks"]:
            assert chunk["metadata"]["caption_kind"] == "auto"
            assert chunk["metadata"]["vimeo_video_id"] == "1223358361"
            assert chunk["metadata"]["summary_lang"] == "nb"

    def test_a_youtube_video_id_is_not_surfaced_under_vimeos_key(self, tmp_path):
        # `_FRONTMATTER_METADATA_FIELDS` is global: every markdown collection is
        # converted here. The YouTube channel fetcher writes a bare `video_id:`
        # into every transcript it saves, so an unqualified `video_id` in the
        # allowlist serves a YouTube id under the key a Vimeo reader reads.
        from main.sources.files.files_document_converter import FilesDocumentConverter
        text = (
            "---\ntitle: A parenting video\nvideo_id: -AbCdEf1234\n"
            'channel: Some Channel\nurl: "https://www.youtube.com/watch?v=-AbCdEf1234"\n'
            "upload_date: 2024-01-03\n---\n\nBody.\n"
        )
        path = tmp_path / "yt.md"
        path.write_text(text, encoding="utf-8")
        converted = FilesDocumentConverter().convert({
            "fileRelativePath": "markdown/Channel/yt.md",
            "fileFullPath": str(path),
            "modifiedTime": "2026-09-01T00:00:00Z",
            "content": [{"text": text}],
        })[0]
        assert "video_id" not in converted["metadata"]
        assert "vimeo_video_id" not in converted["metadata"]
        for chunk in converted["chunks"]:
            assert "video_id" not in chunk["metadata"]

    def test_duration_is_a_number_not_a_string(self, tmp_path):
        # A quoted "3220" sorts and compares as text; every consumer would have
        # to re-coerce it, and the ones that forget compare "1000" < "900".
        converted = self._convert(tmp_path)
        assert converted["metadata"]["duration_sec"] == 3220
        assert isinstance(converted["metadata"]["duration_sec"], int)
        assert converted["chunks"][0]["metadata"]["duration_sec"] == 3220

    def test_unparseable_duration_is_dropped_rather_than_served_as_text(self, tmp_path):
        # Nothing stops a hand-edited page from carrying `duration_sec: soon`.
        # Same rule as the collection routes' score coercion: a field that is not
        # a number is absent, so "key missing" is the single no-value signal.
        from main.sources.files.files_document_converter import FilesDocumentConverter
        text = (
            '---\ndate: "2026-09-01"\nurl: "https://vimeo.com/1"\n'
            'duration_sec: soon\ncategory: "ai/general"\n---\n\nBody.\n'
        )
        path = tmp_path / "page.md"
        path.write_text(text, encoding="utf-8")
        converted = FilesDocumentConverter().convert({
            "fileRelativePath": "ai/general/page.md",
            "fileFullPath": str(path),
            "modifiedTime": "2026-09-01T00:00:00Z",
            "content": [{"text": text}],
        })[0]
        assert "duration_sec" not in converted["metadata"]


class TestWriteSummaryBodySuffix:
    """``body_suffix`` distinguishes "nothing to append" from "append nothing"."""

    def _write(self, tmp_path, title, **over):
        from main.ingest._summary_ingest import write_summary
        kwargs = {
            "root": str(tmp_path),
            "title": title,
            "url": f"https://example.com/{title}",
            "summary": "Body text.",
            "category": "ai/general",
            "date": "2026-09-04",
        }
        kwargs.update(over)
        result = write_summary(**kwargs)
        return (tmp_path / result["file_path"]).read_text(encoding="utf-8")

    def test_absent_suffix_leaves_the_summary_as_the_whole_body(self, tmp_path):
        assert self._write(tmp_path, "absent", body_suffix=None).endswith("Body text.")

    def test_empty_suffix_is_appended_rather_than_skipped(self, tmp_path):
        # A falsy check conflates "" with None, so a caller that computes an
        # empty section gets the silent skip the docstring does not promise.
        assert self._write(tmp_path, "empty", body_suffix="").endswith("Body text.\n\n")

    def test_non_empty_suffix_is_separated_by_one_blank_line(self, tmp_path):
        written = self._write(tmp_path, "filled", body_suffix="## Transcript\n\nx\n")
        assert written.endswith("Body text.\n\n## Transcript\n\nx\n")

    def test_a_summary_ending_in_newlines_still_gets_exactly_one_blank_line(self, tmp_path):
        # Nothing stops a summarizer from ending its answer with a newline; the
        # separator is normalized rather than appended to whatever came back, or
        # the transcript heading drifts one line further down per trailing
        # newline and the section stops looking like a section.
        written = self._write(
            tmp_path, "trailing", summary="Body text.\n\n\n",
            body_suffix="## Transcript\n\nx\n",
        )
        assert written.endswith("Body text.\n\n## Transcript\n\nx\n")


class TestVimeoTranscriptCap:
    """An unbounded transcript is a whole-file write and an index rebuild.

    Muninn caps the VTT it harvests at 2 MB; the receiving end refuses the same
    size rather than trusting the sender to have done it.
    """

    def _client(self):
        from main.runtime.knowledge_store import KnowledgeStore, get_store
        app.dependency_overrides[get_store] = lambda: KnowledgeStore()
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    #: Spelled out here rather than imported so the size tests are a BEHAVIOUR
    #: check (a 422 the route did not answer before) and not a tautology against
    #: whatever the module happens to declare. The module's own constant is
    #: pinned to this number by `test_cap_is_muninns_vtt_cap` below.
    _CAP = 2 * 1024 * 1024

    def _body(self, transcript):
        return {
            "title": "Capped talk",
            "url": "https://vimeo.com/1223358361",
            "summary": "A talk.",
            "category": "ai/general",
            "transcript_markdown": transcript,
        }

    def test_cap_is_muninns_vtt_cap(self):
        from main.ingest.vimeo import VIMEO_TRANSCRIPT_MAX_BYTES
        assert VIMEO_TRANSCRIPT_MAX_BYTES == self._CAP

    #: The rendered frontmatter's bound, spelled out (see `_summary_ingest`):
    #: 75% of the 8192-character head `read_frontmatter_from_path` parses.
    _HEAD_CAP = 6144

    def _near_bound_body(self, tags):
        # A hashed url, every capped string at its cap, and `tags` distinct
        # tags: 250 renders under the bound, 400 over it (6773, measured).
        return {**self._body("t"), "url": "https://vimeo.com/1223358361?h=" + "a" * 2000,
                "tags": [f"tag{i}" for i in range(tags)],
                "speaker": "s" * 500, "author": "a" * 500,
                "thumbnail_url": "https://i.vimeocdn.com/" + "x" * 480}

    def test_the_head_cap_is_pinned(self):
        from main.ingest._summary_ingest import FRONTMATTER_MAX_CHARS
        assert FRONTMATTER_MAX_CHARS == self._HEAD_CAP

    def test_a_frontmatter_that_would_overrun_the_head_is_refused_413_whatever_carries_it(self, tmp_path):
        # The OUTPUT is bounded, so the door does not depend on which field is
        # the long one: url, a numeric field, or many small tags.
        _set_ingest("vimeo", str(tmp_path), None)
        client = self._client()
        long_url = "https://vimeo.com/1223358361?h=" + "a" * 20000
        assert client.post("/api/vimeo/ingest", json={**self._body("t"), "url": long_url}).status_code == 413
        # A bare numeric field has no string cap at all: 4000 digits beside a
        # url that alone is fine.
        raw = json.dumps({**self._body("t"), "url": "https://vimeo.com/1223358361?h=" + "a" * 2500, "duration_sec": 0})
        raw = raw.replace('"duration_sec": 0', '"duration_sec": ' + "9" * 4000)
        assert client.post("/api/vimeo/ingest", content=raw, headers={"content-type": "application/json"}).status_code == 413
        many = client.post("/api/vimeo/ingest", json={**self._body("t"), "tags": [f"t{i}" for i in range(3000)]})
        assert many.status_code == 413
        # And the case that sits BETWEEN the bound and the readers' 8192 head —
        # every field under its own cap, 6773 rendered characters (measured):
        # this is the case a raised bound would let through.
        mid = client.post("/api/vimeo/ingest", json=self._near_bound_body(tags=400))
        assert mid.status_code == 413, mid.text
        assert (tmp_path / "ai" / "general").exists() is False or not list((tmp_path / "ai" / "general").glob("*.md"))

    def test_whatever_the_writer_accepts_parses_back_and_never_forks(self, tmp_path):
        # Just under the bound, with the longest thing a real sender could carry
        # (a hashed url) and distinct tags: the head still parses, so a
        # re-ingest lands on the SAME file.
        from main.utils.frontmatter import read_frontmatter_from_path
        _set_ingest("vimeo", str(tmp_path), None)
        client = self._client()
        body = self._near_bound_body(tags=250)
        url = body["url"]
        first = client.post("/api/vimeo/ingest", json=body)
        assert first.status_code == 200, first.text
        second = client.post("/api/vimeo/ingest", json=body).json()
        assert second["file_path"] == first.json()["file_path"]
        head = read_frontmatter_from_path(str(tmp_path / first.json()["file_path"]))
        assert head.get("url") == url

    def test_the_http_response_carries_author(self, tmp_path):
        _set_ingest("vimeo", str(tmp_path), None)
        body = self._client().post("/api/vimeo/ingest", json={**self._body("t"), "author": "JavaZone"}).json()
        assert body["author"] == "JavaZone"
        # And null, not absent, when the request carried none — the key is in
        # the response contract either way.
        without = self._client().post(
            "/api/vimeo/ingest", json={**self._body("t"), "url": "https://vimeo.com/1223358362"}
        ).json()
        assert "author" in without and without["author"] is None

    def test_oversize_transcript_is_refused_with_422(self, tmp_path):
        _set_ingest("vimeo", str(tmp_path), None)
        resp = self._client().post(
            "/api/vimeo/ingest", json=self._body("a" * (self._CAP + 1))
        )
        assert resp.status_code == 422
        assert not list(tmp_path.rglob("*.md"))

    def test_cap_counts_bytes_not_characters(self, tmp_path):
        # Half a cap's worth of two-byte characters is over the byte cap while
        # comfortably under it by len(), which is what a naive check measures.
        _set_ingest("vimeo", str(tmp_path), None)
        transcript = "é" * (self._CAP // 2 + 1)
        assert len(transcript) < self._CAP
        resp = self._client().post("/api/vimeo/ingest", json=self._body(transcript))
        assert resp.status_code == 422
        assert not list(tmp_path.rglob("*.md"))

    def test_a_transcript_at_the_cap_is_accepted(self, tmp_path):
        _set_ingest("vimeo", str(tmp_path), None)
        resp = self._client().post(
            "/api/vimeo/ingest", json=self._body("a" * self._CAP)
        )
        assert resp.status_code == 200
        assert len(list(tmp_path.rglob("*.md"))) == 1


# --- Ingest registry: parametrized suite over the summary push sources ---------

def _summary_sources():
    """Registry sources that write via the shared write_summary helper (all but Jira)."""
    from main.ingest.registry import INGEST_SOURCES
    return [s for s in INGEST_SOURCES if s.name != "jira"]


_SUMMARY_TEXT = "Parametrized summary body for the shared write path."


def _summary_req(src, **over):
    """Build a minimal valid request for any registry source.

    A `summary` is always supplied so the YouTube source skips its transcript
    fetch + Claude call. Keys the model does not declare are dropped, so
    `author` / `issueKey` only reach the models that have them.
    """
    fields = src.request_model.model_fields
    base = {
        "title": "Registry parametrized title",
        "url": "https://example.com/registry-item",
        "summary": _SUMMARY_TEXT,
        "category": "ai/claude-code",
        "date": "2026-07-04",
        "issueKey": _CONTRACT_ISSUE_KEY,
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

    def test_empty_auto_category_fails_loudly(self, tmp_path, monkeypatch):
        """A malformed Claude response (empty category) must 400, not silently
        fall through to write_summary's ai/general default."""
        import main.ingest.youtube as yt
        from fastapi import HTTPException
        monkeypatch.setattr(yt, "_call_claude_headless", lambda prompt: "whatever")
        monkeypatch.setattr(yt, "_parse_claude_response", lambda resp: ("", "a summary"))
        with pytest.raises(HTTPException) as exc:
            yt.ingest_youtube(
                self._req(summary=None, category=None, transcript="some transcript"),
                transcripts_path=str(tmp_path),
            )
        assert exc.value.status_code == 400
        assert "Invalid category ''" in exc.value.detail


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


class TestReaderPatterns:
    """`_reader_patterns` mirrors the update factory's effective localFiles defaults."""

    def test_localfiles_explicit_patterns_passthrough(self):
        from main.routes.collections import _reader_patterns
        manifest = {"reader": {"type": "localFiles",
                               "includePatterns": ["^life/.*"],
                               "excludePatterns": ["^index\\.md$"]}}
        assert _reader_patterns(manifest) == (["^life/.*"], ["^index\\.md$"])

    def test_localfiles_omitted_include_defaults_to_index_all(self):
        # Mirrors _build_local_files: include defaults to [".*"], exclude to [].
        from main.routes.collections import _reader_patterns
        assert _reader_patterns({"reader": {"type": "localFiles"}}) == ([".*"], [])

    def test_non_localfiles_reader_has_no_file_patterns(self):
        from main.routes.collections import _reader_patterns
        assert _reader_patterns({"reader": {"type": "jira"}}) == ([], [])

    def test_missing_reader_block(self):
        from main.routes.collections import _reader_patterns
        assert _reader_patterns({}) == ([], [])


class TestListCollectionsReaderPatterns:
    """GET /api/collections exposes each collection's reader include/exclude rules."""

    class _FakeIndexer:
        def get_size(self):
            return 42

    class _FakeSearcher:
        def __init__(self):
            self.indexer = TestListCollectionsReaderPatterns._FakeIndexer()

    class _FakePersister:
        def __init__(self, manifests):
            self._manifests = manifests

        def read_text_file(self, path):
            name = path.split("/")[0]
            if name not in self._manifests:
                raise FileNotFoundError(path)
            return json.dumps(self._manifests[name])

    class _FakeStore:
        def __init__(self, manifests):
            self._searchers = {n: TestListCollectionsReaderPatterns._FakeSearcher() for n in manifests}
            self.disk_persister = TestListCollectionsReaderPatterns._FakePersister(manifests)

        def get_searchers(self):
            return self._searchers

    def _client(self, store) -> TestClient:
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def teardown_method(self):
        from main.runtime.knowledge_store import get_store
        app.dependency_overrides.pop(get_store, None)

    def test_wiki_collection_reports_meta_denylist(self):
        store = self._FakeStore({
            "wiki": {
                "numberOfDocuments": 729,
                "reader": {
                    "type": "localFiles",
                    "includePatterns": [".*"],
                    "excludePatterns": ["^index\\.md$", "^log\\.md$", "^plans/.*"],
                },
            },
        })
        body = self._client(store).get("/api/collections").json()
        entry = next(c for c in body["collections"] if c["name"] == "wiki")
        assert entry["includePatterns"] == [".*"]
        assert entry["excludePatterns"] == ["^index\\.md$", "^log\\.md$", "^plans/.*"]
        assert entry["document_count"] == 729

    def test_localfiles_without_patterns_defaults_index_all(self):
        store = self._FakeStore({"notes": {"reader": {"type": "localFiles"}}})
        body = self._client(store).get("/api/collections").json()
        entry = next(c for c in body["collections"] if c["name"] == "notes")
        assert entry["includePatterns"] == [".*"]
        assert entry["excludePatterns"] == []
