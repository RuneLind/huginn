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


class TestModelConfig:
    """Verify model configuration to prevent MPS memory explosion on Apple Silicon."""

    def test_cross_encoder_max_length_capped(self):
        """max_length=8192 causes O(n²) attention memory (~48GB per layer). Cap to 512."""
        from main.indexes.reranking.cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        assert reranker.model.max_length <= 512, (
            f"CrossEncoder max_length should be ≤512, got {reranker.model.max_length}"
        )

