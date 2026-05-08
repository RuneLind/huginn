"""Tests for main.core.mcp_search_tool.build_search_tool_fn.

The unit that ties GraphSearchAugmenter, DocumentCollectionSearcher, and
shape_search_results together for each MCP search tool — shared by both
multi_collection_search_mcp_adapter and collection_search_mcp_stdio_adapter.
"""
import json
import pytest

from main.core.mcp_search_tool import build_search_tool_fn
from main.graph.knowledge_graph import KnowledgeGraph
from main.graph.graph_search_augmenter import GraphSearchAugmenter


@pytest.fixture
def graph(tmp_path):
    data = {
        "nodes": [
            {"id": "entity:lovvalg", "type": "concept",
             "label": "lovvalg", "properties": {"definition": "valg av trygderegelverk"}},
            {"id": "entity:utsending", "type": "concept",
             "label": "utsending", "properties": {}},
        ],
        "edges": [
            {"source": "entity:lovvalg", "target": "entity:utsending",
             "type": "relates_to", "properties": {}},
        ],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data))
    return KnowledgeGraph(p)


class _FakeSearcher:
    """Captures the call args; returns a canned raw search response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.response


def _raw_response(*, reranked=True, low_confidence=False):
    response = {
        "results": [
            {
                "id": "doc-1",
                "url": "https://example.com/lovvalg",
                "modifiedTime": "2025-04-01",
                "path": "wiki/Lovvalg-and-utsending.json",
                "matchedChunks": [
                    {
                        "content": {
                            "indexedData": "Lovvalg is the choice of social-security regulation when a worker moves between EEA states.",
                            "heading": "Background",
                        },
                        "score": -0.45,
                    },
                    {
                        "content": {
                            "indexedData": "Utsending refers to a posted worker temporarily assigned abroad.",
                            "heading": "Definitions",
                        },
                        "score": -0.30,
                    },
                ],
            }
        ],
        "reranked": reranked,
    }
    if low_confidence:
        response["lowConfidence"] = True
    return response


class TestNoGraphConfigured:

    def test_augmenter_with_no_graph_passes_query_through_and_shapes_response(self):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
        )

        result = json.loads(fn("what is lovvalg"))

        assert searcher.calls[0]["query"] == "what is lovvalg"
        assert "graph_answer" not in result
        assert len(result["results"]) == 1
        first = result["results"][0]
        assert first["collection"] == "wiki"
        assert first["title"] == "Lovvalg-and-utsending"
        assert "relevance" in first and 0.0 <= first["relevance"] <= 1.0
        assert "_score" not in first
        assert "_reranked" not in first
        for chunk in first["matchedChunks"]:
            assert "score" not in chunk
            assert "relevance" in chunk


class TestWithGraph:

    def test_query_is_expanded_with_neighbor_terms_before_search(self, graph):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(graph)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
        )

        fn("hva betyr lovvalg")

        sent_query = searcher.calls[0]["query"]
        assert sent_query.startswith("hva betyr lovvalg")
        assert "utsending" in sent_query

    def test_title_boost_query_remains_the_user_query(self, graph):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(graph)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
        )

        fn("hva betyr lovvalg")

        assert searcher.calls[0]["title_boost_query"] == "hva betyr lovvalg"

    def test_graph_answer_and_context_attached_when_entities_detected(self, graph):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(graph)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
        )

        result = json.loads(fn("hva er lovvalg"))

        # Title contains "Lovvalg" and "utsending" — both graph entities.
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY in result["results"][0]


class TestLowConfidence:

    def test_low_confidence_propagates(self):
        searcher = _FakeSearcher(_raw_response(low_confidence=True))
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
        )

        result = json.loads(fn("anything"))

        assert result.get("lowConfidence") is True


class TestTrace:

    def test_trace_attached_when_trace_default_true(self):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
            trace_default=True,
        )

        result = json.loads(fn("anything"))

        assert "trace" in result
        assert result["trace"]["query"]["raw"] == "anything"

    def test_no_trace_when_trace_default_false(self):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=False,
            trace_default=False,
        )

        result = json.loads(fn("anything"))

        assert "trace" not in result


class TestSearchArgs:

    def test_full_text_mode_passes_include_text_content(self):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=20,
            max_number_of_documents=10,
            include_full_text=True,
        )

        fn("anything")

        call = searcher.calls[0]
        assert call["include_text_content"] is True
        assert call["include_matched_chunks_content"] is False

    def test_chunk_and_document_caps_passed_through(self):
        searcher = _FakeSearcher(_raw_response())
        augmenter = GraphSearchAugmenter(None)
        fn = build_search_tool_fn(
            searcher, "wiki", augmenter,
            max_number_of_chunks=42,
            max_number_of_documents=7,
            include_full_text=False,
        )

        fn("anything")

        call = searcher.calls[0]
        assert call["max_number_of_chunks"] == 42
        assert call["max_number_of_documents"] == 7
