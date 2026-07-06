"""Direct unit tests for main.core.search_pipeline.

``search_and_shape`` and ``run_search_request`` are the two shared stages the
HTTP route (``routes/search.py``) and the in-process MCP tool
(``core/mcp_search_tool.py``) both run. They were previously only exercised
transitively through those two call sites; these tests pin them down directly.
"""
import pytest

from main.core.search_pipeline import run_search_request, search_and_shape
from main.core.search_trace import create_trace
from main.graph.graph_search_augmenter import GraphSearchAugmenter


class _FakeSearcher:
    """Records call args; returns a canned raw search response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.response


def _raw_response(*, doc_id="doc-1", url="https://example.com/a", reranked=True, low_confidence=False):
    response = {
        "results": [
            {
                "id": doc_id,
                "url": url,
                "path": f"wiki/{doc_id}.json",
                "matchedChunks": [
                    {"content": {"indexedData": "Some relevant content here.", "heading": "H"},
                     "score": -0.45},
                ],
            }
        ],
        "reranked": reranked,
    }
    if low_confidence:
        response["lowConfidence"] = True
    return response


def _search_kwargs():
    return dict(
        max_number_of_chunks=20,
        max_number_of_documents=10,
        include_matched_chunks_content=True,
    )


class _RecordingAugmenter(GraphSearchAugmenter):
    """No-graph augmenter that records whether enrich_results ran."""

    def __init__(self):
        super().__init__(None)
        self.enrich_calls = []

    def enrich_results(self, results, detected_entities):
        self.enrich_calls.append((results, detected_entities))
        return super().enrich_results(results, detected_entities)


class TestSearchAndShape:

    def test_searches_every_collection_and_shapes(self):
        s1 = _FakeSearcher(_raw_response(doc_id="a", url="https://example.com/a"))
        s2 = _FakeSearcher(_raw_response(doc_id="b", url="https://example.com/b"))
        augmenter = _RecordingAugmenter()

        results, per_collection, any_low = search_and_shape(
            {"c1": s1, "c2": s2},
            "the query",
            augmenter=augmenter,
            detected_entities=[],
            trace=create_trace(False),
            title_boost_query="the query",
            search_kwargs=_search_kwargs(),
            shape_kwargs=dict(limit=10),
        )

        assert s1.calls[0]["query"] == "the query"
        assert s2.calls[0]["query"] == "the query"
        assert s1.calls[0]["title_boost_query"] == "the query"
        assert [name for name, _ in per_collection] == ["c1", "c2"]
        assert len(results) == 2
        assert any_low is False
        # enrichment ran on the shaped results
        assert augmenter.enrich_calls and augmenter.enrich_calls[0][0] is results

    def test_low_confidence_propagates(self):
        s1 = _FakeSearcher(_raw_response(low_confidence=True))
        results, _, any_low = search_and_shape(
            {"c1": s1},
            "q",
            augmenter=GraphSearchAugmenter(None),
            detected_entities=[],
            trace=create_trace(False),
            title_boost_query="q",
            search_kwargs=_search_kwargs(),
            shape_kwargs=dict(limit=10),
        )
        assert any_low is True

    def test_forwards_search_kwargs(self):
        s1 = _FakeSearcher(_raw_response())
        search_and_shape(
            {"c1": s1},
            "q",
            augmenter=GraphSearchAugmenter(None),
            detected_entities=[],
            trace=create_trace(False),
            title_boost_query="q",
            search_kwargs=dict(max_number_of_chunks=42, max_number_of_documents=7,
                               include_matched_chunks_content=True),
            shape_kwargs=dict(limit=10),
        )
        assert s1.calls[0]["max_number_of_chunks"] == 42
        assert s1.calls[0]["max_number_of_documents"] == 7


def _run(target_searchers, *, raw_query="q", search_query=None, augmenter=None,
         graph_answer=None, min_relevance=None, corrective_mode="off", limit=10):
    return run_search_request(
        target_searchers,
        raw_query=raw_query,
        search_query=search_query if search_query is not None else raw_query,
        augmenter=augmenter or GraphSearchAugmenter(None),
        detected_entities=[],
        graph_answer=graph_answer,
        trace=create_trace(False),
        search_kwargs=_search_kwargs(),
        shape_kwargs=dict(limit=limit),
        min_relevance=min_relevance,
        corrective_mode=corrective_mode,
        limit=limit,
    )


class TestRunSearchRequest:

    def test_returns_shaped_response_without_internal_fields(self):
        response = _run({"c1": _FakeSearcher(_raw_response())})
        assert len(response["results"]) == 1
        first = response["results"][0]
        assert first["collection"] == "c1"
        assert "relevance" in first and 0.0 <= first["relevance"] <= 1.0
        assert "_score" not in first
        assert "_reranked" not in first
        for chunk in first["matchedChunks"]:
            assert "score" not in chunk

    def test_uses_search_query_for_search_and_raw_query_for_title_boost(self):
        searcher = _FakeSearcher(_raw_response())
        _run({"c1": searcher}, raw_query="raw", search_query="expanded query")
        assert searcher.calls[0]["query"] == "expanded query"
        assert searcher.calls[0]["title_boost_query"] == "raw"

    def test_graph_answer_merged_only_when_truthy(self):
        with_answer = _run({"c1": _FakeSearcher(_raw_response())}, graph_answer="the answer")
        assert with_answer["graph_answer"] == "the answer"
        without = _run({"c1": _FakeSearcher(_raw_response())}, graph_answer=None)
        assert "graph_answer" not in without

    def test_low_confidence_merged(self):
        response = _run({"c1": _FakeSearcher(_raw_response(low_confidence=True))})
        assert response["lowConfidence"] is True
        clean = _run({"c1": _FakeSearcher(_raw_response(low_confidence=False))})
        assert "lowConfidence" not in clean

    def test_reranked_true_when_all_collections_reranked(self):
        response = _run({
            "c1": _FakeSearcher(_raw_response(doc_id="a", url="https://example.com/a", reranked=True)),
            "c2": _FakeSearcher(_raw_response(doc_id="b", url="https://example.com/b", reranked=True)),
        })
        assert response["reranked"] is True

    def test_reranked_false_when_any_collection_not_reranked(self):
        # Mirrors the HTTP route's ``all(...)`` honesty rule: one non-reranked
        # collection makes the whole response non-reranked.
        response = _run({
            "c1": _FakeSearcher(_raw_response(doc_id="a", url="https://example.com/a", reranked=True)),
            "c2": _FakeSearcher(_raw_response(doc_id="b", url="https://example.com/b", reranked=False)),
        })
        assert response["reranked"] is False

    def test_min_relevance_empties_results_and_signals(self):
        response = _run({"c1": _FakeSearcher(_raw_response())}, min_relevance=0.99)
        assert response["results"] == []
        assert response["noConfidentResults"] is True
