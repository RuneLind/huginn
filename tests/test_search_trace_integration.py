"""Integration tests for search-time trace recording.

Exercises the new tracing paths in:
- HybridSearchIndexer (return_breakdown)
- CrossEncoderReranker (return_ce_scores)
- KnowledgeGraph.detect_entities (with_spans)
- DocumentCollectionSearcher.search (trace= parameter)
"""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.search_trace import SearchTrace, create_trace
from main.indexes.indexers.hybrid_search_indexer import HybridSearchIndexer
from main.indexes.reranking.cross_encoder_reranker import CrossEncoderReranker


# --- Hybrid indexer: return_breakdown ---


class _FakeIndexer:
    def __init__(self, results):
        self._results = results

    def search(self, text, number_of_results=10):
        items = self._results[:number_of_results]
        if not items:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        ids = np.array([[r[0] for r in items]], dtype=np.int64)
        scores = np.array([[r[1] for r in items]], dtype=np.float32)
        return scores, ids

    def get_size(self):
        return len(self._results)


class TestHybridReturnBreakdown:
    def test_default_returns_two_tuple(self):
        hybrid = HybridSearchIndexer(_FakeIndexer([(1, 0.1)]), _FakeIndexer([(1, -1.0)]))
        out = hybrid.search("q", number_of_results=5)
        assert len(out) == 2  # backward compatible

    def test_breakdown_includes_all_three_stages(self):
        faiss = _FakeIndexer([(1, 0.1), (2, 0.2), (3, 0.3)])
        bm25 = _FakeIndexer([(2, 8.0), (4, 5.0), (1, 2.0)])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids, breakdown = hybrid.search("q", number_of_results=5, return_breakdown=True)
        assert "faiss" in breakdown and "bm25" in breakdown and "rrf" in breakdown
        # FAISS breakdown: 3 entries, ranks 0/1/2
        faiss_entries = breakdown["faiss"]
        assert [e[0] for e in faiss_entries] == [1, 2, 3]
        assert [e[1] for e in faiss_entries] == [0, 1, 2]
        # BM25 breakdown: 3 entries
        assert [e[0] for e in breakdown["bm25"]] == [2, 4, 1]
        # RRF entries match the returned order
        assert [e[0] for e in breakdown["rrf"]] == ids[0].tolist()

    def test_breakdown_skips_negative_ids(self):
        faiss = _FakeIndexer([(1, 0.1), (-1, 0.0)])
        bm25 = _FakeIndexer([(1, -1.0)])
        hybrid = HybridSearchIndexer(faiss, bm25)
        _, _, breakdown = hybrid.search("q", return_breakdown=True)
        assert all(e[0] != -1 for e in breakdown["faiss"])

    def test_empty_breakdown_shape(self):
        hybrid = HybridSearchIndexer(_FakeIndexer([]), _FakeIndexer([]))
        scores, ids, breakdown = hybrid.search("q", return_breakdown=True)
        assert breakdown == {"faiss": [], "bm25": [], "rrf": []}


# --- Cross-encoder: return_ce_scores ---


def _make_reranker(predict_scores):
    with patch.object(CrossEncoderReranker, "__init__", lambda self, **kwargs: None):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = "mock"
        reranker.model = MagicMock()
        reranker.model.predict.return_value = np.array(predict_scores, dtype=np.float32)
        return reranker


class TestRerankerReturnCEScores:
    def test_default_returns_two_tuple(self):
        reranker = _make_reranker([0.5, 0.1, 0.9])
        out = reranker.rerank(
            "q",
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[10, 20, 30]], dtype=np.int64),
            ["a", "b", "c"],
            top_k=3,
        )
        assert len(out) == 2

    def test_ce_breakdown_includes_all_candidates(self):
        reranker = _make_reranker([0.5, 0.1, 0.9, 0.3])
        scores, indexes, breakdown = reranker.rerank(
            "q",
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            np.array([[10, 20, 30, 40]], dtype=np.int64),
            ["a", "b", "c", "d"],
            top_k=2,
            return_ce_scores=True,
        )
        # Top-k cut to 2 chunks
        assert len(indexes[0]) == 2
        # But breakdown contains all 4 candidates, in CE-descending order
        assert len(breakdown) == 4
        assert breakdown[0] == (30, pytest.approx(0.9))
        assert breakdown[1] == (10, pytest.approx(0.5))
        assert breakdown[2] == (40, pytest.approx(0.3))
        assert breakdown[3] == (20, pytest.approx(0.1))

    def test_empty_returns_empty_breakdown(self):
        reranker = _make_reranker([])
        scores, indexes, breakdown = reranker.rerank(
            "q",
            np.array([[]], dtype=np.float32),
            np.array([[]], dtype=np.int64),
            [],
            top_k=5,
            return_ce_scores=True,
        )
        assert breakdown == []


# --- Knowledge graph: detect_entities with_spans ---


def _make_graph(nodes, edges=None):
    """Build a KnowledgeGraph from in-memory data."""
    from main.graph.knowledge_graph import KnowledgeGraph
    payload = {"nodes": nodes, "edges": edges or []}
    g = KnowledgeGraph.__new__(KnowledgeGraph)
    g.nodes = {}
    from collections import defaultdict
    g.outgoing = defaultdict(list)
    g.incoming = defaultdict(list)
    for node in payload["nodes"]:
        g.nodes[node["id"]] = node
    for edge in payload["edges"]:
        g.outgoing[edge["source"]].append(edge)
        g.incoming[edge["target"]].append(edge)
    g._entity_patterns = []
    for node_id, node in g.nodes.items():
        if node_id.startswith("entity:") and len(node["label"]) >= 3:
            g._entity_patterns.append((node["label"].lower(), node_id))
    return g


class TestDetectEntitiesWithSpans:
    def test_default_returns_ids_only(self):
        g = _make_graph([
            {"id": "buc:LA_BUC_02", "type": "BUC", "label": "LA_BUC_02", "properties": {}},
        ])
        out = g.detect_entities("hva er LA_BUC_02")
        assert out == ["buc:LA_BUC_02"]

    def test_with_spans_returns_pairs(self):
        g = _make_graph([
            {"id": "buc:LA_BUC_02", "type": "BUC", "label": "LA_BUC_02", "properties": {}},
            {"id": "sed:A003", "type": "SED", "label": "A003", "properties": {}},
        ])
        out = g.detect_entities("LA_BUC_02 inneholder A003", with_spans=True)
        ids = [pair[0] for pair in out]
        spans = dict(out)
        assert "buc:LA_BUC_02" in ids
        assert "sed:A003" in ids
        assert spans["buc:LA_BUC_02"] == "LA_BUC_02"
        assert spans["sed:A003"] == "A003"

    def test_normalized_buc_span_preserves_user_text(self):
        """User typed 'LA BUC 02' (spaces); span should reflect their text, not normalized id."""
        g = _make_graph([
            {"id": "buc:LA_BUC_02", "type": "BUC", "label": "LA_BUC_02", "properties": {}},
        ])
        out = g.detect_entities("hva er LA BUC 02 da", with_spans=True)
        assert out[0][0] == "buc:LA_BUC_02"
        assert out[0][1] == "LA BUC 02"


# --- End-to-end: searcher with trace ---


def _make_e2e_setup(reranker=None):
    mapping = {
        "0": {"documentId": "doc-A", "documentUrl": "http://x/a",
              "documentPath": "col/documents/doc-A.json", "chunkNumber": 0},
        "1": {"documentId": "doc-A", "documentUrl": "http://x/a",
              "documentPath": "col/documents/doc-A.json", "chunkNumber": 1},
        "2": {"documentId": "doc-B", "documentUrl": "http://x/b",
              "documentPath": "col/documents/doc-B.json", "chunkNumber": 0},
    }
    doc_a = {"id": "doc-A", "text": "A text", "chunks": [
        {"indexedData": "chunk A0"}, {"indexedData": "chunk A1"},
    ]}
    doc_b = {"id": "doc-B", "text": "B text", "chunks": [
        {"indexedData": "chunk B0"},
    ]}

    persister = MagicMock()
    def read_text(path):
        if "index_document_mapping" in path:
            return json.dumps(mapping)
        if "doc-A" in path:
            return json.dumps(doc_a)
        if "doc-B" in path:
            return json.dumps(doc_b)
        raise FileNotFoundError(path)
    persister.read_text_file.side_effect = read_text

    indexer = MagicMock()
    indexer.get_name.return_value = "test_indexer"
    # No rrf_k attribute → searcher won't try to capture index breakdown
    indexer.search.return_value = (
        np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int64),
    )

    return DocumentCollectionSearcher(
        collection_name="col",
        indexer=indexer,
        persister=persister,
        reranker=reranker,
    )


class TestSearcherTraceWiring:
    def test_no_trace_param_unchanged_response(self):
        searcher = _make_e2e_setup()
        result = searcher.search("test", max_number_of_chunks=3)
        assert "results" in result
        assert result["reranked"] is False

    def test_trace_no_reranker_records_skip_reason(self):
        searcher = _make_e2e_setup()
        trace = create_trace(True)
        searcher.search("test query here", max_number_of_chunks=3, trace=trace)
        d = trace.to_dict()
        assert d["query"]["rerankerSkipped"] is True
        assert d["query"]["rerankerSkipReason"] == "no_reranker"

    def test_trace_caller_skip_reranker(self):
        reranker = MagicMock()
        searcher = _make_e2e_setup(reranker=reranker)
        trace = create_trace(True)
        searcher.search("xyz nonsense", max_number_of_chunks=3, trace=trace, skip_reranker=True)
        d = trace.to_dict()
        assert d["query"]["rerankerSkipReason"] == "caller_opted_out"
        # Reranker should NOT have been called
        reranker.rerank.assert_not_called()

    def test_trace_records_collection_with_final_stage(self):
        searcher = _make_e2e_setup()
        trace = create_trace(True)
        searcher.search("nordmenn rettigheter", max_number_of_chunks=3, trace=trace)
        d = trace.to_dict()
        assert len(d["collections"]) == 1
        coll = d["collections"][0]
        assert coll["name"] == "col"
        assert coll["indexer"] == "test_indexer"
        # All three chunks should be recorded with a "final" stage
        chunk_ids = {c["chunkId"] for c in coll["candidates"]}
        assert chunk_ids == {0, 1, 2}
        for c in coll["candidates"]:
            assert "final" in c["stages"]
        # Doc IDs should be annotated
        doc_ids = {c["documentId"] for c in coll["candidates"]}
        assert doc_ids == {"doc-A", "doc-B"}

    def test_trace_records_ce_stage_when_reranker_present(self):
        reranker = MagicMock()
        reranker.rerank.return_value = (
            np.array([[-0.9, -0.5, -0.1]], dtype=np.float32),
            np.array([[2, 0, 1]], dtype=np.int64),
            [(2, 0.9), (0, 0.5), (1, 0.1)],
        )
        searcher = _make_e2e_setup(reranker=reranker)
        trace = create_trace(True)
        searcher.search("nordmenn rettigheter", max_number_of_chunks=3, trace=trace)
        # Verify reranker was called with return_ce_scores=True
        call_kwargs = reranker.rerank.call_args.kwargs
        assert call_kwargs.get("return_ce_scores") is True

        coll = trace.to_dict()["collections"][0]
        ce_ranks = {c["chunkId"]: c["stages"]["ce"]["rank"]
                    for c in coll["candidates"] if "ce" in c["stages"]}
        assert ce_ranks == {2: 0, 0: 1, 1: 2}

    def test_trace_records_confidence_block(self):
        searcher = _make_e2e_setup()
        trace = create_trace(True)
        searcher.search("test", max_number_of_chunks=3, trace=trace)
        coll = trace.to_dict()["collections"][0]
        # No reranker → confidence filtering is skipped, but trace still records
        # the (default) state with low_confidence=False from response.get default.
        assert coll["confidence"] is not None
        assert "lowConfidenceThreshold" in coll["confidence"]

    def test_trace_records_timings(self):
        searcher = _make_e2e_setup()
        trace = create_trace(True)
        searcher.search("test", max_number_of_chunks=3, trace=trace)
        coll = trace.to_dict()["collections"][0]
        timings = coll["timingsMs"]
        for key in ("indexFetch", "chunkLoad", "rerank", "titleBoost", "assembly", "total"):
            assert key in timings
            assert isinstance(timings[key], int)

    def test_disabled_trace_does_not_call_breakdown_apis(self):
        """When trace is disabled, indexer.search should be called with old 2-arg form."""
        searcher = _make_e2e_setup()
        searcher.search("test", max_number_of_chunks=3)  # no trace
        # The MagicMock indexer's .search was called; verify return_breakdown wasn't passed
        call_kwargs = searcher.indexer.search.call_args.kwargs
        assert "return_breakdown" not in call_kwargs
