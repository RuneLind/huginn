"""Tests for GraphSearchAugmenter — query augmentation and result enrichment."""
import json
import pytest

from main.core.search_trace import SearchTrace, NullSearchTrace
from main.graph.knowledge_graph import KnowledgeGraph
from main.graph.graph_search_augmenter import GraphSearchAugmenter


@pytest.fixture
def graph(tmp_path):
    data = {
        "nodes": [
            {"id": "buc:LA_BUC_01", "type": "BUC",
             "label": "LA_BUC_01 Søknad om unntak", "properties": {}},
            {"id": "buc:LA_BUC_02", "type": "BUC",
             "label": "LA_BUC_02 Beslutning om lovvalg", "properties": {}},
            {"id": "sed:A001", "type": "SED",
             "label": "A001", "properties": {"title": "Søknad om unntak"}},
            {"id": "sed:A003", "type": "SED",
             "label": "A003", "properties": {"title": "Beslutning om lovvalg"}},
            {"id": "artikkel:13", "type": "Artikkel",
             "label": "Artikkel 13", "properties": {"forordning": "883/2004"}},
        ],
        "edges": [
            {"source": "buc:LA_BUC_01", "target": "sed:A001",
             "type": "inneholder_sed", "properties": {}},
            {"source": "buc:LA_BUC_02", "target": "sed:A003",
             "type": "inneholder_sed", "properties": {}},
            {"source": "buc:LA_BUC_02", "target": "artikkel:13",
             "type": "hjemlet_i", "properties": {}},
        ],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data))
    return KnowledgeGraph(p)


class TestAugmentQueryNoGraph:

    def test_returns_passthrough_when_graph_is_none(self):
        aug = GraphSearchAugmenter(None)
        trace = NullSearchTrace()
        search_q, answer, entities = aug.augment_query("LA_BUC_01", trace, trace_enabled=False)
        assert search_q == "LA_BUC_01"
        assert answer is None
        assert entities == []


class TestAugmentQueryNoEntitiesDetected:

    def test_no_entities_returns_original_query(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = NullSearchTrace()
        search_q, answer, entities = aug.augment_query("hello world", trace, trace_enabled=False)
        assert search_q == "hello world"
        assert answer is None
        assert entities == []


class TestAugmentQueryWithEntities:

    def test_detects_entities_and_returns_them(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = NullSearchTrace()
        _, _, entities = aug.augment_query("Hva er LA_BUC_01?", trace, trace_enabled=False)
        assert "buc:LA_BUC_01" in entities

    def test_expands_query_with_neighbor_terms(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = NullSearchTrace()
        search_q, _, _ = aug.augment_query("LA_BUC_01", trace, trace_enabled=False)
        # Expansion appends neighbor labels — query grows past the original.
        assert search_q.startswith("LA_BUC_01 ")
        assert len(search_q) > len("LA_BUC_01")

    def test_returns_graph_answer_for_relational_question(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = NullSearchTrace()
        _, answer, _ = aug.augment_query("Hvilke SEDer inneholder LA_BUC_01?", trace, trace_enabled=False)
        assert answer is not None
        assert "A001" in answer

    def test_no_graph_answer_for_non_question(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = NullSearchTrace()
        _, answer, _ = aug.augment_query("LA_BUC_01", trace, trace_enabled=False)
        assert answer is None

    def test_expansion_capped_at_term_limit(self, graph):
        """Expansion is capped at GraphSearchAugmenter.EXPANSION_TERM_LIMIT terms."""
        aug = GraphSearchAugmenter(graph)
        trace = NullSearchTrace()

        # Make get_expansion_terms return more than the cap so the slice is observable.
        many_terms = [f"term{i}" for i in range(20)]
        graph.get_expansion_terms = lambda ids: many_terms

        search_q, _, _ = aug.augment_query("LA_BUC_01", trace, trace_enabled=False)
        appended = search_q.removeprefix("LA_BUC_01 ").split(" ")
        assert appended == many_terms[: GraphSearchAugmenter.EXPANSION_TERM_LIMIT]


class TestAugmentQueryTraceMarkers:

    def test_records_detected_entities_when_trace_enabled(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace, trace_enabled=True)
        d = trace.to_dict()["query"]
        assert any(e["id"] == "buc:LA_BUC_01" for e in d["detectedEntities"])

    def test_does_not_record_entities_when_trace_disabled(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace, trace_enabled=False)
        assert trace.to_dict()["query"]["detectedEntities"] == []

    def test_records_graph_answered_when_question_answered(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("Hvilke SEDer inneholder LA_BUC_01?", trace, trace_enabled=True)
        assert trace.to_dict()["query"]["graphAnswered"] is True

    def test_records_graph_answered_false_for_non_question(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace, trace_enabled=True)
        # Entities matched but no relational question → answer is None → graphAnswered stays False.
        assert trace.to_dict()["query"]["graphAnswered"] is False

    def test_records_expansion_terms(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace, trace_enabled=True)
        d = trace.to_dict()["query"]
        assert d["expansionTerms"]
        assert d["expanded"] and d["expanded"] != d["raw"]


class TestEnrichResults:

    def test_no_op_when_graph_is_none(self):
        aug = GraphSearchAugmenter(None)
        results = [{"title": "LA_BUC_01"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert "graph_context" not in results[0]

    def test_no_op_when_no_detected_entities(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{"title": "LA_BUC_01"}]
        aug.enrich_results(results, [])
        assert "graph_context" not in results[0]

    def test_adds_graph_context_for_title_match(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{"title": "LA_BUC_01 oversikt"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert "graph_context" in results[0]
        assert results[0]["graph_context"]

    def test_skips_results_without_title_match(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{"title": "unrelated document"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert "graph_context" not in results[0]

    def test_caps_contexts_per_result(self, graph):
        aug = GraphSearchAugmenter(graph)
        # Force get_entity_context to return many distinct values; force detect_entities
        # to return many distinct entities so we can observe the [:3] slice.
        graph.detect_entities = lambda text, with_spans=False: [f"entity:{i}" for i in range(10)]
        graph.get_entity_context = lambda eid: f"context for {eid}"
        results = [{"title": "anything"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert len(results[0]["graph_context"]) == GraphSearchAugmenter.CONTEXT_PER_RESULT_LIMIT

    def test_handles_missing_title(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        # Empty title produces no entities — no graph_context, no crash.
        assert "graph_context" not in results[0]
