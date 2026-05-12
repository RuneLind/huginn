"""Tests for GraphSearchAugmenter — query augmentation and result enrichment."""
import json
import pytest

from main.core.search_trace import SearchTrace, NullSearchTrace
from main.graph.knowledge_graph import KnowledgeGraph
from main.graph.graph_search_augmenter import GraphSearchAugmenter, _broaden_query


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
        search_q, answer, entities = aug.augment_query("LA_BUC_01", NullSearchTrace())
        assert search_q == "LA_BUC_01"
        assert answer is None
        assert entities == []


class TestAugmentQueryNoEntitiesDetected:

    def test_no_entities_returns_original_query(self, graph):
        aug = GraphSearchAugmenter(graph)
        search_q, answer, entities = aug.augment_query("hello world", NullSearchTrace())
        assert search_q == "hello world"
        assert answer is None
        assert entities == []


class TestAugmentQueryWithEntities:

    def test_detects_entities_and_returns_them(self, graph):
        aug = GraphSearchAugmenter(graph)
        _, _, entities = aug.augment_query("Hva er LA_BUC_01?", NullSearchTrace())
        assert "buc:LA_BUC_01" in entities

    def test_expands_query_with_neighbor_terms(self, graph):
        aug = GraphSearchAugmenter(graph)
        search_q, _, _ = aug.augment_query("LA_BUC_01", NullSearchTrace())
        assert search_q.startswith("LA_BUC_01 ")
        assert len(search_q) > len("LA_BUC_01")

    def test_returns_graph_answer_for_relational_question(self, graph):
        aug = GraphSearchAugmenter(graph)
        _, answer, _ = aug.augment_query("Hvilke SEDer inneholder LA_BUC_01?", NullSearchTrace())
        assert answer is not None
        assert "A001" in answer

    def test_no_graph_answer_for_non_question(self, graph):
        aug = GraphSearchAugmenter(graph)
        _, answer, _ = aug.augment_query("LA_BUC_01", NullSearchTrace())
        assert answer is None

    def test_expansion_capped_at_term_limit(self, graph):
        aug = GraphSearchAugmenter(graph)
        many_terms = [f"term{i}" for i in range(20)]
        graph.get_expansion_terms = lambda ids: many_terms

        search_q, _, _ = aug.augment_query("LA_BUC_01", NullSearchTrace())
        appended = search_q.removeprefix("LA_BUC_01 ").split(" ")
        assert appended == many_terms[: GraphSearchAugmenter.EXPANSION_TERM_LIMIT]


class TestAugmentQueryTraceMarkers:

    def test_records_detected_entities(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace)
        assert any(e["id"] == "buc:LA_BUC_01"
                   for e in trace.to_dict()["query"]["detectedEntities"])

    def test_records_graph_answered_when_question_answered(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("Hvilke SEDer inneholder LA_BUC_01?", trace)
        assert trace.to_dict()["query"]["graphAnswered"] is True

    def test_records_graph_answered_false_for_non_question(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace)
        assert trace.to_dict()["query"]["graphAnswered"] is False

    def test_records_expansion_terms(self, graph):
        aug = GraphSearchAugmenter(graph)
        trace = SearchTrace()
        aug.augment_query("LA_BUC_01", trace)
        d = trace.to_dict()["query"]
        assert d["expansionTerms"]
        assert d["expanded"] and d["expanded"] != d["raw"]


class TestEnrichResults:

    def test_no_op_when_graph_is_none(self):
        aug = GraphSearchAugmenter(None)
        results = [{"title": "LA_BUC_01"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY not in results[0]

    def test_no_op_when_no_detected_entities(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{"title": "LA_BUC_01"}]
        aug.enrich_results(results, [])
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY not in results[0]

    def test_adds_graph_context_for_title_match(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{"title": "LA_BUC_01 oversikt"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY in results[0]
        assert results[0][GraphSearchAugmenter.GRAPH_CONTEXT_KEY]

    def test_skips_results_without_title_match(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{"title": "unrelated document"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY not in results[0]

    def test_caps_contexts_per_result(self, graph):
        aug = GraphSearchAugmenter(graph)
        # Force get_entity_context to return many distinct values; force detect_entities
        # to return many distinct entities so we can observe the [:3] slice.
        graph.detect_entities = lambda text, with_spans=False: [f"entity:{i}" for i in range(10)]
        graph.get_entity_context = lambda eid: f"context for {eid}"
        results = [{"title": "anything"}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        assert len(results[0][GraphSearchAugmenter.GRAPH_CONTEXT_KEY]) == GraphSearchAugmenter.CONTEXT_PER_RESULT_LIMIT

    def test_handles_missing_title(self, graph):
        aug = GraphSearchAugmenter(graph)
        results = [{}]
        aug.enrich_results(results, ["buc:LA_BUC_01"])
        # Empty title produces no entities — no graph_context, no crash.
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY not in results[0]


class TestGraphContextKey:
    def test_value_is_pinned(self):
        """Renaming GRAPH_CONTEXT_KEY's value would silently break external
        clients (Muninn, bots) that read the raw JSON response by key."""
        assert GraphSearchAugmenter.GRAPH_CONTEXT_KEY == "graph_context"


class TestBroadenQuery:
    def test_keeps_first_conjunct(self):
        assert _broaden_query("lovvalg and utsending") == "lovvalg"
        assert _broaden_query("trygd og pensjon her") == "trygd"
        assert _broaden_query("FAISS versus BM25 tradeoffs") == "FAISS"

    def test_strips_trailing_parenthetical(self):
        assert _broaden_query("trygdeavgift beregning (for selvstendig)") == "trygdeavgift beregning"

    def test_unquotes(self):
        assert _broaden_query('"exact phrase here"') == "exact phrase here"

    def test_drops_last_word_when_query_long_enough(self):
        assert _broaden_query("alpha beta gamma delta epsilon") == "alpha beta gamma delta"

    def test_returns_none_for_short_query(self):
        assert _broaden_query("two words") is None
        assert _broaden_query("one") is None
        assert _broaden_query("") is None


class TestGetRetryHints:
    def test_no_graph_offers_broader_heuristic_only(self):
        aug = GraphSearchAugmenter(None)
        assert aug.get_retry_hints("alpha beta gamma delta", []) == {"broaderQuery": "alpha beta gamma"}

    def test_no_graph_short_query_returns_none(self):
        aug = GraphSearchAugmenter(None)
        assert aug.get_retry_hints("short query", []) is None

    def test_with_graph_offers_entities_related_terms_and_narrower(self, graph):
        aug = GraphSearchAugmenter(graph)
        detected = graph.detect_entities("LA_BUC_02 details")
        hints = aug.get_retry_hints("LA_BUC_02 details", detected)
        assert "LA_BUC_02" in hints["detectedEntities"][0]
        assert any("A003" in t for t in hints["relatedTerms"])
        assert hints["narrowerQuery"].startswith("LA_BUC_02 details ")

    def test_excludes_terms_already_present_in_query(self, graph):
        aug = GraphSearchAugmenter(graph)
        detected = graph.detect_entities("artikkel 13")
        hints = aug.get_retry_hints("artikkel 13", detected) or {}
        assert all(t.lower() != "artikkel 13" for t in hints.get("relatedTerms", []))

    def test_returns_none_when_nothing_useful(self):
        aug = GraphSearchAugmenter(None)
        assert aug.get_retry_hints("ab cd", []) is None
