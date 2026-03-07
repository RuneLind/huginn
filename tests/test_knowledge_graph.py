"""Tests for KnowledgeGraph entity detection, expansion, and query answering."""
import json
import pytest
from pathlib import Path
from main.graph.knowledge_graph import KnowledgeGraph


@pytest.fixture
def sample_graph(tmp_path):
    """Create a minimal test graph."""
    graph_data = {
        "nodes": [
            {"id": "buc:LA_BUC_01", "type": "BUC", "label": "LA_BUC_01 Søknad om unntak", "properties": {}},
            {"id": "buc:LA_BUC_02", "type": "BUC", "label": "LA_BUC_02 Beslutning om lovvalg", "properties": {}},
            {"id": "sed:A001", "type": "SED", "label": "A001", "properties": {"title": "Søknad/anmodning om unntak"}},
            {"id": "sed:A002", "type": "SED", "label": "A002", "properties": {"title": "Svar på anmodning om unntak"}},
            {"id": "sed:A003", "type": "SED", "label": "A003", "properties": {"title": "Beslutning om lovvalg"}},
            {"id": "sed:X001", "type": "SED", "label": "X001", "properties": {"title": "Forespørsel om informasjon", "type": "administrativ"}},
            {"id": "artikkel:12", "type": "Artikkel", "label": "Artikkel 12", "properties": {"forordning": "883/2004"}},
            {"id": "artikkel:13", "type": "Artikkel", "label": "Artikkel 13", "properties": {"forordning": "883/2004"}},
            {"id": "artikkel:13.1", "type": "Artikkel", "label": "Artikkel 13 nr. 1", "properties": {"forordning": "883/2004"}},
            {"id": "artikkel:16", "type": "Artikkel", "label": "Artikkel 16", "properties": {"forordning": "883/2004"}},
            {"id": "forordning:883/2004", "type": "Forordning", "label": "Forordning 883/2004", "properties": {}},
        ],
        "edges": [
            {"source": "buc:LA_BUC_01", "target": "sed:A001", "type": "inneholder_sed", "properties": {}},
            {"source": "buc:LA_BUC_01", "target": "sed:A002", "type": "inneholder_sed", "properties": {}},
            {"source": "buc:LA_BUC_01", "target": "artikkel:16", "type": "hjemlet_i", "properties": {}},
            {"source": "buc:LA_BUC_02", "target": "sed:A003", "type": "inneholder_sed", "properties": {}},
            {"source": "buc:LA_BUC_02", "target": "artikkel:13", "type": "hjemlet_i", "properties": {}},
            {"source": "artikkel:13.1", "target": "artikkel:13", "type": "del_av", "properties": {}},
            {"source": "artikkel:12", "target": "forordning:883/2004", "type": "del_av", "properties": {}},
        ],
    }
    graph_file = tmp_path / "test_graph.json"
    graph_file.write_text(json.dumps(graph_data))
    return KnowledgeGraph(graph_file)


class TestEntityDetection:
    def test_detect_buc_underscore(self, sample_graph):
        assert "buc:LA_BUC_01" in sample_graph.detect_entities("Hva er LA_BUC_01?")

    def test_detect_buc_space(self, sample_graph):
        assert "buc:LA_BUC_02" in sample_graph.detect_entities("LA BUC 02 prosessen")

    def test_detect_buc_compact(self, sample_graph):
        assert "buc:LA_BUC_01" in sample_graph.detect_entities("LABUC01 detaljer")

    def test_detect_sed_a(self, sample_graph):
        assert "sed:A003" in sample_graph.detect_entities("Hva inneholder A003?")

    def test_detect_sed_x(self, sample_graph):
        assert "sed:X001" in sample_graph.detect_entities("X001 brukes til forespørsler")

    def test_detect_artikkel(self, sample_graph):
        assert "artikkel:13" in sample_graph.detect_entities("artikkel 13 gjelder arbeid i flere land")

    def test_detect_artikkel_abbreviated(self, sample_graph):
        assert "artikkel:12" in sample_graph.detect_entities("art. 12 utsendte arbeidstakere")

    def test_detect_artikkel_with_sub(self, sample_graph):
        entities = sample_graph.detect_entities("art 13 nr. 1 regler")
        assert "artikkel:13.1" in entities
        assert "artikkel:13" in entities

    def test_detect_forordning(self, sample_graph):
        assert "forordning:883/2004" in sample_graph.detect_entities("forordning 883/2004 artikkel 12")

    def test_detect_multiple(self, sample_graph):
        entities = sample_graph.detect_entities("LA_BUC_01 inneholder A001 og A002")
        assert "buc:LA_BUC_01" in entities
        assert "sed:A001" in entities
        assert "sed:A002" in entities

    def test_detect_sed_case_insensitive(self, sample_graph):
        assert "sed:A003" in sample_graph.detect_entities("hva er a003?")

    def test_detect_x_sed_case_insensitive(self, sample_graph):
        assert "sed:X001" in sample_graph.detect_entities("x001 info")

    def test_detect_unknown_entity(self, sample_graph):
        assert sample_graph.detect_entities("A999 finnes ikke") == []

    def test_detect_deduplicated(self, sample_graph):
        entities = sample_graph.detect_entities("A001 og A001 igjen")
        assert entities.count("sed:A001") == 1

    def test_detect_empty(self, sample_graph):
        assert sample_graph.detect_entities("ingen entiteter her") == []


class TestQueryExpansion:
    def test_expansion_includes_node_label(self, sample_graph):
        terms = sample_graph.get_expansion_terms(["buc:LA_BUC_01"])
        assert "LA_BUC_01 Søknad om unntak" in terms

    def test_expansion_includes_sed_neighbors(self, sample_graph):
        terms = sample_graph.get_expansion_terms(["buc:LA_BUC_01"])
        assert any("A001" in t for t in terms)

    def test_expansion_includes_artikkel_neighbors(self, sample_graph):
        terms = sample_graph.get_expansion_terms(["buc:LA_BUC_01"])
        assert any("Artikkel 16" in t for t in terms)

    def test_expansion_empty_for_unknown(self, sample_graph):
        assert sample_graph.get_expansion_terms(["buc:LA_BUC_99"]) == []

    def test_expansion_sed_includes_parent_buc(self, sample_graph):
        terms = sample_graph.get_expansion_terms(["sed:A001"])
        assert any("LA_BUC_01" in t for t in terms)


class TestEntityContext:
    def test_context_for_buc(self, sample_graph):
        ctx = sample_graph.get_entity_context("buc:LA_BUC_01")
        assert "LA_BUC_01" in ctx
        assert "A001" in ctx
        assert "Artikkel 16" in ctx

    def test_context_for_sed(self, sample_graph):
        ctx = sample_graph.get_entity_context("sed:A001")
        assert "Søknad/anmodning om unntak" in ctx
        assert "LA_BUC_01" in ctx

    def test_context_for_artikkel(self, sample_graph):
        ctx = sample_graph.get_entity_context("artikkel:13")
        assert "Artikkel 13" in ctx
        assert "LA_BUC_02" in ctx

    def test_context_for_unknown(self, sample_graph):
        assert sample_graph.get_entity_context("buc:LA_BUC_99") is None


class TestGraphQueryAnswering:
    def test_answer_seds_in_buc(self, sample_graph):
        answer = sample_graph.answer_graph_query(
            ["buc:LA_BUC_01"], "Hvilke SEDer inneholder LA_BUC_01?"
        )
        assert answer is not None
        assert "A001" in answer
        assert "A002" in answer

    def test_answer_bucs_for_sed(self, sample_graph):
        answer = sample_graph.answer_graph_query(
            ["sed:A003"], "Which BUCs contain A003?"
        )
        assert answer is not None
        assert "LA_BUC_02" in answer

    def test_answer_artikkel_for_buc(self, sample_graph):
        answer = sample_graph.answer_graph_query(
            ["buc:LA_BUC_01"], "Hva er hjemmelen for LA_BUC_01?"
        )
        assert answer is not None
        assert "Artikkel 16" in answer

    def test_no_answer_for_non_relational(self, sample_graph):
        answer = sample_graph.answer_graph_query(
            ["buc:LA_BUC_01"], "Forklar hva LA_BUC_01 betyr"
        )
        assert answer is None

    def test_no_answer_for_empty_entities(self, sample_graph):
        assert sample_graph.answer_graph_query([], "hvilke SEDer?") is None


@pytest.fixture
def jira_graph(tmp_path):
    """Create a minimal Jira test graph."""
    graph_data = {
        "nodes": [
            {"id": "epic:PROJECT-6079", "type": "Epic", "label": "PROJECT-6079: Required membership",
             "properties": {"summary": "Required membership for workers", "issue_count": 3}},
            {"id": "epic:PROJECT-5203", "type": "Epic", "label": "PROJECT-5203: Step selector",
             "properties": {"summary": "Step selector for decisions", "issue_count": 2}},
            {"id": "issue:PROJECT-6587", "type": "Issue", "label": "PROJECT-6587: Routines required",
             "properties": {"status": "Ferdig", "issue_type": "Historie"}},
            {"id": "issue:PROJECT-6588", "type": "Issue", "label": "PROJECT-6588: Frontend required",
             "properties": {"status": "Under arbeid", "issue_type": "Historie"}},
            {"id": "issue:PROJECT-6787", "type": "Issue", "label": "PROJECT-6787: Address bug",
             "properties": {"status": "Ferdig", "issue_type": "Feil"}},
            {"id": "issue:PROJECT-7770", "type": "Issue", "label": "PROJECT-7770: Create case",
             "properties": {"status": "Ferdig", "issue_type": "Deloppgave"}},
        ],
        "edges": [
            {"source": "issue:PROJECT-6587", "target": "epic:PROJECT-6079", "type": "tilhører_epic", "properties": {}},
            {"source": "issue:PROJECT-6588", "target": "epic:PROJECT-6079", "type": "tilhører_epic", "properties": {}},
            {"source": "issue:PROJECT-6787", "target": "epic:PROJECT-5203", "type": "tilhører_epic", "properties": {}},
            {"source": "issue:PROJECT-6587", "target": "issue:PROJECT-6588", "type": "refererer_til", "properties": {}},
            {"source": "issue:PROJECT-7770", "target": "issue:PROJECT-6787", "type": "refererer_til", "properties": {}},
        ],
    }
    graph_file = tmp_path / "jira_graph.json"
    graph_file.write_text(json.dumps(graph_data))
    return KnowledgeGraph(graph_file)


@pytest.fixture
def merged_graph(tmp_path, sample_graph):
    """Load both EESSI and Jira graphs into one instance."""
    eessi_data = {
        "nodes": [
            {"id": "buc:LA_BUC_01", "type": "BUC", "label": "LA_BUC_01 Unntak", "properties": {}},
            {"id": "sed:A001", "type": "SED", "label": "A001", "properties": {"title": "Søknad"}},
        ],
        "edges": [
            {"source": "buc:LA_BUC_01", "target": "sed:A001", "type": "inneholder_sed", "properties": {}},
        ],
    }
    jira_data = {
        "nodes": [
            {"id": "epic:PROJECT-100", "type": "Epic", "label": "PROJECT-100: Test epic",
             "properties": {"summary": "Test epic", "issue_count": 1}},
            {"id": "issue:PROJECT-101", "type": "Issue", "label": "PROJECT-101: Test issue",
             "properties": {"status": "Ferdig"}},
        ],
        "edges": [
            {"source": "issue:PROJECT-101", "target": "epic:PROJECT-100", "type": "tilhører_epic", "properties": {}},
        ],
    }
    eessi_file = tmp_path / "eessi.json"
    jira_file = tmp_path / "jira.json"
    eessi_file.write_text(json.dumps(eessi_data))
    jira_file.write_text(json.dumps(jira_data))
    return KnowledgeGraph([eessi_file, jira_file])


class TestJiraEntityDetection:
    def test_detect_issue_key(self, jira_graph):
        assert "issue:PROJECT-6587" in jira_graph.detect_entities("Se PROJECT-6587 for detaljer")

    def test_detect_epic_key(self, jira_graph):
        assert "epic:PROJECT-6079" in jira_graph.detect_entities("Epic PROJECT-6079 required membership")

    def test_detect_issue_prefers_issue_over_epic(self, jira_graph):
        # PROJECT-6587 exists as issue, not epic
        entities = jira_graph.detect_entities("PROJECT-6587")
        assert "issue:PROJECT-6587" in entities

    def test_detect_unknown_issue(self, jira_graph):
        assert jira_graph.detect_entities("PROJECT-9999") == []

    def test_detect_multiple_issues(self, jira_graph):
        entities = jira_graph.detect_entities("PROJECT-6587 refererer til PROJECT-6787")
        assert "issue:PROJECT-6587" in entities
        assert "issue:PROJECT-6787" in entities


class TestJiraQueryExpansion:
    def test_epic_expansion_includes_summary(self, jira_graph):
        terms = jira_graph.get_expansion_terms(["epic:PROJECT-6079"])
        assert any("Required membership" in t for t in terms)

    def test_issue_expansion_includes_epic(self, jira_graph):
        terms = jira_graph.get_expansion_terms(["issue:PROJECT-6587"])
        assert any("PROJECT-6079" in t for t in terms)

    def test_issue_expansion_includes_cross_ref(self, jira_graph):
        terms = jira_graph.get_expansion_terms(["issue:PROJECT-6587"])
        assert any("PROJECT-6588" in t for t in terms)


class TestJiraEntityContext:
    def test_context_for_epic(self, jira_graph):
        ctx = jira_graph.get_entity_context("epic:PROJECT-6079")
        assert "PROJECT-6079" in ctx
        assert "3 issues" in ctx

    def test_context_for_issue_with_epic(self, jira_graph):
        ctx = jira_graph.get_entity_context("issue:PROJECT-6587")
        assert "PROJECT-6587" in ctx
        assert "Epic:" in ctx

    def test_context_for_issue_without_epic(self, jira_graph):
        ctx = jira_graph.get_entity_context("issue:PROJECT-7770")
        assert "PROJECT-7770" in ctx


class TestJiraGraphQueryAnswering:
    def test_answer_issues_in_epic(self, jira_graph):
        answer = jira_graph.answer_graph_query(
            ["epic:PROJECT-6079"], "Hvilke issues tilhører PROJECT-6079?"
        )
        assert answer is not None
        assert "PROJECT-6587" in answer
        assert "PROJECT-6588" in answer

    def test_answer_epic_for_issue(self, jira_graph):
        answer = jira_graph.answer_graph_query(
            ["issue:PROJECT-6587"], "Hvilken epic tilhører PROJECT-6587?"
        )
        assert answer is not None
        assert "PROJECT-6079" in answer

    def test_no_answer_for_non_relational(self, jira_graph):
        answer = jira_graph.answer_graph_query(
            ["issue:PROJECT-6587"], "Forklar PROJECT-6587"
        )
        assert answer is None


class TestMergedGraph:
    def test_merged_has_both_node_types(self, merged_graph):
        assert "buc:LA_BUC_01" in merged_graph.nodes
        assert "issue:PROJECT-101" in merged_graph.nodes
        assert "epic:PROJECT-100" in merged_graph.nodes

    def test_merged_detects_both_entity_types(self, merged_graph):
        entities = merged_graph.detect_entities("LA_BUC_01 og PROJECT-101")
        assert "buc:LA_BUC_01" in entities
        assert "issue:PROJECT-101" in entities

    def test_merged_node_count(self, merged_graph):
        assert merged_graph.node_count() == 4  # 2 EESSI + 2 Jira


class TestGraphCounts:
    def test_node_count(self, sample_graph):
        assert sample_graph.node_count() == 11

    def test_edge_count(self, sample_graph):
        assert sample_graph.edge_count() == 7

    def test_node_detail(self, sample_graph):
        detail = sample_graph.get_node_detail("buc:LA_BUC_01")
        assert detail is not None
        assert detail["type"] == "BUC"
        assert len(detail["outgoing"]) == 3  # 2 SEDs + 1 artikkel
        assert len(detail["incoming"]) == 0

    def test_node_detail_unknown(self, sample_graph):
        assert sample_graph.get_node_detail("buc:LA_BUC_99") is None
