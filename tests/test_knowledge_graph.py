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


@pytest.fixture
def llm_entity_graph(tmp_path):
    """Graph with LLM-extracted entity:* nodes whose labels are short, common
    substrings — the case that exposed the unanchored-match bug (H6)."""
    graph_data = {
        "nodes": [
            {"id": "entity:nav", "type": "Entity", "label": "NAV", "properties": {}},
            {"id": "entity:api", "type": "Entity", "label": "API", "properties": {}},
            {"id": "entity:sed", "type": "Entity", "label": "SED", "properties": {}},
            {"id": "entity:trygdeavgift", "type": "Entity", "label": "trygdeavgift", "properties": {}},
        ],
        "edges": [],
    }
    graph_file = tmp_path / "llm_graph.json"
    graph_file.write_text(json.dumps(graph_data))
    return KnowledgeGraph(graph_file)


class TestLlmEntityWordBoundary:
    """LLM-entity labels must match on word boundaries, not as bare substrings (H6)."""

    def test_matches_standalone_word(self, llm_entity_graph):
        assert "entity:nav" in llm_entity_graph.detect_entities("ansatt i NAV siden 2020")

    def test_case_insensitive(self, llm_entity_graph):
        assert "entity:api" in llm_entity_graph.detect_entities("kall mot api-et")

    def test_does_not_match_inside_larger_word(self, llm_entity_graph):
        # "nav" inside "navnet", "api" inside "rapid", "sed" inside "used" must NOT match.
        entities = llm_entity_graph.detect_entities("navnet på rapid prototyping ble used")
        assert "entity:nav" not in entities
        assert "entity:api" not in entities
        assert "entity:sed" not in entities

    def test_multi_token_label_still_matches(self, llm_entity_graph):
        assert "entity:trygdeavgift" in llm_entity_graph.detect_entities(
            "beregning av trygdeavgift for utsendte"
        )

    def test_recorded_span_preserves_original_case(self, llm_entity_graph):
        spans = llm_entity_graph.detect_entities("søk i NAV-systemet", with_spans=True)
        nav = [span for nid, span in spans if nid == "entity:nav"]
        assert nav == ["NAV"]


@pytest.fixture
def chain_graph(tmp_path):
    """Entity graph with a 2-hop chain, a hub, and sibling edges.

    ``entity:hub`` stands in for a node like ``entity:claude_code`` (hundreds of
    edges in the real graphs): everything links to it, so walking *through* it
    would pull unrelated facts into every context.
    """
    nodes = [
        {"id": "entity:refund_policy", "type": "Policy", "label": "Refund Policy", "properties": {}},
        {"id": "entity:ops_manager", "type": "Role", "label": "Ops Manager", "properties": {}},
        {"id": "entity:sarah", "type": "Person", "label": "Sarah", "properties": {"mention_count": 4}},
        {"id": "entity:marcus", "type": "Person", "label": "Marcus", "properties": {}},
        {"id": "entity:hub", "type": "Product", "label": "Hub", "properties": {}},
        {"id": "entity:payroll", "type": "Policy", "label": "Payroll", "properties": {}},
    ]
    edges = [
        {"source": "entity:refund_policy", "target": "entity:ops_manager",
         "type": "approved_by", "properties": {"mention_count": 5}},
        {"source": "entity:ops_manager", "target": "entity:sarah",
         "type": "held_by", "properties": {"mention_count": 4}},
        {"source": "entity:sarah", "target": "entity:marcus",
         "type": "delegates_to", "properties": {"mention_count": 3}},
        # A sibling of the chain: Payroll also points *into* ops_manager, so it is
        # two hops from the policy but says nothing about it.
        {"source": "entity:payroll", "target": "entity:ops_manager",
         "type": "approved_by", "properties": {"mention_count": 9}},
        # The hub is one hop from the policy and one hop from many unrelated nodes.
        {"source": "entity:refund_policy", "target": "entity:hub",
         "type": "related_to", "properties": {"mention_count": 1}},
    ]
    for i in range(60):
        nodes.append({"id": f"entity:noise{i}", "type": "Entity",
                      "label": f"Noise {i}", "properties": {}})
        edges.append({"source": "entity:hub", "target": f"entity:noise{i}",
                      "type": "related_to", "properties": {"mention_count": 2}})
    # A thin entity one hop from a well-connected (but not hub-sized) node: the
    # shape where the second hop carries more facts than the chain cap allows.
    nodes.append({"id": "entity:leaf", "type": "Entity", "label": "Leaf", "properties": {}})
    nodes.append({"id": "entity:mid", "type": "Entity", "label": "Mid", "properties": {}})
    edges.append({"source": "entity:leaf", "target": "entity:mid",
                  "type": "uses", "properties": {"mention_count": 3}})
    for i in range(6):
        nodes.append({"id": f"entity:mid{i}", "type": "Entity",
                      "label": f"Mid {i}", "properties": {}})
        edges.append({"source": "entity:mid", "target": f"entity:mid{i}",
                      "type": "uses", "properties": {"mention_count": 1}})
    graph_file = tmp_path / "chain_graph.json"
    graph_file.write_text(json.dumps({"nodes": nodes, "edges": edges}))
    return KnowledgeGraph(graph_file)


def _labels(triples):
    return {(t["source_label"], t["predicate"], t["target_label"]) for t in triples}


class TestWalkTriples:
    def test_reaches_second_hop(self, chain_graph):
        triples = chain_graph.walk_triples(["entity:refund_policy"])
        assert ("Ops Manager", "held_by", "Sarah") in _labels(triples)

    def test_triples_are_depth_ordered(self, chain_graph):
        depths = [t["depth"] for t in chain_graph.walk_triples(["entity:refund_policy"])]
        assert depths == sorted(depths)

    def test_triples_carry_node_ids(self, chain_graph):
        first = chain_graph.walk_triples(["entity:refund_policy"], limit=1)[0]
        assert first["source"] == "entity:refund_policy"
        assert first["target"] == "entity:ops_manager"

    def test_hops_bound_the_walk(self, chain_graph):
        labels = {t["target_label"] for t in chain_graph.walk_triples(["entity:refund_policy"])}
        assert "Marcus" not in labels  # 3 hops out, past the default
        deeper = {t["target_label"]
                  for t in chain_graph.walk_triples(["entity:refund_policy"], hops=3)}
        assert "Marcus" in deeper

    def test_second_hop_continues_direction(self, chain_graph):
        """A depth-2 fact must chain off the seed, not be a sibling of the neighbour."""
        triples = chain_graph.walk_triples(["entity:refund_policy"])
        # Payroll --approved_by--> Ops Manager is the strongest edge on ops_manager,
        # but it points the wrong way: it is a fact about Payroll, not the policy.
        assert ("Payroll", "approved_by", "Ops Manager") not in _labels(triples)
        assert ("Ops Manager", "held_by", "Sarah") in _labels(triples)

    def test_does_not_expand_through_a_hub(self, chain_graph):
        labels = {t["target_label"] for t in chain_graph.walk_triples(["entity:refund_policy"])}
        assert "Hub" in labels  # reached as a direct neighbour...
        assert not any(label.startswith("Noise") for label in labels)  # ...never walked through

    def test_hub_seed_is_still_expanded(self, chain_graph):
        # The rule is about walking *through* a hub, not about naming one.
        triples = chain_graph.walk_triples(["entity:hub"])
        assert triples and any(t["target_label"].startswith("Noise") for t in triples)

    def test_strongest_edges_first(self, chain_graph):
        # entity:hub's noise edges (mention_count 2) outrank the policy edge (1).
        first = chain_graph.walk_triples(["entity:hub"], hops=1, beam=1)
        assert first[0]["target_label"].startswith("Noise")

    def test_limit_caps_total_triples(self, chain_graph):
        assert len(chain_graph.walk_triples(["entity:hub"], limit=3)) == 3

    def test_zero_and_negative_bounds_yield_nothing(self, chain_graph):
        for kwargs in ({"limit": 0}, {"limit": -1}, {"beam": 0}, {"hops": 0}):
            assert chain_graph.walk_triples(["entity:hub"], **kwargs) == [], kwargs

    def test_min_depth_spends_the_budget_on_deeper_facts(self, chain_graph):
        triples = chain_graph.walk_triples(["entity:leaf"], limit=3, min_depth=2)
        assert len(triples) == 3
        assert all(t["depth"] == 2 for t in triples)

    def test_limit_spreads_across_neighbours(self, chain_graph):
        """Round-robin: a tight limit must not drain one neighbour's whole beam."""
        triples = chain_graph.walk_triples(["entity:hub"], hops=1, limit=3, beam=4)
        assert len({t["source"] for t in triples} | {t["target"] for t in triples}) > 2

    def test_edge_between_two_seeds_is_emitted(self, chain_graph):
        triples = chain_graph.walk_triples(["entity:refund_policy", "entity:ops_manager"])
        assert ("Refund Policy", "approved_by", "Ops Manager") in _labels(triples)

    def test_each_node_reached_once(self, chain_graph):
        triples = chain_graph.walk_triples(["entity:refund_policy"], hops=3)
        reached = [t["target"] for t in triples if t["depth"] > 1]
        assert len(reached) == len(set(reached))

    def test_unknown_seed_yields_nothing(self, chain_graph):
        assert chain_graph.walk_triples(["entity:nope"]) == []

    def test_isolated_seed_yields_nothing(self, chain_graph):
        isolated = {"nodes": [{"id": "entity:alone", "type": "Entity", "label": "Alone",
                               "properties": {}}], "edges": []}
        assert KnowledgeGraph(isolated).walk_triples(["entity:alone"]) == []

    def test_duplicate_seeds_are_harmless(self, chain_graph):
        once = chain_graph.walk_triples(["entity:refund_policy"])
        twice = chain_graph.walk_triples(["entity:refund_policy", "entity:refund_policy"])
        assert _labels(once) == _labels(twice)


class TestWalkRobustness:
    """Merged graphs come from arbitrary JSON; a malformed edge must not 500 a search."""

    def test_null_mention_count_does_not_raise(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": "entity:a", "type": "E", "label": "A", "properties": {}},
                      {"id": "entity:b", "type": "E", "label": "B", "properties": {}}],
            "edges": [{"source": "entity:a", "target": "entity:b", "type": "uses",
                       "properties": {"mention_count": None}}],
        })
        assert _labels(graph.walk_triples(["entity:a"])) == {("A", "uses", "B")}

    def test_non_numeric_mention_count_does_not_raise(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": "entity:a", "type": "E", "label": "A", "properties": {}},
                      {"id": "entity:b", "type": "E", "label": "B", "properties": {}}],
            "edges": [{"source": "entity:a", "target": "entity:b", "type": "uses",
                       "properties": {"mention_count": "many"}}],
        })
        assert graph.walk_triples(["entity:a"])

    def test_node_without_label_falls_back_to_id(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": "entity:a", "type": "E", "label": "A", "properties": {}},
                      {"id": "entity:b", "type": "E", "properties": {}}],
            "edges": [{"source": "entity:a", "target": "entity:b", "type": "uses",
                       "properties": {}}],
        })
        assert graph.walk_triples(["entity:a"])[0]["target_label"] == "entity:b"

    def test_dangling_edge_is_skipped(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": "entity:a", "type": "E", "label": "A", "properties": {}}],
            "edges": [{"source": "entity:a", "target": "entity:gone", "type": "uses",
                       "properties": {}}],
        })
        assert graph.walk_triples(["entity:a"]) == []

    def test_cycle_terminates(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": f"entity:{c}", "type": "E", "label": c.upper(), "properties": {}}
                      for c in "abc"],
            "edges": [{"source": "entity:a", "target": "entity:b", "type": "uses", "properties": {}},
                      {"source": "entity:b", "target": "entity:c", "type": "uses", "properties": {}},
                      {"source": "entity:c", "target": "entity:a", "type": "uses", "properties": {}}],
        })
        assert len(graph.walk_triples(["entity:a"], hops=10)) <= 3

    def test_self_loop_counts_once_in_degree(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": "entity:a", "type": "E", "label": "A", "properties": {}},
                      {"id": "entity:b", "type": "E", "label": "B", "properties": {}}],
            "edges": [{"source": "entity:a", "target": "entity:a", "type": "uses", "properties": {}},
                      {"source": "entity:a", "target": "entity:b", "type": "uses", "properties": {}}],
        })
        assert graph.degree("entity:a") == 1

    def test_parallel_edges_count_once_in_degree(self):
        graph = KnowledgeGraph({
            "nodes": [{"id": "entity:a", "type": "E", "label": "A", "properties": {}},
                      {"id": "entity:b", "type": "E", "label": "B", "properties": {}}],
            "edges": [{"source": "entity:a", "target": "entity:b", "type": "uses", "properties": {}},
                      {"source": "entity:a", "target": "entity:b", "type": "improves", "properties": {}}],
        })
        assert graph.degree("entity:a") == 1


class TestWalkProvenance:
    """The runtime merges every graph file into one KnowledgeGraph. A shared node
    must not become a bridge that leaks one corpus's facts into another's."""

    @pytest.fixture
    def two_corpora(self, tmp_path):
        shared = {"id": "entity:shared", "type": "E", "label": "Shared", "properties": {}}
        left = tmp_path / "left"
        right = tmp_path / "right"
        left.mkdir()
        right.mkdir()
        (left / "a_llm_graph.json").write_text(json.dumps({
            "nodes": [shared, {"id": "entity:left", "type": "E", "label": "Left", "properties": {}}],
            "edges": [{"source": "entity:left", "target": "entity:shared", "type": "uses",
                       "properties": {"mention_count": 1}}],
        }))
        (right / "b_llm_graph.json").write_text(json.dumps({
            "nodes": [shared, {"id": "entity:right", "type": "E", "label": "Right", "properties": {}}],
            "edges": [{"source": "entity:shared", "target": "entity:right", "type": "uses",
                       "properties": {"mention_count": 9}}],
        }))
        return KnowledgeGraph([left / "a_llm_graph.json", right / "b_llm_graph.json"])

    def test_walk_does_not_cross_into_another_graph_dir(self, two_corpora):
        labels = _labels(two_corpora.walk_triples(["entity:left"]))
        assert ("Left", "uses", "Shared") in labels
        assert ("Shared", "uses", "Right") not in labels

    def test_node_present_in_both_dirs_may_walk_both(self, two_corpora):
        labels = _labels(two_corpora.walk_triples(["entity:shared"]))
        assert ("Shared", "uses", "Right") in labels
        assert ("Left", "uses", "Shared") in labels


class TestEntityContextChain:
    def test_context_includes_second_hop_chain(self, chain_graph):
        ctx = chain_graph.get_entity_context("entity:refund_policy")
        assert "via: Ops Manager held_by Sarah" in ctx

    def test_context_chain_is_capped(self, chain_graph):
        ctx = chain_graph.get_entity_context("entity:leaf")
        via = ctx.split("via: ", 1)[1]
        # entity:mid has 6 further neighbours; only CONTEXT_CHAIN_LIMIT are carried.
        assert len(via.split(", ")) == KnowledgeGraph.CONTEXT_CHAIN_LIMIT

    def test_well_connected_entity_gets_no_chain(self, chain_graph):
        # The gate: a context already carrying its full quota of direct relations
        # spends nothing on a second hop, which is where the drift lives.
        ctx = chain_graph.get_entity_context("entity:hub")
        assert "via:" not in ctx

    def test_context_without_chain_is_unchanged(self, llm_entity_graph):
        # No edges at all -> no "via:" segment, and the 1-hop shape is untouched.
        assert llm_entity_graph.get_entity_context("entity:nav") == "NAV (Entity)"


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

    def test_bare_hva_on_epic_does_not_dump_issues(self, jira_graph):
        # "Hva er <epic>" is a definitional question, not a request to enumerate
        # the epic's child issues — it must fall through to normal RAG (M4).
        answer = jira_graph.answer_graph_query(
            ["epic:PROJECT-6079"], "Hva er PROJECT-6079?"
        )
        assert answer is None

    def test_contains_intent_lists_epic_issues(self, jira_graph):
        # Explicit containment intent still lists the epic's issues.
        answer = jira_graph.answer_graph_query(
            ["epic:PROJECT-6079"], "Hva inneholder PROJECT-6079?"
        )
        assert answer is not None
        assert "PROJECT-6587" in answer


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


class TestGraphMerge:
    """Merging multiple graph files: duplicate-node merge + edge dedup (M1, M3)."""

    def test_duplicate_node_merges_properties(self, tmp_path):
        a = {"nodes": [{"id": "n:1", "type": "T", "label": "L", "properties": {"x": 1}}], "edges": []}
        b = {"nodes": [{"id": "n:1", "type": "T", "label": "L", "properties": {"y": 2}}], "edges": []}
        fa, fb = tmp_path / "a.json", tmp_path / "b.json"
        fa.write_text(json.dumps(a))
        fb.write_text(json.dumps(b))
        g = KnowledgeGraph([fa, fb])
        assert g.node_count() == 1
        assert g.nodes["n:1"]["properties"] == {"x": 1, "y": 2}

    def test_duplicate_node_without_properties_key_does_not_raise(self, tmp_path):
        # First copy lacks a "properties" key entirely — must not KeyError on merge.
        a = {"nodes": [{"id": "n:1", "type": "T", "label": "L"}], "edges": []}
        b = {"nodes": [{"id": "n:1", "type": "T", "label": "L", "properties": {"y": 2}}], "edges": []}
        fa, fb = tmp_path / "a.json", tmp_path / "b.json"
        fa.write_text(json.dumps(a))
        fb.write_text(json.dumps(b))
        g = KnowledgeGraph([fa, fb])
        assert g.nodes["n:1"]["properties"] == {"y": 2}

    def test_merge_does_not_mutate_source_dict(self, tmp_path):
        a = {"nodes": [{"id": "n:1", "type": "T", "label": "L", "properties": {"x": 1}}], "edges": []}
        b = {"nodes": [{"id": "n:1", "type": "T", "label": "L", "properties": {"y": 2}}], "edges": []}
        fa, fb = tmp_path / "a.json", tmp_path / "b.json"
        fa.write_text(json.dumps(a))
        fb.write_text(json.dumps(b))
        KnowledgeGraph([fa, fb])
        # Re-read the first file from disk: its node must be untouched by the merge.
        reread = json.loads(fa.read_text())
        assert reread["nodes"][0]["properties"] == {"x": 1}

    def test_duplicate_edge_across_files_is_deduped(self, tmp_path):
        edge = {"source": "n:1", "target": "n:2", "type": "rel", "properties": {}}
        nodes = [
            {"id": "n:1", "type": "T", "label": "1", "properties": {}},
            {"id": "n:2", "type": "T", "label": "2", "properties": {}},
        ]
        a = {"nodes": nodes, "edges": [edge]}
        b = {"nodes": nodes, "edges": [edge]}
        fa, fb = tmp_path / "a.json", tmp_path / "b.json"
        fa.write_text(json.dumps(a))
        fb.write_text(json.dumps(b))
        g = KnowledgeGraph([fa, fb])
        assert g.edge_count() == 1
        assert len(g.outgoing["n:1"]) == 1
        assert len(g.incoming["n:2"]) == 1


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

    def test_node_detail_filters_by_edge_type(self, sample_graph):
        detail = sample_graph.get_node_detail("buc:LA_BUC_01", edge_types={"inneholder_sed"})
        assert len(detail["outgoing"]) == 2
        assert all(e["type"] == "inneholder_sed" for e in detail["outgoing"])


@pytest.fixture
def epic_subtree_graph(tmp_path):
    """Epic with 2 stories; one story has 2 subtasks (mirrors the MELOSYS-7464 shape)."""
    graph_data = {
        "nodes": [
            {"id": "epic:E-1", "type": "Epic", "label": "E-1: Root epic", "properties": {}},
            {"id": "issue:S-1", "type": "Issue", "label": "S-1: Story 1", "properties": {"issue_type": "Historie"}},
            {"id": "issue:S-2", "type": "Issue", "label": "S-2: Story 2", "properties": {"issue_type": "Historie"}},
            {"id": "issue:T-1", "type": "Issue", "label": "T-1: Subtask 1", "properties": {"issue_type": "Deloppgave"}},
            {"id": "issue:T-2", "type": "Issue", "label": "T-2: Subtask 2", "properties": {"issue_type": "Deloppgave"}},
            {"id": "issue:OTHER", "type": "Issue", "label": "OTHER: Unrelated", "properties": {}},
        ],
        "edges": [
            {"source": "issue:S-1", "target": "epic:E-1", "type": "tilhører_epic", "properties": {}},
            {"source": "issue:S-2", "target": "epic:E-1", "type": "tilhører_epic", "properties": {}},
            {"source": "issue:T-1", "target": "issue:S-1", "type": "er_subtask_av", "properties": {}},
            {"source": "issue:T-2", "target": "issue:S-1", "type": "er_subtask_av", "properties": {}},
            {"source": "issue:S-1", "target": "issue:S-2", "type": "refererer_til", "properties": {}},
        ],
    }
    graph_file = tmp_path / "subtree.json"
    graph_file.write_text(json.dumps(graph_data))
    return KnowledgeGraph(graph_file)


class TestGetSubtree:
    def test_epic_depth_2_returns_stories_and_subtasks(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree("epic:E-1", direction="incoming", max_depth=2)
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"epic:E-1", "issue:S-1", "issue:S-2", "issue:T-1", "issue:T-2"}
        assert result["stats"]["node_count"] == 5

    def test_depth_1_only_returns_direct_neighbors(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree("epic:E-1", direction="incoming", max_depth=1)
        ids = {n["id"] for n in result["nodes"]}
        assert ids == {"epic:E-1", "issue:S-1", "issue:S-2"}

    def test_edge_type_filter_excludes_refererer(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree(
            "epic:E-1",
            direction="incoming",
            edge_types={"tilhører_epic", "er_subtask_av"},
            max_depth=2,
        )
        edge_types = {e["type"] for e in result["edges"]}
        assert edge_types == {"tilhører_epic", "er_subtask_av"}

    def test_unknown_root_returns_none(self, epic_subtree_graph):
        assert epic_subtree_graph.get_subtree("epic:DOES-NOT-EXIST") is None

    def test_stats_count_edge_types(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree("epic:E-1", direction="incoming", max_depth=2)
        assert result["stats"]["by_edge_type"]["tilhører_epic"] == 2
        assert result["stats"]["by_edge_type"]["er_subtask_av"] == 2

    def test_other_unrelated_node_not_included(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree("epic:E-1", direction="incoming", max_depth=5)
        ids = {n["id"] for n in result["nodes"]}
        assert "issue:OTHER" not in ids

    def test_max_nodes_truncates_and_flags(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree(
            "epic:E-1", direction="incoming", max_depth=5, max_nodes=2
        )
        assert result["stats"]["truncated"] is True
        assert result["stats"]["node_count"] <= 2

    def test_not_truncated_by_default(self, epic_subtree_graph):
        result = epic_subtree_graph.get_subtree("epic:E-1", direction="incoming", max_depth=2)
        assert result["stats"]["truncated"] is False

    def test_dangling_edge_is_skipped(self, tmp_path):
        # An edge whose source node is absent from the graph must not be emitted,
        # and must not introduce a phantom node id into the result (M2).
        graph_data = {
            "nodes": [
                {"id": "epic:E-1", "type": "Epic", "label": "E-1", "properties": {}},
                {"id": "issue:S-1", "type": "Issue", "label": "S-1", "properties": {}},
            ],
            "edges": [
                {"source": "issue:S-1", "target": "epic:E-1", "type": "tilhører_epic", "properties": {}},
                # Dangling: issue:GHOST is not a declared node.
                {"source": "issue:GHOST", "target": "epic:E-1", "type": "tilhører_epic", "properties": {}},
            ],
        }
        graph_file = tmp_path / "dangling.json"
        graph_file.write_text(json.dumps(graph_data))
        g = KnowledgeGraph(graph_file)
        result = g.get_subtree("epic:E-1", direction="incoming", max_depth=2)
        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"epic:E-1", "issue:S-1"}
        assert "issue:GHOST" not in node_ids
        # Every emitted edge must reference only nodes present in the result.
        for e in result["edges"]:
            assert e["source"] in node_ids
            assert e["target"] in node_ids
