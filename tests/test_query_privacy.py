"""Query-side half of build-time aliasing: de-alias seam + alias pin.

Four things this file pins:

1. ``dealias_query`` rewrites a typed real name into the alias the index holds,
   is idempotent, and is a no-op without a registry.
2. The pipeline applies it **per collection**, so a mixed request searches an
   aliased collection in alias space and an out-of-scope one in name space.
3. Nothing that persists or echoes — trace ``query.raw``, the query log,
   ``retryHints``, ``corrective.queriesTried`` — ever carries the typed name.
4. The alias pin rescues an exact alias-token hit that the cross-encoder scores
   as noise, and stays completely out of the way otherwise.

Invented names only; the real map is never read here.
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from knowledge_api_server import app
from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.search_pipeline import run_search_request, search_and_shape
from main.core.search_trace import create_trace
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.privacy import alias_registry as alias_registry_module
from main.privacy.alias_registry import AliasRegistry
from main.privacy.query_privacy import (
    alias_token_pattern,
    alias_tokens_in,
    dealias_query,
    dealias_value,
    first_armed_registry,
    text_contains_any_token,
    unaliased_expansion_twin,
)
from main.runtime.knowledge_store import KnowledgeStore, get_store
from tests.test_privacy_pipeline import FIXTURE_MAP

NAME = "Ada Example"          # -> dev-01
ALIAS = "dev-01"
OTHER_NAME = "Bo Tester"      # -> fag-01
OTHER_ALIAS = "fag-01"

# A public scope entry, so resolve_registry() arms on the collection name alone.
IN_SCOPE_COLLECTION = "jira-issues"


@pytest.fixture
def registry():
    return AliasRegistry(FIXTURE_MAP)


# --- 1. the substituter, used as a query rewriter ---------------------------

class TestDealiasQuery:

    def test_rewrites_a_typed_name_into_the_alias(self, registry):
        assert dealias_query(f"hva gjorde {NAME} i saken", registry) == \
            f"hva gjorde {ALIAS} i saken"

    def test_is_idempotent(self, registry):
        once = dealias_query(f"{NAME} og {OTHER_NAME}", registry)
        assert dealias_query(once, registry) == once
        assert once == f"{ALIAS} og {OTHER_ALIAS}"

    def test_no_registry_is_a_no_op(self):
        assert dealias_query(f"hva gjorde {NAME}", None) == f"hva gjorde {NAME}"

    def test_empty_query_survives(self, registry):
        assert dealias_query("", registry) == ""

    def test_first_armed_registry_skips_the_none_entries(self, registry):
        assert first_armed_registry({"a": None, "b": registry, "c": None}) is registry
        assert first_armed_registry({"a": None}) is None
        assert first_armed_registry({}) is None


class TestAliasTokenPattern:

    def test_matches_the_alias_vocabulary(self, registry):
        pattern = alias_token_pattern(registry)
        assert pattern.findall(f"se {ALIAS} og {OTHER_ALIAS}") == [ALIAS, OTHER_ALIAS]

    def test_never_matches_the_redaction_tokens(self, registry):
        # They mean "someone", not someone in particular: thousands of chunks
        # carry them, so pinning on one would promote pure noise.
        pattern = alias_token_pattern(registry)
        assert pattern.findall("[~person] @person [~ukjent-person]") == []

    def test_prefixes_come_from_the_map_not_a_hardcoded_list(self):
        other = AliasRegistry({**FIXTURE_MAP, "entries": [
            {"alias": "zz-07", "name": "Cy Sample", "variants": ["Cy Sample"]},
        ]})
        pattern = alias_token_pattern(other)
        assert pattern.search("zz-07")
        assert pattern.search(ALIAS) is None

    def test_a_longer_number_is_a_different_token(self, registry):
        """Token equality, not substring: a query for dev-01 must not match a
        chunk that only mentions dev-011."""
        pattern = alias_token_pattern(registry)
        assert alias_tokens_in("se dev-011", pattern) == {"dev-011"}
        assert text_contains_any_token("se dev-011", {ALIAS}, pattern) is False
        assert text_contains_any_token(f"se {ALIAS}.", {ALIAS}, pattern) is True

    def test_none_registry_has_no_pattern(self):
        assert alias_token_pattern(None) is None

    def test_pattern_is_cached_on_the_registry(self, registry):
        assert alias_token_pattern(registry) is alias_token_pattern(registry)


class TestUnaliasedExpansionTwin:

    def test_none_when_nothing_was_dealiased(self):
        assert unaliased_expansion_twin("q", "q", "q terms") is None

    def test_carries_the_expansion_suffix_over_to_the_raw_query(self):
        assert unaliased_expansion_twin(NAME, ALIAS, f"{ALIAS} sak vedtak") == \
            f"{NAME} sak vedtak"

    def test_falls_back_to_the_raw_query_if_expansion_is_not_a_suffix(self):
        assert unaliased_expansion_twin(NAME, ALIAS, "something else") == NAME


# --- 2. per-collection application in the pipeline --------------------------

class _FakeSearcher:
    def __init__(self, response=None):
        self.response = response or {"results": [], "reranked": True}
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.response


def _search_kwargs():
    return dict(max_number_of_chunks=20, max_number_of_documents=10,
                include_matched_chunks_content=True)


def _shape(**kwargs):
    return search_and_shape(
        augmenter=GraphSearchAugmenter(None),
        detected_entities=[],
        trace=create_trace(False),
        search_kwargs=_search_kwargs(),
        shape_kwargs=dict(limit=10),
        **kwargs,
    )


class TestPerCollectionQueryText:

    def test_mixed_request_searches_each_collection_in_its_own_space(self, registry):
        armed, plain = _FakeSearcher(), _FakeSearcher()
        _shape(
            target_searchers={"armed": armed, "plain": plain},
            query=f"{ALIAS} sak",
            title_boost_query=ALIAS,
            alias_registries={"armed": registry, "plain": None},
            unaliased_query=f"{NAME} sak",
            unaliased_title_boost_query=NAME,
        )
        assert armed.calls[0]["query"] == f"{ALIAS} sak"
        assert armed.calls[0]["title_boost_query"] == ALIAS
        assert plain.calls[0]["query"] == f"{NAME} sak"
        assert plain.calls[0]["title_boost_query"] == NAME

    def test_armed_collection_is_dealiased_even_if_the_caller_forgot(self, registry):
        """The seam is idempotent, so it is also a backstop."""
        armed = _FakeSearcher()
        _shape(
            target_searchers={"armed": armed},
            query=f"{NAME} sak",
            title_boost_query=f"{NAME} sak",
            alias_registries={"armed": registry},
        )
        assert armed.calls[0]["query"] == f"{ALIAS} sak"

    def test_out_of_scope_only_request_is_untouched(self):
        plain = _FakeSearcher()
        _shape(
            target_searchers={"plain": plain},
            query=f"{NAME} sak",
            title_boost_query=f"{NAME} sak",
            alias_registries={"plain": None},
        )
        assert plain.calls[0]["query"] == f"{NAME} sak"
        assert "alias_pattern" not in plain.calls[0]

    def test_no_registries_at_all_changes_nothing(self):
        plain = _FakeSearcher()
        _shape(target_searchers={"plain": plain}, query="q", title_boost_query="q")
        assert plain.calls[0]["query"] == "q"
        assert "alias_pattern" not in plain.calls[0]

    def test_armed_collection_gets_the_alias_pattern(self, registry):
        armed = _FakeSearcher()
        _shape(
            target_searchers={"armed": armed},
            query=ALIAS,
            title_boost_query=ALIAS,
            alias_registries={"armed": registry},
        )
        assert armed.calls[0]["alias_pattern"].search(ALIAS)

    def test_alias_pinned_counts_are_summed_onto_the_response(self, registry):
        raw = {"results": [], "reranked": True, "aliasPinned": 2}
        response = run_search_request(
            {"a": _FakeSearcher(raw), "b": _FakeSearcher(raw)},
            raw_query=ALIAS,
            search_query=ALIAS,
            augmenter=GraphSearchAugmenter(None),
            detected_entities=[],
            graph_answer=None,
            trace=create_trace(False),
            search_kwargs=_search_kwargs(),
            shape_kwargs=dict(limit=10),
            min_relevance=None,
            corrective_mode="off",
            alias_registries={"a": registry, "b": registry},
        )
        assert response["aliasPinned"] == 4

    def test_no_alias_pinned_key_when_nothing_was_pinned(self, registry):
        response = run_search_request(
            {"a": _FakeSearcher()},
            raw_query=ALIAS,
            search_query=ALIAS,
            augmenter=GraphSearchAugmenter(None),
            detected_entities=[],
            graph_answer=None,
            trace=create_trace(False),
            search_kwargs=_search_kwargs(),
            shape_kwargs=dict(limit=10),
            min_relevance=None,
            corrective_mode="off",
            alias_registries={"a": registry},
        )
        assert "aliasPinned" not in response


# --- 3. nothing that persists or echoes carries the name --------------------

class _StubStore:
    """Enough store surface for /api/search."""

    def __init__(self, searchers, registries):
        self._searchers = searchers
        self._registries = registries
        self.graph = None

    def has_collection(self, name):
        return name in self._searchers

    def get_searchers(self, names=None):
        if names:
            return {n: self._searchers[n] for n in names if n in self._searchers}
        return dict(self._searchers)

    def get_alias_registries(self, names=None):
        names = list(names) if names else list(self._searchers)
        return {n: self._registries.get(n) for n in names}


class TestNoNameEscapesTheRequest:

    def setup_method(self):
        self.armed = _FakeSearcher()
        self.plain = _FakeSearcher()

    def teardown_method(self):
        app.dependency_overrides.pop(get_store, None)

    def _client(self, registry):
        store = _StubStore({"armed": self.armed, "plain": self.plain},
                           {"armed": registry, "plain": None})
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def test_trace_hints_and_queries_tried_hold_the_alias_never_the_name(
            self, registry, tmp_path, monkeypatch):
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(tmp_path / "query-log.jsonl"))
        client = self._client(registry)

        response = client.get("/api/search", params={
            "q": f"{NAME} rolle i saken", "trace": "true", "corrective": "force",
        })
        assert response.status_code == 200
        body = response.json()
        blob = json.dumps(body, ensure_ascii=False)

        assert NAME not in blob
        assert body["trace"]["query"]["raw"] == f"{ALIAS} rolle i saken"
        assert body["retryHints"]["broaderQuery"].startswith(f"{ALIAS} rolle")
        assert body["corrective"]["queriesTried"][0] == f"{ALIAS} rolle i saken"
        assert body["corrective"]["queriesTried"][1] == body["retryHints"]["broaderQuery"]

        # …and the two searchers each got the text their own index spells.
        assert self.armed.calls[0]["query"] == f"{ALIAS} rolle i saken"
        assert self.plain.calls[0]["query"] == f"{NAME} rolle i saken"

        log_text = (tmp_path / "query-log.jsonl").read_text()
        assert NAME not in log_text
        assert json.loads(log_text.splitlines()[0])["query"] == f"{ALIAS} rolle i saken"

    def test_an_all_unarmed_request_still_logs_the_name_it_searched_with(
            self, tmp_path, monkeypatch):
        """The de-alias seam must not touch the 28 out-of-scope collections."""
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(tmp_path / "query-log.jsonl"))
        client = self._client(None)
        response = client.get("/api/search", params={"q": NAME, "collection": "plain"})
        assert response.status_code == 200
        assert self.plain.calls[0]["query"] == NAME
        record = json.loads((tmp_path / "query-log.jsonl").read_text().splitlines()[0])
        assert record["query"] == NAME


# --- 4. the alias pin at the reranker seam ----------------------------------

CHUNK_TEXTS = {
    "doc-A": ["saksgang og vedtak", "frister i saken"],
    "doc-B": [f"{OTHER_ALIAS} eier rutinen"],   # the only exact alias hit
}

# What the live cross-encoder does with an opaque token: every candidate scores
# as the same near-zero noise (measured 0.00095), which
# search_policy.apply_confidence_filtering then drops wholesale.
CE_NOISE = -0.00095


def _noise_reranker():
    reranker = MagicMock()

    def rerank(query, scores, indexes, chunk_texts, top_k, return_ce_scores=False):
        ranked = list(indexes[0])[:top_k]
        out = (np.array([[CE_NOISE] * len(ranked)], dtype=np.float32),
               np.array([ranked], dtype=np.int64))
        return (*out, [(int(c), -CE_NOISE) for c in ranked]) if return_ce_scores else out

    reranker.rerank.side_effect = rerank
    return reranker


def _alias_searcher(reranker):
    mapping, documents = {}, {}
    chunk_id = 0
    for doc_id, chunks in CHUNK_TEXTS.items():
        documents[doc_id] = {"id": doc_id, "text": " ".join(chunks),
                             "chunks": [{"indexedData": c} for c in chunks]}
        for chunk_number in range(len(chunks)):
            mapping[str(chunk_id)] = {
                "documentId": doc_id,
                "documentUrl": f"http://example.com/{doc_id}",
                "documentPath": f"col/documents/{doc_id}.json",
                "chunkNumber": chunk_number,
            }
            chunk_id += 1

    persister = MagicMock()

    def read_text(path):
        if "index_document_mapping" in path:
            return json.dumps(mapping)
        for doc_id, document in documents.items():
            if doc_id in path:
                return json.dumps(document)
        raise FileNotFoundError(path)

    persister.read_text_file.side_effect = read_text

    indexer = MagicMock()
    indexer.get_name.return_value = "test_indexer"
    indexer.supports_breakdown = False
    indexer.search.return_value = (
        # RRF order: the alias chunk is retrieved LAST but is the exact hit.
        np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    return DocumentCollectionSearcher(collection_name="col", indexer=indexer,
                                      persister=persister, reranker=reranker)


class TestAliasPin:

    def test_without_the_pin_a_noise_scoring_ce_empties_the_result_set(self):
        """The bug the pin exists for, pinned as a test."""
        result = _alias_searcher(_noise_reranker()).search(
            OTHER_ALIAS, max_number_of_chunks=3)
        assert result["results"] == []
        assert "aliasPinned" not in result

    def test_the_pin_keeps_the_exact_alias_hit(self, registry):
        result = _alias_searcher(_noise_reranker()).search(
            OTHER_ALIAS, max_number_of_chunks=3,
            alias_pattern=alias_token_pattern(registry))
        assert [doc["id"] for doc in result["results"]] == ["doc-B"]
        assert result["aliasPinned"] == 1
        # Scored below the low-confidence band, so the response is not flagged.
        assert "lowConfidence" not in result
        assert result["results"][0]["matchedChunks"][0]["score"] == pytest.approx(-0.15)

    def test_pin_is_absent_when_the_query_has_no_alias_token(self, registry):
        result = _alias_searcher(_noise_reranker()).search(
            "vedtak og frister", max_number_of_chunks=3,
            alias_pattern=alias_token_pattern(registry))
        assert "aliasPinned" not in result

    def test_pin_is_absent_when_no_chunk_carries_the_queried_alias(self, registry):
        result = _alias_searcher(_noise_reranker()).search(
            ALIAS, max_number_of_chunks=3,   # dev-01 is in no chunk; fag-01 is
            alias_pattern=alias_token_pattern(registry))
        assert "aliasPinned" not in result

    def test_pin_does_not_run_without_the_reranker(self, registry):
        result = _alias_searcher(_noise_reranker()).search(
            OTHER_ALIAS, max_number_of_chunks=3, skip_reranker=True,
            alias_pattern=alias_token_pattern(registry))
        assert "aliasPinned" not in result
        assert result["reranked"] is False

    def test_trace_records_the_pinned_chunk(self, registry):
        trace = create_trace(True)
        _alias_searcher(_noise_reranker()).search(
            OTHER_ALIAS, max_number_of_chunks=3, trace=trace,
            alias_pattern=alias_token_pattern(registry))
        stages = [c["stages"] for c in trace.to_dict()["collections"][0]["candidates"]
                  if "aliasPin" in c["stages"]]
        assert len(stages) == 1
        assert stages[0]["aliasPin"]["rank"] == 0
        assert stages[0]["aliasPin"]["score"] == pytest.approx(-0.15)

    def test_a_confident_ce_ranking_keeps_its_best_score(self, registry):
        """The synthetic score is min(best CE, band): a strong CE hit is not
        weakened by the pin, only reordered."""
        reranker = MagicMock()
        reranker.rerank.return_value = (
            np.array([[-0.9, -0.8, -0.7]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )
        result = _alias_searcher(reranker).search(
            OTHER_ALIAS, max_number_of_chunks=3,
            alias_pattern=alias_token_pattern(registry))
        assert result["results"][0]["id"] == "doc-B"
        assert result["results"][0]["matchedChunks"][0]["score"] == pytest.approx(-0.9)


# --- 5. the store resolves, degrades and re-resolves ------------------------

class TestStoreRegistryResolution:

    def _store(self, manifest):
        store = KnowledgeStore()
        persister = MagicMock()
        persister.read_text_file.return_value = json.dumps(manifest)
        store.disk_persister = persister
        return store

    def test_missing_map_for_an_armed_collection_logs_and_serves(
            self, tmp_path, caplog, monkeypatch):
        """Indexing fails closed; serving must not. The index on disk is already
        aliased, so a missing map costs de-aliasing, not privacy."""
        # An empty root: no huginn-*/privacy/aliases.json to discover.
        monkeypatch.setattr(alias_registry_module, "REPO_ROOT", str(tmp_path))
        store = self._store({"reader": {"basePath": str(tmp_path / "src")},
                             "privacy": {"policy_version": 1, "map_version": 1}})
        with caplog.at_level("ERROR"):
            assert store._resolve_alias_registry(IN_SCOPE_COLLECTION) is None
        assert "WITHOUT query de-aliasing" in caplog.text

    def test_out_of_scope_collection_resolves_to_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(alias_registry_module, "REPO_ROOT", str(tmp_path))
        store = self._store({"reader": {"basePath": str(tmp_path / "src")}})
        assert store._resolve_alias_registry("some-public-collection") is None

    def test_reload_re_resolves_the_registry(self, registry, monkeypatch):
        store = KnowledgeStore()
        store._build_aux_indexes = False
        store.alias_registries["col"] = None
        monkeypatch.setattr(store, "_build_searcher", lambda name: MagicMock())
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda **kw: None)
        monkeypatch.setattr(store, "_resolve_alias_registry", lambda name: registry)

        store.reload_collection("col")

        assert store.alias_registries["col"] is registry
        assert store.get_alias_registries(["col"]) == {"col": registry}

    def test_get_alias_registries_only_reports_served_collections(self, registry):
        store = KnowledgeStore()
        store.searchers = {"a": MagicMock()}
        store.alias_registries = {"a": registry}
        assert store.get_alias_registries(["a", "missing"]) == {"a": registry}
        assert store.get_alias_registries() == {"a": registry}


# --- 6. the pre-alias knowledge graph -------------------------------------

class _FakeGraph:
    """A graph extracted before the corpus was aliased: its labels are names."""

    def __init__(self):
        self.nodes = {"e1": {"type": "Person", "label": NAME}}

    def detect_entities(self, text, with_spans=False):
        if "sak" not in (text or ""):
            return []
        return [("e1", "sak")] if with_spans else ["e1"]

    def answer_graph_query(self, entities, q):
        return f"{NAME} eier saken"

    def get_expansion_terms(self, entities):
        return [OTHER_NAME, "vedtak"]

    def get_entity_context(self, eid):
        return {"label": NAME, "facts": [f"{NAME} er utvikler"]}


class TestGraphDerivedTextIsDealiased:
    """The graph JSONs predate aliasing, so their labels and expansion terms
    are as much a leak as the query echo."""

    def test_expansion_labels_and_answer_are_rewritten(self, registry):
        augmenter = GraphSearchAugmenter(_FakeGraph(), alias_registry=registry)
        trace = create_trace(True)
        search_q, graph_answer, entities = augmenter.augment_query("hvem eier saken", trace)
        assert NAME not in search_q and OTHER_ALIAS in search_q
        assert graph_answer == f"{ALIAS} eier saken"
        assert trace.to_dict()["query"]["detectedEntities"][0]["label"] == ALIAS
        assert NAME not in json.dumps(trace.to_dict(), ensure_ascii=False)

    def test_retry_hints_and_context_are_rewritten(self, registry):
        augmenter = GraphSearchAugmenter(_FakeGraph(), alias_registry=registry)
        hints = augmenter.get_retry_hints("hvem eier saken", ["e1"])
        assert hints["detectedEntities"] == [ALIAS]
        assert OTHER_ALIAS in hints["relatedTerms"]
        assert NAME not in json.dumps(hints, ensure_ascii=False)

        results = [{"title": "saken", "id": "d1"}]
        augmenter.enrich_results(results, ["e1"])
        assert NAME not in json.dumps(results, ensure_ascii=False)

    def test_without_a_registry_the_graph_is_untouched(self):
        """An out-of-scope-only request must behave exactly as before."""
        augmenter = GraphSearchAugmenter(_FakeGraph())
        search_q, graph_answer, _ = augmenter.augment_query(
            "hvem eier saken", create_trace(False))
        assert OTHER_NAME in search_q
        assert graph_answer == f"{NAME} eier saken"

    def test_dealias_value_does_not_mutate_the_graph_structure(self, registry):
        """The structures belong to the shared, long-lived graph object."""
        node = {"label": NAME, "facts": [NAME]}
        out = dealias_value(node, registry)
        assert out == {"label": ALIAS, "facts": [ALIAS]}
        assert node == {"label": NAME, "facts": [NAME]}
        assert dealias_value(node, None) is node
