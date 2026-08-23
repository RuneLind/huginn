"""Query-side half of build-time aliasing: the de-alias seam.

Four things this file pins:

1. ``dealias_query`` rewrites a typed real name into the alias the index holds,
   is idempotent, and is a no-op without a registry.
2. The pipeline applies it **per collection**, so a mixed request searches an
   aliased collection in alias space and an out-of-scope one in name space —
   and an out-of-scope collection never receives an alias token, on either the
   initial or the corrective-rescue leg.
3. Nothing that persists or echoes — trace ``query.raw``, the query log,
   ``retryHints``, ``corrective.queriesTried`` — ever carries the typed name,
   whichever collection the request is scoped to.
4. The graph is a pre-alias artifact, so every string it contributes is
   de-aliased, and the terms that would re-identify a person beside their own
   alias are dropped.

Invented names only; the real map is never read here.
"""
import builtins
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import collection_search_cmd_adapter as search_cli
from knowledge_api_server import app
from main.core.mcp_search_tool import build_search_tool_fn
from main.core.search_pipeline import search_and_shape
from main.core.search_policy import apply_title_boost
from main.core.search_trace import create_trace
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.privacy import alias_registry as alias_registry_module
from main.privacy.alias_registry import AliasRegistry
from main.privacy.query_privacy import (
    dealias_query,
    dealias_value,
    first_armed_registry,
    keep_expansion_term,
    name_space_twin,
    prepare_aliased_request,
    unaliased_expansion_twin,
)
from main.routes import ingest as ingest_routes
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


def _private_map_root(tmp_path, monkeypatch):
    """A tmp REPO_ROOT holding one discoverable alias map, and nothing else."""
    map_path = tmp_path / "huginn-x" / "privacy" / "aliases.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(json.dumps(FIXTURE_MAP), encoding="utf-8")
    monkeypatch.setattr(alias_registry_module, "REPO_ROOT", str(tmp_path))
    return map_path


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

    def test_dealias_value_walks_tuples_too(self, registry):
        """Graph payloads are arbitrary JSON-ish; a tuple used to pass straight
        through with the name still in it."""
        assert dealias_value((NAME, [NAME], {"k": NAME}), registry) == \
            (ALIAS, [ALIAS], {"k": ALIAS})

    def test_dealias_value_does_not_mutate_the_graph_structure(self, registry):
        """The structures belong to the shared, long-lived graph object."""
        node = {"label": NAME, "facts": [NAME]}
        out = dealias_value(node, registry)
        assert out == {"label": ALIAS, "facts": [ALIAS]}
        assert node == {"label": NAME, "facts": [NAME]}
        assert dealias_value(node, None) is node


class TestUnaliasedExpansionTwin:

    def test_none_when_nothing_was_dealiased_and_nothing_was_expanded(self):
        assert unaliased_expansion_twin("q", "q", "q") is None

    def test_carries_the_expansion_suffix_over_to_the_raw_query(self):
        assert unaliased_expansion_twin(NAME, ALIAS, f"{ALIAS} sak vedtak") == \
            f"{NAME} sak vedtak"

    def test_falls_back_to_the_raw_query_if_expansion_is_not_a_suffix(self):
        assert unaliased_expansion_twin(NAME, ALIAS, "something else") == NAME

    def test_raw_expansion_terms_win_even_when_the_query_itself_was_clean(self):
        """The graph predates aliasing, so a clean query can still pick up a
        de-aliased suffix — the out-of-scope twin needs the name-space terms."""
        assert unaliased_expansion_twin(
            "hvem eier saken", "hvem eier saken",
            f"hvem eier saken {OTHER_ALIAS} vedtak",
            [OTHER_NAME, "vedtak"]) == f"hvem eier saken {OTHER_NAME} vedtak"


class TestNameSpaceTwin:

    def test_maps_an_alias_token_back_to_the_name_the_other_indexes_spell(self, registry):
        assert name_space_twin(f"{ALIAS} rolle", registry) == f"{NAME} rolle"

    def test_leaves_a_text_with_no_alias_token_alone(self, registry):
        assert name_space_twin("hvem eier saken", registry) is None

    def test_does_not_fire_on_a_longer_token_that_merely_starts_with_an_alias(self, registry):
        assert name_space_twin(f"{ALIAS}9 rolle", registry) is None

    def test_without_a_registry_there_is_no_twin(self):
        assert name_space_twin(f"{ALIAS} rolle", None) is None


class TestExpansionTermFilter:

    def test_drops_the_redaction_tokens(self, registry):
        for token in ("[~ukjent-person]", "[~person]", "@person"):
            assert keep_expansion_term(token, registry) is False

    def test_drops_a_bare_given_name_of_a_mapped_entry(self, registry):
        """A first name beside its own alias re-pairs the two. Bare given names
        survive the substitution by design (they would alias half the corpus)."""
        assert keep_expansion_term("Ada", registry) is False
        assert keep_expansion_term("ada", registry) is False
        assert keep_expansion_term("Bo", registry) is False

    def test_keeps_an_ordinary_term(self, registry):
        assert keep_expansion_term("vedtak", registry) is True

    def test_without_a_registry_nothing_is_filtered(self):
        assert keep_expansion_term("Ada", None) is True
        assert keep_expansion_term("[~person]", None) is True


# --- 2. per-collection application in the pipeline --------------------------

class _FakeSearcher:
    def __init__(self, response=None):
        self.response = response or {"results": [], "reranked": True}
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.response

    def queries(self):
        return [call["query"] for call in self.calls] + \
               [call.get("title_boost_query") for call in self.calls]


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

    def test_no_registries_at_all_changes_nothing(self):
        plain = _FakeSearcher()
        _shape(target_searchers={"plain": plain}, query="q", title_boost_query="q")
        assert plain.calls[0]["query"] == "q"


class TestTitleBoostStaysInNameSpace:
    """Document filenames are never aliased (an id is a join key — check 7's
    whole premise), so an alias token cannot legitimately match one. What it can
    do is match by accident."""

    def test_an_alias_token_boosts_every_numbered_filename(self):
        """`apply_title_boost` tokenises with `\\w+`, so `dev-01` contributes the
        token `01` — and boosts every `…-01-…` document in the collection."""
        mapping = {"0": {"documentId": "d0", "documentPath": "c/documents/notat-01-plan.md.json"},
                   "1": {"documentId": "d1", "documentPath": "c/documents/rapport.md.json"}}
        scores, indexes = np.array([[0.10, 0.20]]), np.array([[0, 1]])

        boosted, _ = apply_title_boost(ALIAS, scores, indexes, mapping)
        assert boosted[0][0] < 0.10, "the -01- filename was boosted by the alias token"

        untouched, _ = apply_title_boost(NAME, scores, indexes, mapping)
        assert list(untouched[0]) == [0.10, 0.20]

    def test_a_typed_alias_is_mapped_back_for_title_matching(self, registry):
        """The raw text is not always name space: an alias query IS the alias,
        and it would boost every `…-01-…` filename. Measured live on
        melosys-confluence-v3: 8 candidates boosted by the token `01`, three
        extra documents surfacing, where the same person's name boosted none."""
        prepared = prepare_aliased_request(f"{ALIAS} rolle", {"armed": registry})
        assert prepared.title_boost_text() == f"{NAME} rolle"

    def test_the_two_spellings_agree_on_what_titles_to_boost(self, registry):
        """…which is what keeps an alias query and a name query returning the
        same documents: the search text was already identical, the title-boost
        text is what had diverged."""
        typed_alias = prepare_aliased_request(f"{ALIAS} rolle", {"armed": registry})
        typed_name = prepare_aliased_request(f"{NAME} rolle", {"armed": registry})
        assert typed_alias.title_boost_text() == typed_name.title_boost_text()

    def test_an_unarmed_request_has_no_title_boost_twin_at_all(self):
        assert prepare_aliased_request(f"{NAME} rolle", {"plain": None}).title_boost_text() \
            is None

    def test_an_armed_collection_gets_the_raw_title_boost_text(self, registry):
        armed = _FakeSearcher()
        _shape(
            target_searchers={"armed": armed},
            query=f"{ALIAS} sak",
            title_boost_query=ALIAS,
            alias_registries={"armed": registry},
            unaliased_query=f"{NAME} sak",
            unaliased_title_boost_query=NAME,
        )
        assert armed.calls[0]["query"] == f"{ALIAS} sak"
        assert armed.calls[0]["title_boost_query"] == NAME


# --- 3. nothing that persists or echoes carries the name --------------------

class _FakeGraph:
    """A graph extracted before the corpus was aliased: its labels are names."""

    def __init__(self, expansion=None):
        self.nodes = {f"entity:{NAME.lower()}": {"type": "Person", "label": NAME}}
        self.entity_id = f"entity:{NAME.lower()}"
        self.expansion = [OTHER_NAME, "vedtak"] if expansion is None else expansion

    def detect_entities(self, text, with_spans=False):
        if "sak" not in (text or ""):
            return []
        return [(self.entity_id, "sak")] if with_spans else [self.entity_id]

    def answer_graph_query(self, entities, q):
        return f"{NAME} eier saken"

    def get_expansion_terms(self, entities):
        return list(self.expansion)

    def get_entity_context(self, eid):
        return {"label": NAME, "facts": [f"{NAME} er utvikler"]}


class _StubStore:
    """Enough store surface for /api/search."""

    def __init__(self, searchers, registries, graph=None):
        self._searchers = searchers
        self._registries = registries
        self.graph = graph

    def has_collection(self, name):
        return name in self._searchers

    def get_searchers(self, names=None):
        if names:
            return {n: self._searchers[n] for n in names if n in self._searchers}
        return dict(self._searchers)

    def get_alias_registries(self, names=None):
        names = list(names) if names else list(self._searchers)
        return {n: self._registries.get(n) for n in names}

    def get_searchers_and_registries(self, names=None):
        return self.get_searchers(names), dict(self._registries)


class _RouteCase:
    """One /api/search call against a two-collection store, one armed."""

    def setup_method(self):
        self.armed = _FakeSearcher()
        self.plain = _FakeSearcher()

    def teardown_method(self):
        app.dependency_overrides.pop(get_store, None)

    def _client(self, registry, graph=None):
        store = _StubStore({"armed": self.armed, "plain": self.plain},
                           {"armed": registry, "plain": None}, graph=graph)
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app)

    def _get(self, registry, graph=None, **params):
        response = self._client(registry, graph).get("/api/search", params=params)
        assert response.status_code == 200
        return response.json()


class TestRouteDealiasesTheSharedText(_RouteCase):

    def test_the_trace_records_the_alias_not_the_typed_name(self, registry):
        body = self._get(registry, q=f"{NAME} rolle i saken", trace="true")
        assert body["trace"]["query"]["raw"] == f"{ALIAS} rolle i saken"
        assert NAME not in json.dumps(body, ensure_ascii=False)

    def test_the_retry_hints_and_queries_tried_hold_the_alias(self, registry):
        body = self._get(registry, q=f"{NAME} rolle i saken", corrective="force")
        assert body["retryHints"]["broaderQuery"].startswith(f"{ALIAS} rolle")
        assert body["corrective"]["queriesTried"][0] == f"{ALIAS} rolle i saken"
        assert body["corrective"]["queriesTried"][1] == body["retryHints"]["broaderQuery"]

    def test_a_request_scoped_to_an_unarmed_collection_still_dealiases(
            self, registry, tmp_path, monkeypatch):
        """The armed collection is served but NOT targeted. The shared text is
        the request's, not the target set's: scoping to the out-of-scope
        collection used to write the typed name straight into the query log."""
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(tmp_path / "query-log.jsonl"))
        body = self._get(registry, q=f"{NAME} rolle", collection="plain", trace="true")

        assert body["trace"]["query"]["raw"] == f"{ALIAS} rolle"
        log_text = (tmp_path / "query-log.jsonl").read_text()
        assert NAME not in log_text
        assert json.loads(log_text.splitlines()[0])["query"] == f"{ALIAS} rolle"
        # …and the collection it actually searched still got name space.
        assert self.plain.calls[0]["query"] == f"{NAME} rolle"
        assert not self.armed.calls


class TestQueryLogNeverHoldsTheTypedName(_RouteCase):

    def test_the_logged_query_is_the_alias(self, registry, tmp_path, monkeypatch):
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(tmp_path / "query-log.jsonl"))
        self._get(registry, q=f"{NAME} rolle i saken")
        log_text = (tmp_path / "query-log.jsonl").read_text()
        assert NAME not in log_text
        assert json.loads(log_text.splitlines()[0])["query"] == f"{ALIAS} rolle i saken"

    def test_an_all_unarmed_request_still_logs_the_name_it_searched_with(
            self, tmp_path, monkeypatch):
        """The de-alias seam must not touch the out-of-scope collections."""
        monkeypatch.setenv("HUGINN_QUERY_LOG", str(tmp_path / "query-log.jsonl"))
        self._get(None, q=NAME, collection="plain")
        assert self.plain.calls[0]["query"] == NAME
        record = json.loads((tmp_path / "query-log.jsonl").read_text().splitlines()[0])
        assert record["query"] == NAME


class TestOutOfScopeCollectionsNeverSeeAnAliasToken(_RouteCase):
    """Their indexes were never aliased: an alias token retrieves nothing there,
    and the alias vocabulary has no business in a name-space index's query."""

    def _alias_tokens_in(self, searcher):
        return [q for q in searcher.queries()
                if q and (ALIAS in q or OTHER_ALIAS in q)]

    def test_with_a_name_in_the_typed_query(self, registry):
        self._get(registry, q=f"{NAME} rolle i saken", corrective="force")
        assert len(self.plain.calls) == 2, "both the initial and the rescue leg ran"
        assert self._alias_tokens_in(self.plain) == []
        assert self.plain.calls[1]["query"].startswith(NAME)

    def test_without_a_name_in_the_typed_query(self, registry):
        """The expansion terms come from the pre-alias graph, so a clean query
        still picks up a de-aliased suffix the out-of-scope index cannot match."""
        self._get(registry, q="hvem eier saken", corrective="force",
                  graph=_FakeGraph())
        assert len(self.plain.calls) == 2
        assert self._alias_tokens_in(self.plain) == []
        assert OTHER_NAME in self.plain.calls[0]["query"]

    def test_the_armed_collection_still_searches_in_alias_space(self, registry):
        self._get(registry, q=f"{NAME} rolle i saken", corrective="force",
                  graph=_FakeGraph())
        assert all(NAME not in call["query"] for call in self.armed.calls)
        assert self.armed.calls[0]["query"].startswith(ALIAS)


class TestPerCollectionRegistryPlumbing(_RouteCase):
    """The route must hand the pipeline a registry PER collection, not one
    verdict for the whole request."""

    def test_each_collection_gets_the_text_its_own_index_spells(self, registry):
        self._get(registry, q=f"{NAME} rolle i saken")
        assert self.armed.calls[0]["query"] == f"{ALIAS} rolle i saken"
        assert self.plain.calls[0]["query"] == f"{NAME} rolle i saken"

    def test_both_spellings_reach_the_searchers_identically(self, registry):
        """End to end, the property the acceptance sweep measures: an alias
        query and a name query search every collection with the same text."""
        self._get(registry, q=f"{NAME} rolle i saken")
        typed_name = [dict(call) for call in self.armed.calls + self.plain.calls]
        self.setup_method()
        self._get(registry, q=f"{ALIAS} rolle i saken")
        typed_alias = [dict(call) for call in self.armed.calls + self.plain.calls]
        assert typed_alias == typed_name


# --- 4. the store resolves, degrades and re-resolves ------------------------

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

    def test_an_unimportable_privacy_module_does_not_kill_the_load(
            self, tmp_path, caplog, monkeypatch):
        """A clone without the privacy package must still serve. The import used
        to sit outside the try, so it took the whole collection load with it."""
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "main.privacy.alias_registry":
                raise ImportError("no privacy package in this checkout")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        store = self._store({"reader": {"basePath": "/src"}})
        with caplog.at_level("ERROR"):
            assert store._resolve_alias_registry("some-collection") is None
        assert "WITHOUT query de-aliasing" in caplog.text

    def test_a_map_version_the_index_was_not_built_from_warns(
            self, tmp_path, caplog, monkeypatch):
        """Queries de-alias to whatever the CURRENT map says; the index holds
        whatever the map at build time said. Divergence is a silent miss."""
        _private_map_root(tmp_path, monkeypatch)
        store = self._store({"reader": {"basePath": str(tmp_path / "src")},
                             "privacy": {"policy_version": 1, "map_version": 6}})
        with caplog.at_level("WARNING"):
            assert store._resolve_alias_registry(IN_SCOPE_COLLECTION) is not None
        assert "map_version" in caplog.text
        assert "6" in caplog.text and str(FIXTURE_MAP["version"]) in caplog.text

    def test_a_matching_map_version_is_quiet(self, tmp_path, caplog, monkeypatch):
        _private_map_root(tmp_path, monkeypatch)
        store = self._store({"reader": {"basePath": str(tmp_path / "src")},
                             "privacy": {"policy_version": 1,
                                         "map_version": FIXTURE_MAP["version"]}})
        with caplog.at_level("WARNING"):
            assert store._resolve_alias_registry(IN_SCOPE_COLLECTION) is not None
        assert "map_version" not in caplog.text

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

    def test_load_collections_stores_the_resolved_registry(self, registry, monkeypatch):
        """The normal path, not just reload: a load that dropped the resolution
        on the floor would serve every collection unaliased and say nothing."""
        store = KnowledgeStore()
        monkeypatch.setattr(store, "_build_searcher", lambda name: MagicMock())
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda **kw: None)
        monkeypatch.setattr(store, "_resolve_alias_registry", lambda name: registry)
        monkeypatch.setattr(store, "_KnowledgeStore__detect_shared_faiss_index",
                            lambda names: "idx")
        monkeypatch.setattr("main.runtime.knowledge_store.create_embedder",
                            lambda name: MagicMock())
        monkeypatch.setattr("main.runtime.knowledge_store.create_reranker",
                            lambda: MagicMock())

        store.load_collections(["col"], build_aux_indexes=False)

        assert store.alias_registries["col"] is registry


class TestSingleLockAccessor:
    """Reading the searchers and the registries under two separate lock
    acquisitions let a reload land between them, so a request could search a
    just-rebuilt aliased index with the previous (or absent) registry. There is
    no way to assert on that race from a test; what IS assertable is that the
    accessor hands back one consistent pair."""

    def test_returns_the_targeted_searchers_and_every_served_registry(self, registry):
        store = KnowledgeStore()
        store.searchers = {"a": MagicMock(), "b": MagicMock()}
        store.alias_registries = {"a": registry, "b": None}

        searchers, registries = store.get_searchers_and_registries(["b"])

        assert list(searchers) == ["b"]
        # Every served registry, not the targeted subset: the shared text is
        # de-aliased when ANY served collection is armed.
        assert registries == {"a": registry, "b": None}

    def test_the_pair_is_consistent_after_a_reload(self, registry, monkeypatch):
        store = KnowledgeStore()
        store._build_aux_indexes = False
        searcher = MagicMock()
        monkeypatch.setattr(store, "_build_searcher", lambda name: searcher)
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda **kw: None)
        monkeypatch.setattr(store, "_resolve_alias_registry", lambda name: registry)

        store.reload_collection("col")
        searchers, registries = store.get_searchers_and_registries(["col"])

        assert searchers == {"col": searcher}
        assert registries == {"col": registry}

    def test_an_unserved_collection_is_absent_from_both(self, registry):
        store = KnowledgeStore()
        store.searchers = {"a": MagicMock()}
        store.alias_registries = {"a": registry}
        searchers, registries = store.get_searchers_and_registries(["a", "missing"])
        assert list(searchers) == ["a"]
        assert "missing" not in registries


# --- 5. the pre-alias knowledge graph -------------------------------------

class TestGraphDerivedTextIsDealiased:
    """The graph JSONs predate aliasing, so their labels, ids and expansion
    terms are as much a leak as the query echo."""

    def test_expansion_labels_and_answer_are_rewritten(self, registry):
        augmenter = GraphSearchAugmenter(_FakeGraph(), alias_registry=registry)
        trace = create_trace(True)
        search_q, graph_answer, entities, raw_terms = augmenter.augment_query(
            "hvem eier saken", trace)
        assert NAME not in search_q and OTHER_ALIAS in search_q
        assert graph_answer == f"{ALIAS} eier saken"
        assert trace.to_dict()["query"]["detectedEntities"][0]["label"] == ALIAS
        assert NAME not in json.dumps(trace.to_dict(), ensure_ascii=False)
        # The raw twin the out-of-scope collections need, alongside.
        assert raw_terms == [OTHER_NAME, "vedtak"]

    def test_the_detected_entity_id_is_rewritten_too(self, registry):
        """`entity:<name>` ids reach the trace verbatim; the id is as much a name
        as the label. The lookup keeps using the raw id."""
        graph = _FakeGraph()
        augmenter = GraphSearchAugmenter(graph, alias_registry=registry)
        trace = create_trace(True)
        _, _, entities, _ = augmenter.augment_query("hvem eier saken", trace)

        recorded = trace.to_dict()["query"]["detectedEntities"][0]
        assert recorded["id"] == f"entity:{ALIAS}"
        assert entities == [graph.entity_id], "the graph lookups keep the raw id"

    def test_retry_hints_and_context_are_rewritten(self, registry):
        augmenter = GraphSearchAugmenter(_FakeGraph(), alias_registry=registry)
        hints = augmenter.get_retry_hints("hvem eier saken", [f"entity:{NAME.lower()}"])
        assert hints["detectedEntities"] == [ALIAS]
        assert OTHER_ALIAS in hints["relatedTerms"]
        assert NAME not in json.dumps(hints, ensure_ascii=False)

        results = [{"title": "saken", "id": "d1"}]
        augmenter.enrich_results(results, [f"entity:{NAME.lower()}"])
        assert NAME not in json.dumps(results, ensure_ascii=False)

    @pytest.mark.parametrize("term", ["[~ukjent-person]", "[~person]", "@person", "Ada"])
    def test_a_reidentifying_term_never_reaches_the_related_terms(self, registry, term):
        """A redaction token is noise; a bare given name standing beside its own
        alias is the pairing the whole campaign exists to prevent."""
        graph = _FakeGraph(expansion=[term, "vedtak"])
        augmenter = GraphSearchAugmenter(graph, alias_registry=registry)
        hints = augmenter.get_retry_hints("hvem eier saken", [graph.entity_id])
        assert hints["relatedTerms"] == ["vedtak"]

        search_q, _, _, raw_terms = augmenter.augment_query("hvem eier saken",
                                                            create_trace(False))
        assert search_q == "hvem eier saken vedtak"
        assert raw_terms == ["vedtak"]

    def test_without_a_registry_nothing_is_filtered_or_rewritten(self):
        """An out-of-scope-only request must behave exactly as before."""
        graph = _FakeGraph(expansion=["Ada", "[~person]", "vedtak"])
        augmenter = GraphSearchAugmenter(graph)
        search_q, graph_answer, _, raw_terms = augmenter.augment_query(
            "hvem eier saken", create_trace(False))
        assert search_q == "hvem eier saken Ada [~person] vedtak"
        assert raw_terms == ["Ada", "[~person]", "vedtak"]
        assert graph_answer == f"{NAME} eier saken"


# --- 6. the MCP seam --------------------------------------------------------

class TestMcpSearchToolDealiases:
    """The stdio tools run the same pipeline as the route and used to be the one
    seam with no de-alias at all."""

    def _run(self, registry, query, graph=None, **kwargs):
        searcher = _FakeSearcher()
        search_fn = build_search_tool_fn(
            searcher, "armed", GraphSearchAugmenter(graph),
            max_number_of_chunks=20, max_number_of_documents=5,
            include_full_text=False, alias_registry=registry, **kwargs)
        return searcher, json.loads(search_fn(query))

    def test_the_searcher_and_the_response_see_only_the_alias(self, registry, caplog):
        with caplog.at_level(logging.INFO):
            searcher, body = self._run(registry, f"{NAME} rolle i saken")
        assert searcher.calls[0]["query"] == f"{ALIAS} rolle i saken"
        assert NAME not in json.dumps(body, ensure_ascii=False)
        assert NAME not in caplog.text

    def test_the_graph_contribution_is_dealiased(self, registry):
        """The augmenter is handed in already built, so the tool has to re-wrap
        it with the registry — otherwise the graph's names ride out in
        `graph_answer` while the query itself is clean."""
        _, body = self._run(registry, "hvem eier saken", graph=_FakeGraph())
        assert body["graph_answer"] == f"{ALIAS} eier saken"
        assert NAME not in json.dumps(body, ensure_ascii=False)

    def test_an_unarmed_tool_is_unchanged(self):
        searcher, body = self._run(None, f"{NAME} rolle")
        assert searcher.calls[0]["query"] == f"{NAME} rolle"

    def test_an_out_of_scope_tool_beside_an_armed_one_still_dealiases_its_log(
            self, registry, caplog):
        """`shared_registry` is the multi-collection adapter's whole point: one
        served armed collection means the request's shared text is de-aliased,
        whichever tool was called."""
        with caplog.at_level(logging.INFO):
            searcher, body = self._run(None, f"{NAME} rolle", shared_registry=registry)
        assert NAME not in caplog.text
        # …while the collection it searches is still spelled in name space.
        assert searcher.calls[0]["query"] == f"{NAME} rolle"


# --- 7. the remaining callers ----------------------------------------------

class TestIngestSimilarityDealiases:

    def test_the_similarity_query_is_rewritten_for_an_armed_collection(self, registry):
        searcher = _FakeSearcher({"results": []})
        store = _StubStore({"armed": searcher}, {"armed": registry})
        ingest_routes._similar_for_collection(
            store, "armed", f"{NAME} rolle", lambda doc: False)
        assert searcher.calls[0]["query"] == f"{ALIAS} rolle"

    def test_an_out_of_scope_collection_is_untouched(self):
        searcher = _FakeSearcher({"results": []})
        store = _StubStore({"plain": searcher}, {"plain": None})
        ingest_routes._similar_for_collection(
            store, "plain", f"{NAME} rolle", lambda doc: False)
        assert searcher.calls[0]["query"] == f"{NAME} rolle"


class TestSearchCliDealiases:

    def _collections(self, tmp_path, monkeypatch, manifest):
        collections = tmp_path / "collections"
        (collections / "demo").mkdir(parents=True)
        (collections / "demo" / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(search_cli, "COLLECTIONS_DIR", collections)

    def test_an_armed_collection_is_searched_in_alias_space(self, tmp_path, monkeypatch):
        _private_map_root(tmp_path, monkeypatch)
        self._collections(tmp_path, monkeypatch,
                          {"collectionName": "demo",
                           "reader": {"basePath": str(tmp_path / "src")},
                           "privacy": {"policy_version": 1,
                                       "map_version": FIXTURE_MAP["version"]}})
        assert search_cli.dealiased_query("demo", f"{NAME} rolle") == f"{ALIAS} rolle"

    def test_an_out_of_scope_collection_is_untouched(self, tmp_path, monkeypatch):
        _private_map_root(tmp_path, monkeypatch)
        self._collections(tmp_path, monkeypatch,
                          {"collectionName": "demo",
                           "reader": {"basePath": str(tmp_path / "src")}})
        assert search_cli.dealiased_query("demo", f"{NAME} rolle") == f"{NAME} rolle"

    def test_a_missing_manifest_is_not_fatal(self, tmp_path, monkeypatch):
        _private_map_root(tmp_path, monkeypatch)
        monkeypatch.setattr(search_cli, "COLLECTIONS_DIR", tmp_path / "nope")
        assert search_cli.dealiased_query("demo", f"{NAME} rolle") == f"{NAME} rolle"


# --- 8. what de-aliasing does NOT change -----------------------------------

class TestRerankerSkipDecision:
    """De-aliasing rewrites the text the language detector and the word count
    see, so the reranker-skip decision is taken per collection on that text.
    This pins today's behaviour for the shortest interesting case."""

    def _searcher(self):
        from main.core.documents_collection_searcher import DocumentCollectionSearcher
        return DocumentCollectionSearcher.__new__(DocumentCollectionSearcher)

    def test_a_two_token_query_never_reaches_langdetect(self):
        searcher = self._searcher()
        assert searcher._should_skip_reranker(NAME) is False
        assert searcher._should_skip_reranker(ALIAS) is False

    def test_the_skip_log_line_carries_no_query_text(self, caplog, monkeypatch):
        from main.core import documents_collection_searcher as module
        monkeypatch.setattr(module, "_langdetect_available", True)
        monkeypatch.setattr(module, "detect", lambda text: "en")
        searcher = self._searcher()
        with caplog.at_level(logging.INFO):
            assert searcher._should_skip_reranker(f"who is {NAME} really") is True
        assert NAME not in caplog.text
        assert "Ada" not in caplog.text


# --- 9. the request preamble both transports share --------------------------

class TestPrepareAliasedRequest:

    def test_picks_the_armed_registry_and_dealiases_once(self, registry):
        prepared = prepare_aliased_request(f"{NAME} rolle",
                                           {"plain": None, "armed": registry})
        assert prepared.armed is registry
        assert prepared.public_query == f"{ALIAS} rolle"
        assert prepared.title_boost_text() == f"{NAME} rolle"

    def test_an_unarmed_request_carries_no_twin_at_all(self):
        prepared = prepare_aliased_request(f"{NAME} rolle", {"plain": None})
        assert prepared.armed is None
        assert prepared.public_query == f"{NAME} rolle"
        assert prepared.title_boost_text() is None
        assert prepared.expansion_twin(f"{NAME} rolle vedtak", ["vedtak"]) is None

    def test_a_clean_query_still_gets_an_expansion_twin(self, registry):
        prepared = prepare_aliased_request("hvem eier saken", {"armed": registry})
        assert prepared.expansion_twin(f"hvem eier saken {OTHER_ALIAS}",
                                       [OTHER_NAME]) == f"hvem eier saken {OTHER_NAME}"
