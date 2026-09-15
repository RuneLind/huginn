"""HUGINN_RERANK (server-side reranker switch) and GET /ready."""
import dataclasses
import types

import pytest
from fastapi.testclient import TestClient

import main.runtime.knowledge_store as knowledge_store_module
from knowledge_api_server import app
from main.runtime.knowledge_store import KnowledgeStore, get_store


class TestRerankSwitch:
    def _load(self, monkeypatch, env_value):
        if env_value is None:
            monkeypatch.delenv("HUGINN_RERANK", raising=False)
        else:
            monkeypatch.setenv("HUGINN_RERANK", env_value)
        calls = []
        monkeypatch.setattr(knowledge_store_module, "create_embedder",
                            lambda name: types.SimpleNamespace(model_name="emb"))
        monkeypatch.setattr(knowledge_store_module, "create_reranker",
                            lambda: calls.append(1) or types.SimpleNamespace(model_name="ce"))
        store = KnowledgeStore()
        monkeypatch.setattr(store, "_build_searcher",
                            lambda name: types.SimpleNamespace(
                                reranker=store.shared_reranker,
                                indexer=types.SimpleNamespace(get_size=lambda: 3)))
        monkeypatch.setattr(store, "_resolve_alias_registry", lambda name: None)
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda extra_paths=None: None)
        store.load_collections(["c"], faiss_index_name="idx", build_aux_indexes=False)
        return store, calls

    @pytest.mark.parametrize("value", [None, "", "1", "true", "on", "ON"])
    def test_default_and_truthy_values_load_the_reranker(self, monkeypatch, value):
        store, calls = self._load(monkeypatch, value)
        assert calls == [1]
        assert store.searchers["c"].reranker is not None

    @pytest.mark.parametrize("value", ["0", "false", "off", " Off "])
    def test_off_values_never_construct_the_reranker(self, monkeypatch, value):
        store, calls = self._load(monkeypatch, value)
        assert calls == []
        assert store.shared_reranker is None
        assert store.searchers["c"].reranker is None

    def test_unknown_value_refuses_before_loading_any_model(self, monkeypatch):
        monkeypatch.setenv("HUGINN_RERANK", "maybe")
        monkeypatch.setattr(knowledge_store_module, "create_embedder",
                            lambda name: pytest.fail("embedder loaded before the env check"))
        with pytest.raises(ValueError, match="HUGINN_RERANK"):
            KnowledgeStore().load_collections(["c"], faiss_index_name="idx")


class _FakeStore:
    def __init__(self, sizes):
        self._sizes = sizes

    def collection_sizes(self):
        return dict(self._sizes)


class TestReadiness:
    def _get(self, requested, sizes):
        original = app.state.config
        app.state.config = dataclasses.replace(original, collections=requested)
        app.dependency_overrides[get_store] = lambda: _FakeStore(sizes)
        try:
            return TestClient(app).get("/ready")
        finally:
            app.dependency_overrides.pop(get_store, None)
            app.state.config = original

    def test_ready_when_every_requested_collection_has_chunks(self):
        response = self._get(["a", "b"], {"a": 10, "b": 2})
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "collections": {"a": 10, "b": 2},
                                   "missing": [], "empty": []}

    def test_missing_collection_is_503_and_named(self):
        response = self._get(["a", "b"], {"a": 10})
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["missing"] == ["b"]
        assert body["empty"] == []
        assert body["collections"] == {"a": 10}

    def test_empty_collection_is_503_and_named(self):
        response = self._get(["a", "b"], {"a": 10, "b": 0})
        assert response.status_code == 503
        assert response.json()["empty"] == ["b"]

    def test_no_requested_collections_is_not_ready(self):
        response = self._get([], {})
        assert response.status_code == 503

    def test_served_but_unrequested_collection_is_listed_and_does_not_gate(self):
        response = self._get(["a"], {"a": 1, "extra": 0})
        assert response.status_code == 200
        assert response.json()["collections"] == {"a": 1, "extra": 0}

    def test_real_store_reports_indexer_sizes(self):
        store = KnowledgeStore()
        store.searchers = {"a": types.SimpleNamespace(indexer=types.SimpleNamespace(get_size=lambda: 7))}
        assert store.collection_sizes() == {"a": 7}
