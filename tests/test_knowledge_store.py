from types import SimpleNamespace

from main.runtime.knowledge_store import KnowledgeStore, run_collection_update


class TestSimilarityGraphCacheAccessors:
    def test_get_returns_none_when_unset(self):
        store = KnowledgeStore()
        assert store.get_cached_similarity_graph("foo") is None

    def test_set_then_get_roundtrips(self):
        store = KnowledgeStore()
        graph = {"nodes": [], "all_edges": []}
        store.set_cached_similarity_graph("foo", graph)
        assert store.get_cached_similarity_graph("foo") is graph

    def test_set_overwrites(self):
        store = KnowledgeStore()
        store.set_cached_similarity_graph("foo", {"v": 1})
        store.set_cached_similarity_graph("foo", {"v": 2})
        assert store.get_cached_similarity_graph("foo") == {"v": 2}

    def test_separate_collections_independent(self):
        store = KnowledgeStore()
        store.set_cached_similarity_graph("foo", {"v": 1})
        store.set_cached_similarity_graph("bar", {"v": 2})
        assert store.get_cached_similarity_graph("foo") == {"v": 1}
        assert store.get_cached_similarity_graph("bar") == {"v": 2}


class TestAuthorGraphCacheAccessors:
    def test_get_returns_none_when_unset(self):
        store = KnowledgeStore()
        assert store.get_cached_author_graph("foo") is None

    def test_set_then_get_roundtrips(self):
        store = KnowledgeStore()
        graph = {"nodes": [], "edges": []}
        store.set_cached_author_graph("foo", graph)
        assert store.get_cached_author_graph("foo") is graph

    def test_similarity_and_author_caches_are_independent(self):
        """Same collection name in both caches must not collide."""
        store = KnowledgeStore()
        sim = {"kind": "similarity"}
        aut = {"kind": "author"}
        store.set_cached_similarity_graph("foo", sim)
        store.set_cached_author_graph("foo", aut)
        assert store.get_cached_similarity_graph("foo") == sim
        assert store.get_cached_author_graph("foo") == aut


class TestUpdateState:
    def test_idle_when_never_updated(self):
        store = KnowledgeStore()
        # idle returns the same key shape as a present state (nulls), not a truncated dict
        assert store.get_update_status("c") == {
            "collection": "c", "status": "idle",
            "startedAt": None, "finishedAt": None, "error": None,
        }

    def test_try_begin_reserves_and_blocks_concurrent(self):
        store = KnowledgeStore()
        assert store.try_begin_update("c") is True
        # a second update for the same collection is the H4 race — must be refused
        assert store.try_begin_update("c") is False
        status = store.get_update_status("c")
        assert status["status"] == "running"
        assert status["startedAt"]

    def test_begin_allowed_again_after_success(self):
        store = KnowledgeStore()
        store.try_begin_update("c")
        store.mark_update_succeeded("c")
        assert store.get_update_status("c")["status"] == "succeeded"
        assert store.try_begin_update("c") is True

    def test_begin_allowed_again_after_failure_records_error(self):
        store = KnowledgeStore()
        store.try_begin_update("c")
        store.mark_update_failed("c", RuntimeError("boom"))
        status = store.get_update_status("c")
        assert status["status"] == "failed"
        assert status["error"] == "boom"
        assert store.try_begin_update("c") is True

    def test_different_collections_are_independent(self):
        store = KnowledgeStore()
        assert store.try_begin_update("a") is True
        assert store.try_begin_update("b") is True


class TestRunCollectionUpdate:
    def _patch_updater(self, monkeypatch, run_fn):
        monkeypatch.setattr(
            "main.factories.update_collection_factory.create_collection_updater",
            lambda name: SimpleNamespace(run=run_fn),
        )

    def test_success_reloads_and_marks_succeeded(self, monkeypatch):
        store = KnowledgeStore()
        store.try_begin_update("c")
        reloaded = []
        monkeypatch.setattr(store, "reload_collection", lambda n: reloaded.append(n))
        self._patch_updater(monkeypatch, lambda: None)

        run_collection_update("c", store)

        assert reloaded == ["c"]
        assert store.get_update_status("c")["status"] == "succeeded"

    def test_failure_marks_failed_and_skips_reload(self, monkeypatch):
        store = KnowledgeStore()
        store.try_begin_update("c")
        reloaded = []
        monkeypatch.setattr(store, "reload_collection", lambda n: reloaded.append(n))

        def boom():
            raise RuntimeError("indexing failed")

        self._patch_updater(monkeypatch, boom)

        run_collection_update("c", store)

        # reload must be gated on a successful rebuild (H5) — no silent stale reload
        assert reloaded == []
        status = store.get_update_status("c")
        assert status["status"] == "failed"
        assert "indexing failed" in status["error"]


class TestLoadCollectionsResilience:
    def test_broken_collection_is_skipped_others_load(self, monkeypatch):
        store = KnowledgeStore()
        monkeypatch.setattr("main.runtime.knowledge_store.detect_faiss_index", lambda *a, **k: "fake_index")
        monkeypatch.setattr("main.runtime.knowledge_store.create_embedder", lambda name: SimpleNamespace(model_name="fake"))
        monkeypatch.setattr("main.runtime.knowledge_store.create_reranker", lambda: SimpleNamespace(model_name="fake"))
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda extra_paths=None: None)

        def fake_build(name):
            if name == "bad":
                raise RuntimeError("corrupt index/mapping")
            return SimpleNamespace(indexer=SimpleNamespace(get_size=lambda: 5))

        monkeypatch.setattr(store, "_build_searcher", fake_build)

        store.load_collections(["bad", "good"], build_aux_indexes=False)

        assert store.has_collection("good")
        assert not store.has_collection("bad")

    def test_embedder_detection_falls_through_broken_first_collection(self, monkeypatch):
        store = KnowledgeStore()

        def detect(name, persister):
            if name == "bad":
                raise ValueError("No FAISS index found")
            return "fake_index"

        monkeypatch.setattr("main.runtime.knowledge_store.detect_faiss_index", detect)
        monkeypatch.setattr("main.runtime.knowledge_store.create_embedder", lambda name: SimpleNamespace(model_name="fake"))
        monkeypatch.setattr("main.runtime.knowledge_store.create_reranker", lambda: SimpleNamespace(model_name="fake"))
        monkeypatch.setattr(store, "_load_knowledge_graph", lambda extra_paths=None: None)
        monkeypatch.setattr(store, "_build_searcher",
                            lambda name: SimpleNamespace(indexer=SimpleNamespace(get_size=lambda: 1)))

        # the broken first collection must not abort embedder detection / startup
        store.load_collections(["bad", "good"], build_aux_indexes=False)

        assert store.has_collection("good")
