import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from main.runtime.knowledge_store import KnowledgeStore, run_collection_update


def _notion_mapping(pages):
    """Build an index_document_mapping.json payload from {hex: path} pages."""
    return json.dumps({
        str(i): {
            "documentUrl": f"https://www.notion.so/Title-{hex_id}",
            "documentPath": path,
            "documentId": path,
            "chunkNumber": 0,
        }
        for i, (hex_id, path) in enumerate(pages.items())
    })


def _store_with_notion(mappings):
    """KnowledgeStore whose persister serves a settable per-collection mapping.

    `mappings` is {collection_name: {notion_hex: doc_path}}; mutate it between
    _build_notion_id_lookup calls to simulate an update rewriting the mapping.
    """
    store = KnowledgeStore()
    persister = MagicMock()

    def read_text(path):
        for name, pages in mappings.items():
            if path == f"{name}/indexes/index_document_mapping.json":
                return _notion_mapping(pages)
        raise FileNotFoundError(path)

    persister.read_text_file.side_effect = read_text
    store.disk_persister = persister
    return store


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


HEX_A = "a" * 32
HEX_B = "b" * 32
HEX_C = "c" * 32


class TestNotionIdLookupRebuild:
    """Per-collection slices, re-merged on (re)load — drops stale, no cross-collection clobber (M9)."""

    def test_reload_drops_entries_a_collection_no_longer_contains(self):
        mappings = {"collA": {HEX_A: "collA/documents/a.json", HEX_B: "collA/documents/b.json"}}
        store = _store_with_notion(mappings)

        store._build_notion_id_lookup("collA")
        assert set(store.notion_id_to_doc) == {HEX_A, HEX_B}

        # An update removes page B from the collection's mapping.
        mappings["collA"] = {HEX_A: "collA/documents/a.json"}
        store._build_notion_id_lookup("collA")

        # Stale page B is gone, not accumulated.
        assert set(store.notion_id_to_doc) == {HEX_A}

    def test_rebuilding_one_collection_keeps_another_collections_entries(self):
        mappings = {
            "collA": {HEX_A: "collA/documents/a.json"},
            "collB": {HEX_C: "collB/documents/c.json"},
        }
        store = _store_with_notion(mappings)

        store._build_notion_id_lookup("collA")
        store._build_notion_id_lookup("collB")
        assert set(store.notion_id_to_doc) == {HEX_A, HEX_C}

        # Rebuilding collA must not drop collB's entry.
        store._build_notion_id_lookup("collA")
        assert set(store.notion_id_to_doc) == {HEX_A, HEX_C}

    def test_lookup_dict_is_swapped_not_mutated_in_place(self):
        """Readers holding the old dict reference never see it mutated (M10)."""
        mappings = {"collA": {HEX_A: "collA/documents/a.json"}}
        store = _store_with_notion(mappings)
        store._build_notion_id_lookup("collA")

        before = store.notion_id_to_doc  # a reader grabs the current reference

        mappings["collA"] = {HEX_A: "collA/documents/a.json", HEX_B: "collA/documents/b.json"}
        store._build_notion_id_lookup("collA")

        # The reference the reader holds is untouched; a new dict was swapped in.
        assert set(before) == {HEX_A}
        assert set(store.notion_id_to_doc) == {HEX_A, HEX_B}
        assert store.notion_id_to_doc is not before


class TestTagCountsRebuild:
    def test_tag_counts_published_via_locked_accessor(self):
        store = KnowledgeStore()
        persister = MagicMock()

        def read_text(path):
            return json.dumps({"metadata": {"tags": "alpha, beta, alpha"}})

        persister.read_text_file.side_effect = read_text
        persister.read_folder_files.return_value = ["0.json"]
        store.disk_persister = persister

        store._build_tag_counts("collA")
        counts = store.get_tag_counts(["collA"])["collA"]
        assert counts == {"alpha": 2, "beta": 1}
