from main.runtime.knowledge_store import KnowledgeStore


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
