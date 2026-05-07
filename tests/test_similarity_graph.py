"""Tests for main.graph.similarity_graph (Louvain community detection)."""


class TestDetectCommunities:
    """Tests for detect_communities (Louvain community detection on similarity graphs)."""

    def _make_nodes(self, n, categories=None):
        return [
            {"id": f"doc_{i}", "title": f"Doc {i}", "category": categories[i] if categories else "default", "tags": [categories[i]] if categories else []}
            for i in range(n)
        ]

    def test_two_clear_clusters(self):
        import numpy as np
        from main.graph.similarity_graph import detect_communities

        sim = np.array([
            [1.0, 0.9, 0.85, 0.1, 0.1, 0.1],
            [0.9, 1.0, 0.88, 0.1, 0.1, 0.1],
            [0.85, 0.88, 1.0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 1.0, 0.92, 0.87],
            [0.1, 0.1, 0.1, 0.92, 1.0, 0.9],
            [0.1, 0.1, 0.1, 0.87, 0.9, 1.0],
        ], dtype=np.float32)
        doc_ids = [f"doc_{i}" for i in range(6)]
        categories = ["ai"] * 3 + ["health"] * 3
        nodes = self._make_nodes(6, categories)

        communities = detect_communities(sim, doc_ids, nodes)

        assert len(communities) == 2
        sizes = sorted(c["size"] for c in communities)
        assert sizes == [3, 3]
        assert nodes[0]["community"] == nodes[1]["community"]
        assert nodes[3]["community"] == nodes[4]["community"]
        assert nodes[0]["community"] != nodes[3]["community"]

    def test_all_isolated_nodes(self):
        import numpy as np
        from main.graph.similarity_graph import detect_communities

        sim = np.eye(4, dtype=np.float32)
        doc_ids = [f"doc_{i}" for i in range(4)]
        nodes = self._make_nodes(4)

        communities = detect_communities(sim, doc_ids, nodes)

        assert len(communities) == 0
        for n in nodes:
            assert "community" in n

    def test_community_metadata(self):
        import numpy as np
        from main.graph.similarity_graph import detect_communities

        sim = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ], dtype=np.float32)
        doc_ids = ["a", "b", "c"]
        nodes = [
            {"id": "a", "title": "Alpha", "category": "ai", "tags": ["ai", "ml"]},
            {"id": "b", "title": "Beta", "category": "ai", "tags": ["ai", "nlp"]},
            {"id": "c", "title": "Gamma", "category": "health", "tags": ["health"]},
        ]

        communities = detect_communities(sim, doc_ids, nodes)

        cluster = [c for c in communities if c["size"] == 2]
        assert len(cluster) == 1
        assert "top_tags" in cluster[0]
        assert "representative_docs" in cluster[0]
        assert cluster[0]["top_tags"][0]["tag"] == "ai"
