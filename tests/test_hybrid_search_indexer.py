import numpy as np

from main.indexes.indexers.hybrid_search_indexer import HybridSearchIndexer


class FakeIndexer:
    """Minimal indexer stub for testing hybrid search."""

    def __init__(self, results):
        """results: list of (id, score) tuples, sorted by relevance."""
        self._results = results

    def search(self, text, number_of_results=10):
        items = self._results[:number_of_results]
        if not items:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        ids = np.array([[r[0] for r in items]], dtype=np.int64)
        scores = np.array([[r[1] for r in items]], dtype=np.float32)
        return scores, ids

    def get_size(self):
        return len(self._results)


class TestHybridSearch:
    def test_combines_both_retrievers(self):
        faiss = FakeIndexer([(1, 0.1), (2, 0.2), (3, 0.3)])
        bm25 = FakeIndexer([(2, -1.5), (4, -1.0), (1, -0.5)])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids = hybrid.search("query", number_of_results=5)
        result_ids = ids[0].tolist()

        # Doc 1 and 2 appear in both retrievers, should be ranked highest
        assert result_ids[0] in (1, 2)
        assert result_ids[1] in (1, 2)
        # All unique docs should be present
        assert set(result_ids) == {1, 2, 3, 4}

    def test_scores_are_negated(self):
        faiss = FakeIndexer([(1, 0.1)])
        bm25 = FakeIndexer([(1, -1.0)])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids = hybrid.search("query")
        assert scores[0][0] < 0

    def test_respects_number_of_results(self):
        faiss = FakeIndexer([(i, float(i)) for i in range(20)])
        bm25 = FakeIndexer([(i, float(-i)) for i in range(20)])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids = hybrid.search("query", number_of_results=5)
        assert len(ids[0]) == 5

    def test_empty_retrievers(self):
        faiss = FakeIndexer([])
        bm25 = FakeIndexer([])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids = hybrid.search("query")
        assert ids.shape == (1, 0) or len(ids[0]) == 0

    def test_one_empty_retriever(self):
        faiss = FakeIndexer([(1, 0.1), (2, 0.2)])
        bm25 = FakeIndexer([])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids = hybrid.search("query")
        assert set(ids[0].tolist()) == {1, 2}

    def test_negative_ids_ignored(self):
        """FAISS returns -1 for unfilled slots; hybrid should skip them."""
        faiss = FakeIndexer([(1, 0.1), (-1, 0.0)])
        bm25 = FakeIndexer([(1, -1.0)])
        hybrid = HybridSearchIndexer(faiss, bm25)

        scores, ids = hybrid.search("query")
        assert -1 not in ids[0].tolist()

    def test_rrf_k_affects_scores(self):
        faiss = FakeIndexer([(1, 0.1)])
        bm25 = FakeIndexer([(1, -1.0)])

        hybrid_low_k = HybridSearchIndexer(faiss, bm25, rrf_k=1)
        hybrid_high_k = HybridSearchIndexer(faiss, bm25, rrf_k=100)

        scores_low, _ = hybrid_low_k.search("query")
        scores_high, _ = hybrid_high_k.search("query")

        # Lower k gives higher RRF scores (more negative when negated)
        assert scores_low[0][0] < scores_high[0][0]

    def test_get_size_delegates_to_faiss(self):
        faiss = FakeIndexer([(1, 0.1), (2, 0.2)])
        bm25 = FakeIndexer([(1, -1.0)])
        hybrid = HybridSearchIndexer(faiss, bm25)
        assert hybrid.get_size() == 2

    def test_get_name(self):
        faiss = FakeIndexer([])
        bm25 = FakeIndexer([])
        hybrid = HybridSearchIndexer(faiss, bm25)
        assert hybrid.get_name() == "hybrid_FAISS_BM25"

    def test_index_texts_raises(self):
        faiss = FakeIndexer([])
        bm25 = FakeIndexer([])
        hybrid = HybridSearchIndexer(faiss, bm25)
        try:
            hybrid.index_texts([0], ["text"])
            assert False, "Should have raised"
        except NotImplementedError:
            pass

    def test_serialize_raises(self):
        faiss = FakeIndexer([])
        bm25 = FakeIndexer([])
        hybrid = HybridSearchIndexer(faiss, bm25)
        try:
            hybrid.serialize()
            assert False, "Should have raised"
        except NotImplementedError:
            pass
