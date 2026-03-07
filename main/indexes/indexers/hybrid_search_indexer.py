import numpy as np


class HybridSearchIndexer:
    """Search-time wrapper combining FAISS and BM25 results via Reciprocal Rank Fusion."""

    def __init__(self, faiss_indexer, bm25_indexer, rrf_k=60):
        self.faiss_indexer = faiss_indexer
        self.bm25_indexer = bm25_indexer
        self.rrf_k = rrf_k

    def get_name(self):
        return "hybrid_FAISS_BM25"

    def search(self, text, number_of_results=10):
        # Fetch more candidates from each retriever to improve fusion quality
        fetch_k = number_of_results * 3

        faiss_scores, faiss_ids = self.faiss_indexer.search(text, fetch_k)
        bm25_scores, bm25_ids = self.bm25_indexer.search(text, fetch_k)

        # Build RRF score map: score = sum(1 / (k + rank)) across retrievers
        rrf_scores = {}

        for rank, doc_id in enumerate(faiss_ids[0]):
            doc_id = int(doc_id)
            if doc_id < 0:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank)

        for rank, doc_id in enumerate(bm25_ids[0]):
            doc_id = int(doc_id)
            if doc_id < 0:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score descending, take top-k
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:number_of_results]

        if not sorted_items:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)

        result_ids = np.array([[item[0] for item in sorted_items]], dtype=np.int64)
        # Negate RRF scores so lower = better (consistent with L2 convention)
        result_scores = np.array([[-item[1] for item in sorted_items]], dtype=np.float32)

        return result_scores, result_ids

    def get_size(self):
        return self.faiss_indexer.get_size()

    def index_texts(self, ids, texts):
        raise NotImplementedError("HybridSearchIndexer is search-only")

    def serialize(self):
        raise NotImplementedError("HybridSearchIndexer is search-only")

    def remove_ids(self, ids):
        raise NotImplementedError("HybridSearchIndexer is search-only")
