"""Shared test helpers."""

from unittest.mock import MagicMock, patch

import numpy as np

from main.indexes.reranking.cross_encoder_reranker import CrossEncoderReranker


class FakeIndexer:
    """Minimal indexer stub: returns a fixed list of (id, score) pairs."""

    def __init__(self, results):
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


def make_mock_reranker(predict_scores, model_name="mock-reranker"):
    """Build a CrossEncoderReranker that bypasses model loading and returns fixed scores."""
    with patch.object(CrossEncoderReranker, "__init__", lambda self, **kwargs: None):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = model_name
        reranker.model = MagicMock()
        reranker.model.predict.return_value = np.array(predict_scores, dtype=np.float32)
        return reranker
