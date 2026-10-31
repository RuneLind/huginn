import numpy as np

from tests.conftest import make_mock_reranker as _make_reranker


class TestCrossEncoderReranker:
    def test_reranks_by_cross_encoder_score(self):
        # Original order: chunk 10, 20, 30 with scores [0.1, 0.2, 0.3]
        # Cross-encoder says: chunk 30 is best (0.9), chunk 10 next (0.5), chunk 20 worst (0.1)
        reranker = _make_reranker([0.5, 0.1, 0.9])

        scores = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        indexes = np.array([[10, 20, 30]], dtype=np.int64)
        chunk_texts = ["text A", "text B", "text C"]

        new_scores, new_indexes = reranker.rerank("query", scores, indexes, chunk_texts, top_k=3)

        # Should be reordered: 30 (0.9), 10 (0.5), 20 (0.1)
        assert new_indexes[0].tolist() == [30, 10, 20]

    def test_respects_top_k(self):
        reranker = _make_reranker([0.5, 0.1, 0.9, 0.3])

        scores = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        indexes = np.array([[10, 20, 30, 40]], dtype=np.int64)
        chunk_texts = ["A", "B", "C", "D"]

        new_scores, new_indexes = reranker.rerank("query", scores, indexes, chunk_texts, top_k=2)

        assert len(new_indexes[0]) == 2
        assert len(new_scores[0]) == 2

    def test_empty_inputs(self):
        reranker = _make_reranker([])

        scores = np.array([[]], dtype=np.float32)
        indexes = np.array([[]], dtype=np.int64)
        chunk_texts = []

        new_scores, new_indexes = reranker.rerank("query", scores, indexes, chunk_texts, top_k=5)

        assert len(new_indexes[0]) == 0

    def test_scores_are_negated(self):
        # Cross-encoder score is 0.8 (positive = relevant)
        reranker = _make_reranker([0.8])

        scores = np.array([[1.0]], dtype=np.float32)
        indexes = np.array([[5]], dtype=np.int64)
        chunk_texts = ["some text"]

        new_scores, new_indexes = reranker.rerank("query", scores, indexes, chunk_texts, top_k=1)

        # Score should be negated: -0.8
        assert new_scores[0][0] < 0
        assert abs(new_scores[0][0] - (-0.8)) < 1e-5

    def test_preserves_index_score_pairing(self):
        # Verify that after reranking, each index is paired with its correct score
        reranker = _make_reranker([0.2, 0.8, 0.5])

        scores = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        indexes = np.array([[100, 200, 300]], dtype=np.int64)
        chunk_texts = ["A", "B", "C"]

        new_scores, new_indexes = reranker.rerank("query", scores, indexes, chunk_texts, top_k=3)

        # Expected order by CE score: 200 (0.8), 300 (0.5), 100 (0.2)
        assert new_indexes[0].tolist() == [200, 300, 100]
        assert abs(new_scores[0][0] - (-0.8)) < 1e-5
        assert abs(new_scores[0][1] - (-0.5)) < 1e-5
        assert abs(new_scores[0][2] - (-0.2)) < 1e-5

    def test_model_name_property(self):
        reranker = _make_reranker([])
        assert reranker.model_name == "mock-reranker"


class TestBucketedTokenizer:
    """The reranker pads every batch to a multiple of PAD_TO_MULTIPLE_OF, so the
    model sees a bounded set of input shapes (MPS keeps memory per shape)."""

    def test_forces_pad_to_multiple_of_on_every_call(self):
        from unittest.mock import MagicMock

        from main.indexes.reranking.cross_encoder_reranker import (
            PAD_TO_MULTIPLE_OF,
            _BucketedTokenizer,
        )

        inner = MagicMock()
        wrapped = _BucketedTokenizer(inner, PAD_TO_MULTIPLE_OF)
        wrapped([("q", "doc")], padding=True, truncation=True, return_tensors="pt")

        assert inner.call_args.kwargs["pad_to_multiple_of"] == PAD_TO_MULTIPLE_OF
        assert inner.call_args.kwargs["padding"] is True

    def test_delegates_attributes(self):
        from unittest.mock import MagicMock

        from main.indexes.reranking.cross_encoder_reranker import _BucketedTokenizer

        inner = MagicMock()
        inner.model_max_length = 512
        assert _BucketedTokenizer(inner, 128).model_max_length == 512

    def test_constructor_installs_wrapper(self):
        from unittest.mock import MagicMock, patch

        from main.indexes.reranking import cross_encoder_reranker as mod

        fake_model = MagicMock()
        original_tokenizer = fake_model.tokenizer
        with patch.object(mod, "CrossEncoder", return_value=fake_model):
            reranker = mod.CrossEncoderReranker("any-model")

        assert isinstance(reranker.model.tokenizer, mod._BucketedTokenizer)
        reranker.model.tokenizer(["x"], padding=True)
        assert original_tokenizer.call_args.kwargs["pad_to_multiple_of"] == mod.PAD_TO_MULTIPLE_OF
