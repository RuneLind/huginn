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


class _PlainTokenizer:
    """Picklable stand-in: records the kwargs of its last call."""

    def __init__(self):
        self.model_max_length = 512
        self.last_kwargs = None

    def __call__(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return kwargs

    def __deepcopy__(self, memo):
        # Real tokenizers can define this; the wrapper must not forward to it.
        return _PlainTokenizer()


class TestBucketedTokenizerAsAttributeHolder:
    def test_attribute_writes_reach_the_inner_tokenizer(self):
        # CrossEncoder.max_length's setter writes tokenizer.model_max_length.
        from main.indexes.reranking.cross_encoder_reranker import _BucketedTokenizer

        inner = _PlainTokenizer()
        wrapped = _BucketedTokenizer(inner, 128)
        wrapped.model_max_length = 256

        assert inner.model_max_length == 256

    def test_deepcopy_and_pickle_round_trip(self):
        import copy
        import pickle

        from main.indexes.reranking.cross_encoder_reranker import _BucketedTokenizer

        wrapped = _BucketedTokenizer(_PlainTokenizer(), 128)
        for clone in (copy.deepcopy(wrapped), pickle.loads(pickle.dumps(wrapped))):
            assert isinstance(clone, _BucketedTokenizer)
            clone(["x"], padding=True)
            assert clone._tokenizer.last_kwargs["pad_to_multiple_of"] == 128

    def test_overrides_a_caller_supplied_multiple(self):
        from main.indexes.reranking.cross_encoder_reranker import _BucketedTokenizer

        inner = _PlainTokenizer()
        _BucketedTokenizer(inner, 128)(["x"], padding=True, pad_to_multiple_of=8)

        assert inner.last_kwargs["pad_to_multiple_of"] == 128


def test_real_tokenizer_lands_every_batch_on_a_bucket():
    """The property the wrapper exists for: token lengths are 128, 256, 384 or 512."""
    import pytest

    transformers = pytest.importorskip("transformers")
    from main.indexes.reranking.cross_encoder_reranker import _BucketedTokenizer

    try:
        inner = transformers.AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-v2-m3", local_files_only=True
        )
    except OSError:
        pytest.skip("bge-reranker-v2-m3 tokenizer not in the local HF cache")
    inner.model_max_length = 512
    wrapped = _BucketedTokenizer(inner, 128)

    lengths = set()
    for words in (1, 40, 150, 300, 2000):
        batch = [("query", "word " * words), ("query", "word")]
        features = wrapped(batch, padding=True, truncation=True, return_tensors="pt")
        lengths.add(features["input_ids"].shape[1])

    assert lengths <= {128, 256, 384, 512}
    assert len(lengths) > 1
