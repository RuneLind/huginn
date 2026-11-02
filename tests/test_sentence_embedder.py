from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from main.indexes.bucketed_tokenizer import PAD_TO_MULTIPLE_OF, BucketedTokenizer
from main.indexes.embeddings import sentence_embeder as mod

E5 = "intfloat/multilingual-e5-base"
BUCKETS = {128, 256, 384, 512}


def _fake_embedder(device="mps", **kwargs):
    """An embedder whose encode returns one row per text: the row's input index."""
    fake_model = MagicMock()
    fake_model.device.type = device
    fake_model.encode.side_effect = lambda texts, **_: np.arange(len(texts)).reshape(-1, 1)
    with patch.object(mod, "SentenceTransformer", return_value=fake_model):
        return mod.SentenceEmbedder("any-model", **kwargs), fake_model


class TestEmbedPassages:
    def test_fills_the_last_batch_and_returns_only_the_real_rows(self):
        embedder, model = _fake_embedder(passage_prefix="passage: ")

        vectors = embedder.embed_passages(["a", "b", "c"])

        encoded = model.encode.call_args.args[0]
        assert len(encoded) == mod.PASSAGE_BATCH_SIZE
        assert encoded[:3] == ["passage: a", "passage: b", "passage: c"]
        assert set(encoded[3:]) == {""}
        assert model.encode.call_args.kwargs["batch_size"] == mod.PASSAGE_BATCH_SIZE
        assert vectors.tolist() == [[0], [1], [2]]

    def test_a_whole_number_of_batches_gets_no_fill(self):
        embedder, model = _fake_embedder()

        vectors = embedder.embed_passages(["x"] * mod.PASSAGE_BATCH_SIZE * 2)

        assert "" not in model.encode.call_args.args[0]
        assert len(vectors) == mod.PASSAGE_BATCH_SIZE * 2

    def test_no_texts_returns_no_rows(self):
        embedder, model = _fake_embedder()

        assert len(embedder.embed_passages([])) == 0
        assert model.encode.call_args.args[0] == []


def test_constructor_installs_the_bucketed_tokenizer():
    embedder, model = _fake_embedder()

    assert isinstance(model.tokenizer, BucketedTokenizer)
    assert embedder.model.tokenizer._multiple == PAD_TO_MULTIPLE_OF


def test_a_cpu_model_gets_no_padding_and_no_fill():
    """Only the Mac GPU keeps memory per shape; on the CPU padding only costs time."""
    embedder, model = _fake_embedder(device="cpu", passage_prefix="passage: ")

    vectors = embedder.embed_passages(["a", "b", "c"])

    assert not isinstance(model.tokenizer, BucketedTokenizer)
    assert model.encode.call_args.args[0] == ["passage: a", "passage: b", "passage: c"]
    assert vectors.tolist() == [[0], [1], [2]]


def test_embed_query_applies_the_query_prefix():
    embedder, model = _fake_embedder(query_prefix="query: ", passage_prefix="passage: ")

    embedder.embed_query("x")

    assert model.encode.call_args.args[0] == "query: x"


def test_benchmark_sizes_need_no_fill():
    """A filled batch encodes rows the benchmark's per-second figure does not count."""
    from benchmarks.speed.bench_indexing import bench_embedding_throughput

    ctx = MagicMock()
    ctx._embedder.model_name = "any-model"

    bench_embedding_throughput(ctx)

    sizes = {len(call.args[0]) for call in ctx._embedder.embed_passages.call_args_list}
    assert sizes and all(size % mod.PASSAGE_BATCH_SIZE == 0 for size in sizes)


@pytest.fixture(scope="module")
def real_embedder():
    """The real e5 model on the CPU with the Mac GPU's shape bound on, so the test holds no GPU memory."""
    real_class = mod.SentenceTransformer
    try:
        with (
            patch.object(mod, "SentenceTransformer", lambda name: real_class(name, device="cpu")),
            patch.object(mod, "_needs_bounded_shapes", lambda _model: True),
        ):
            return mod.SentenceEmbedder(E5, query_prefix="query: ", passage_prefix="passage: ")
    except OSError:
        pytest.skip("e5 model not in the local HF cache")


def _record_input_shapes(embedder):
    shapes = []

    def hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        shapes.append(tuple(input_ids.shape))

    return shapes, embedder.model[0].auto_model.register_forward_pre_hook(hook, with_kwargs=True)


def test_every_passage_batch_has_a_bounded_shape(real_embedder):
    """The property the padding exists for, through sentence-transformers' own batching."""
    shapes, handle = _record_input_shapes(real_embedder)
    try:
        for words in (1, 7, 40, 90, 200):
            count = mod.PASSAGE_BATCH_SIZE + words
            real_embedder.embed_passages([f"word {i} " * (1 + (words * i) // count) for i in range(count)])
    finally:
        handle.remove()

    assert len(shapes) >= 10
    assert {size for size, _ in shapes} == {mod.PASSAGE_BATCH_SIZE}
    assert {length for _, length in shapes} <= BUCKETS
    assert {128, 256} <= {length for _, length in shapes}


def test_query_lengths_land_on_a_bucket(real_embedder):
    shapes, handle = _record_input_shapes(real_embedder)
    try:
        for words in (1, 12, 60):
            real_embedder.embed_query("word " * words)
    finally:
        handle.remove()

    assert {size for size, _ in shapes} == {1}
    assert {length for _, length in shapes} <= BUCKETS


def test_padding_and_fill_leave_the_vectors_unchanged(real_embedder):
    """An index built before this change stays valid: rows match the unpadded model."""
    texts = [f"trygdeavgift og medlemskap {i} " * (1 + i * 3) for i in range(5)]
    prefixed = ["passage: " + t for t in texts]
    wrapper = real_embedder.model.tokenizer

    real_embedder.model.tokenizer = wrapper._tokenizer
    try:
        unpadded = real_embedder.model.encode(prefixed)
        unpadded_query = real_embedder.model.encode("query: " + texts[0])
    finally:
        real_embedder.model.tokenizer = wrapper

    np.testing.assert_allclose(real_embedder.embed_passages(texts), unpadded, atol=1e-5)
    np.testing.assert_allclose(real_embedder.embed_query(texts[0]), unpadded_query, atol=1e-5)
