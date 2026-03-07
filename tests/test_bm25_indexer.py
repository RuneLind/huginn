import numpy as np

from main.indexes.indexers.bm25_indexer import BM25Indexer, _tokenize


# BM25Okapi IDF formula: log((N - n + 0.5) / (n + 0.5))
# With N=2, n=1: log(1.5/1.5) = 0 → need 3+ docs for non-zero scores.


def _make_indexer(*texts):
    """Create a BM25Indexer with the given texts indexed."""
    indexer = BM25Indexer()
    indexer.index_texts(list(range(len(texts))), list(texts))
    return indexer


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert _tokenize("hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers_kept(self):
        assert _tokenize("item 42") == ["item", "42"]


class TestIndexAndSearch:
    def test_finds_matching_document(self):
        indexer = _make_indexer(
            "python programming language",
            "the cat sat on the mat",
            "a fish swam in the sea",
        )
        scores, ids = indexer.search("python")
        assert 0 in ids[0].tolist()

    def test_multiple_documents_ranking(self):
        indexer = _make_indexer(
            "the cat sat on the mat",
            "python programming language",
            "the python snake is long",
        )
        scores, ids = indexer.search("python")
        returned_ids = set(ids[0].tolist())
        assert 1 in returned_ids
        assert 2 in returned_ids
        assert 0 not in returned_ids

    def test_no_match_returns_empty(self):
        indexer = _make_indexer("hello world", "good morning", "nice day")
        scores, ids = indexer.search("python")
        assert ids.shape == (1, 0) or len(ids[0]) == 0

    def test_empty_index_returns_empty(self):
        indexer = BM25Indexer()
        scores, ids = indexer.search("anything")
        assert ids.shape == (1, 0) or len(ids[0]) == 0

    def test_scores_are_negated(self):
        """BM25 scores are negated so lower = better (consistent with L2 convention)."""
        indexer = _make_indexer(
            "python is great",
            "the weather is nice",
            "a car drove past",
        )
        scores, ids = indexer.search("python")
        assert scores[0][0] < 0

    def test_number_of_results_limits_output(self):
        indexer = _make_indexer(
            *[f"document about topic {i}" for i in range(10)]
        )
        scores, ids = indexer.search("document topic", number_of_results=3)
        assert len(ids[0]) <= 3

    def test_incremental_indexing(self):
        indexer = BM25Indexer()
        indexer.index_texts([0], ["first document about cats"])
        indexer.index_texts([1], ["second document about python"])
        indexer.index_texts([2], ["third document about birds"])
        assert indexer.get_size() == 3
        scores, ids = indexer.search("python")
        assert 1 in ids[0].tolist()


class TestRemoveIds:
    def test_remove_single(self):
        indexer = _make_indexer("aaa bbb", "ccc ddd", "eee fff")
        indexer.remove_ids([1])
        assert indexer.get_size() == 2

    def test_remove_all(self):
        indexer = _make_indexer("aaa", "bbb")
        indexer.remove_ids([0, 1])
        assert indexer.get_size() == 0

    def test_remove_nonexistent_noop(self):
        indexer = _make_indexer("hello")
        indexer.remove_ids([99])
        assert indexer.get_size() == 1


class TestSerializeDeserialize:
    def test_roundtrip(self):
        indexer = _make_indexer(
            "hello world about cats",
            "python code example",
            "java spring boot",
        )

        state = indexer.serialize()
        restored = BM25Indexer(serialized_state=state)

        assert restored.get_size() == 3
        scores, ids = restored.search("python")
        assert 1 in ids[0].tolist()

    def test_empty_roundtrip(self):
        indexer = BM25Indexer()
        state = indexer.serialize()
        restored = BM25Indexer(serialized_state=state)
        assert restored.get_size() == 0


class TestGetSize:
    def test_empty(self):
        assert BM25Indexer().get_size() == 0

    def test_after_indexing(self):
        indexer = _make_indexer("a", "b", "c")
        assert indexer.get_size() == 3


class TestGetName:
    def test_default(self):
        assert BM25Indexer().get_name() == "indexer_BM25"

    def test_custom(self):
        assert BM25Indexer(name="custom").get_name() == "custom"
