import pytest

from main.indexes.indexer_factory import (
    create_embedder,
    create_indexer,
    detect_faiss_index,
    load_search_indexer,
    FAISS_INDEX_PREFERENCE,
)
from main.indexes.indexers.bm25_indexer import BM25Indexer


class FakePersister:
    """Minimal persister stub that reports which paths exist."""

    def __init__(self, existing_paths=None, files=None):
        self._existing = set(existing_paths or [])
        self._files = files or {}

    def is_path_exists(self, path):
        return path in self._existing

    def read_bin_file(self, path):
        return self._files.get(path)


class TestCreateEmbedder:
    def test_multilingual_e5(self):
        embedder = create_embedder("indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base")
        assert embedder is not None
        assert "e5" in embedder.model_name
        assert embedder.query_prefix == "query: "
        assert embedder.passage_prefix == "passage: "

    def test_minilm(self):
        embedder = create_embedder("indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2")
        assert embedder is not None
        assert "MiniLM" in embedder.model_name
        assert embedder.query_prefix == ""

    def test_mpnet(self):
        embedder = create_embedder("indexer_FAISS_IndexFlatL2__embeddings_all-mpnet-base-v2")
        assert embedder is not None

    def test_unknown_returns_none(self):
        assert create_embedder("unknown_model_xyz") is None


class TestCreateIndexer:
    def test_bm25(self):
        indexer = create_indexer("indexer_BM25")
        assert isinstance(indexer, BM25Indexer)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown indexer"):
            create_indexer("totally_unknown_indexer")


class TestDetectFaissIndex:
    def test_prefers_multilingual(self):
        persister = FakePersister(existing_paths={
            "my-col/indexes/indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base/indexer",
            "my-col/indexes/indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2/indexer",
        })
        result = detect_faiss_index("my-col", persister)
        assert "multilingual-e5-base" in result

    def test_falls_back_to_minilm(self):
        persister = FakePersister(existing_paths={
            "my-col/indexes/indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2/indexer",
        })
        result = detect_faiss_index("my-col", persister)
        assert "MiniLM" in result

    def test_raises_when_nothing_found(self):
        persister = FakePersister()
        with pytest.raises(ValueError, match="No FAISS index found"):
            detect_faiss_index("missing-col", persister)


class TestLoadSearchIndexer:
    def test_faiss_only_when_no_bm25(self):
        e5_path = "col/indexes/indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base/indexer"
        bm25_path = "col/indexes/indexer_BM25/indexer"

        # Only FAISS exists, no BM25
        persister = FakePersister(
            existing_paths={e5_path},
        )

        # We need a real embedder to construct the FaissIndexer
        embedder = create_embedder("indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base")

        indexer = load_search_indexer(
            "col",
            persister,
            faiss_index_name="indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base",
            shared_embedder=embedder,
        )

        # Should be plain FAISS, not hybrid
        assert indexer.get_name() != "hybrid_FAISS_BM25"

    def test_hybrid_when_bm25_exists(self):
        e5_name = "indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base"
        e5_path = f"col/indexes/{e5_name}/indexer"
        bm25_path = "col/indexes/indexer_BM25/indexer"

        # Create a serialized BM25 state
        bm25 = BM25Indexer()
        bm25.index_texts([0], ["test document"])
        bm25_state = bm25.serialize()

        persister = FakePersister(
            existing_paths={e5_path, bm25_path},
            files={bm25_path: bm25_state},
        )

        embedder = create_embedder(e5_name)
        indexer = load_search_indexer(
            "col", persister, faiss_index_name=e5_name, shared_embedder=embedder
        )

        assert indexer.get_name() == "hybrid_FAISS_BM25"


class TestFaissIndexPreference:
    def test_preference_order(self):
        assert "multilingual-e5-base" in FAISS_INDEX_PREFERENCE[0]
        assert "MiniLM" in FAISS_INDEX_PREFERENCE[1]
