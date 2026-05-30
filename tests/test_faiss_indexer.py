"""Tests for FaissIndexer dimension guard + embedding metadata (Phase 4a / H15)."""

import pytest

from main.indexes.indexers.faiss_indexer import FaissIndexer


class FakeEmbedder:
    """Embedder stub with a controllable dimension — avoids loading a real model."""

    def __init__(self, dim, model_name="fake/model"):
        self._dim = dim
        self.model_name = model_name

    def get_number_of_dimensions(self):
        return self._dim


class TestDimensionGuard:
    def test_same_dimension_reloads(self):
        blob = FaissIndexer("ix", FakeEmbedder(8)).serialize()
        reloaded = FaissIndexer("ix", FakeEmbedder(8), blob)
        assert reloaded.get_size() == 0

    def test_mismatched_dimension_raises_with_actionable_message(self):
        # Persist a 4-dim index, then load it with an 8-dim embedder.
        blob = FaissIndexer("ix", FakeEmbedder(4)).serialize()
        with pytest.raises(ValueError, match="dimension mismatch"):
            FaissIndexer("ix", FakeEmbedder(8, model_name="intfloat/multilingual-e5-base"), blob)

    def test_mismatch_message_names_model_and_dimensions(self):
        blob = FaissIndexer("ix", FakeEmbedder(4)).serialize()
        with pytest.raises(ValueError) as exc:
            FaissIndexer("ix", FakeEmbedder(768, model_name="the-model"), blob)
        msg = str(exc.value)
        assert "the-model" in msg and "4" in msg and "768" in msg


class TestEmbeddingMetadata:
    def test_metadata_carries_model_dimension_and_versions(self):
        idx = FaissIndexer("ix", FakeEmbedder(384, model_name="intfloat/multilingual-e5-base"))
        meta = idx.get_embedding_metadata()
        assert meta["model"] == "intfloat/multilingual-e5-base"
        assert meta["dimension"] == 384
        assert "faissVersion" in meta  # records the library that wrote the index
