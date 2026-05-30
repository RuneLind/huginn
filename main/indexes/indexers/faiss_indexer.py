import faiss
import numpy as np


class FaissIndexer:
    def __init__(self, name, embedder, serialized_index=None):
        self.name = name
        self.embedder = embedder
        if serialized_index is not None:
            self.faiss_index = faiss.deserialize_index(serialized_index)
            self.__assert_dimension_matches_embedder()
        else:
            self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedder.get_number_of_dimensions()))

    def __assert_dimension_matches_embedder(self):
        """Fail loudly if the embedding model changed under a persisted index.

        Mismatched vector dimensions would otherwise surface as an opaque crash
        deep in FAISS C++ at query time; catching it at load names the model and
        the fix (H15).
        """
        persisted_dim = self.faiss_index.d
        embedder_dim = self.embedder.get_number_of_dimensions()
        if persisted_dim != embedder_dim:
            raise ValueError(
                f"Index dimension mismatch for '{self.name}': the persisted FAISS index "
                f"has dimension {persisted_dim}, but embedder '{self.embedder.model_name}' "
                f"produces {embedder_dim}-dim vectors. The embedding model changed under "
                f"the index — re-index the collection."
            )

    def get_name(self):
        return self.name

    def get_embedding_metadata(self):
        """Identity of the embedding model + library versions, recorded in the
        manifest so a future load can detect a model/version drift (H15)."""
        from importlib.metadata import version, PackageNotFoundError
        meta = {
            "model": self.embedder.model_name,
            "dimension": self.embedder.get_number_of_dimensions(),
            "faissVersion": faiss.__version__,
        }
        try:
            meta["sentenceTransformersVersion"] = version("sentence-transformers")
        except PackageNotFoundError:
            pass
        return meta

    def index_texts(self, ids, texts):
        self.faiss_index.add_with_ids(self.embedder.embed_passages(texts), np.array(ids, dtype=np.int64))

    def remove_ids(self, ids):
        self.faiss_index.remove_ids(ids)

    def serialize(self):
        return faiss.serialize_index(self.faiss_index)

    def search(self, text, number_of_results=10):
        return self.faiss_index.search(np.expand_dims(self.embedder.embed_query(text), axis=0), number_of_results)
    
    def get_size(self):
        return self.faiss_index.ntotal
