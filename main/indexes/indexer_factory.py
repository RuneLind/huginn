import logging

from .indexers.faiss_indexer import FaissIndexer
from .indexers.bm25_indexer import BM25Indexer
from .indexers.hybrid_search_indexer import HybridSearchIndexer
from .embeddings.sentence_embeder import SentenceEmbedder
from .reranking.cross_encoder_reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

FAISS_INDEX_PREFERENCE = [
    "indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base",
    "indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2",
    "indexer_FAISS_IndexFlatL2__embeddings_all-mpnet-base-v2",
]

def create_embedder(indexer_name):
    """Create the appropriate SentenceEmbedder for a given indexer name."""
    if "multilingual-e5-base" in indexer_name:
        return SentenceEmbedder(
            model_name="intfloat/multilingual-e5-base",
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
    if "all-MiniLM-L6-v2" in indexer_name:
        return SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if "all-mpnet-base-v2" in indexer_name:
        return SentenceEmbedder(model_name="sentence-transformers/all-mpnet-base-v2")
    if "multi-qa-distilbert-cos-v1" in indexer_name:
        return SentenceEmbedder(model_name="sentence-transformers/multi-qa-distilbert-cos-v1")
    return None

def create_indexer(indexer_name):
    if indexer_name == "indexer_BM25":
        return BM25Indexer(indexer_name)

    embedder = create_embedder(indexer_name)
    if embedder:
        return FaissIndexer(indexer_name, embedder)

    raise ValueError(f"Unknown indexer name: {indexer_name}")

def load_indexer(indexer_name, collection_name, persister):
    if indexer_name == "indexer_BM25":
        serialized_state = persister.read_bin_file(f"{collection_name}/indexes/{indexer_name}/indexer")
        return BM25Indexer(indexer_name, serialized_state=serialized_state)

    embedder = create_embedder(indexer_name)
    if embedder:
        serialized_index = persister.read_bin_file(f"{collection_name}/indexes/{indexer_name}/indexer")
        return FaissIndexer(indexer_name, embedder, serialized_index)

    raise ValueError(f"Unknown indexer name: {indexer_name}")

def detect_faiss_index(collection_name, persister):
    """Auto-detect which FAISS index exists on disk, preferring newer models."""
    for name in FAISS_INDEX_PREFERENCE:
        if persister.is_path_exists(f"{collection_name}/indexes/{name}/indexer"):
            return name
    raise ValueError(f"No FAISS index found for collection {collection_name}")

def load_search_indexer(collection_name, persister, faiss_index_name=None, shared_embedder=None):
    """Load a search indexer, auto-detecting FAISS index and BM25 presence for hybrid search.

    If shared_embedder is provided, it is used instead of creating a new one (saves memory
    when loading multiple collections with the same model).
    """
    if faiss_index_name is None:
        faiss_index_name = detect_faiss_index(collection_name, persister)

    if shared_embedder:
        serialized_index = persister.read_bin_file(f"{collection_name}/indexes/{faiss_index_name}/indexer")
        faiss_indexer = FaissIndexer(faiss_index_name, shared_embedder, serialized_index)
    else:
        faiss_indexer = load_indexer(faiss_index_name, collection_name, persister)

    bm25_path = f"{collection_name}/indexes/indexer_BM25/indexer"
    if persister.is_path_exists(bm25_path):
        bm25_indexer = BM25Indexer("indexer_BM25", serialized_state=persister.read_bin_file(bm25_path))
        return HybridSearchIndexer(faiss_indexer, bm25_indexer)

    return faiss_indexer


def create_reranker(model_name=None):
    """Create a cross-encoder reranker instance."""
    if model_name is None:
        model_name = "BAAI/bge-reranker-v2-m3"
    return CrossEncoderReranker(model_name=model_name)
