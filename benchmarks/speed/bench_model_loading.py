"""Benchmark: model and collection loading times."""

import time

from benchmarks.results import BenchmarkResult
from main.indexes.indexer_factory import create_embedder, create_reranker, detect_faiss_index, load_search_indexer
from main.persisters.disk_persister import DiskPersister


def bench_embedder_load() -> BenchmarkResult:
    """Time SentenceEmbedder cold load (from HF cache)."""
    t0 = time.monotonic()
    embedder = create_embedder("indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base")
    duration = (time.monotonic() - t0) * 1000

    return BenchmarkResult(
        name="embedder_load",
        category="speed",
        metrics={"load_ms": duration, "dimensions": embedder.get_number_of_dimensions()},
        duration_ms=duration,
        metadata={"model": embedder.model_name},
    )


def bench_reranker_load() -> BenchmarkResult:
    """Time CrossEncoderReranker cold load (from HF cache)."""
    t0 = time.monotonic()
    reranker = create_reranker()
    duration = (time.monotonic() - t0) * 1000

    return BenchmarkResult(
        name="reranker_load",
        category="speed",
        metrics={"load_ms": duration},
        duration_ms=duration,
        metadata={"model": reranker.model_name},
    )


def bench_collection_load(persister: DiskPersister, collection_name: str) -> BenchmarkResult:
    """Time loading a collection's indexes from disk."""
    t0 = time.monotonic()
    indexer = load_search_indexer(collection_name, persister)
    duration = (time.monotonic() - t0) * 1000

    return BenchmarkResult(
        name=f"collection_load_{collection_name}",
        category="speed",
        metrics={"load_ms": duration, "index_size": indexer.get_size()},
        duration_ms=duration,
        metadata={"collection": collection_name},
    )
