"""Benchmark: embedding throughput and indexing speed."""

import time
import statistics

from benchmarks.context import BenchmarkContext, load_documents_for_collection
from benchmarks.results import BenchmarkResult
from main.indexes.indexers.faiss_indexer import FaissIndexer
from main.indexes.indexers.bm25_indexer import BM25Indexer


def bench_embedding_throughput(ctx: BenchmarkContext) -> BenchmarkResult:
    """Measure raw embedding speed at different batch sizes."""
    embedder = ctx._embedder
    sample_texts = [
        "artikkel 12 utsendte arbeidstakere lovvalg",
        "Pliktig medlemskap i folketrygden for yrkesaktive personer",
        "Søknad om unntak fra lovvalgsreglene i EØS-avtalen",
        "Behandling av saker om trygdekoordinering i henhold til forordning 883/2004",
        "Onboarding for nye utviklere i team MELOSYS med tilganger og oppsett",
    ] * 20  # 100 texts

    t_start = time.monotonic()
    metrics = {}

    for batch_size in [10, 50, 100]:
        batch = sample_texts[:batch_size]
        times = []
        for _ in range(3):
            t0 = time.monotonic()
            embedder.embed_passages(batch)
            times.append((time.monotonic() - t0) * 1000)

        median_ms = statistics.median(times)
        metrics[f"batch_{batch_size}_ms"] = median_ms
        metrics[f"batch_{batch_size}_per_second"] = batch_size / (median_ms / 1000)

    total_duration = (time.monotonic() - t_start) * 1000

    return BenchmarkResult(
        name="embedding_throughput",
        category="speed",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"model": embedder.model_name},
    )


def bench_indexing_speed(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Measure FAISS + BM25 indexing speed using real chunks from a collection.

    Creates fresh indexes (no side effects on existing data).
    """
    documents = load_documents_for_collection(ctx.persister, collection_name)

    # Extract chunks from documents
    all_chunks = []
    for doc in documents:
        for i, chunk in enumerate(doc.get("chunks", [])):
            text = chunk.get("indexedData", str(chunk)) if isinstance(chunk, dict) else str(chunk)
            if text:
                all_chunks.append(text)

    t_start = time.monotonic()
    metrics = {"total_chunks_available": len(all_chunks)}

    for n in [100, 500, min(1000, len(all_chunks))]:
        if n > len(all_chunks):
            continue
        chunk_subset = all_chunks[:n]
        ids = list(range(n))

        # FAISS indexing
        faiss_indexer = FaissIndexer(
            "bench_faiss",
            ctx._embedder,
        )
        t0 = time.monotonic()
        faiss_indexer.index_texts(ids, chunk_subset)
        faiss_ms = (time.monotonic() - t0) * 1000
        metrics[f"faiss_{n}_chunks_ms"] = faiss_ms

        # BM25 indexing
        bm25_indexer = BM25Indexer("bench_bm25")
        t0 = time.monotonic()
        bm25_indexer.index_texts(ids, chunk_subset)
        bm25_ms = (time.monotonic() - t0) * 1000
        metrics[f"bm25_{n}_chunks_ms"] = bm25_ms

        metrics[f"total_{n}_chunks_ms"] = faiss_ms + bm25_ms
        metrics[f"chunks_per_second_{n}"] = n / ((faiss_ms + bm25_ms) / 1000)

    total_duration = (time.monotonic() - t_start) * 1000

    return BenchmarkResult(
        name=f"indexing_speed_{collection_name}",
        category="speed",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name},
    )
