"""Benchmark: search latency measurements."""

import statistics
import time

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult

STANDARD_QUERIES = [
    "artikkel 12 utsendte arbeidstakere",
    "how to deploy to production",
    "pliktig medlemskap folketrygdloven",
    "onboarding nye utviklere",
    "feilhåndtering integrasjon",
]


def _measure_search(searcher, query: str, skip_reranker: bool = False, max_chunks: int = 15) -> float:
    """Run a search and return wall-clock milliseconds."""
    t0 = time.monotonic()
    searcher.search(query, max_number_of_chunks=max_chunks, skip_reranker=skip_reranker)
    return (time.monotonic() - t0) * 1000


def bench_search_latency(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Measure search latency with and without reranker.

    Runs each query 3 times, discards first (warmup), uses last 2 for stats.
    """
    searcher = ctx.get_searcher(collection_name)
    times_with = []
    times_without = []

    t_start = time.monotonic()

    for query in STANDARD_QUERIES:
        # Warmup
        _measure_search(searcher, query, skip_reranker=False)
        _measure_search(searcher, query, skip_reranker=True)

        # Timed runs
        for _ in range(2):
            times_with.append(_measure_search(searcher, query, skip_reranker=False))
            times_without.append(_measure_search(searcher, query, skip_reranker=True))

    total_duration = (time.monotonic() - t_start) * 1000

    metrics = {
        "with_reranker_p50_ms": statistics.median(times_with),
        "with_reranker_p90_ms": _percentile(times_with, 0.9),
        "with_reranker_mean_ms": statistics.mean(times_with),
        "without_reranker_p50_ms": statistics.median(times_without),
        "without_reranker_p90_ms": _percentile(times_without, 0.9),
        "without_reranker_mean_ms": statistics.mean(times_without),
        "reranker_overhead_ms": statistics.mean(times_with) - statistics.mean(times_without),
    }

    return BenchmarkResult(
        name=f"search_latency_{collection_name}",
        category="speed",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name, "queries": len(STANDARD_QUERIES), "runs_per_query": 2},
    )


def bench_search_scaling(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Measure latency vs max_number_of_chunks."""
    searcher = ctx.get_searcher(collection_name)
    query = "artikkel 12 utsendte arbeidstakere"
    chunk_counts = [5, 10, 15, 30, 50]

    t_start = time.monotonic()
    metrics = {}

    for k in chunk_counts:
        # Warmup
        _measure_search(searcher, query, skip_reranker=False, max_chunks=k)
        # Measure
        times = [_measure_search(searcher, query, skip_reranker=False, max_chunks=k) for _ in range(3)]
        metrics[f"chunks_{k}_ms"] = statistics.median(times)

    total_duration = (time.monotonic() - t_start) * 1000

    return BenchmarkResult(
        name=f"search_scaling_{collection_name}",
        category="speed",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name, "query": query},
    )


def bench_retriever_breakdown(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Compare individual retriever latency: FAISS vs BM25 vs Hybrid."""
    searcher = ctx.get_searcher(collection_name)
    indexer = searcher.indexer
    query = "artikkel 12 utsendte arbeidstakere"

    t_start = time.monotonic()
    metrics = {}

    # Hybrid search (default)
    times = [_measure_search(searcher, query, skip_reranker=True) for _ in range(5)]
    metrics["hybrid_p50_ms"] = statistics.median(times)

    # Direct FAISS and BM25 access (if hybrid indexer)
    if hasattr(indexer, 'faiss_indexer') and hasattr(indexer, 'bm25_indexer'):
        faiss_times = []
        bm25_times = []
        for _ in range(5):
            t0 = time.monotonic()
            indexer.faiss_indexer.search(query, 15)
            faiss_times.append((time.monotonic() - t0) * 1000)

            t0 = time.monotonic()
            indexer.bm25_indexer.search(query, 15)
            bm25_times.append((time.monotonic() - t0) * 1000)

        metrics["faiss_p50_ms"] = statistics.median(faiss_times)
        metrics["bm25_p50_ms"] = statistics.median(bm25_times)
        metrics["rrf_overhead_ms"] = metrics["hybrid_p50_ms"] - max(metrics["faiss_p50_ms"], metrics["bm25_p50_ms"])

    total_duration = (time.monotonic() - t_start) * 1000

    return BenchmarkResult(
        name=f"retriever_breakdown_{collection_name}",
        category="speed",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name},
    )


def _percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]
