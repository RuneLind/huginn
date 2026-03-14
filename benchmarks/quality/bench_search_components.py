"""Benchmark: compare search quality across pipeline stages.

Tests the value contribution of each component:
  FAISS-only vs BM25-only vs Hybrid vs Hybrid+Reranker
"""

import random
import time

from benchmarks.context import BenchmarkContext, load_documents_for_collection
from benchmarks.quality.bench_self_retrieval import _extract_title
from benchmarks.results import BenchmarkResult


def bench_component_comparison(ctx: BenchmarkContext, collection_name: str, sample_size: int = 30) -> BenchmarkResult:
    """Compare recall@5 across pipeline configurations using self-retrieval queries."""
    documents = load_documents_for_collection(ctx.persister, collection_name)
    searcher = ctx.get_searcher(collection_name)
    indexer = searcher.indexer

    # Sample documents
    rng = random.Random(42)
    sample = rng.sample(documents, min(sample_size, len(documents)))
    titled = [(doc, _extract_title(doc)) for doc in sample]
    titled = [(doc, title) for doc, title in titled if title and len(title) > 5]

    t_start = time.monotonic()

    has_hybrid = hasattr(indexer, 'faiss_indexer') and hasattr(indexer, 'bm25_indexer')

    # Track hits per configuration
    configs = {
        "hybrid_reranker": {"hits_at_5": 0, "mrr_sum": 0},
        "hybrid_no_reranker": {"hits_at_5": 0, "mrr_sum": 0},
    }
    if has_hybrid:
        configs["faiss_only"] = {"hits_at_5": 0, "mrr_sum": 0}
        configs["bm25_only"] = {"hits_at_5": 0, "mrr_sum": 0}

    for doc, title in titled:
        doc_id = doc.get("id", "")

        # Hybrid + Reranker
        results = searcher.search(title, max_number_of_chunks=30, skip_reranker=False)
        _score_results(configs["hybrid_reranker"], results, doc_id)

        # Hybrid, no reranker
        results = searcher.search(title, max_number_of_chunks=30, skip_reranker=True)
        _score_results(configs["hybrid_no_reranker"], results, doc_id)

        if has_hybrid:
            # FAISS only
            scores, indexes = indexer.faiss_indexer.search(title, 30)
            _score_raw(configs["faiss_only"], scores, indexes, doc_id, searcher)

            # BM25 only
            scores, indexes = indexer.bm25_indexer.search(title, 30)
            _score_raw(configs["bm25_only"], scores, indexes, doc_id, searcher)

    total_duration = (time.monotonic() - t_start) * 1000
    n = len(titled)

    metrics = {}
    for config_name, stats in configs.items():
        metrics[f"{config_name}_recall_at_5"] = stats["hits_at_5"] / n if n else 0
        metrics[f"{config_name}_mrr"] = stats["mrr_sum"] / n if n else 0

    metrics["sample_size"] = n

    # Calculate reranker lift
    if n:
        metrics["reranker_lift_recall_at_5"] = (
            metrics["hybrid_reranker_recall_at_5"] - metrics["hybrid_no_reranker_recall_at_5"]
        )
        if has_hybrid:
            metrics["bm25_lift_recall_at_5"] = (
                metrics["hybrid_no_reranker_recall_at_5"] - metrics["faiss_only_recall_at_5"]
            )

    return BenchmarkResult(
        name=f"component_comparison_{collection_name}",
        category="quality",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name},
    )


def _score_results(stats: dict, results: dict, doc_id: str):
    """Score a search result dict for a specific document."""
    for rank, result in enumerate(results.get("results", []), 1):
        if result["id"] == doc_id:
            if rank <= 5:
                stats["hits_at_5"] += 1
            stats["mrr_sum"] += 1.0 / rank
            return
    # Not found


def _score_raw(stats: dict, scores, indexes, doc_id: str, searcher):
    """Score raw indexer results by mapping chunk IDs back to documents."""
    import json
    mapping = searcher._load_mapping()
    seen_docs = []

    for chunk_id in indexes[0]:
        chunk_id_str = str(int(chunk_id))
        entry = mapping.get(chunk_id_str)
        if not entry:
            continue
        did = entry["documentId"]
        if did not in seen_docs:
            seen_docs.append(did)

    for rank, did in enumerate(seen_docs, 1):
        if did == doc_id:
            if rank <= 5:
                stats["hits_at_5"] += 1
            stats["mrr_sum"] += 1.0 / rank
            return
