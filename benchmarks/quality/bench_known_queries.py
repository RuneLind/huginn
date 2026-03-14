"""Benchmark: curated known-answer queries.

Runs a set of manually curated queries with expected top results.
These serve as regression tests for search quality.
"""

import json
import time

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult


def _find_query_file(ctx: BenchmarkContext, prefix: str, collection_name: str):
    """Search data dirs for a query file, trying collection name and base name."""
    for name in [collection_name, collection_name.rsplit("-v", 1)[0] if "-v" in collection_name else None]:
        if name:
            path = ctx.find_data_file(f"{prefix}_{name}.json")
            if path:
                return path
    return None


def bench_known_queries(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Run curated query set and check if expected docs appear in top-k.

    Searches all data directories for known_queries_{collection}.json.
    """
    query_file = _find_query_file(ctx, "known_queries", collection_name)
    if not query_file:
        return BenchmarkResult(
            name=f"known_queries_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": f"No query file found for {collection_name}"},
        )

    data = json.loads(query_file.read_text())
    queries = data["queries"]
    searcher = ctx.get_searcher(collection_name)

    t_start = time.monotonic()

    hits = 0
    reciprocal_ranks = []
    query_details = []

    for entry in queries:
        query = entry["query"]
        expected_ids = entry["expected_doc_ids"]
        top_k = entry.get("expected_in_top_k", 5)

        results = searcher.search(
            query,
            max_number_of_chunks=max(top_k * 3, 15),
            skip_reranker=False,
        )

        result_ids = [r["id"] for r in results.get("results", [])]

        # Find best rank of any expected document
        best_rank = None
        for expected_id in expected_ids:
            for rank, rid in enumerate(result_ids, 1):
                # Support partial matching (expected_id is a substring of rid)
                if expected_id in rid:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                    break

        hit = best_rank is not None and best_rank <= top_k
        if hit:
            hits += 1
        reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)

        query_details.append({
            "query": query,
            "hit": hit,
            "best_rank": best_rank,
            "expected_in_top_k": top_k,
            "top_3_results": [r["id"][:80] for r in results.get("results", [])[:3]],
        })

    total_duration = (time.monotonic() - t_start) * 1000
    n = len(queries)

    metrics = {
        "hit_rate": hits / n if n else 0,
        "mrr": sum(reciprocal_ranks) / n if n else 0,
        "total_queries": n,
        "hits": hits,
        "misses": n - hits,
    }

    return BenchmarkResult(
        name=f"known_queries_{collection_name}",
        category="quality",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name, "query_details": query_details},
    )
