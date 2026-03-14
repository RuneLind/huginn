"""Benchmark: realistic natural-language queries.

Tests with the kind of questions developers actually ask via MCP,
not just exact title lookups. Measures quality with and without
reranker to quantify the reranker's real-world value.
"""

import json
import time
from pathlib import Path

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_queries(collection_name: str) -> dict | None:
    """Load realistic query file for a collection."""
    for suffix in [collection_name, collection_name.rsplit("-v", 1)[0] if "-v" in collection_name else collection_name]:
        path = DATA_DIR / f"realistic_queries_{suffix}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def _evaluate(searcher, queries: list[dict], skip_reranker: bool) -> dict:
    """Run queries and compute hit rate + MRR."""
    hits = 0
    reciprocal_ranks = []
    details = []

    for entry in queries:
        query = entry["query"]
        expected_ids = entry["expected_doc_ids"]
        top_k = entry.get("expected_in_top_k", 5)

        results = searcher.search(
            query,
            max_number_of_chunks=max(top_k * 3, 15),
            skip_reranker=skip_reranker,
        )

        result_ids = [r["id"] for r in results.get("results", [])]

        # Find best rank of any expected document (partial match)
        best_rank = None
        matched_id = None
        for expected_id in expected_ids:
            for rank, rid in enumerate(result_ids, 1):
                if expected_id in rid:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        matched_id = rid
                    break

        hit = best_rank is not None and best_rank <= top_k
        if hit:
            hits += 1
        reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)

        details.append({
            "query": query,
            "hit": hit,
            "best_rank": best_rank,
            "matched": matched_id[:60] if matched_id else None,
            "top_3": [rid.rsplit("/", 1)[-1][:50] for rid in result_ids[:3]],
        })

    n = len(queries)
    return {
        "hit_rate": hits / n if n else 0,
        "mrr": sum(reciprocal_ranks) / n if n else 0,
        "hits": hits,
        "total": n,
        "details": details,
    }


def bench_realistic_queries(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Run realistic queries and measure quality with and without reranker.

    This benchmark uses natural-language queries (the kind developers ask via MCP)
    rather than exact title lookups. It directly measures the reranker's value.
    """
    data = _load_queries(collection_name)
    if not data:
        return BenchmarkResult(
            name=f"realistic_queries_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": f"No realistic query file for {collection_name}"},
        )

    queries = data["queries"]
    searcher = ctx.get_searcher(collection_name)

    t_start = time.monotonic()

    # With reranker
    with_rr = _evaluate(searcher, queries, skip_reranker=False)

    # Without reranker
    without_rr = _evaluate(searcher, queries, skip_reranker=True)

    total_duration = (time.monotonic() - t_start) * 1000

    metrics = {
        "with_reranker_hit_rate": with_rr["hit_rate"],
        "with_reranker_mrr": with_rr["mrr"],
        "without_reranker_hit_rate": without_rr["hit_rate"],
        "without_reranker_mrr": without_rr["mrr"],
        "reranker_hit_rate_lift": with_rr["hit_rate"] - without_rr["hit_rate"],
        "reranker_mrr_lift": with_rr["mrr"] - without_rr["mrr"],
        "total_queries": len(queries),
    }

    return BenchmarkResult(
        name=f"realistic_queries_{collection_name}",
        category="quality",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={
            "collection": collection_name,
            "with_reranker_details": with_rr["details"],
            "without_reranker_details": without_rr["details"],
        },
    )
