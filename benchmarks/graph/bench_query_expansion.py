"""Benchmark: knowledge graph query expansion effectiveness."""

import time

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult

# Queries containing entities, with expected documents that should rank better with expansion
EXPANSION_TEST_QUERIES = [
    "LA_BUC_01 søknad unntak",
    "A003 beslutning lovvalg",
    "artikkel 12 utsendte arbeidstakere",
    "artikkel 13 arbeid flere land",
]


def bench_query_expansion(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Measure whether graph expansion improves search ranking.

    For entity-rich queries, compares search results with original query
    vs query expanded with graph neighbor terms.
    """
    graph = ctx.graph
    if not graph:
        return BenchmarkResult(
            name=f"query_expansion_{collection_name}",
            category="graph",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "No knowledge graph loaded"},
        )

    searcher = ctx.get_searcher(collection_name)
    t_start = time.monotonic()

    original_results_count = []
    expanded_results_count = []
    expansion_term_counts = []
    rank_changes = []

    # Combine static + dynamic queries
    queries = list(EXPANSION_TEST_QUERIES)
    queries.extend(_dynamic_expansion_queries(graph))

    for query in queries:
        entities = graph.detect_entities(query)
        if not entities:
            continue

        expansion_terms = graph.get_expansion_terms(entities)
        expansion_term_counts.append(len(expansion_terms))

        # Search with original query
        original = searcher.search(query, max_number_of_chunks=15, skip_reranker=True)
        original_ids = [r["id"] for r in original.get("results", [])]
        original_results_count.append(len(original_ids))

        # Search with expanded query
        expanded_query = query + " " + " ".join(expansion_terms[:5])  # limit expansion
        expanded = searcher.search(expanded_query, max_number_of_chunks=15, skip_reranker=True)
        expanded_ids = [r["id"] for r in expanded.get("results", [])]
        expanded_results_count.append(len(expanded_ids))

        # Compare: how many results are shared, and do ranks shift?
        shared = set(original_ids[:5]) & set(expanded_ids[:5])
        rank_changes.append({
            "query": query,
            "entities": entities,
            "expansion_terms": len(expansion_terms),
            "original_top5": original_ids[:5],
            "expanded_top5": expanded_ids[:5],
            "shared_in_top5": len(shared),
        })

    total_duration = (time.monotonic() - t_start) * 1000

    metrics = {
        "queries_with_entities": len(rank_changes),
        "avg_expansion_terms": sum(expansion_term_counts) / len(expansion_term_counts) if expansion_term_counts else 0,
        "avg_original_results": sum(original_results_count) / len(original_results_count) if original_results_count else 0,
        "avg_expanded_results": sum(expanded_results_count) / len(expanded_results_count) if expanded_results_count else 0,
        "avg_shared_top5": sum(r["shared_in_top5"] for r in rank_changes) / len(rank_changes) if rank_changes else 0,
    }

    return BenchmarkResult(
        name=f"query_expansion_{collection_name}",
        category="graph",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name, "rank_changes": rank_changes},
    )


def _dynamic_expansion_queries(graph) -> list[str]:
    """Generate expansion queries from actual graph entities."""
    import random
    rng = random.Random(42)
    queries = []

    epics = [nid for nid in graph.nodes if graph.nodes[nid]["type"] == "Epic"]
    for epic_id in rng.sample(epics, min(3, len(epics))):
        key = epic_id.split(":", 1)[1]
        summary = graph.nodes[epic_id].get("properties", {}).get("summary", "")
        if summary:
            # Use first few words of summary as query
            short = " ".join(summary.split()[:4])
            queries.append(f"{key} {short}")

    return queries
