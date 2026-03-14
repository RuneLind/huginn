"""Benchmark: replay real MCP query traces.

Uses actual query-document pairs captured from MCP usage sessions.
Each trace represents a multi-step search where the agent tried multiple
queries across collections, and the fetched_docs are the documents that
were actually used to answer the question.

This is the highest-fidelity quality benchmark — it measures whether the
system can reproduce the same retrievals that worked in real sessions.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult


def _load_trace_data(data_dir: Path) -> list[dict] | None:
    """Load query-doc-pairs.jsonl from the data directory."""
    path = data_dir / "query-doc-pairs.jsonl"
    if not path.exists():
        return None
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


def _load_relevant_docs(data_dir: Path, collection: str) -> set[str]:
    """Load the expected relevant doc IDs for a collection."""
    docs_file = data_dir / f"{collection}-docs.jsonl"
    if not docs_file.exists():
        return set()
    doc_ids = set()
    for line in docs_file.read_text().splitlines():
        line = line.strip()
        if line:
            entry = json.loads(line)
            doc_ids.add(entry["doc_id"])
    return doc_ids


def bench_trace_replay(ctx: BenchmarkContext, collection_name: str, trace_data_dir: str | Path = None) -> BenchmarkResult:
    """Replay real MCP traces and measure retrieval quality.

    For each query that was used in a real session, runs it through
    the current search pipeline and checks if the same documents
    that were fetched in the session appear in the results.
    """
    # Find trace data
    if trace_data_dir:
        data_dir = Path(trace_data_dir)
    else:
        # Search context data dirs for query-doc-pairs.jsonl
        data_dir = None
        for d in ctx.data_dirs:
            if (d / "query-doc-pairs.jsonl").exists():
                data_dir = d
                break

    if not data_dir:
        return BenchmarkResult(
            name=f"trace_replay_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "No query-doc-pairs.jsonl found in data dirs"},
        )

    pairs = _load_trace_data(data_dir)
    if not pairs:
        return BenchmarkResult(
            name=f"trace_replay_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "Empty trace data"},
        )

    # Normalize collection name matching (handle jira-issues vs jira)
    collection_variants = {collection_name}
    if "-" in collection_name:
        collection_variants.add(collection_name.split("-")[0])  # jira-issues -> jira
    if collection_name.count("-") >= 2:
        # melosys-confluence-v3 -> melosys-confluence-v3, melosys-confluence
        collection_variants.add(collection_name.rsplit("-v", 1)[0])

    # Filter pairs for this collection
    collection_pairs = [
        p for p in pairs
        if p.get("collection") in collection_variants or p.get("collection") == collection_name
    ]

    # Only keep pairs that actually fetched documents (non-empty results)
    pairs_with_docs = [p for p in collection_pairs if p.get("fetched_docs")]
    # Also track pairs where nothing was fetched (potential misses)
    pairs_without_docs = [p for p in collection_pairs if not p.get("fetched_docs")]

    if not pairs_with_docs:
        return BenchmarkResult(
            name=f"trace_replay_{collection_name}",
            category="quality",
            metrics={"skipped": 1, "pairs_without_docs": len(pairs_without_docs)},
            duration_ms=0,
            metadata={"reason": f"No pairs with fetched docs for {collection_name}"},
        )

    searcher = ctx.get_searcher(collection_name)
    relevant_docs = _load_relevant_docs(data_dir, collection_name)

    t_start = time.monotonic()

    # Deduplicate queries (same query may appear in multiple traces)
    seen_queries = set()
    unique_pairs = []
    for p in pairs_with_docs:
        q = p["query"]
        if q not in seen_queries:
            seen_queries.add(q)
            unique_pairs.append(p)

    total_hits = 0
    total_expected = 0
    reciprocal_ranks = []
    query_details = []

    for pair in unique_pairs:
        query = pair["query"]
        expected_docs = pair["fetched_docs"]
        tags = pair.get("tags")

        results = searcher.search(
            query,
            max_number_of_chunks=20,
            skip_reranker=False,
        )

        result_ids = [r["id"] for r in results.get("results", [])]

        # Check how many of the originally fetched docs we can find
        hits_for_query = 0
        best_rank = None

        for expected_doc in expected_docs:
            total_expected += 1
            found = False
            for rank, rid in enumerate(result_ids, 1):
                # Match by doc filename (expected_doc is typically the filename)
                if expected_doc in rid or rid.endswith(expected_doc):
                    hits_for_query += 1
                    found = True
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                    break

        total_hits += hits_for_query
        reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)

        query_details.append({
            "query": query[:80],
            "expected_docs": len(expected_docs),
            "found": hits_for_query,
            "best_rank": best_rank,
            "tags": tags,
        })

    total_duration = (time.monotonic() - t_start) * 1000
    n = len(unique_pairs)

    # Calculate metrics
    doc_recall = total_hits / total_expected if total_expected else 0
    query_hit_rate = sum(1 for d in query_details if d["found"] > 0) / n if n else 0
    mrr = sum(reciprocal_ranks) / n if n else 0

    # Find queries where we missed all expected docs
    missed_queries = [d for d in query_details if d["found"] == 0]

    metrics = {
        "doc_recall": doc_recall,
        "query_hit_rate": query_hit_rate,
        "mrr": mrr,
        "unique_queries": n,
        "total_expected_docs": total_expected,
        "total_found_docs": total_hits,
        "missed_queries": len(missed_queries),
        "pairs_without_docs": len(pairs_without_docs),
    }

    return BenchmarkResult(
        name=f"trace_replay_{collection_name}",
        category="quality",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={
            "collection": collection_name,
            "missed_queries": missed_queries[:10],
            "trace_data_dir": str(data_dir),
        },
    )
