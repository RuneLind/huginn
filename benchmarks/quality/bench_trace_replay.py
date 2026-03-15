"""Benchmark: replay real MCP query traces.

Uses actual query-document pairs captured from MCP usage sessions.
Each trace represents a multi-step search where the agent tried multiple
queries across collections, and the fetched_docs are the documents that
were actually used to answer the question.

This is the highest-fidelity quality benchmark — it measures whether the
system can reproduce the same retrievals that worked in real sessions.
"""

import json
import re
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


def _normalize(name: str) -> str:
    """Normalize a document name for comparison: lowercase, collapse whitespace/underscores."""
    return re.sub(r'[\s_]+', '_', name.strip()).lower().rstrip('.')


def _extract_issue_key(name: str) -> str | None:
    """Extract Jira issue key (e.g., MELOSYS-7855) from a string."""
    m = re.search(r'[A-Z][A-Z0-9]+-\d+', name, re.IGNORECASE)
    return m.group(0).upper() if m else None


def _doc_matches(expected_doc: str, result_id: str) -> bool:
    """Check if an expected document matches a result ID.

    Handles filename normalization (spaces vs underscores, trailing dots)
    and falls back to issue key matching for Jira documents.
    """
    # Direct substring match
    if expected_doc in result_id or result_id.endswith(expected_doc):
        return True

    # Normalized match (handles spaces vs underscores, trailing dots)
    if _normalize(expected_doc) in _normalize(result_id):
        return True

    # Issue key match: if expected doc has an issue key, check if result has the same key
    expected_key = _extract_issue_key(expected_doc)
    if expected_key:
        result_key = _extract_issue_key(result_id)
        if result_key and expected_key == result_key:
            return True

    return False


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
                if _doc_matches(expected_doc, rid):
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


def _find_trace_data_dir(ctx: BenchmarkContext, trace_data_dir: str | Path | None) -> Path | None:
    """Find the directory containing query-doc-pairs.jsonl."""
    if trace_data_dir:
        return Path(trace_data_dir)
    for d in ctx.data_dirs:
        if (d / "query-doc-pairs.jsonl").exists():
            return d
    return None


def _collection_variants(collection_name: str) -> set[str]:
    """Generate collection name variants for matching (e.g. jira-issues -> jira)."""
    variants = {collection_name}
    if "-" in collection_name:
        variants.add(collection_name.split("-")[0])
    if collection_name.count("-") >= 2:
        variants.add(collection_name.rsplit("-v", 1)[0])
    return variants


def bench_session_replay(ctx: BenchmarkContext, collection_name: str, trace_data_dir: str | Path = None) -> BenchmarkResult:
    """Replay traces at session level — group queries by trace_id.

    This is a fairer benchmark than per-query replay because it mirrors
    how the MCP agent actually works: it tries multiple query variations
    within a session and the union of all results is what matters.

    For each session (trace_id), runs all queries that targeted this
    collection, collects the union of all result documents, and checks
    if all expected documents from the session appear in that union.
    """
    data_dir = _find_trace_data_dir(ctx, trace_data_dir)
    if not data_dir:
        return BenchmarkResult(
            name=f"session_replay_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "No query-doc-pairs.jsonl found in data dirs"},
        )

    pairs = _load_trace_data(data_dir)
    if not pairs:
        return BenchmarkResult(
            name=f"session_replay_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "Empty trace data"},
        )

    variants = _collection_variants(collection_name)

    # Group pairs by trace_id (session)
    sessions: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        if p.get("collection") in variants:
            sessions[p["trace_id"]].append(p)

    # Only keep sessions that have at least one pair with fetched docs
    active_sessions = {
        tid: queries for tid, queries in sessions.items()
        if any(q.get("fetched_docs") for q in queries)
    }

    if not active_sessions:
        return BenchmarkResult(
            name=f"session_replay_{collection_name}",
            category="quality",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": f"No sessions with fetched docs for {collection_name}"},
        )

    searcher = ctx.get_searcher(collection_name)
    t_start = time.monotonic()

    total_expected_docs = 0
    total_found_docs = 0
    sessions_with_full_recall = 0
    session_details = []

    for trace_id, session_pairs in active_sessions.items():
        # Collect all expected docs for this session
        expected_docs = set()
        for p in session_pairs:
            for doc in p.get("fetched_docs", []):
                expected_docs.add(doc)

        if not expected_docs:
            continue

        # Run all queries and collect union of results
        all_result_ids = []
        queries_run = []
        for p in session_pairs:
            query = p["query"]
            results = searcher.search(
                query,
                max_number_of_chunks=20,
                skip_reranker=False,
            )
            result_ids = [r["id"] for r in results.get("results", [])]
            all_result_ids.extend(result_ids)
            queries_run.append(query)

        # Check which expected docs were found across all queries
        found_docs = set()
        missed_docs = set()
        for expected_doc in expected_docs:
            if any(_doc_matches(expected_doc, rid) for rid in all_result_ids):
                found_docs.add(expected_doc)
            else:
                missed_docs.add(expected_doc)

        total_expected_docs += len(expected_docs)
        total_found_docs += len(found_docs)
        if not missed_docs:
            sessions_with_full_recall += 1

        session_details.append({
            "trace_id": trace_id[:8],
            "queries": len(queries_run),
            "expected_docs": len(expected_docs),
            "found_docs": len(found_docs),
            "missed": sorted(missed_docs) if missed_docs else [],
        })

    total_duration = (time.monotonic() - t_start) * 1000
    n_sessions = len(session_details)

    doc_recall = total_found_docs / total_expected_docs if total_expected_docs else 0
    session_hit_rate = sessions_with_full_recall / n_sessions if n_sessions else 0

    missed_sessions = [s for s in session_details if s["missed"]]

    metrics = {
        "doc_recall": doc_recall,
        "session_hit_rate": session_hit_rate,
        "total_sessions": n_sessions,
        "sessions_full_recall": sessions_with_full_recall,
        "total_expected_docs": total_expected_docs,
        "total_found_docs": total_found_docs,
    }

    return BenchmarkResult(
        name=f"session_replay_{collection_name}",
        category="quality",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={
            "collection": collection_name,
            "missed_sessions": missed_sessions[:10],
            "trace_data_dir": str(data_dir),
        },
    )
