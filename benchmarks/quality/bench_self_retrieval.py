"""Benchmark: self-retrieval test.

Index a document, search for its title, check if it ranks in top-k.
This is the core quality metric — if the system can't find its own documents
by title, something is fundamentally wrong.
"""

import random
import time

from benchmarks.context import BenchmarkContext, load_documents_for_collection
from benchmarks.results import BenchmarkResult


def _extract_title(doc: dict) -> str | None:
    """Extract a searchable title from a document."""
    metadata = doc.get("metadata", {})

    # Jira: use "ISSUE_KEY: title"
    issue_key = metadata.get("issue_key")
    title = metadata.get("title", "")
    if issue_key and title:
        return f"{issue_key} {title}"

    # Confluence: use title from metadata
    if title:
        return title

    # Fallback: use document id (filename stem)
    doc_id = doc.get("id", "")
    if doc_id:
        return doc_id.rsplit("/", 1)[-1].replace(".md", "").replace("_", " ")

    return None


def bench_self_retrieval(ctx: BenchmarkContext, collection_name: str, sample_size: int = 50) -> BenchmarkResult:
    """Self-retrieval: search for document titles and check if they rank in top-k.

    Metrics: recall@1, recall@3, recall@5, mrr (mean reciprocal rank).
    """
    documents = load_documents_for_collection(ctx.persister, collection_name)
    searcher = ctx.get_searcher(collection_name)

    # Sample documents with seed for reproducibility
    rng = random.Random(42)
    sample = rng.sample(documents, min(sample_size, len(documents)))

    # Filter to documents with titles
    titled = [(doc, _extract_title(doc)) for doc in sample]
    titled = [(doc, title) for doc, title in titled if title and len(title) > 5]

    t_start = time.monotonic()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    failures = []

    for doc, title in titled:
        doc_id = doc.get("id", "")
        results = searcher.search(
            title,
            max_number_of_chunks=30,
            skip_reranker=False,
        )

        # Check if the source document appears in results
        found_rank = None
        for rank, result in enumerate(results.get("results", []), 1):
            if result["id"] == doc_id:
                found_rank = rank
                break

        if found_rank:
            reciprocal_ranks.append(1.0 / found_rank)
            if found_rank <= 1:
                hits_at_1 += 1
            if found_rank <= 3:
                hits_at_3 += 1
            if found_rank <= 5:
                hits_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)
            failures.append({"doc_id": doc_id, "title": title[:80]})

    total_duration = (time.monotonic() - t_start) * 1000
    n = len(titled)

    metrics = {
        "recall_at_1": hits_at_1 / n if n else 0,
        "recall_at_3": hits_at_3 / n if n else 0,
        "recall_at_5": hits_at_5 / n if n else 0,
        "mrr": sum(reciprocal_ranks) / n if n else 0,
        "sample_size": n,
        "failures": len(failures),
    }

    return BenchmarkResult(
        name=f"self_retrieval_{collection_name}",
        category="quality",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={
            "collection": collection_name,
            "total_docs": len(documents),
            "failures": failures[:10],  # keep top 10 for debugging
        },
    )
