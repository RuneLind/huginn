"""Benchmark: tagging coverage and distribution (no LLM calls)."""

import time
from collections import Counter

from benchmarks.context import BenchmarkContext, load_documents_for_collection
from benchmarks.results import BenchmarkResult


def bench_tagging_coverage(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Measure tag distribution and coverage across a collection.

    Reads metadata.tags from all documents. No LLM calls.
    """
    documents = load_documents_for_collection(ctx.persister, collection_name)

    t_start = time.monotonic()

    total_docs = len(documents)
    tagged_docs = 0
    tag_counter = Counter()
    tags_per_doc = []

    for doc in documents:
        metadata = doc.get("metadata", {})
        tags_str = metadata.get("tags", "")

        if tags_str and tags_str.strip():
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            if tags:
                tagged_docs += 1
                tags_per_doc.append(len(tags))
                tag_counter.update(tags)

    total_duration = (time.monotonic() - t_start) * 1000

    metrics = {
        "total_docs": total_docs,
        "tagged_docs": tagged_docs,
        "tagged_fraction": tagged_docs / total_docs if total_docs else 0,
        "untagged_docs": total_docs - tagged_docs,
        "unique_tags": len(tag_counter),
        "tags_per_doc_mean": sum(tags_per_doc) / len(tags_per_doc) if tags_per_doc else 0,
        "tags_per_doc_min": min(tags_per_doc) if tags_per_doc else 0,
        "tags_per_doc_max": max(tags_per_doc) if tags_per_doc else 0,
    }

    # Top 20 tags
    top_tags = tag_counter.most_common(20)
    tag_distribution = {tag: count for tag, count in top_tags}

    return BenchmarkResult(
        name=f"tagging_coverage_{collection_name}",
        category="tagging",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={
            "collection": collection_name,
            "tag_distribution": tag_distribution,
        },
    )
