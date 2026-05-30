"""Shared search orchestration for the HTTP route and the MCP stdio tools.

Both ``/api/search`` (``routes/search.py``) and ``build_search_tool_fn``
(``mcp_search_tool.py``) run the same per-collection sequence: search each
target searcher, shape the combined results, and apply graph-context
enrichment. They run it twice each — once for the initial query and once for
the corrective-rescue rerun. ``search_and_shape`` owns that sequence so all
four call sites share one copy.
"""
from main.core.search_response_formatter import shape_search_results


def search_and_shape(
    target_searchers,
    query,
    *,
    augmenter,
    detected_entities,
    trace,
    title_boost_query,
    search_kwargs,
    shape_kwargs,
):
    """Search every target searcher with ``query``, shape, and graph-enrich.

    Args:
        target_searchers: ``{collection_name: searcher}`` (a single entry for the
            single-collection MCP runtimes).
        query: the (possibly graph-expanded) query passed to each searcher.
        augmenter: ``GraphSearchAugmenter`` whose ``enrich_results`` runs in place.
        detected_entities: entities from query augmentation, for enrichment.
        trace: the per-request search trace (or a no-op trace).
        title_boost_query: raw query used for title-boost token matching.
        search_kwargs: forwarded to ``searcher.search`` (caps, include flags,
            ``skip_reranker``).
        shape_kwargs: forwarded to ``shape_search_results`` (limit, brief,
            filters, ...).

    Returns:
        ``(results, per_collection, any_low_confidence)``.
    """
    per_collection = []
    for coll_name, searcher in target_searchers.items():
        search_result = searcher.search(
            query,
            trace=trace,
            title_boost_query=title_boost_query,
            **search_kwargs,
        )
        per_collection.append((coll_name, search_result))

    results, any_low_confidence = shape_search_results(per_collection, **shape_kwargs)
    augmenter.enrich_results(results, detected_entities)
    return results, per_collection, any_low_confidence
