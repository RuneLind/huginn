"""Shared search orchestration for the HTTP route and the MCP stdio tools.

Both ``/api/search`` (``routes/search.py``) and ``build_search_tool_fn``
(``mcp_search_tool.py``) run the same per-collection sequence: search each
target searcher, shape the combined results, and apply graph-context
enrichment. They run it twice each — once for the initial query and once for
the corrective-rescue rerun. ``search_and_shape`` owns that sequence so all
four call sites share one copy.
"""
from main.core.search_response_formatter import (
    run_corrective_search,
    shape_search_results,
)


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


def run_search_request(
    target_searchers,
    *,
    raw_query,
    search_query,
    augmenter,
    detected_entities,
    graph_answer,
    trace,
    search_kwargs,
    shape_kwargs,
    min_relevance,
    corrective_mode,
    limit,
):
    """Run the full search→shape→enrich→corrective envelope and return the
    response dict.

    This owns the ~40-line sequence the HTTP route and the MCP stdio tool used
    to duplicate: the initial ``search_and_shape``, the ``reranked`` honesty
    computation, the corrective-rescue rerun closure, the ``run_corrective_search``
    call, and the ``graph_answer`` / ``lowConfidence`` merge onto the response.

    Callers keep only transport-specific glue: building ``search_kwargs`` /
    ``shape_kwargs`` from their own request params, running query augmentation
    to obtain ``search_query`` / ``graph_answer`` / ``detected_entities``, and
    trace-field merge + serialization of the returned dict (the two transports
    differ there — HTTP has pointer-mode/`traceId`, MCP dumps to JSON).

    Args:
        target_searchers: ``{collection_name: searcher}`` (single entry for the
            single-collection MCP runtimes).
        raw_query: the user's original query, used for title-boost on the initial
            search and as the ``query`` fed to the corrective signal.
        search_query: the (possibly graph-expanded) query for the initial search.
        augmenter: ``GraphSearchAugmenter`` used for enrichment and rescue.
        detected_entities: entities from query augmentation.
        graph_answer: direct graph answer (or falsy); merged onto the response.
        trace: the per-request search trace (or a no-op trace).
        search_kwargs: forwarded to ``searcher.search``.
        shape_kwargs: forwarded to ``shape_search_results``.
        min_relevance: drop-weak threshold (or ``None``).
        corrective_mode: ``"auto" | "off" | "force"``.
        limit: result limit passed to ``run_corrective_search``.

    Returns:
        The response dict (``graph_answer`` / ``lowConfidence`` merged in;
        trace-field merge is left to the caller).
    """
    results, per_collection, any_low_confidence = search_and_shape(
        target_searchers,
        search_query,
        augmenter=augmenter,
        detected_entities=detected_entities,
        trace=trace,
        title_boost_query=raw_query,
        search_kwargs=search_kwargs,
        shape_kwargs=shape_kwargs,
    )

    reranked = (
        all(sr.get("reranked", True) for _, sr in per_collection)
        if per_collection
        else True
    )

    def rerun_search_fn(rescue_query: str):
        rescue_results, _, _ = search_and_shape(
            target_searchers,
            rescue_query,
            augmenter=augmenter,
            detected_entities=detected_entities,
            trace=trace,
            title_boost_query=rescue_query,
            search_kwargs=search_kwargs,
            shape_kwargs=shape_kwargs,
        )
        return rescue_results

    results, response = run_corrective_search(
        results,
        query=raw_query,
        augmenter=augmenter,
        detected_entities=detected_entities,
        min_relevance=min_relevance,
        trace=trace,
        reranked=reranked,
        mode=corrective_mode,
        rerun_search_fn=rerun_search_fn,
        limit=limit,
    )
    if graph_answer:
        response["graph_answer"] = graph_answer
    if any_low_confidence:
        response["lowConfidence"] = True
    return response
