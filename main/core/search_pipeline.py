"""Shared search orchestration for the HTTP route and the MCP stdio tools.

Both ``/api/search`` (``routes/search.py``) and ``build_search_tool_fn``
(``mcp_search_tool.py``) run the same per-collection sequence: search each
target searcher, shape the combined results, and apply graph-context
enrichment. They run it twice each — once for the initial query and once for
the corrective-rescue rerun. ``search_and_shape`` owns that sequence so all
four call sites share one copy.
"""
from main.core.query_log import log_search_request
from main.core.search_response_formatter import (
    run_corrective_search,
    shape_search_results,
)
from main.privacy.query_privacy import alias_token_pattern, dealias_query


def _collection_text(text, unaliased_text, registry):
    """The text to search one collection with.

    Armed (in privacy scope, index built aliased): the alias-space text.
    ``dealias_query`` is idempotent, so passing an already de-aliased query is
    a no-op and passing a raw one is still corrected here.

    Out of scope: the *unaliased* twin when the caller has one. This is what
    makes a mixed request work — searching a personal wiki that was never
    aliased for "dev-06" finds nothing, and searching an aliased collection for
    a real name finds nothing. Each collection gets the text its own index
    spells. ``unaliased_text`` is None for a corrective rescue query (it is
    derived from the already de-aliased hints and has no raw twin), in which
    case every collection uses the one text.
    """
    if registry is not None:
        return dealias_query(text, registry)
    return text if unaliased_text is None else unaliased_text


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
    alias_registries=None,
    unaliased_query=None,
    unaliased_title_boost_query=None,
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
        alias_registries: ``{collection_name: AliasRegistry | None}``. Absent or
            all-None, nothing about the search changes.
        unaliased_query / unaliased_title_boost_query: the pre-de-alias twins of
            ``query`` / ``title_boost_query``, used for the out-of-scope
            collections of a mixed request (see ``_collection_text``).

    Returns:
        ``(results, per_collection, any_low_confidence)``.
    """
    registries = alias_registries or {}
    per_collection = []
    for coll_name, searcher in target_searchers.items():
        registry = registries.get(coll_name)
        extra = {}
        if registry is not None:
            # Lets the searcher pin exact alias-token hits past a cross-encoder
            # that scores opaque tokens as uniform noise.
            extra["alias_pattern"] = alias_token_pattern(registry)
        search_result = searcher.search(
            _collection_text(query, unaliased_query, registry),
            trace=trace,
            title_boost_query=_collection_text(
                title_boost_query, unaliased_title_boost_query, registry),
            **search_kwargs,
            **extra,
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
    alias_registries=None,
    unaliased_raw_query=None,
    unaliased_search_query=None,
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
        corrective_mode: ``"auto" | "off" | "force"``. The result limit for
            corrective rescue is read from ``shape_kwargs["limit"]`` so the
            two can't drift apart.
        alias_registries: ``{collection_name: AliasRegistry | None}`` for the
            targeted collections.
        unaliased_raw_query / unaliased_search_query: the pre-de-alias twins of
            ``raw_query`` / ``search_query``, or None when the caller did not
            de-alias. **Only** the de-aliased text reaches the query log, the
            retry hints, ``corrective.queriesTried`` and the trace — a real name
            must not persist anywhere — while the twins keep out-of-scope
            collections searching in name space.

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
        alias_registries=alias_registries,
        unaliased_query=unaliased_search_query,
        unaliased_title_boost_query=unaliased_raw_query,
    )

    # AND across collections: any non-reranked collection makes the whole
    # response non-reranked (the honest form — see docs/search-invariants.md).
    # The pre-extraction MCP path checked only the first collection; identical
    # while MCP tools wrap a single searcher, and safer if that ever changes.
    reranked = (
        all(sr.get("reranked", True) for _, sr in per_collection)
        if per_collection
        else True
    )

    def rerun_search_fn(rescue_query: str):
        # No unaliased twin: the rescue query is built from the retry hints,
        # which are themselves derived from the de-aliased text.
        rescue_results, _, _ = search_and_shape(
            target_searchers,
            rescue_query,
            augmenter=augmenter,
            detected_entities=detected_entities,
            trace=trace,
            title_boost_query=rescue_query,
            search_kwargs=search_kwargs,
            shape_kwargs=shape_kwargs,
            alias_registries=alias_registries,
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
        limit=shape_kwargs["limit"],
    )
    if graph_answer:
        response["graph_answer"] = graph_answer
    if any_low_confidence:
        response["lowConfidence"] = True
    alias_pinned = sum(sr.get("aliasPinned", 0) for _, sr in per_collection)
    if alias_pinned:
        response["aliasPinned"] = alias_pinned
    log_search_request(
        collections=target_searchers.keys(),
        query=raw_query,
        response=response,
    )
    return response
