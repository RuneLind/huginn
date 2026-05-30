"""Build the per-collection search function used by MCP stdio adapters.

Both ``multi_collection_search_mcp_adapter`` and
``collection_search_mcp_stdio_adapter`` register MCP tools that share the
same orchestration: graph-aware query expansion, search via
``DocumentCollectionSearcher``, post-processing via
``shape_search_results``, and per-result graph context enrichment. Mirrors
``knowledge_api_server.py`` ``/api/search`` so all three runtimes return the
same response shape.
"""
import json
import logging
from typing import Callable, Literal

from main.core.search_pipeline import search_and_shape
from main.core.search_response_formatter import run_corrective_search
from main.core.search_trace import create_trace
from main.graph.graph_search_augmenter import GraphSearchAugmenter


CorrectiveMode = Literal["auto", "off", "force"]


def build_search_tool_fn(
    searcher,
    collection_name: str,
    augmenter: GraphSearchAugmenter,
    *,
    max_number_of_chunks: int,
    max_number_of_documents: int,
    include_full_text: bool,
    trace_default: bool = False,
    min_relevance: float | None = None,
    corrective_default: CorrectiveMode = "auto",
) -> Callable[..., str]:
    """Return the ``(query, corrective="auto") -> str`` callable an MCP tool
    handler invokes.

    Pass ``GraphSearchAugmenter(None)`` when the runtime has no knowledge
    graph configured — augmentation and enrichment then become no-ops while
    the rest of the pipeline still runs. ``min_relevance`` drops weak results
    (and triggers ``noConfidentResults`` + ``retryHints`` when it empties the
    set), mirroring the ``/api/search`` query param.

    ``corrective_default`` sets the runtime default for the returned callable's
    ``corrective`` arg, which controls huginn-side rescue retrieval. Normally
    leave the per-call value as ``"auto"``; set to ``"off"`` only to reproduce
    pre-corrective behaviour for testing. ``"force"`` is a debug knob.
    """
    def search_fn(query: str, corrective: CorrectiveMode = corrective_default) -> str:
        logging.info(f"Searching in {collection_name}: {query}")
        trace = create_trace(trace_default)
        trace.set_query_raw(query)

        search_q, graph_answer, detected_entities = augmenter.augment_query(query, trace)

        target_searchers = {collection_name: searcher}
        search_kwargs = dict(
            max_number_of_chunks=max_number_of_chunks,
            max_number_of_documents=max_number_of_documents,
            include_text_content=include_full_text,
            include_matched_chunks_content=not include_full_text,
        )
        shape_kwargs = dict(limit=max_number_of_documents)

        results, per_collection, any_low_confidence = search_and_shape(
            target_searchers,
            search_q,
            augmenter=augmenter,
            detected_entities=detected_entities,
            trace=trace,
            title_boost_query=query,
            search_kwargs=search_kwargs,
            shape_kwargs=shape_kwargs,
        )

        reranked = bool(per_collection[0][1].get("reranked", True))

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
            query=query,
            augmenter=augmenter,
            detected_entities=detected_entities,
            min_relevance=min_relevance,
            trace=trace,
            reranked=reranked,
            mode=corrective,
            rerun_search_fn=rerun_search_fn,
            limit=max_number_of_documents,
        )
        if graph_answer:
            response["graph_answer"] = graph_answer
        if any_low_confidence:
            response["lowConfidence"] = True
        if trace_default:
            response["trace"] = trace.to_dict()
        return json.dumps(response, indent=2, ensure_ascii=False)

    return search_fn
