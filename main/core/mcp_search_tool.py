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
from typing import Callable

from main.core.search_response_formatter import apply_corrective_signal, shape_search_results
from main.core.search_trace import create_trace
from main.graph.graph_search_augmenter import GraphSearchAugmenter


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
) -> Callable[[str], str]:
    """Return the ``(query: str) -> str`` callable an MCP tool handler invokes.

    Pass ``GraphSearchAugmenter(None)`` when the runtime has no knowledge
    graph configured — augmentation and enrichment then become no-ops while
    the rest of the pipeline still runs. ``min_relevance`` drops weak results
    (and triggers ``noConfidentResults`` + ``retryHints`` when it empties the
    set), mirroring the ``/api/search`` query param.
    """
    def search_fn(query: str) -> str:
        logging.info(f"Searching in {collection_name}: {query}")
        trace = create_trace(trace_default)
        trace.set_query_raw(query)

        search_q, graph_answer, detected_entities = augmenter.augment_query(query, trace)

        raw = searcher.search(
            search_q,
            max_number_of_chunks=max_number_of_chunks,
            max_number_of_documents=max_number_of_documents,
            include_text_content=include_full_text,
            include_matched_chunks_content=not include_full_text,
            trace=trace,
            title_boost_query=query,
        )
        results, any_low_confidence = shape_search_results(
            [(collection_name, raw)],
            limit=max_number_of_documents,
        )
        augmenter.enrich_results(results, detected_entities)

        results, response = apply_corrective_signal(
            results,
            query=query,
            augmenter=augmenter,
            detected_entities=detected_entities,
            min_relevance=min_relevance,
            trace=trace,
        )
        if graph_answer:
            response["graph_answer"] = graph_answer
        if any_low_confidence:
            response["lowConfidence"] = True
        if trace_default:
            response["trace"] = trace.to_dict()
        return json.dumps(response, indent=2, ensure_ascii=False)

    return search_fn
