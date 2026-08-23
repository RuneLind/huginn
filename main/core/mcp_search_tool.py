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

from main.core.search_pipeline import run_search_request
from main.core.search_trace import create_trace
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.privacy.query_privacy import dealias_query


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
    alias_registry=None,
) -> Callable[..., str]:
    """Return the ``(query, corrective="auto") -> str`` callable an MCP tool
    handler invokes.

    Pass ``GraphSearchAugmenter(None)`` when the runtime has no knowledge
    graph configured — augmentation and enrichment then become no-ops while
    the rest of the pipeline still runs. ``min_relevance`` drops weak results
    (and triggers ``noConfidentResults`` + ``retryHints`` when it empties the
    set), mirroring the ``/api/search`` query param.

    ``alias_registry`` is the collection's ``AliasRegistry`` when it is served
    from a privacy-aliased index (``KnowledgeStore.get_alias_registries``), else
    ``None``. With one, the query is rewritten into alias space before anything
    records or echoes it — the tool wraps a single collection, so there is no
    out-of-scope sibling needing the name-space text.

    ``corrective_default`` sets the runtime default for the returned callable's
    ``corrective`` arg, which controls huginn-side rescue retrieval. Normally
    leave the per-call value as ``"auto"``; set to ``"off"`` only to reproduce
    pre-corrective behaviour for testing. ``"force"`` is a debug knob.
    """
    if alias_registry is not None:
        # Same reason as the HTTP route: the knowledge graph still spells people
        # by name, so its contribution to the response is de-aliased too.
        augmenter = GraphSearchAugmenter(augmenter.graph, alias_registry=alias_registry)

    def search_fn(query: str, corrective: CorrectiveMode = corrective_default) -> str:
        # Everything below — the log line included — uses the de-aliased text.
        public_query = dealias_query(query, alias_registry)
        logging.info(f"Searching in {collection_name}: {public_query}")
        trace = create_trace(trace_default)
        trace.set_query_raw(public_query)

        search_q, graph_answer, detected_entities = augmenter.augment_query(public_query, trace)

        target_searchers = {collection_name: searcher}
        search_kwargs = dict(
            max_number_of_chunks=max_number_of_chunks,
            max_number_of_documents=max_number_of_documents,
            include_text_content=include_full_text,
            include_matched_chunks_content=not include_full_text,
        )
        shape_kwargs = dict(limit=max_number_of_documents)

        response = run_search_request(
            target_searchers,
            raw_query=public_query,
            search_query=search_q,
            augmenter=augmenter,
            detected_entities=detected_entities,
            graph_answer=graph_answer,
            trace=trace,
            search_kwargs=search_kwargs,
            shape_kwargs=shape_kwargs,
            min_relevance=min_relevance,
            corrective_mode=corrective,
            alias_registries={collection_name: alias_registry},
        )
        if trace_default:
            response["trace"] = trace.to_dict()
        return json.dumps(response, indent=2, ensure_ascii=False)

    return search_fn
