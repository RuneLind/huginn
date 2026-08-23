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
from main.privacy.query_privacy import prepare_aliased_request


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
    shared_registry=None,
) -> Callable[..., str]:
    """Return the ``(query, corrective="auto") -> str`` callable an MCP tool
    handler invokes.

    Pass ``GraphSearchAugmenter(None)`` when the runtime has no knowledge
    graph configured — augmentation and enrichment then become no-ops while
    the rest of the pipeline still runs. ``min_relevance`` drops weak results
    (and triggers ``noConfidentResults`` + ``retryHints`` when it empties the
    set), mirroring the ``/api/search`` query param.

    ``alias_registry`` is THIS collection's ``AliasRegistry`` when it is served
    from a privacy-aliased index (``KnowledgeStore.get_alias_registries``), else
    ``None``; it decides the space this collection is searched in.
    ``shared_registry`` decides the space everything that records or echoes is
    written in, and is the first armed registry among ALL served collections —
    the multi-collection adapter registers one tool per collection, so a real
    name typed at an out-of-scope tool must still not reach the log or the
    trace. It defaults to this collection's own, which is the single-collection
    runtime's whole world.

    ``corrective_default`` sets the runtime default for the returned callable's
    ``corrective`` arg, which controls huginn-side rescue retrieval. Normally
    leave the per-call value as ``"auto"``; set to ``"off"`` only to reproduce
    pre-corrective behaviour for testing. ``"force"`` is a debug knob.
    """
    if shared_registry is None:
        shared_registry = alias_registry
    if shared_registry is not None:
        # Same reason as the HTTP route: the knowledge graph still spells people
        # by name, so its contribution to the response is de-aliased too.
        augmenter = GraphSearchAugmenter(augmenter.graph, alias_registry=shared_registry)

    def search_fn(query: str, corrective: CorrectiveMode = corrective_default) -> str:
        # Everything below — the log line included — uses the de-aliased text.
        prepared = prepare_aliased_request(query, {collection_name: shared_registry})
        public_query = prepared.public_query
        logging.info(f"Searching in {collection_name}: {public_query}")
        trace = create_trace(trace_default)
        trace.set_query_raw(public_query)

        search_q, graph_answer, detected_entities, raw_terms = augmenter.augment_query(
            public_query, trace)

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
            shared_registry=shared_registry,
            unaliased_title_boost_query=prepared.title_boost_text(),
            unaliased_search_query=prepared.expansion_twin(search_q, raw_terms),
        )
        if trace_default:
            response["trace"] = trace.to_dict()
        return json.dumps(response, indent=2, ensure_ascii=False)

    return search_fn
