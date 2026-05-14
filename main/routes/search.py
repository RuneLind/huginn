"""Search and trace routes."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from main.core.search_response_formatter import apply_corrective_signal, shape_search_results
from main.core.search_trace import create_trace
from main.core.trace_store import any_trace_enabled, default_trace_store, pointer_mode_enabled
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.runtime.knowledge_store import KnowledgeStore, get_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/search")
def search(
    q: str = Query(..., description="Search query"),
    collection: list[str] = Query(None, description="Collection name(s) (searches all if omitted)"),
    limit: int = Query(10, description="Max number of results"),
    brief: bool = Query(False, description="Return brief results (title, url, snippet only)"),
    rerank: bool | None = Query(None, description="Enable cross-encoder reranking (default: true for full, false for brief)"),
    max_chunk_chars: int = Query(None, description="Truncate each chunk's content to N characters"),
    max_chunks_per_doc: int = Query(3, ge=1, description="Max matched chunks per document"),
    project: str = Query(None, description="Filter by project metadata"),
    git_branch: str = Query(None, description="Filter by gitBranch metadata"),
    tags: str = Query(None, description="Filter by tags (comma-separated, matches any)"),
    min_relevance: float = Query(None, ge=0.0, le=1.0, description="Drop results below this relevance (0.0-1.0). If all are below, returns empty results plus retryHints and noConfidentResults=true."),
    trace: bool = Query(False, description="Return per-stage search trace (entities, scores, timings) for debugging"),
    store: KnowledgeStore = Depends(get_store),
):
    if collection:
        for c in collection:
            if not store.has_collection(c):
                raise HTTPException(status_code=404, detail=f"Collection '{c}' not found")
    target_searchers = store.get_searchers(collection)

    has_filters = bool(project or git_branch or tags)
    overfetch = 5 if has_filters else 3

    skip_reranker = not rerank if rerank is not None else brief

    trace_enabled = trace or any_trace_enabled()
    trace_obj = create_trace(trace_enabled)
    trace_obj.set_query_raw(q)

    augmenter = GraphSearchAugmenter(store.graph)
    search_q, graph_answer, detected_entities = augmenter.augment_query(q, trace_obj)
    if search_q != q:
        logger.debug(f"Graph-expanded query: {search_q[:200]}")

    per_collection = []
    for coll_name, searcher in target_searchers.items():
        search_result = searcher.search(
            search_q,
            max_number_of_chunks=limit * overfetch,
            max_number_of_documents=limit * (3 if has_filters else 1),
            include_matched_chunks_content=True,
            skip_reranker=skip_reranker,
            trace=trace_obj,
            title_boost_query=q,
        )
        per_collection.append((coll_name, search_result))

    results, any_low_confidence = shape_search_results(
        per_collection,
        limit=limit,
        brief=brief,
        max_chunk_chars=max_chunk_chars,
        max_chunks_per_doc=max_chunks_per_doc,
        project=project,
        git_branch=git_branch,
        tags=tags,
    )

    augmenter.enrich_results(results, detected_entities)

    reranked = all(sr.get("reranked", True) for _, sr in per_collection) if per_collection else True
    results, response = apply_corrective_signal(
        results,
        query=q,
        augmenter=augmenter,
        detected_entities=detected_entities,
        min_relevance=min_relevance,
        trace=trace_obj,
        reranked=reranked,
    )
    if graph_answer:
        response["graph_answer"] = graph_answer
    if any_low_confidence:
        response["lowConfidence"] = True
    if trace_enabled:
        trace_dict = trace_obj.to_dict()
        if pointer_mode_enabled():
            response["traceId"] = default_trace_store().put(trace_dict)
        else:
            response["trace"] = trace_dict
    return response


@router.get("/api/trace/{trace_id}")
def get_search_trace(trace_id: str):
    """Fetch a stored search trace by ID. 404 once expired (TTL ~5 min)."""
    trace = default_trace_store().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found or expired")
    return trace
