"""Search and trace routes."""
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from main.core.search_pipeline import run_search_request
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
    corrective: Literal["auto", "off", "force"] = Query("auto", description="Corrective-rescue mode: 'auto' (default; rescue when first search is weak and a usable hint exists), 'off' (today's behaviour, no rescue), 'force' (rescue whenever a hint exists — test knob)."),
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

    search_kwargs = dict(
        max_number_of_chunks=limit * overfetch,
        max_number_of_documents=limit * (3 if has_filters else 1),
        include_matched_chunks_content=True,
        skip_reranker=skip_reranker,
    )
    shape_kwargs = dict(
        limit=limit,
        brief=brief,
        max_chunk_chars=max_chunk_chars,
        max_chunks_per_doc=max_chunks_per_doc,
        project=project,
        git_branch=git_branch,
        tags=tags,
    )

    response = run_search_request(
        target_searchers,
        raw_query=q,
        search_query=search_q,
        augmenter=augmenter,
        detected_entities=detected_entities,
        graph_answer=graph_answer,
        trace=trace_obj,
        search_kwargs=search_kwargs,
        shape_kwargs=shape_kwargs,
        min_relevance=min_relevance,
        corrective_mode=corrective,
    )
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
