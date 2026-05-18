"""Knowledge-graph and per-collection graph routes."""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from main.graph.author_graph import build_author_graph
from main.graph.similarity_graph import (
    EMPTY_GRAPH,
    build_similarity_graph,
    shape_similarity_response,
)
from main.runtime.knowledge_store import KnowledgeStore, get_store

router = APIRouter()


def _parse_edge_types(edge_types: str | None) -> set[str] | None:
    if not edge_types:
        return None
    return {t.strip() for t in edge_types.split(",") if t.strip()}


@router.get("/api/graph/{node_id:path}/subtree")
def get_graph_subtree(
    node_id: str,
    depth: int = Query(2, ge=1, le=5),
    direction: str = Query("incoming", pattern="^(incoming|outgoing|both)$"),
    edge_types: str | None = Query(None, description="Comma-separated edge type names to include"),
    max_nodes: int = Query(1000, ge=1, le=5000),
    store: KnowledgeStore = Depends(get_store),
):
    """Return a BFS subtree from a node. For epics, defaults walk stories + subtasks in one call."""
    if not store.graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    subtree = store.graph.get_subtree(
        node_id,
        direction=direction,
        edge_types=_parse_edge_types(edge_types),
        max_depth=depth,
        max_nodes=max_nodes,
    )
    if not subtree:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return subtree


@router.get("/api/graph/{node_id:path}")
def get_graph_node(
    node_id: str,
    edge_types: str | None = Query(None, description="Comma-separated edge type names to include"),
    store: KnowledgeStore = Depends(get_store),
):
    """Inspect a knowledge graph node and its relationships."""
    if not store.graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    detail = store.graph.get_node_detail(node_id, edge_types=_parse_edge_types(edge_types))
    if not detail:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return detail


@router.get("/api/collection/{name}/similarity-graph")
def collection_similarity_graph(
    name: str,
    top_k: int = Query(5, ge=1, le=20),
    min_similarity: float = Query(0.65, ge=0.0, le=1.0),
    store: KnowledgeStore = Depends(get_store),
):
    """Build a document similarity graph from FAISS embeddings (mean-pooled per document)."""
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    cached = store.get_cached_similarity_graph(name)
    if not cached:
        searcher = store.get_searchers([name])[name]
        cached = build_similarity_graph(name, searcher, store.disk_persister)
        if not cached:
            return EMPTY_GRAPH
        store.set_cached_similarity_graph(name, cached)

    return shape_similarity_response(cached, top_k, min_similarity)


@router.get("/api/collection/{name}/author-graph")
def collection_author_graph(
    request: Request,
    name: str,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    min_tweets: int = Query(3, ge=1, le=100),
    min_interactions: int = Query(1, ge=1, le=100),
    store: KnowledgeStore = Depends(get_store),
):
    """Serve the author interaction graph for a collection.

    Reads pre-computed author scores from huginn-jarvis and transforms
    them into the same node/edge/community format as similarity-graph.
    Only includes authors that have at least one interaction edge (no isolates).
    Results are cached per collection; invalidated on collection reload.
    """
    cached = store.get_cached_author_graph(name)
    if cached:
        return cached

    huginn_root: Path = request.app.state.huginn_root
    scores_path = huginn_root / "huginn-jarvis" / "data" / f"{name}-author-scores.json"
    if not scores_path.exists():
        raise HTTPException(status_code=404, detail=f"No author graph found for '{name}'")

    scores = json.loads(scores_path.read_text())
    result = build_author_graph(scores, name, store.disk_persister, min_score, min_tweets, min_interactions)
    store.set_cached_author_graph(name, result)
    return result
