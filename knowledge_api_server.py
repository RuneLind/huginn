#!/usr/bin/env python3
"""
Knowledge API Server — long-running HTTP API for vector search.

Loads embedding model and FAISS indexes once at startup, serves search
results via HTTP. Designed for low-latency responses (<50ms after warmup).

Usage:
    uv run knowledge_api_server.py --collections my-notion --port 8321
"""
import json
import math
import argparse
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import datetime as dt
import subprocess
import urllib.request
import urllib.error

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import detect_faiss_index, create_embedder, load_search_indexer, create_reranker
from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.search_trace import create_trace
from main.utils.env import env_bool
from main.sources.notion.notion_document_reader import NotionDocumentReader
from main.utils.logger import setup_root_logger
from main.utils.filename import sanitize_filename
from main.fetchers.youtube.youtube_transcript_downloader import YouTubeTranscriptDownloader
from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer

setup_root_logger()
logger = logging.getLogger(__name__)

_pii_sanitizer = PiiSanitizer()


class KnowledgeStore:
    """Holds all server state: embedder, persister, searchers, and Notion ID lookup."""

    def __init__(self):
        self.shared_embedder = None
        self.shared_reranker = None
        self.disk_persister = None
        self.searchers = {}
        self.graph = None  # KnowledgeGraph or None
        # Maps normalized notion ID (no dashes) -> {doc_path, url} for fast lookup
        self.notion_id_to_doc = {}
        # Maps collection name -> {tag: count} for tag distribution
        self.tag_counts = {}
        # Cached similarity graphs: collection_name -> {nodes, all_edges}
        self._similarity_graph_cache = {}
        self._author_graph_cache = {}
        self._lock = threading.Lock()

    def load_collections(self, collection_names, data_path="./data/collections"):
        self.disk_persister = DiskPersister(base_path=data_path)

        # Auto-detect model from the first collection's FAISS index
        first_index_name = detect_faiss_index(collection_names[0], self.disk_persister)
        self.detected_index_name = first_index_name

        logger.info(f"Loading shared embedding model for index: {first_index_name}")
        self.shared_embedder = create_embedder(first_index_name)
        logger.info(f"Embedding model loaded: {self.shared_embedder.model_name}")

        self.shared_reranker = create_reranker()
        logger.info(f"Reranker loaded: {self.shared_reranker.model_name}")

        for name in collection_names:
            logger.info(f"Loading collection: {name}")
            searcher = self._build_searcher(name)
            with self._lock:
                self.searchers[name] = searcher
            logger.info(f"Collection {name} loaded with {searcher.indexer.get_size()} embeddings")
            self._build_tag_counts(name)
            self._build_notion_id_lookup(name)

        self._load_knowledge_graph()

    def _load_knowledge_graph(self):
        graph_paths = []
        eessi_path_str = os.environ.get("KNOWLEDGE_GRAPH_PATH", "")
        jira_path_str = os.environ.get("JIRA_GRAPH_PATH", "")
        llm_graph_str = os.environ.get("LLM_GRAPH_PATH", "")
        if eessi_path_str and Path(eessi_path_str).exists():
            graph_paths.append(Path(eessi_path_str))
        if jira_path_str and Path(jira_path_str).exists():
            graph_paths.append(Path(jira_path_str))
        if llm_graph_str and Path(llm_graph_str).exists():
            graph_paths.append(Path(llm_graph_str))
        # Auto-detect LLM graphs in private repo dirs and fallback to local scripts
        for search_dir in [
            Path("./huginn-jarvis/scripts/knowledge_graph"),
            Path("./huginn-nav/scripts/knowledge_graph"),
            Path("./scripts/knowledge_graph"),
        ]:
            for p in search_dir.glob("*_llm_graph.json"):
                if p not in graph_paths:
                    graph_paths.append(p)
        if graph_paths:
            from main.graph.knowledge_graph import KnowledgeGraph
            self.graph = KnowledgeGraph(graph_paths)
            logger.info(f"Knowledge graph loaded from {len(graph_paths)} file(s): "
                        f"{self.graph.node_count()} nodes, {self.graph.edge_count()} edges")
        else:
            logger.info("No knowledge graph found — graph features disabled")

    def reload_collection(self, collection_name):
        searcher = self._build_searcher(collection_name)
        with self._lock:
            self.searchers[collection_name] = searcher
            self._similarity_graph_cache.pop(collection_name, None)
            self._author_graph_cache.pop(collection_name, None)
        self._build_tag_counts(collection_name)
        self._build_notion_id_lookup(collection_name)
        self._load_knowledge_graph()
        logger.info(f"Collection {collection_name} reloaded ({searcher.indexer.get_size()} embeddings)")

    def get_searchers(self, collection_names=None):
        with self._lock:
            if collection_names:
                return {c: self.searchers[c] for c in collection_names if c in self.searchers}
            return dict(self.searchers)

    def has_collection(self, name):
        with self._lock:
            return name in self.searchers

    def collection_names(self):
        with self._lock:
            return list(self.searchers.keys())

    def total_embeddings(self):
        with self._lock:
            return sum(s.indexer.get_size() for s in self.searchers.values())

    def _build_searcher(self, name):
        indexer = load_search_indexer(name, self.disk_persister, shared_embedder=self.shared_embedder)
        logger.info(f"Collection {name}: using {indexer.get_name()}")
        return DocumentCollectionSearcher(
            collection_name=name,
            indexer=indexer,
            persister=self.disk_persister,
            reranker=self.shared_reranker,
        )

    def get_tag_counts(self, collection_names=None):
        with self._lock:
            if collection_names:
                return {c: self.tag_counts.get(c, {}) for c in collection_names if c in self.tag_counts}
            return dict(self.tag_counts)

    def _build_tag_counts(self, name):
        """Scan document metadata to build tag frequency counts for a collection."""
        from collections import Counter
        tag_counts = Counter()
        docs_dir = f"{name}/documents"
        try:
            doc_files = self.disk_persister.read_folder_files(docs_dir)
        except Exception:
            doc_files = []
        for doc_file in doc_files:
            if not doc_file.endswith(".json"):
                continue
            try:
                doc = json.loads(self.disk_persister.read_text_file(f"{docs_dir}/{doc_file}"))
                tags_str = (doc.get("metadata") or {}).get("tags", "")
                for tag in tags_str.split(","):
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] += 1
            except Exception:
                continue
        self.tag_counts[name] = dict(tag_counts.most_common())
        logger.info(f"Built tag counts for {name}: {len(tag_counts)} unique tags")

    def _build_notion_id_lookup(self, name):
        try:
            mapping_text = self.disk_persister.read_text_file(
                f"{name}/indexes/index_document_mapping.json"
            )
            mapping = json.loads(mapping_text)
            seen_paths = set()
            for entry in mapping.values():
                doc_url = entry.get("documentUrl", "")
                doc_path = entry.get("documentPath", "")
                if doc_path in seen_paths or not doc_url:
                    continue
                seen_paths.add(doc_path)
                # Extract notion ID from URL (last 32 hex chars)
                url_tail = doc_url.rstrip("/").split("/")[-1] if doc_url else ""
                notion_hex = url_tail[-32:] if len(url_tail) >= 32 else ""
                if notion_hex and all(c in "0123456789abcdef" for c in notion_hex):
                    self.notion_id_to_doc[notion_hex] = {"doc_path": doc_path, "url": doc_url}
            logger.info(f"Built Notion ID lookup with {len(self.notion_id_to_doc)} pages")
        except Exception as e:
            logger.warning(f"Could not build Notion ID lookup for {name}: {e}")


store = KnowledgeStore()


def run_collection_update(collection_name):
    """Run incremental collection update in background."""
    from main.factories.update_collection_factory import create_collection_updater

    logger.info(f"Starting background update for collection: {collection_name}")
    try:
        updater = create_collection_updater(collection_name)
        updater.run()
        store.reload_collection(collection_name)
    except Exception as e:
        logger.error(f"Failed to update collection {collection_name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_collections(app.state.collection_names, data_path=app.state.data_path)
    yield


app = FastAPI(title="Knowledge API", lifespan=lifespan)

# CORS for Chrome extension and local dev access
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|http://localhost(:\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class YouTubeIngestRequest(BaseModel):
    title: str
    url: str
    video_id: Optional[str] = None
    transcript: Optional[str] = None  # if provided, skip fetching
    summary: Optional[str] = None  # if provided, skip Claude summarization
    date: Optional[str] = None
    category: Optional[str] = None  # auto-detected if not provided


class XArticleIngestRequest(BaseModel):
    """X/Twitter article content summarized by the Chrome extension."""
    title: str
    url: str
    author: str  # @handle of the article author
    summary: str  # pre-made summary from the extension
    date: Optional[str] = None
    category: Optional[str] = None  # auto-detected if not provided
    tags: Optional[list[str]] = None


class JiraIngestComment(BaseModel):
    author: str = "Unknown"
    date: str = ""
    body: str = ""


class JiraIngestRequest(BaseModel):
    """Jira issue content scraped from the page DOM by the Chrome extension."""
    issueKey: str  # e.g., "PROJECT-1234"
    url: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    status: Optional[str] = None
    type: Optional[str] = None
    priority: Optional[str] = None
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    labels: Optional[list[str]] = None
    description: Optional[str] = None
    comments: Optional[list[JiraIngestComment]] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    epicLink: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collections": store.collection_names(),
        "totalEmbeddings": store.total_embeddings(),
    }


@app.get("/api/collections")
def list_collections():
    result = []
    for name, searcher in store.get_searchers().items():
        try:
            manifest_text = store.disk_persister.read_text_file(f"{name}/manifest.json")
            manifest = json.loads(manifest_text)
        except FileNotFoundError:
            manifest = {}
        result.append({
            "name": name,
            "document_count": manifest.get("numberOfDocuments", 0),
            "chunk_count": manifest.get("numberOfChunks", 0),
            "embedding_count": searcher.indexer.get_size(),
            "updatedTime": manifest.get("updatedTime"),
        })
    return {"collections": result}


@app.get("/api/tags")
def list_tags(
    collection: str = Query(None, description="Collection name (all if omitted)"),
):
    """Return tag distribution for a collection (or all collections). Cached at startup."""
    target_names = [collection] if collection else store.collection_names()
    result = {}
    for name in target_names:
        if not store.has_collection(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        tags = store.get_tag_counts([name]).get(name, {})
        result[name] = {
            "unique_tags": len(tags),
            "tags": tags,
        }
    return result


@app.get("/api/search")
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
    trace: bool = Query(False, description="Return per-stage search trace (entities, scores, timings) for debugging"),
):
    if collection:
        for c in collection:
            if not store.has_collection(c):
                raise HTTPException(status_code=404, detail=f"Collection '{c}' not found")
    target_searchers = store.get_searchers(collection)

    has_filters = bool(project or git_branch or tags)
    overfetch = 5 if has_filters else 3

    # Reranking: explicit param > brief default (skip for brief) > always rerank
    skip_reranker = not rerank if rerank is not None else brief

    trace_enabled = trace or env_bool("HUGINN_TRACE_DEFAULT")
    trace_obj = create_trace(trace_enabled)
    trace_obj.set_query_raw(q)

    # Graph-enhanced search: entity detection + query expansion
    search_q = q
    graph_answer = None
    detected_entities = []
    if store.graph:
        if trace_enabled:
            entity_pairs = store.graph.detect_entities(q, with_spans=True)
            detected_entities = [eid for eid, _ in entity_pairs]
            for eid, span in entity_pairs:
                node = store.graph.nodes.get(eid, {})
                trace_obj.add_detected_entity(
                    entity_id=eid,
                    entity_type=node.get("type", ""),
                    label=node.get("label", ""),
                    matched_span=span,
                )
        else:
            detected_entities = store.graph.detect_entities(q)
        if detected_entities:
            graph_answer = store.graph.answer_graph_query(detected_entities, q)
            trace_obj.set_graph_answered(graph_answer is not None)
            expansion_terms = store.graph.get_expansion_terms(detected_entities)[:5]
            if expansion_terms:
                search_q = q + " " + " ".join(expansion_terms)
                trace_obj.set_expansion(search_q, expansion_terms)
                logger.debug(f"Graph-expanded query: {search_q[:200]}")

    all_results = []
    any_low_confidence = False
    for coll_name, searcher in target_searchers.items():
        search_result = searcher.search(
            search_q,
            max_number_of_chunks=limit * overfetch,
            max_number_of_documents=limit * (3 if has_filters else 1),
            include_matched_chunks_content=True,
            skip_reranker=skip_reranker,
            trace=trace_obj,
        )
        if search_result.get("lowConfidence"):
            any_low_confidence = True
        is_reranked = search_result.get("reranked", True)
        for doc in search_result.get("results", []):
            matched_chunks = []
            for chunk in doc.get("matchedChunks", []):
                raw = chunk.get("content", "")
                entry = {
                    "content": _extract_chunk_text(raw),
                    "score": chunk.get("score", 0),
                    "heading": _extract_chunk_heading(raw),
                }
                chunk_meta = _extract_chunk_metadata(raw)
                if chunk_meta:
                    entry["metadata"] = dict(chunk_meta)
                matched_chunks.append(entry)
            if not matched_chunks:
                continue

            # Limit chunks per document (sorted by score, lower=better)
            matched_chunks.sort(key=lambda c: c["score"])
            matched_chunks = matched_chunks[:max_chunks_per_doc]

            title = doc.get("path", "").rsplit("/", 1)[-1].replace(".json", "")
            url = doc.get("url", "")
            modified_time = doc.get("modifiedTime")

            best_score = matched_chunks[0]["score"]
            relevance = _normalize_score(best_score, is_reranked)

            # Separate metadata from chunk content, collect breadcrumb
            doc_breadcrumb = None
            for chunk in matched_chunks:
                clean_content, text_metadata, breadcrumb = _separate_metadata(chunk["content"])
                chunk["content"] = clean_content
                # Merge: chunk dict metadata (from JSON) + text-parsed metadata (text overrides)
                merged = {}
                if chunk.get("metadata"):
                    merged.update(chunk["metadata"])
                if text_metadata:
                    merged.update(text_metadata)
                if merged:
                    chunk["metadata"] = merged
                if breadcrumb and not doc_breadcrumb:
                    doc_breadcrumb = breadcrumb

            if brief:
                best_chunk = matched_chunks[0]
                snippet = _truncate_snippet(best_chunk["content"])
                if not snippet and best_chunk.get("metadata"):
                    snippet = " | ".join(f"{k}: {v}" for k, v in best_chunk["metadata"].items())
                result = {
                    "collection": coll_name,
                    "id": doc.get("id"),
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "relevance": round(relevance, 3),
                    "_score": best_score,
                    "_reranked": is_reranked,
                }
                if modified_time:
                    result["modifiedTime"] = modified_time
                if doc_breadcrumb:
                    result["breadcrumb"] = doc_breadcrumb
                if best_chunk.get("heading"):
                    result["heading"] = best_chunk["heading"]
                if best_chunk.get("metadata"):
                    result["metadata"] = best_chunk["metadata"]
                all_results.append(result)
            else:
                if max_chunk_chars is not None:
                    for chunk in matched_chunks:
                        if len(chunk["content"]) > max_chunk_chars:
                            chunk["content"] = chunk["content"][:max_chunk_chars] + "…"
                # Add relevance to each chunk
                for chunk in matched_chunks:
                    chunk["relevance"] = round(_normalize_score(chunk["score"], is_reranked), 3)
                result = {
                    "collection": coll_name,
                    "id": doc.get("id"),
                    "title": title,
                    "url": url,
                    "relevance": round(relevance, 3),
                    "matchedChunks": matched_chunks,
                    "_score": best_score,
                    "_reranked": is_reranked,
                }
                if modified_time:
                    result["modifiedTime"] = modified_time
                if doc_breadcrumb:
                    result["breadcrumb"] = doc_breadcrumb
                best_meta = matched_chunks[0].get("metadata") if matched_chunks else None
                if best_meta:
                    result["metadata"] = best_meta
                all_results.append(result)

    # Apply metadata filters (post-retrieval)
    if has_filters:
        all_results = _apply_metadata_filters(all_results, project=project, git_branch=git_branch, tags=tags)

    # Sort by best chunk score (lower = better: L2 distance for FAISS, negated RRF for hybrid)
    all_results.sort(key=lambda r: r["_score"])

    # Override relevance for non-reranked results with rank-based scoring
    # (absolute hybrid/FAISS scores aren't meaningful as relevance values)
    # Capped at 0.75 because without cross-encoder validation we can't
    # claim high confidence — avoids inflated scores on irrelevant results
    NON_RERANKED_MAX_RELEVANCE = 0.75
    for i, r in enumerate(all_results[:limit]):
        if not r.get("_reranked"):
            rank_relevance = round(NON_RERANKED_MAX_RELEVANCE / (1.0 + 0.12 * i), 3)
            r["relevance"] = rank_relevance
            for j, chunk in enumerate(r.get("matchedChunks", [])):
                chunk["relevance"] = round(max(0.1, rank_relevance * (1.0 - 0.1 * j)), 3)

    # Graph context enrichment: annotate results with graph entity context
    if store.graph and detected_entities:
        for r in all_results[:limit]:
            title = r.get("title", "")
            result_entities = store.graph.detect_entities(title)
            contexts = []
            for eid in result_entities:
                ctx = store.graph.get_entity_context(eid)
                if ctx:
                    contexts.append(ctx)
            if contexts:
                r["graph_context"] = contexts[:3]

    for r in all_results:
        r.pop("_score", None)
        r.pop("_reranked", None)
        for chunk in r.get("matchedChunks", []):
            chunk.pop("score", None)
    response = {"results": all_results[:limit]}
    if graph_answer:
        response["graph_answer"] = graph_answer
    if any_low_confidence:
        response["lowConfidence"] = True
    if trace_enabled:
        response["trace"] = trace_obj.to_dict()
    return response


@app.get("/api/graph/{node_id:path}")
def get_graph_node(node_id: str):
    """Inspect a knowledge graph node and its relationships."""
    if not store.graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    detail = store.graph.get_node_detail(node_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return detail


@app.get("/api/collection/{name}/documents")
def list_collection_documents(name: str):
    """List all documents in a collection with their IDs and URLs."""
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    # Use index mapping for O(1) lookup instead of reading every document file
    try:
        mapping_text = store.disk_persister.read_text_file(
            f"{name}/indexes/index_document_mapping.json"
        )
        mapping = json.loads(mapping_text)
    except Exception:
        return {"documents": []}

    seen_ids = set()
    documents = []
    docs_prefix = f"{name}/documents/"
    for entry in mapping.values():
        doc_id = entry.get("documentId", "")
        doc_url = entry.get("documentUrl", "")
        if doc_id in seen_ids or not doc_url:
            continue
        seen_ids.add(doc_id)
        documents.append({"id": doc_id, "url": doc_url})

    return {"documents": documents}


_EMPTY_GRAPH = {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0, "avg_similarity": 0.0},
                 "communities": []}


def _detect_communities(sim_matrix, doc_ids, nodes, min_similarity=0.5):
    """Run Louvain community detection on the similarity matrix.

    Builds a networkx graph from document pairs above min_similarity,
    then finds communities. Returns list of community dicts and updates
    each node with its community ID.
    """
    import numpy as np
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    num_docs = len(doc_ids)
    G = nx.Graph()
    G.add_nodes_from(range(num_docs))

    # Vectorized edge filtering — avoids O(n^2) Python loop
    rows, cols = np.where(np.triu(sim_matrix, k=1) >= min_similarity)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c), weight=float(sim_matrix[r, c]))

    # Remove isolated nodes (no edges above threshold) before community detection
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_nodes() == 0:
        # All nodes are isolated — assign each to its own community
        for i, node in enumerate(nodes):
            node["community"] = i
        return []

    communities = louvain_communities(G, weight="weight", resolution=1.0, seed=42)

    # Sort communities by size (largest first)
    communities = sorted(communities, key=len, reverse=True)

    # Build node-to-community mapping
    node_to_community = {}
    for comm_id, members in enumerate(communities):
        for member_idx in members:
            node_to_community[member_idx] = comm_id

    # Assign isolated nodes to their own communities starting after detected ones
    next_comm = len(communities)
    for idx in isolates:
        node_to_community[idx] = next_comm
        next_comm += 1

    # Update nodes with community ID
    for i, node in enumerate(nodes):
        node["community"] = node_to_community.get(i, -1)

    # Build community summaries
    community_info = []
    for comm_id, members in enumerate(communities):
        member_nodes = [nodes[idx] for idx in members]
        # Count tags across members
        tag_counts = {}
        for mn in member_nodes:
            for tag in mn.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]

        # Count categories
        cat_counts = {}
        for mn in member_nodes:
            cat = mn.get("category", "")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        top_categories = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]

        # Representative titles (top 3 most connected within community)
        member_set = set(members)
        internal_degree = {}
        for idx in members:
            deg = sum(1 for neighbor in G.neighbors(idx) if neighbor in member_set)
            internal_degree[idx] = deg
        top_members = sorted(members, key=lambda x: -internal_degree.get(x, 0))[:3]
        representative_titles = [nodes[idx]["title"] for idx in top_members]

        # Generate a readable community name from top tags/categories
        if top_tags:
            name_parts = [t for t, _ in top_tags[:2]]
        elif top_categories:
            name_parts = [c for c, _ in top_categories[:2]]
        else:
            name_parts = [f"Cluster {comm_id}"]
        community_name = " + ".join(name_parts)

        community_info.append({
            "id": comm_id,
            "name": community_name,  # may be deduplicated below
            "size": len(members),
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "top_categories": [{"category": c, "count": cnt} for c, cnt in top_categories],
            "representative_docs": representative_titles,
        })

    # Deduplicate names: append representative doc title for collisions
    name_counts = {}
    for c in community_info:
        name_counts[c["name"]] = name_counts.get(c["name"], 0) + 1
    for name_val, count in name_counts.items():
        if count <= 1:
            continue
        for c in community_info:
            if c["name"] == name_val and c["representative_docs"]:
                # Shorten the first representative doc title for disambiguation
                doc_hint = c["representative_docs"][0]
                if len(doc_hint) > 30:
                    doc_hint = doc_hint[:27] + "..."
                c["name"] = f"{name_val}: {doc_hint}"

    return community_info


def _build_similarity_graph(name, searcher):
    """Compute the full similarity graph for a collection (cached by caller)."""
    import faiss
    import numpy as np

    indexer = searcher.indexer
    faiss_indexer = indexer.faiss_indexer if hasattr(indexer, "faiss_indexer") else indexer
    idx = faiss_indexer.faiss_index  # IndexIDMap wrapping IndexFlatL2

    n_vectors = idx.ntotal
    if n_vectors == 0:
        return None

    all_vectors = idx.index.reconstruct_n(0, n_vectors)
    id_map = faiss.vector_to_array(idx.id_map)

    try:
        mapping_text = store.disk_persister.read_text_file(
            f"{name}/indexes/index_document_mapping.json"
        )
        mapping = json.loads(mapping_text)
    except Exception:
        logger.warning(f"Could not load index mapping for {name}")
        return None

    doc_chunks = {}
    doc_meta = {}
    for vec_idx, chunk_id in enumerate(id_map):
        entry = mapping.get(str(int(chunk_id)))
        if not entry:
            continue
        doc_url = entry.get("documentUrl", "")
        doc_id = entry["documentId"]
        doc_chunks.setdefault(doc_id, []).append(vec_idx)
        if doc_id not in doc_meta:
            doc_meta[doc_id] = {"url": doc_url, "path": entry.get("documentPath", "")}

    if not doc_chunks:
        return None

    # Mean-pool chunk vectors into document vectors, normalize for cosine similarity
    doc_ids = list(doc_chunks.keys())
    dim = all_vectors.shape[1]
    doc_vectors = np.zeros((len(doc_ids), dim), dtype=np.float32)
    for i, doc_id in enumerate(doc_ids):
        doc_vectors[i] = all_vectors[doc_chunks[doc_id]].mean(axis=0)
    faiss.normalize_L2(doc_vectors)

    # Compute full pairwise cosine similarity via inner product
    # A single matrix multiply is faster than building a FAISS index at this scale
    sim_matrix = doc_vectors @ doc_vectors.T

    nodes = []
    for doc_id in doc_ids:
        meta = doc_meta[doc_id]
        title = doc_id.rsplit("/", 1)[-1].replace(".md", "")
        category = doc_id.split("/")[0] if "/" in doc_id else "uncategorized"
        doc_date = None
        headings = []
        summary = ""
        tags_list = []
        try:
            doc_json = json.loads(store.disk_persister.read_text_file(
                f"{name}/documents/{doc_id}.json"
            ))
            stored_meta = doc_json.get("metadata") or {}
            chunk_meta = (doc_json.get("chunks") or [{}])[0].get("metadata", {})
            doc_date = chunk_meta.get("date") or stored_meta.get("date")

            # Derive category: chunk category (YouTube), first tag (Jira/Confluence), epic, fallback
            if chunk_meta.get("category"):
                category = chunk_meta["category"]
            elif stored_meta.get("tags"):
                first_tag = stored_meta["tags"].split(",")[0].strip()
                if first_tag:
                    category = first_tag
            elif stored_meta.get("epic_summary"):
                category = stored_meta["epic_summary"]

            if stored_meta.get("title"):
                title = stored_meta["title"]
            if stored_meta.get("tags"):
                tags_list = [t.strip() for t in stored_meta["tags"].split(",") if t.strip()]

            headings = [c["heading"] for c in doc_json.get("chunks", []) if c.get("heading")]
            text = doc_json.get("text", "")
            if text:
                summary = text[:500].rstrip() + ("..." if len(text) > 500 else "")
        except Exception:
            logger.debug(f"Could not read metadata for {doc_id}")
        if not tags_list:
            tags_list = [t.strip() for t in category.split("/") if t.strip()]
        nodes.append({
            "id": doc_id,
            "title": title,
            "url": meta["url"],
            "category": category,
            "tags": tags_list,
            "date": doc_date,
            "headings": headings,
            "summary": summary,
        })

    # Run community detection on the full similarity matrix
    # Use 75th percentile as threshold — keeps top 25% of connections
    import numpy as np
    upper_tri = sim_matrix[np.triu_indices(len(doc_ids), k=1)]
    p75 = float(np.percentile(upper_tri, 75)) if len(upper_tri) > 0 else 0.5
    communities = _detect_communities(sim_matrix, doc_ids, nodes, min_similarity=p75)

    return {"nodes": nodes, "sim_matrix": sim_matrix, "doc_ids": doc_ids, "communities": communities}


@app.get("/api/collection/{name}/similarity-graph")
def collection_similarity_graph(
    name: str,
    top_k: int = Query(5, ge=1, le=20),
    min_similarity: float = Query(0.65, ge=0.0, le=1.0),
):
    """Build a document similarity graph from FAISS embeddings (mean-pooled per document)."""
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    # Use cached graph or compute on first request
    cached = store._similarity_graph_cache.get(name)
    if not cached:
        searcher = store.get_searchers([name])[name]
        cached = _build_similarity_graph(name, searcher)
        if not cached:
            return _EMPTY_GRAPH
        store._similarity_graph_cache[name] = cached

    import numpy as np

    nodes = cached["nodes"]
    sim_matrix = cached["sim_matrix"]
    doc_ids = cached["doc_ids"]
    n = len(doc_ids)
    k = min(top_k, n - 1)

    # For each document, find top-k neighbors above threshold
    edges = []
    seen_pairs = set()
    for i in range(n):
        row = sim_matrix[i]
        # Get top-(k+1) indices, exclude self
        top_indices = np.argpartition(row, -(k + 1))[-(k + 1):]
        for idx in top_indices:
            j = int(idx)
            if j == i:
                continue
            sim = float(row[j])
            if sim < min_similarity:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({
                "source": doc_ids[i],
                "target": doc_ids[j],
                "similarity": round(sim, 4),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": cached.get("communities", []),
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "community_count": len(cached.get("communities", [])),
            "avg_similarity": round(sum(e["similarity"] for e in edges) / max(len(edges), 1), 4),
        },
    }


@app.get("/api/collection/{name}/author-graph")
def collection_author_graph(
    name: str,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    min_tweets: int = Query(3, ge=1, le=100),
    min_interactions: int = Query(1, ge=1, le=100),
):
    """Serve the author interaction graph for a collection.

    Reads pre-computed author scores from huginn-jarvis and transforms
    them into the same node/edge/community format as similarity-graph.
    Only includes authors that have at least one interaction edge (no isolates).
    Results are cached per collection; invalidated on collection reload.
    """
    from collections import defaultdict

    cache_key = name
    cached = store._author_graph_cache.get(cache_key)
    if cached:
        return cached

    scores_path = Path(__file__).parent / "huginn-jarvis" / "data" / f"{name}-author-scores.json"
    if not scores_path.exists():
        raise HTTPException(status_code=404, detail=f"No author graph found for '{name}'")

    data = json.loads(scores_path.read_text())

    # Pre-filter candidates by score and tweet count
    candidates = {
        handle for handle, info in data.items()
        if info.get("author_score", 0) >= min_score
        and info.get("tweet_count", 0) >= min_tweets
    }

    # Build edges first so we can filter to connected nodes only
    tweet_dir = Path(__file__).parent / "data" / "sources" / name
    interaction_counts: dict[tuple[str, str], float] = defaultdict(float)
    if tweet_dir.exists():
        re_handle = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+?)_\d+\.md$")
        re_quoted = re.compile(r"> \*\*Quoted @(\w+):")
        re_mention = re.compile(r"(?<![.\w])@(\w{1,15})(?!\.\w)")

        for f in tweet_dir.glob("*.md"):
            m = re_handle.match(f.name)
            if not m:
                continue
            src = m.group(1).lower()
            if src not in candidates:
                continue
            content = f.read_text(encoding="utf-8")
            body = content
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    body = content[end + 3:]

            for qh in re_quoted.findall(body):
                tgt = qh.lower()
                if tgt in candidates and tgt != src:
                    interaction_counts[(src, tgt)] += 3.0
            for line in body.split("\n"):
                if line.startswith("# @") or line.startswith("> **Quoted @") or line.startswith("- **Engagement"):
                    continue
                for mh in re_mention.findall(line):
                    tgt = mh.lower()
                    if tgt in candidates and tgt != src:
                        interaction_counts[(src, tgt)] += 1.0

    # Filter to only connected handles (have at least one edge)
    connected = set()
    for (src, tgt), weight in interaction_counts.items():
        if weight >= min_interactions:
            connected.add(src)
            connected.add(tgt)

    # Remap sparse community IDs to contiguous 0, 1, 2...
    orig_communities = set()
    for handle in connected:
        orig_communities.add(data[handle].get("community", -1))
    comm_remap = {old: new for new, old in enumerate(sorted(orig_communities))}

    # Build nodes (only connected authors)
    nodes = []
    for handle in connected:
        info = data[handle]
        nodes.append({
            "id": handle,
            "title": f"@{handle}",
            "url": f"https://x.com/{handle}",
            "category": f"community-{comm_remap.get(info.get('community', -1), 0)}",
            "tags": [f"tweets:{info.get('tweet_count', 0)}"],
            "date": None,
            "headings": [],
            "summary": (
                f"Score: {info.get('author_score', 0):.3f} | "
                f"PageRank: {info.get('pagerank_norm', 0):.3f} | "
                f"Avg engagement: {info.get('avg_engagement', 0):.1f} | "
                f"Tweets: {info.get('tweet_count', 0)}"
            ),
            "community": comm_remap.get(info.get("community", -1), -1),
            "score": info.get("author_score", 0),
        })

    # Sort nodes by score descending
    nodes.sort(key=lambda n: -n["score"])

    # Build edges with normalized weights
    edges = []
    max_weight = max(interaction_counts.values()) if interaction_counts else 1.0
    for (src, tgt), weight in interaction_counts.items():
        if weight < min_interactions or src not in connected or tgt not in connected:
            continue
        edges.append({
            "source": src,
            "target": tgt,
            "similarity": round(weight / max_weight, 4),
        })

    # Build community summaries
    comm_members: dict[int, list] = defaultdict(list)
    for node in nodes:
        comm_members[node["community"]].append(node)

    communities = []
    for cid, members in sorted(comm_members.items(), key=lambda x: -len(x[1])):
        if len(members) < 2:
            continue
        top_authors = sorted(members, key=lambda n: -n["score"])
        name_parts = [n["title"] for n in top_authors[:2]]
        communities.append({
            "id": cid,
            "name": " + ".join(name_parts),
            "size": len(members),
            "top_tags": [],
            "top_categories": [],
            "representative_docs": [n["title"] for n in top_authors[:3]],
        })

    result = {
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "community_count": len(communities),
            "avg_similarity": round(sum(e["similarity"] for e in edges) / max(len(edges), 1), 4),
        },
    }
    store._author_graph_cache[cache_key] = result
    return result


@app.get("/api/document/{collection}/{doc_id:path}")
def get_document(collection: str, doc_id: str):
    if not store.has_collection(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

    # Prevent absolute paths (actual traversal is caught by realpath check below)
    if doc_id.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc_path = f"{collection}/documents/{doc_id}"
    if not doc_id.endswith(".json"):
        doc_path += ".json"

    # Validate resolved path stays within collections directory
    base_dir = os.path.realpath(store.disk_persister.base_path)
    resolved = os.path.realpath(os.path.join(base_dir, doc_path))
    if not resolved.startswith(base_dir + os.sep):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        doc_text = store.disk_persister.read_text_file(doc_path)
        return json.loads(doc_text)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


@app.get("/api/notion/page/{notion_id}")
def get_notion_page(
    notion_id: str,
    source: str = Query("auto", description="Source: auto (live→local fallback), live (API only), local (index only)"),
):
    """Fetch page content from Notion API and/or local index."""
    if source not in ("auto", "live", "local"):
        raise HTTPException(status_code=400, detail=f"Invalid source '{source}'. Must be one of: auto, live, local")
    if source == "local":
        local_content = _find_local_page_by_notion_id(notion_id)
        if local_content:
            return local_content
        raise HTTPException(status_code=404, detail=f"Page '{notion_id}' not found in local index")

    local_content = _find_local_page_by_notion_id(notion_id) if source == "auto" else None

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        if source == "live":
            raise HTTPException(status_code=503, detail="NOTION_TOKEN not configured")
        if local_content:
            return local_content
        raise HTTPException(status_code=503, detail="NOTION_TOKEN not configured and no local content found")

    try:
        from notion_client import Client
        from main.sources.notion.notion_block_to_markdown import convert_blocks_to_markdown, extract_page_properties

        notion = Client(auth=token)
        page = notion.pages.retrieve(page_id=notion_id)
        _resolve_relation_titles(notion, page)

        all_blocks = _fetch_all_blocks(notion, notion_id)
        properties_md = extract_page_properties(page.get("properties", {}))
        blocks_md = convert_blocks_to_markdown(all_blocks)

        content_parts = [p for p in [properties_md, blocks_md] if p]
        markdown = "\n\n".join(content_parts)

        return {
            "id": notion_id,
            "title": NotionDocumentReader.get_page_title(page),
            "url": page.get("url", ""),
            "lastEdited": page.get("last_edited_time", ""),
            "content": markdown,
        }
    except Exception as e:
        logger.error(f"Notion API error for page {notion_id}: {e}")
        if source == "live":
            raise HTTPException(status_code=502, detail=f"Notion API error: {e}")
        if local_content:
            return local_content
        raise HTTPException(status_code=502, detail=f"Notion API error: {e}")


CATEGORIES = [
    "ai/claude-code", "ai/claude", "ai/openclaw", "ai/general", "ai/rag",
    "health", "tech", "career", "parenting", "entertainment", "coding",
]

SUMMARIZE_PROMPT = """Summarize this YouTube video transcript into structured key insights.

Format rules:
- Use ### for section headers that group related points
- Use numbered lists or bullet points with emoji prefixes for each insight
- Use **bold** for key terms, concepts, and important data points
- Keep it concise but capture all important points and actionable takeaways
- Each point should be self-contained and informative

Also pick the single best category from this list: {categories}

Video title: {title}

Transcript:
{transcript}

Respond in this exact format:
CATEGORY: <one category from the list>

SUMMARY:
<your markdown summary>"""


_GENERIC_TITLES = {"youtube", "youtube.com", "(1) youtube", "(2) youtube", "(3) youtube", ""}


def _fetch_youtube_title(video_id: str) -> str | None:
    """Fetch the real video title from YouTube's oembed API."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("title")
    except Exception as e:
        logger.warning(f"Failed to fetch YouTube title for {video_id}: {e}")
        return None


def _extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return as-is."""
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url_or_id)
    if match:
        return match.group(1)
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url_or_id):
        return url_or_id
    raise HTTPException(status_code=400, detail=f"Invalid YouTube URL or video ID: {url_or_id}")


def _fetch_youtube_transcript(video_id_or_url: str) -> str:
    """Fetch YouTube transcript server-side using existing YouTubeTranscriptDownloader."""
    video_id = _extract_video_id(video_id_or_url)
    downloader = YouTubeTranscriptDownloader(max_retries=3, prefer_languages=["en"])

    transcript_data = downloader.download_transcript(video_id)
    if not transcript_data or not transcript_data.get("available"):
        raise HTTPException(status_code=422, detail=f"No transcript available for video {video_id}")

    text = downloader.format_transcript_plain(transcript_data["segments"])
    if not text.strip():
        raise HTTPException(status_code=422, detail="Transcript is empty")

    logger.info(f"Fetched transcript for {video_id}: {len(text)} chars")
    return text


def _call_claude_headless(prompt: str, model: str = None) -> str:
    """Call Claude Code CLI in headless mode (uses Max subscription, no API key needed).

    Passes prompt via stdin (not CLI arg) to avoid OS ARG_MAX limits with long transcripts.
    """
    model = model or os.environ.get("CLAUDE_MODEL", "sonnet")
    cmd = ["claude", "-p", "-", "--output-format", "json", "--model", model]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            input=prompt,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Claude CLI not found. Install Claude Code first.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Claude CLI timed out after 180s")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()[:500] if proc.stderr else "unknown error"
        raise HTTPException(status_code=502, detail=f"Claude CLI error: {stderr}")

    # Parse JSON output — result text is in the "result" field
    try:
        result = json.loads(proc.stdout)
        return result.get("result", proc.stdout)
    except json.JSONDecodeError:
        # Fallback: raw stdout is the response text
        return proc.stdout.strip()


def _parse_claude_response(text: str) -> tuple[str, str]:
    """Parse CATEGORY and SUMMARY from Claude's response."""
    category = "ai/general"
    summary = text

    if "CATEGORY:" in text and "SUMMARY:" in text:
        parts = text.split("SUMMARY:", 1)
        cat_line = parts[0].strip()
        summary = parts[1].strip()
        # Extract category
        for line in cat_line.split("\n"):
            if line.startswith("CATEGORY:"):
                cat = line.replace("CATEGORY:", "").strip()
                if cat in CATEGORIES:
                    category = cat
                break

    return category, summary


@app.post("/api/youtube/ingest")
def youtube_ingest(req: YouTubeIngestRequest, background_tasks: BackgroundTasks):
    """Ingest a YouTube transcript: summarize via Claude, auto-categorize, save, index, return similar."""
    yt_path = app.state.youtube_transcripts_path
    yt_collection = app.state.youtube_collection
    if not yt_path:
        raise HTTPException(status_code=503, detail="YouTube transcripts path not configured")

    date = req.date or dt.date.today().isoformat()

    # Validate title — Chrome extension sometimes sends "YouTube" or the URL before page loads
    title = req.title
    title_lower = title.lower().strip()
    is_generic = title_lower in _GENERIC_TITLES
    is_url = "youtube.com/" in title_lower or "youtu.be/" in title_lower
    if is_generic or is_url:
        video_id = _extract_video_id(req.video_id or req.url)
        real_title = _fetch_youtube_title(video_id)
        if real_title:
            title = real_title
            logger.info(f"Replaced generic title '{req.title}' with '{real_title}'")
        else:
            raise HTTPException(status_code=400, detail=f"Title '{req.title}' is too generic and couldn't fetch real title from YouTube")

    # If pre-made summary provided (e.g. from javrvis streaming), skip transcript fetch + Claude
    if req.summary:
        summary = req.summary
        category = req.category or "ai/general"
    else:
        # Fetch transcript server-side if not provided
        transcript = req.transcript
        if not transcript:
            transcript = _fetch_youtube_transcript(req.video_id or req.url)

        # Call Claude to summarize + categorize
        prompt = SUMMARIZE_PROMPT.format(
            categories=", ".join(CATEGORIES),
            title=title,
            transcript=transcript[:100000],
        )
        claude_response = _call_claude_headless(prompt)
        auto_category, summary = _parse_claude_response(claude_response)
        category = req.category or auto_category
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}")
    tags = ", ".join(category.split("/"))

    # Build markdown content
    frontmatter = f"---\ndate: {date}\nurl: {req.url}\ncategory: {category}\ntags: \"{tags}\"\n---\n\n"
    md_content = frontmatter + summary

    # Save file (detect duplicates by checking existing file's URL in frontmatter)
    category_dir = os.path.join(yt_path, category)
    os.makedirs(category_dir, exist_ok=True)
    base_filename = sanitize_filename(title)
    filename = base_filename + ".md"
    filepath = os.path.join(category_dir, filename)

    if os.path.exists(filepath):
        # Check if it's the same video (same URL) — overwrite is fine
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_content = f.read(500)
            if f"url: {req.url}" not in existing_content:
                # Different video, same title — add numeric suffix
                for i in range(2, 100):
                    filename = f"{base_filename} ({i}).md"
                    filepath = os.path.join(category_dir, filename)
                    if not os.path.exists(filepath):
                        break
        except Exception:
            pass

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    file_rel_path = os.path.join(category, filename)
    logger.info(f"YouTube ingest: saved {file_rel_path} (category: {category})")

    # Search for similar videos
    similar = []
    if yt_collection and store.has_collection(yt_collection):
        searcher = store.get_searchers([yt_collection]).get(yt_collection)
        if searcher:
            search_result = searcher.search(
                summary[:2000],
                max_number_of_chunks=30,
                max_number_of_documents=5,
                include_matched_chunks_content=True,
            )
            for doc in search_result.get("results", []):
                doc_url = doc.get("url", "")
                if doc_url == req.url:
                    continue
                doc_title = doc.get("path", "").rsplit("/", 1)[-1].replace(".json", "")
                chunks = doc.get("matchedChunks", [])
                snippet = ""
                if chunks:
                    raw = chunks[0].get("content", "")
                    snippet = _truncate_snippet(_extract_chunk_text(raw))
                similar.append({
                    "title": doc_title,
                    "url": doc_url,
                    "snippet": snippet,
                })

    # Trigger background reindex
    if yt_collection and store.has_collection(yt_collection):
        background_tasks.add_task(run_collection_update, yt_collection)

    return {
        "status": "ingested",
        "file_path": file_rel_path,
        "category": category,
        "summary": summary,
        "similar": similar[:5],
    }


@app.get("/api/youtube/transcript/{video_id}")
def youtube_transcript(video_id: str):
    """Fetch raw YouTube transcript without summarizing. Used by javrvis to get transcript for its own Claude call."""
    text = _fetch_youtube_transcript(video_id)
    return {"video_id": video_id, "transcript": text, "char_count": len(text)}


@app.get("/api/youtube/categories")
def youtube_categories():
    """List available YouTube transcript categories."""
    yt_path = app.state.youtube_transcripts_path
    if not yt_path:
        raise HTTPException(status_code=503, detail="YouTube transcripts path not configured")
    categories = []
    for root, dirs, files in os.walk(yt_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "project-notes"]
        rel = os.path.relpath(root, yt_path)
        md_count = sum(1 for f in files if f.endswith(".md"))
        if md_count > 0 and rel != ".":
            categories.append({"name": rel, "count": md_count})
    categories.sort(key=lambda c: c["name"])
    return {"categories": categories}


# ── X article ingest ──────────────────────────────────────────────────────


@app.post("/api/x-articles/ingest")
def x_article_ingest(req: XArticleIngestRequest, background_tasks: BackgroundTasks):
    """Ingest an X/Twitter article: save summary as markdown, find similar, reindex."""
    xa_path = app.state.x_articles_sources_path
    xa_collection = app.state.x_articles_collection
    if not xa_path:
        raise HTTPException(status_code=503, detail="X articles sources path not configured (--x-articles-sources-path)")

    date = req.date or dt.date.today().isoformat()

    # Validate / auto-detect category
    category = req.category or "ai/general"
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}")

    # Build tags from explicit tags + category parts
    tag_parts = list(category.split("/"))
    if req.tags:
        for t in req.tags:
            if t not in tag_parts:
                tag_parts.append(t)
    tags = ", ".join(tag_parts)

    # Build markdown content
    frontmatter = (
        f"---\n"
        f"date: {date}\n"
        f"url: {req.url}\n"
        f"author: {req.author}\n"
        f"category: {category}\n"
        f"tags: \"{tags}\"\n"
        f"---\n\n"
    )
    md_content = frontmatter + req.summary

    # Save file under category subdirectory
    category_dir = os.path.join(xa_path, category)
    os.makedirs(category_dir, exist_ok=True)
    base_filename = sanitize_filename(req.title)
    filename = base_filename + ".md"
    filepath = os.path.join(category_dir, filename)

    if os.path.exists(filepath):
        # Check if it's the same article (same URL) — overwrite is fine
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_content = f.read(500)
            if f"url: {req.url}" not in existing_content:
                for i in range(2, 100):
                    filename = f"{base_filename} ({i}).md"
                    filepath = os.path.join(category_dir, filename)
                    if not os.path.exists(filepath):
                        break
        except Exception:
            pass

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    file_rel_path = os.path.join(category, filename)
    logger.info(f"X article ingest: saved {file_rel_path} (author: {req.author}, category: {category})")

    # Search for similar articles
    similar = []
    if xa_collection and store.has_collection(xa_collection):
        searcher = store.get_searchers([xa_collection]).get(xa_collection)
        if searcher:
            search_result = searcher.search(
                req.summary[:2000],
                max_number_of_chunks=30,
                max_number_of_documents=5,
                include_matched_chunks_content=True,
            )
            for doc in search_result.get("results", []):
                doc_url = doc.get("url", "")
                if doc_url == req.url:
                    continue
                doc_title = doc.get("path", "").rsplit("/", 1)[-1].replace(".json", "")
                chunks = doc.get("matchedChunks", [])
                snippet = ""
                if chunks:
                    raw = chunks[0].get("content", "")
                    snippet = _truncate_snippet(_extract_chunk_text(raw))
                similar.append({
                    "title": doc_title,
                    "url": doc_url,
                    "snippet": snippet,
                })

    # Trigger background reindex
    if xa_collection and store.has_collection(xa_collection):
        background_tasks.add_task(run_collection_update, xa_collection)

    return {
        "status": "ingested",
        "file_path": file_rel_path,
        "author": req.author,
        "category": category,
        "summary": req.summary,
        "similar": similar[:5],
    }


# ── Jira ingest ────────────────────────────────────────────────────────────


def _find_existing_jira_file(jira_path: str, issue_key: str) -> Optional[str]:
    """Find an existing markdown file for a Jira issue key.

    Scans files starting with the issue key (handles both underscore and space
    filename conventions) and verifies issue_key in frontmatter.
    """
    if not os.path.isdir(jira_path):
        return None
    prefix = issue_key
    for filename in os.listdir(jira_path):
        if not filename.endswith(".md"):
            continue
        # Match "PROJECT-1234_..." or "PROJECT-1234 ..." or "PROJECT-1234.md"
        if filename.startswith(prefix + "_") or filename.startswith(prefix + " ") or filename == prefix + ".md":
            filepath = os.path.join(jira_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    head = f.read(500)
                if f"issue_key: {issue_key}" in head:
                    return filepath
            except Exception:
                pass
    return None


def _read_existing_frontmatter(filepath: str) -> dict:
    """Read YAML frontmatter from an existing markdown file into a dict.

    Handles multi-line YAML lists (e.g. labels) by collecting list items.
    """
    metadata = {}
    current_list_key = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            in_fm = False
            for line in f:
                if line.strip() == "---" and not in_fm:
                    in_fm = True
                    continue
                if line.strip() == "---" and in_fm:
                    break
                if not in_fm:
                    continue
                # List item (e.g. "  - frontend")
                stripped = line.strip()
                if stripped.startswith("- ") and current_list_key:
                    item = stripped[2:].strip()
                    existing = metadata.get(current_list_key, "")
                    metadata[current_list_key] = (existing + "," + item) if existing else item
                    continue
                current_list_key = None
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key and value:
                        metadata[key] = value
                    elif key and not value:
                        # Key with no value — likely start of a YAML list
                        current_list_key = key
    except Exception:
        pass
    return metadata


def _jira_content_to_markdown(req: "JiraIngestRequest", existing_metadata: Optional[dict] = None) -> str:
    """Convert DOM-scraped Jira issue content to markdown with frontmatter.

    If existing_metadata is provided, preserves fields the Chrome extension
    doesn't capture (epic_summary, project, modifiedTime).
    """
    key = req.issueKey
    summary = req.summary or req.title or ""
    existing = existing_metadata or {}

    def _yaml_escape(val: str) -> str:
        """Wrap value in quotes and escape internal quotes for safe YAML."""
        return '"' + val.replace('\\', '\\\\').replace('"', '\\"') + '"'

    # Frontmatter — use fresh data from extension, fall back to existing
    lines = ["---"]
    lines.append(f"title: {_yaml_escape(summary)}")
    lines.append(f"issue_key: {key}")
    lines.append(f"summary: {_yaml_escape(summary)}")
    lines.append(f"status: {_yaml_escape(req.status or existing.get('status', ''))}")
    lines.append(f"issue_type: {_yaml_escape(req.type or existing.get('issue_type', ''))}")
    lines.append(f"priority: {_yaml_escape(req.priority or existing.get('priority', ''))}")
    lines.append(f"created: {_yaml_escape(req.created or existing.get('created', ''))}")
    updated = req.updated or existing.get('updated', '')
    lines.append(f"updated: {_yaml_escape(updated)}")
    lines.append(f"modifiedTime: {_yaml_escape(updated)}")
    lines.append(f"assignee: {_yaml_escape(req.assignee or existing.get('assignee', ''))}")
    lines.append(f"reporter: {_yaml_escape(req.reporter or existing.get('reporter', ''))}")

    # Labels — write as comma-separated string (not YAML list) for parser compatibility
    if req.labels:
        labels_str = ", ".join(req.labels)
        lines.append(f"labels: {_yaml_escape(labels_str)}")
    elif existing.get('labels'):
        lines.append(f"labels: {_yaml_escape(existing['labels'])}")
    else:
        lines.append(f"labels: {_yaml_escape('')}")

    # Epic — preserve existing epic_summary if extension doesn't provide it
    epic_link = req.epicLink or existing.get('epic_link', '')
    epic_summary = existing.get('epic_summary', '')
    lines.append(f"epic_link: {_yaml_escape(epic_link)}")
    lines.append(f"epic_summary: {_yaml_escape(epic_summary)}")

    # Project — extension doesn't provide, preserve from existing
    project = existing.get('project', key.split('-')[0] if '-' in key else '')
    lines.append(f"project: {_yaml_escape(project)}")

    if req.url:
        lines.append(f"url: {_yaml_escape(req.url)}")
    elif existing.get('url'):
        lines.append(f"url: {_yaml_escape(existing['url'])}")
    lines.append("---\n")

    # Title
    lines.append(f"# {key}: {summary}\n")

    # Epic context in body (if we have it)
    if epic_link and epic_summary:
        base_url = existing.get('url', '').rsplit('/browse/', 1)[0] if existing.get('url') else ''
        if base_url:
            lines.append(f"**Epic:** [{epic_link}]({base_url}/browse/{epic_link}) - {epic_summary}\n")
        else:
            lines.append(f"**Epic:** {epic_link} - {epic_summary}\n")

    # Description
    if req.description:
        lines.append("## Description\n")
        lines.append(req.description + "\n")

    # Comments
    if req.comments:
        lines.append("## Comments\n")
        for comment in req.comments:
            lines.append(f"### {comment.author} ({comment.date})\n")
            lines.append(comment.body + "\n")

    return "\n".join(lines)


@app.post("/api/jira/ingest")
def jira_ingest(req: JiraIngestRequest, background_tasks: BackgroundTasks):
    """Ingest a Jira issue from DOM-scraped content: save as markdown, find similar, reindex.

    If an existing file for this issue_key is found, merges metadata to preserve
    epic_summary, project, and other fields the Chrome extension doesn't capture.
    """
    jira_path = app.state.jira_sources_path
    jira_collection = app.state.jira_collection

    if not jira_path:
        raise HTTPException(status_code=503, detail="Jira sources path not configured (--jira-sources-path)")

    # Validate issue key format
    if not re.match(r"^[A-Z][A-Z0-9]+-\d+$", req.issueKey):
        raise HTTPException(status_code=400, detail=f"Invalid Jira issue key: {req.issueKey}")

    summary_text = req.summary or req.title or "untitled"

    # Check for existing file (any filename convention) to preserve metadata
    os.makedirs(jira_path, exist_ok=True)
    existing_filepath = _find_existing_jira_file(jira_path, req.issueKey)
    existing_metadata = _read_existing_frontmatter(existing_filepath) if existing_filepath else {}

    if existing_metadata:
        logger.info(f"Jira ingest: found existing file for {req.issueKey}, merging metadata")

    # Convert to markdown, merging with existing metadata
    md_content = _jira_content_to_markdown(req, existing_metadata)

    # Sanitize PII before writing
    _sanitize_result = _pii_sanitizer.sanitize(md_content)
    if _sanitize_result.has_pii:
        cats = {}
        for _f in _sanitize_result.findings:
            cats[_f.category] = cats.get(_f.category, 0) + 1
        logger.info(f"Jira ingest PII redacted in {req.issueKey}: "
                    + ", ".join(f"{c}:{n}" for c, n in cats.items()))
        md_content = _sanitize_result.sanitized_text

    # Use existing filename if found, otherwise create new one using underscore convention
    if existing_filepath:
        filepath = existing_filepath
        filename = os.path.basename(filepath)
    else:
        safe_title = re.sub(r'[<>:"/\\|?*]', '', summary_text)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:100].strip('_')
        filename = f"{req.issueKey}_{safe_title}.md"
        filepath = os.path.join(jira_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Set file mtime to issue updated time for correct incremental updates
    updated = req.updated or existing_metadata.get('updated', '')
    if updated:
        try:
            ts = dt.datetime.fromisoformat(updated).timestamp()
            os.utime(filepath, (ts, ts))
        except (ValueError, OSError):
            pass

    logger.info(f"Jira ingest: saved {filename}")

    # Search for similar issues
    similar = []
    if jira_collection and store.has_collection(jira_collection):
        searcher = store.get_searchers([jira_collection]).get(jira_collection)
        if searcher:
            search_q = f"{req.issueKey} {summary_text}"
            search_result = searcher.search(
                search_q,
                max_number_of_chunks=30,
                max_number_of_documents=5,
                include_matched_chunks_content=True,
            )
            for doc in search_result.get("results", []):
                doc_url = doc.get("url", "")
                if req.issueKey in doc_url:
                    continue
                doc_title = doc.get("path", "").rsplit("/", 1)[-1].replace(".json", "")
                chunks = doc.get("matchedChunks", [])
                snippet = ""
                if chunks:
                    raw = chunks[0].get("content", "")
                    snippet = _truncate_snippet(_extract_chunk_text(raw))
                similar.append({
                    "title": doc_title,
                    "url": doc_url,
                    "snippet": snippet,
                })

    # Skip automatic reindex — the daily update script handles both
    # collection reindexing and knowledge graph rebuild in one pass.
    # Use POST /api/collections/{name}/update to trigger manually if needed.

    return {
        "status": "ingested",
        "issue_key": req.issueKey,
        "file_path": filename,
        "summary": summary_text,
        "similar": similar[:5],
    }


@app.post("/api/collections/{name}/update")
def update_collection(name: str, background_tasks: BackgroundTasks):
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    background_tasks.add_task(run_collection_update, name)
    return {"status": "update_started", "collection": name}


def _resolve_relation_titles(notion, page):
    """Resolve relation property IDs to titles for rendering."""
    for prop in page.get("properties", {}).values():
        if prop.get("type") != "relation":
            continue
        for rel in prop.get("relation", []):
            if "id" in rel and "title" not in rel:
                try:
                    related = notion.pages.retrieve(page_id=rel["id"])
                    rel["title"] = NotionDocumentReader.get_page_title(related)
                except Exception:
                    pass


def _find_local_page_by_notion_id(notion_id):
    """Look up locally indexed content by Notion page ID."""
    normalized = notion_id.replace("-", "")
    entry = store.notion_id_to_doc.get(normalized)
    if not entry:
        return None
    try:
        doc = json.loads(store.disk_persister.read_text_file(entry["doc_path"]))
        return {
            "id": notion_id,
            "title": entry["doc_path"].rsplit("/", 1)[-1].replace(".json", ""),
            "url": entry["url"],
            "content": doc.get("text", ""),
            "source": "local_index",
        }
    except Exception:
        return None


def _fetch_all_blocks(notion, block_id, depth=0):
    """Recursively fetch all blocks for a page."""
    if depth > 5:
        return []
    blocks = []
    cursor = None
    while True:
        response = notion.blocks.children.list(block_id=block_id, start_cursor=cursor)
        for block in response.get("results", []):
            blocks.append(block)
            if block.get("has_children"):
                block["children"] = _fetch_all_blocks(notion, block["id"], depth + 1)
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return blocks


def _extract_chunk_text(content):
    """Extract plain text from chunk content (may be dict with indexedData or plain string)."""
    if isinstance(content, dict):
        return content.get("indexedData", str(content))
    return str(content) if content else ""


def _extract_chunk_heading(content):
    """Extract heading from chunk content if available."""
    if isinstance(content, dict):
        return content.get("heading")
    return None


def _extract_chunk_metadata(content):
    """Extract metadata dict from chunk content if available."""
    if isinstance(content, dict):
        return content.get("metadata")
    return None


def _truncate_snippet(text, target=200):
    """Truncate text at a sentence boundary near target length, falling back to word boundary."""
    if not text or len(text) <= target:
        return text
    # Look for sentence boundary in a window around target
    window_start = max(target - 40, 0)
    window_end = min(target + 40, len(text))
    window = text[window_start:window_end]
    # Find last sentence-ending punctuation followed by space in the window
    best = -1
    for m in re.finditer(r'[.!?]\s', window):
        best = m.start() + 1  # include the punctuation
    if best >= 0:
        cut = window_start + best
        return text[:cut].rstrip()
    # Fall back to word boundary
    cut = text.rfind(' ', 0, target + 20)
    if cut > target - 40:
        return text[:cut].rstrip() + "…"
    return text[:target] + "…"


_METADATA_LINE_RE = re.compile(r'^\*\*([^*]+?):\*\*\s*(.+)$')


def _separate_metadata(text):
    """Parse **Key:** Value lines from start of text into a metadata dict.

    Also extracts [Breadcrumb > Path] line for navigation context.
    Returns (clean_content, metadata_dict, breadcrumb_or_None).
    """
    if not text:
        return "", {}, None
    lines = text.split('\n')
    metadata = {}
    breadcrumb = None
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # Skip blank lines at the start
            content_start = i + 1
            continue
        # Extract breadcrumb line
        if stripped.startswith('[') and '>' in stripped and stripped.endswith(']'):
            breadcrumb = stripped[1:-1].strip()
            content_start = i + 1
            continue
        m = _METADATA_LINE_RE.match(stripped)
        if m:
            metadata[m.group(1).strip()] = m.group(2).strip()
            content_start = i + 1
        else:
            break
    clean = '\n'.join(lines[content_start:]).strip()
    return clean, metadata, breadcrumb


def _apply_metadata_filters(results, project=None, git_branch=None, tags=None):
    """Filter results by metadata fields. Checks document-level and chunk-level metadata."""
    filtered = []
    requested_tags = {t.strip() for t in tags.split(",") if t.strip()} if tags else None
    for r in results:
        doc_meta = r.get("metadata") or {}
        # Check chunk-level metadata too (first chunk often has the metadata)
        chunk_meta = {}
        for chunk in r.get("matchedChunks", []):
            if chunk.get("metadata"):
                chunk_meta.update(chunk["metadata"])
        merged = {**doc_meta, **chunk_meta}

        if project and merged.get("project") != project:
            continue
        if git_branch and merged.get("gitBranch") != git_branch:
            continue
        if requested_tags:
            doc_tags = {t.strip() for t in merged.get("tags", "").split(",") if t.strip()}
            if not requested_tags & doc_tags:
                continue
        filtered.append(r)
    return filtered


def _normalize_score(raw_score, is_reranked=True):
    """Convert internal score (lower=better) to 0.0-1.0 relevance (higher=better).

    For reranked results: shifted sigmoid calibrated to cross-encoder score range.
    Maps score -1.0 → ~0.999, -0.5 → ~0.94, -0.15 → ~0.50, -0.01 → ~0.25.

    For non-reranked results: basic sigmoid (rank-based override applied later
    in the search handler for the final response).
    """
    if not is_reranked:
        # Placeholder — search handler overrides with rank-based relevance
        return 0.5

    # Shift and scale to spread cross-encoder scores across 0-1 range
    shifted = (raw_score + 0.15) * 8
    clamped = max(min(shifted, 500), -500)
    return 1.0 / (1.0 + math.exp(clamped))


def main():
    ap = argparse.ArgumentParser(description="Knowledge API Server")
    ap.add_argument(
        "--collections", nargs="+", required=True,
        help="Collections to load (e.g., my-notion)",
    )
    ap.add_argument(
        "--data-path", default=os.environ.get("HUGINN_DATA_PATH", "./data/collections"),
        help="Base path for collection data (default: ./data/collections)",
    )
    ap.add_argument("--port", type=int, default=8321, help="Port to listen on")
    ap.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    ap.add_argument(
        "--youtube-transcripts-path",
        default=os.environ.get("YOUTUBE_TRANSCRIPTS_PATH"),
        help="Path to youtube-transcripts markdown repo",
    )
    ap.add_argument(
        "--youtube-collection",
        default=os.environ.get("YOUTUBE_COLLECTION", "youtube-summaries"),
        help="Collection name for youtube transcripts",
    )
    ap.add_argument(
        "--jira-sources-path",
        default=os.environ.get("JIRA_SOURCES_PATH"),
        help="Path to save Jira issue markdown files",
    )
    ap.add_argument(
        "--jira-collection",
        default=os.environ.get("JIRA_COLLECTION", "jira-issues"),
        help="Collection name for Jira issues",
    )
    ap.add_argument(
        "--x-articles-sources-path",
        default=os.environ.get("X_ARTICLES_SOURCES_PATH"),
        help="Path to save X article summary markdown files",
    )
    ap.add_argument(
        "--x-articles-collection",
        default=os.environ.get("X_ARTICLES_COLLECTION", "x-articles"),
        help="Collection name for X article summaries",
    )
    args = ap.parse_args()

    app.state.data_path = args.data_path
    app.state.collection_names = args.collections
    app.state.youtube_transcripts_path = args.youtube_transcripts_path
    app.state.youtube_collection = args.youtube_collection
    app.state.jira_sources_path = args.jira_sources_path
    app.state.jira_collection = args.jira_collection
    app.state.x_articles_sources_path = args.x_articles_sources_path
    app.state.x_articles_collection = args.x_articles_collection

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
