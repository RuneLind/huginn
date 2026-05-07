#!/usr/bin/env python3
"""
Knowledge API Server — long-running HTTP API for vector search.

Loads embedding model and FAISS indexes once at startup, serves search
results via HTTP. Designed for low-latency responses (<50ms after warmup).

Usage:
    uv run knowledge_api_server.py --collections my-notion --port 8321
"""
import json
import argparse
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import datetime as dt

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import detect_faiss_index, create_embedder, load_search_indexer, create_reranker
from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.search_trace import create_trace
from main.core.trace_store import any_trace_enabled, default_trace_store, pointer_mode_enabled
from main.core.search_response_formatter import (
    extract_chunk_text,
    shape_search_results,
    truncate_snippet,
)
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.graph.similarity_graph import (
    EMPTY_GRAPH,
    build_similarity_graph,
    shape_similarity_response,
)
from main.graph.author_graph import build_author_graph
from main.sources.notion.notion_document_reader import NotionDocumentReader
from main.utils.logger import setup_root_logger
from main.ingest.youtube import (
    YouTubeIngestRequest,
    ingest_youtube,
    fetch_transcript as _fetch_youtube_transcript,
    list_categories as _list_youtube_categories,
)
from main.ingest.x_articles import XArticleIngestRequest, ingest_x_article
from main.ingest.jira import JiraIngestRequest, ingest_jira
from main.ingest.categories import CATEGORIES

setup_root_logger()
logger = logging.getLogger(__name__)


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
        from main.graph.graph_loader import load_default_knowledge_graph
        self.graph = load_default_knowledge_graph()

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

    response = {"results": results}
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


@app.get("/api/trace/{trace_id}")
def get_search_trace(trace_id: str):
    """Fetch a stored search trace by ID. 404 once expired (TTL ~5 min)."""
    trace = default_trace_store().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found or expired")
    return trace


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


@app.get("/api/collection/{name}/similarity-graph")
def collection_similarity_graph(
    name: str,
    top_k: int = Query(5, ge=1, le=20),
    min_similarity: float = Query(0.65, ge=0.0, le=1.0),
):
    """Build a document similarity graph from FAISS embeddings (mean-pooled per document)."""
    if not store.has_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    cached = store._similarity_graph_cache.get(name)
    if not cached:
        searcher = store.get_searchers([name])[name]
        cached = build_similarity_graph(name, searcher, store.disk_persister)
        if not cached:
            return EMPTY_GRAPH
        store._similarity_graph_cache[name] = cached

    return shape_similarity_response(cached, top_k, min_similarity)


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
    cached = store._author_graph_cache.get(name)
    if cached:
        return cached

    scores_path = Path(__file__).parent / "huginn-jarvis" / "data" / f"{name}-author-scores.json"
    if not scores_path.exists():
        raise HTTPException(status_code=404, detail=f"No author graph found for '{name}'")

    scores = json.loads(scores_path.read_text())
    result = build_author_graph(scores, name, store.disk_persister, min_score, min_tweets, min_interactions)
    store._author_graph_cache[name] = result
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


def _find_similar_documents(searcher, query: str, exclude_match) -> list[dict]:
    """Run a similarity search and return up to 5 {title, url, snippet} results.

    exclude_match is called on each result; truthy means skip (e.g. self-link).
    """
    search_result = searcher.search(
        query,
        max_number_of_chunks=30,
        max_number_of_documents=5,
        include_matched_chunks_content=True,
    )
    similar = []
    for doc in search_result.get("results", []):
        if exclude_match(doc):
            continue
        doc_title = doc.get("path", "").rsplit("/", 1)[-1].replace(".json", "")
        chunks = doc.get("matchedChunks", [])
        snippet = ""
        if chunks:
            raw = chunks[0].get("content", "")
            snippet = truncate_snippet(extract_chunk_text(raw))
        similar.append({
            "title": doc_title,
            "url": doc.get("url", ""),
            "snippet": snippet,
        })
    return similar[:5]


@app.post("/api/youtube/ingest")
def youtube_ingest(req: YouTubeIngestRequest, background_tasks: BackgroundTasks):
    """Ingest a YouTube transcript: summarize via Claude, auto-categorize, save, index, return similar."""
    yt_path = app.state.youtube_transcripts_path
    yt_collection = app.state.youtube_collection
    if not yt_path:
        raise HTTPException(status_code=503, detail="YouTube transcripts path not configured")

    result = ingest_youtube(req, transcripts_path=yt_path)

    similar = []
    if yt_collection and store.has_collection(yt_collection):
        searcher = store.get_searchers([yt_collection]).get(yt_collection)
        if searcher:
            similar = _find_similar_documents(
                searcher,
                query=result["summary"][:2000],
                exclude_match=lambda doc: doc.get("url", "") == req.url,
            )
        background_tasks.add_task(run_collection_update, yt_collection)

    return {
        "status": "ingested",
        "file_path": result["file_path"],
        "category": result["category"],
        "summary": result["summary"],
        "similar": similar,
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
    return {"categories": _list_youtube_categories(yt_path)}


# ── X article ingest ──────────────────────────────────────────────────────


@app.post("/api/x-articles/ingest")
def x_article_ingest(req: XArticleIngestRequest, background_tasks: BackgroundTasks):
    """Ingest an X/Twitter article: save summary as markdown, find similar, reindex."""
    xa_path = app.state.x_articles_sources_path
    xa_collection = app.state.x_articles_collection
    if not xa_path:
        raise HTTPException(status_code=503, detail="X articles sources path not configured (--x-articles-sources-path)")

    result = ingest_x_article(req, sources_path=xa_path)

    similar = []
    if xa_collection and store.has_collection(xa_collection):
        searcher = store.get_searchers([xa_collection]).get(xa_collection)
        if searcher:
            similar = _find_similar_documents(
                searcher,
                query=req.summary[:2000],
                exclude_match=lambda doc: doc.get("url", "") == req.url,
            )
        background_tasks.add_task(run_collection_update, xa_collection)

    return {
        "status": "ingested",
        "file_path": result["file_path"],
        "author": result["author"],
        "category": result["category"],
        "summary": result["summary"],
        "similar": similar,
    }


# ── Jira ingest ────────────────────────────────────────────────────────────


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

    result = ingest_jira(req, sources_path=jira_path)

    similar = []
    if jira_collection and store.has_collection(jira_collection):
        searcher = store.get_searchers([jira_collection]).get(jira_collection)
        if searcher:
            similar = _find_similar_documents(
                searcher,
                query=f"{req.issueKey} {result['summary']}",
                exclude_match=lambda doc: req.issueKey in doc.get("url", ""),
            )

    # Skip automatic reindex — the daily update script handles both
    # collection reindexing and knowledge graph rebuild in one pass.
    # Use POST /api/collections/{name}/update to trigger manually if needed.

    return {
        "status": "ingested",
        "issue_key": result["issue_key"],
        "file_path": result["file_path"],
        "summary": result["summary"],
        "similar": similar,
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
