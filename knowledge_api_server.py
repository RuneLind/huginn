#!/usr/bin/env python3
"""
Knowledge API Server — long-running HTTP API for vector search.

Loads embedding model and FAISS indexes once at startup, serves search
results via HTTP. Designed for low-latency responses (<50ms after warmup).

Usage:
    uv run knowledge_api_server.py --collections my-notion --port 8321
"""
import argparse
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.indexes.indexer_factory import (
    create_embedder,
    create_reranker,
    detect_faiss_index,
    load_search_indexer,
)
from main.persisters.disk_persister import DiskPersister
from main.routes.collections import make_collections_router
from main.routes.graph import make_graph_router
from main.routes.ingest import make_ingest_router
from main.routes.notion import make_notion_router
from main.routes.search import make_search_router
from main.utils.logger import setup_root_logger

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
app.state.huginn_root = Path(__file__).parent

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


app.include_router(make_search_router(store))
app.include_router(make_graph_router(store))
app.include_router(make_notion_router(store))
app.include_router(make_collections_router(store, run_collection_update))
app.include_router(make_ingest_router(store, run_collection_update))


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
