"""Server-state holder for the HTTP runtime.

KnowledgeStore loads embedder, reranker, searchers, tag counts, Notion ID
lookups and the knowledge graph. It's the single piece of mutable state the
HTTP routers depend on, exposed via the ``get_store`` FastAPI dependency.
"""
import json
import logging
import threading
from datetime import datetime, timezone

from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.indexes.indexer_factory import (
    create_embedder,
    create_reranker,
    detect_faiss_index,
    load_search_indexer,
)
from main.persisters.disk_persister import DiskPersister

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
        self._extra_graph_paths = None
        self._build_aux_indexes = True
        self._lock = threading.Lock()
        # collection_name -> {status, startedAt, finishedAt, error}; guarded by _lock.
        # The "running" status doubles as a per-collection rebuild mutex (H4).
        self._update_states = {}

    def load_collections(self, collection_names, data_path="./data/collections",
                         faiss_index_name=None, extra_graph_paths=None,
                         build_aux_indexes=True):
        """Load collections into the store.

        ``build_aux_indexes`` controls whether tag-count and Notion-ID lookups
        are built; HTTP endpoints need them, stdio MCP adapters do not.
        """
        self.disk_persister = DiskPersister(base_path=data_path)
        self._extra_graph_paths = extra_graph_paths
        self._build_aux_indexes = build_aux_indexes

        if faiss_index_name is None:
            faiss_index_name = self.__detect_shared_faiss_index(collection_names)

        logger.info(f"Loading shared embedding model for index: {faiss_index_name}")
        self.shared_embedder = create_embedder(faiss_index_name)
        logger.info(f"Embedding model loaded: {self.shared_embedder.model_name}")

        self.shared_reranker = create_reranker()
        logger.info(f"Reranker loaded: {self.shared_reranker.model_name}")

        for name in collection_names:
            logger.info(f"Loading collection: {name}")
            try:
                searcher = self._build_searcher(name)
            except Exception:
                # A corrupt index/mapping for one collection must not take down the
                # whole server — log it and keep loading the rest (H5/M6).
                logger.exception(f"Failed to load collection {name}; skipping it")
                continue
            with self._lock:
                self.searchers[name] = searcher
            logger.info(f"Collection {name} loaded with {searcher.indexer.get_size()} embeddings")
            if build_aux_indexes:
                self._build_tag_counts(name)
                self._build_notion_id_lookup(name)

        self._load_knowledge_graph(extra_paths=extra_graph_paths)

    def __detect_shared_faiss_index(self, collection_names):
        """Pick the shared embedding model from the first collection with a detectable
        FAISS index. Probing only collection_names[0] would abort startup if that one
        happened to be the broken collection (H5/M6)."""
        for name in collection_names:
            try:
                return detect_faiss_index(name, self.disk_persister)
            except Exception:
                logger.warning(f"Could not detect FAISS index for {name}; trying next collection")
        raise ValueError(f"No loadable FAISS index found in any of: {collection_names}")

    def _load_knowledge_graph(self, extra_paths=None):
        from main.graph.graph_loader import load_default_knowledge_graph
        self.graph = load_default_knowledge_graph(extra_paths=extra_paths)

    def reload_collection(self, collection_name):
        searcher = self._build_searcher(collection_name)
        with self._lock:
            self.searchers[collection_name] = searcher
            self._similarity_graph_cache.pop(collection_name, None)
            self._author_graph_cache.pop(collection_name, None)
        if self._build_aux_indexes:
            self._build_tag_counts(collection_name)
            self._build_notion_id_lookup(collection_name)
        self._load_knowledge_graph(extra_paths=self._extra_graph_paths)
        logger.info(f"Collection {collection_name} reloaded ({searcher.indexer.get_size()} embeddings)")

    def try_begin_update(self, collection_name):
        """Reserve the rebuild slot for a collection.

        Returns False if an update is already running for it (the caller should
        return 409 or skip), True if the caller now owns the rebuild. The reserved
        slot is released by mark_update_succeeded / mark_update_failed once the
        background task finishes. Acts as the per-collection rebuild mutex (H4),
        so concurrent updates can't race on disk and clobber each other.
        """
        with self._lock:
            state = self._update_states.get(collection_name)
            if state and state.get("status") == "running":
                return False
            self._update_states[collection_name] = {
                "status": "running",
                "startedAt": self.__now(),
                "finishedAt": None,
                "error": None,
            }
            return True

    def mark_update_succeeded(self, collection_name):
        self.__finish_update(collection_name, "succeeded", None)

    def mark_update_failed(self, collection_name, error):
        self.__finish_update(collection_name, "failed", str(error))

    def __finish_update(self, collection_name, status, error):
        with self._lock:
            started_at = self._update_states.get(collection_name, {}).get("startedAt")
            self._update_states[collection_name] = {
                "status": status,
                "startedAt": started_at,
                "finishedAt": self.__now(),
                "error": error,
            }

    def get_update_status(self, collection_name):
        with self._lock:
            state = self._update_states.get(collection_name) or {
                "status": "idle",
                "startedAt": None,
                "finishedAt": None,
                "error": None,
            }
            return {"collection": collection_name, **state}

    @staticmethod
    def __now():
        return datetime.now(timezone.utc).isoformat()

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

    def get_cached_similarity_graph(self, name):
        with self._lock:
            return self._similarity_graph_cache.get(name)

    def set_cached_similarity_graph(self, name, value):
        with self._lock:
            self._similarity_graph_cache[name] = value

    def get_cached_author_graph(self, name):
        with self._lock:
            return self._author_graph_cache.get(name)

    def set_cached_author_graph(self, name, value):
        with self._lock:
            self._author_graph_cache[name] = value

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


_default_store = KnowledgeStore()


def get_store() -> KnowledgeStore:
    """Return the process-wide ``KnowledgeStore`` singleton.

    Used by the HTTP server (via FastAPI ``Depends``) and by the stdio MCP
    adapters as their entry-point store.
    """
    return _default_store


def run_collection_update(collection_name: str, store: KnowledgeStore):
    """Run incremental collection update in background, then reload the searcher.

    Records the outcome on the store so the failure is surfaced (via the
    update-status endpoint) instead of a successful-looking HTTP 200 hiding a
    silently stale collection (H5). The caller is expected to have reserved the
    slot with try_begin_update; the terminal mark here releases it.
    """
    from main.factories.update_collection_factory import create_collection_updater

    logger.info(f"Starting background update for collection: {collection_name}")
    try:
        updater = create_collection_updater(collection_name)
        updater.run()
        store.reload_collection(collection_name)
    except Exception as e:
        logger.exception(f"Failed to update collection {collection_name}")
        store.mark_update_failed(collection_name, e)
        return
    store.mark_update_succeeded(collection_name)
