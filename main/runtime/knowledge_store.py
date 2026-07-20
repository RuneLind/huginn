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
from main.runtime.indexing_run_ledger import IndexingRunLedger, mint_run_id
from main.utils.frontmatter import parse_tags

logger = logging.getLogger(__name__)


def _to_ledger_ts(value):
    """Normalize an internal isoformat timestamp to the ledger's ...Z form."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except ValueError:
        return value


def _duration_seconds(started_at, finished_at):
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        finish = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    return max(0, int((finish - start).total_seconds()))


class KnowledgeStore:
    """Holds all server state: embedder, persister, searchers, and Notion ID lookup."""

    def __init__(self):
        self.shared_embedder = None
        self.shared_reranker = None
        self.disk_persister = None
        self.searchers = {}
        self.graph = None  # KnowledgeGraph or None
        # Maps normalized notion ID (no dashes) -> {doc_path, url} for fast lookup.
        # Rebuilt by re-merging the per-collection slices below, so a reload drops
        # entries for pages a collection no longer contains instead of accumulating
        # them, and the dict is swapped in whole (never mutated in place) so readers
        # never observe a half-built lookup.
        self.notion_id_to_doc = {}
        # collection name -> {notion_hex: {doc_path, url}}; source of truth re-merged
        # into notion_id_to_doc on every (re)load.
        self._notion_by_collection = {}
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
        # collection_name -> {runId, job, trigger}; guarded by _lock. Deliberately
        # NOT stored in _update_states: (a) __finish_update REPLACES that dict
        # wholesale rather than mutating it, so anything stashed there is dropped
        # before the ledger record is built, and (b) get_update_status returns
        # {"collection": name, **state}, so extra keys would leak onto the public
        # GET /api/collections/{name}/update-status response, which has an
        # exact-dict-equality test and an external consumer.
        self._update_correlation = {}

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
        # Load outside the lock (file I/O), swap the reference in under it so
        # readers see either the whole old graph or the whole new one (M10).
        # Pass the collections base path so stamped graphs can be checked for
        # staleness against their source collection manifests.
        data_path = self.disk_persister.base_path if self.disk_persister else None
        graph = load_default_knowledge_graph(extra_paths=extra_paths, data_path=data_path)
        with self._lock:
            self.graph = graph

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

    def try_begin_update(self, collection_name, run_id=None, job=None, trigger=None,
                         variant="incremental"):
        """Reserve the rebuild slot for a collection.

        Returns False if an update is already running for it (the caller should
        return 409 or skip), True if the caller now owns the rebuild. The reserved
        slot is released by mark_update_succeeded / mark_update_failed once the
        background task finishes. Acts as the per-collection rebuild mutex (H4),
        so concurrent updates can't race on disk and clobber each other.

        ``run_id``/``job``/``trigger`` are the correlation channel: a caller that
        already owns a run id (a launchd shell script wrapping tag+reindex) passes
        it so its own ledger record and huginn's fold into a single run. Absent
        one, huginn mints its own. These live in ``_update_correlation``, never in
        ``_update_states`` — see the comment on that field.
        """
        with self._lock:
            state = self._update_states.get(collection_name)
            if state and state.get("status") == "running":
                return False
            started_at = self.__now()
            self._update_states[collection_name] = {
                "status": "running",
                "startedAt": started_at,
                "finishedAt": None,
                "error": None,
            }
            correlation = {
                "runId": run_id or mint_run_id(collection_name, started_at),
                "job": job,
                "trigger": trigger or "manual",
                "variant": variant or "incremental",
            }
            self._update_correlation[collection_name] = correlation

        # Opening partial, written outside the lock (it is file I/O, and this
        # lock is the one the search hot path takes). Without it a server
        # restarted mid-reindex leaves no trace of the run at all: __finish_update
        # never runs, so nothing is ever appended.
        self.__record_open(collection_name, started_at, correlation)
        return True

    def __record_open(self, collection_name, started_at, correlation):
        try:
            IndexingRunLedger().append({
                "runId": correlation["runId"],
                "collection": collection_name,
                "job": correlation.get("job"),
                "trigger": correlation.get("trigger"),
                "variant": correlation.get("variant"),
                "startedAt": _to_ledger_ts(started_at),
                "source": "huginn",
                "stage": "begin",
            })
        except Exception:
            logger.warning("Could not write opening ledger record for %s",
                           collection_name, exc_info=True)

    def mark_update_succeeded(self, collection_name):
        self.__finish_update(collection_name, "succeeded", None)

    def mark_update_failed(self, collection_name, error):
        self.__finish_update(collection_name, "failed", str(error))

    def __finish_update(self, collection_name, status, error):
        # Everything that touches shared state happens under the lock; the ledger
        # write and the manifest read happen after it is released. This lock is
        # the same one get_searchers() takes on the search hot path, so file I/O
        # inside it would stall every concurrent search for its duration. Same
        # split _build_tag_counts and _load_knowledge_graph already use.
        with self._lock:
            started_at = self._update_states.get(collection_name, {}).get("startedAt")
            finished_at = self.__now()
            self._update_states[collection_name] = {
                "status": status,
                "startedAt": started_at,
                "finishedAt": finished_at,
                "error": error,
            }
            correlation = self._update_correlation.pop(collection_name, None) or {}

        self.__record_run(collection_name, status, error, started_at, finished_at, correlation)

    def __record_run(self, collection_name, status, error, started_at, finished_at, correlation):
        """Append the run to the durable ledger. Never fails the update itself."""
        try:
            phase = {
                "name": "reindex",
                "status": "succeeded" if status == "succeeded" else "failed",
                "durationSeconds": _duration_seconds(started_at, finished_at),
                # The reindex is the one phase whose failure means the run failed;
                # script-side phases (tagging) only degrade it.
                "fatal": True,
            }
            record = {
                "runId": correlation.get("runId") or mint_run_id(collection_name, started_at),
                "collection": collection_name,
                "job": correlation.get("job"),
                "trigger": correlation.get("trigger") or "manual",
                "variant": correlation.get("variant") or "incremental",
                "startedAt": _to_ledger_ts(started_at),
                "finishedAt": _to_ledger_ts(finished_at),
                "durationSeconds": phase["durationSeconds"],
                "status": "succeeded" if status == "succeeded" else "failed",
                "phases": [phase],
                "error": error,
                "source": "huginn",
                "stage": "end",
            }
            if status == "succeeded":
                counts = self.__manifest_counts(collection_name)
                record["documentCount"] = counts.get("numberOfDocuments")
                record["chunkCount"] = counts.get("numberOfChunks")
            IndexingRunLedger().append(record)
        except Exception:
            logger.warning("Could not write indexing run ledger record for %s",
                           collection_name, exc_info=True)

    def __manifest_counts(self, collection_name):
        """Document/chunk counts from the manifest, or empty if it is missing.

        A missing manifest is expected rather than exceptional: the failure path
        never rewrote it, and collection creation removes the whole folder when it
        reads zero documents.
        """
        try:
            return json.loads(
                self.disk_persister.read_text_file(f"{collection_name}/manifest.json")
            )
        except Exception:
            return {}

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
        """(Re)build tag frequency counts for a collection.

        The scan (file I/O) runs outside the lock; only the completed dict is
        swapped in under it, so concurrent get_tag_counts readers never observe
        a partially built map (M10).
        """
        counts = self._compute_tag_counts(name)
        with self._lock:
            self.tag_counts[name] = counts
        logger.info(f"Built tag counts for {name}: {len(counts)} unique tags")

    def _compute_tag_counts(self, name):
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
                for tag in parse_tags(tags_str):
                    tag_counts[tag] += 1
            except Exception:
                continue
        return dict(tag_counts.most_common())

    def _build_notion_id_lookup(self, name):
        """(Re)build this collection's Notion-ID slice and re-merge the lookup.

        Storing per-collection slices lets a reload replace just this
        collection's entries (dropping pages it no longer contains) instead of
        accumulating stale ones (M9). The scan runs outside the lock; the slice
        store and the merged dict are swapped under it (M10).
        """
        slice_ = self._compute_notion_lookup(name)
        with self._lock:
            self._notion_by_collection[name] = slice_
            merged = {}
            for coll_slice in self._notion_by_collection.values():
                merged.update(coll_slice)
            self.notion_id_to_doc = merged
            total = len(merged)
        logger.info(f"Built Notion ID lookup for {name}: {len(slice_)} pages ({total} total)")

    def _compute_notion_lookup(self, name):
        lookup = {}
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
                    lookup[notion_hex] = {"doc_path": doc_path, "url": doc_url}
        except Exception as e:
            logger.warning(f"Could not build Notion ID lookup for {name}: {e}")
        return lookup


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
