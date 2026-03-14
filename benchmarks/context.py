"""Shared benchmark context — loads models and collections once."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import (
    load_search_indexer,
    create_embedder,
    create_reranker,
    detect_faiss_index,
)
from main.core.documents_collection_searcher import DocumentCollectionSearcher

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkContext:
    """Shared state across all benchmarks, loaded once."""
    persister: DiskPersister
    searchers: dict[str, DocumentCollectionSearcher] = field(default_factory=dict)
    graph: object = None  # KnowledgeGraph or None
    data_dirs: list[Path] = field(default_factory=list)  # Directories to search for benchmark data files
    _embedder: object = None  # SentenceEmbedder, shared
    _reranker: object = None  # CrossEncoderReranker, shared

    @property
    def collection_names(self) -> list[str]:
        return list(self.searchers.keys())

    def get_searcher(self, name: str) -> DocumentCollectionSearcher:
        return self.searchers[name]

    def find_data_file(self, filename: str) -> Path | None:
        """Search data directories for a benchmark data file."""
        for d in self.data_dirs:
            path = d / filename
            if path.exists():
                return path
        return None


def load_context(
    data_path: str = "./data/collections",
    collection_filter: list[str] | None = None,
    graph_paths: list[str] | None = None,
    skip_reranker: bool = False,
    extra_data_dirs: list[str] | None = None,
) -> BenchmarkContext:
    """Load benchmark context with shared models.

    Args:
        data_path: Path to collections directory.
        collection_filter: If set, only load these collections. Otherwise auto-detect.
        graph_paths: Paths to knowledge graph JSON files.
        skip_reranker: If True, don't load the cross-encoder reranker (faster startup).
        extra_data_dirs: Additional directories to search for benchmark data files.
    """
    persister = DiskPersister(data_path)

    # Auto-detect collections
    if collection_filter:
        collections = collection_filter
    else:
        collections_dir = Path(data_path)
        collections = [
            d.name for d in sorted(collections_dir.iterdir())
            if d.is_dir() and (d / "indexes").exists()
        ]

    if not collections:
        raise ValueError(f"No collections found in {data_path}")

    # Load shared embedder from the first collection's FAISS index
    t0 = time.monotonic()
    first_faiss = detect_faiss_index(collections[0], persister)
    shared_embedder = create_embedder(first_faiss)
    logger.info(f"Loaded embedder in {(time.monotonic() - t0) * 1000:.0f}ms")

    # Load reranker
    reranker = None
    if not skip_reranker:
        t0 = time.monotonic()
        reranker = create_reranker()
        logger.info(f"Loaded reranker in {(time.monotonic() - t0) * 1000:.0f}ms")

    # Load collections
    searchers = {}
    for name in collections:
        t0 = time.monotonic()
        try:
            indexer = load_search_indexer(name, persister, shared_embedder=shared_embedder)
            searchers[name] = DocumentCollectionSearcher(name, indexer, persister, reranker=reranker)
            logger.info(f"Loaded collection '{name}' in {(time.monotonic() - t0) * 1000:.0f}ms")
        except Exception as e:
            logger.warning(f"Skipping collection '{name}': {e}")

    # Load knowledge graph
    graph = None
    if graph_paths:
        from main.graph.knowledge_graph import KnowledgeGraph
        existing = [p for p in graph_paths if Path(p).exists()]
        if existing:
            graph = KnowledgeGraph([Path(p) for p in existing])
            logger.info(f"Loaded knowledge graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

    # Build data directory search path
    project_root = Path(__file__).parent.parent
    data_dirs = [project_root / "benchmarks" / "data"]

    # Auto-detect domain repo benchmark data (huginn-nav, huginn-capra, etc.)
    for d in sorted(project_root.glob("huginn-*/scripts/benchmarks")):
        if d.is_dir():
            data_dirs.append(d)

    # Add any extra directories from CLI
    if extra_data_dirs:
        data_dirs.extend(Path(d) for d in extra_data_dirs)

    ctx = BenchmarkContext(
        persister=persister,
        searchers=searchers,
        graph=graph,
        data_dirs=data_dirs,
        _embedder=shared_embedder,
        _reranker=reranker,
    )

    return ctx


def load_documents_for_collection(persister: DiskPersister, collection_name: str) -> list[dict]:
    """Load all document JSON files for a collection."""
    docs_path = f"{collection_name}/documents"
    doc_files = persister.read_folder_files(docs_path)
    documents = []
    for f in doc_files:
        if f.endswith(".json"):
            try:
                full_path = f"{docs_path}/{f}"
                doc = json.loads(persister.read_text_file(full_path))
                doc["_relative_path"] = f"{collection_name}/documents/{f}"
                documents.append(doc)
            except Exception as e:
                logger.debug(f"Skipping {f}: {e}")
    return documents
