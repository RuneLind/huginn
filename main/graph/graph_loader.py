"""Default discovery of knowledge graph JSON files.

Centralizes the env-var + auto-glob convention used by both the HTTP API server
and the multi-collection MCP adapter so they pick up the same set of graphs
without re-implementing the resolution.
"""
import json
import logging
import os
from pathlib import Path

from main.graph.knowledge_graph import KnowledgeGraph


logger = logging.getLogger(__name__)


_ENV_PATH_VARS = ("KNOWLEDGE_GRAPH_PATH", "JIRA_GRAPH_PATH", "LLM_GRAPH_PATH")
_AUTO_GLOB_DIRS = (
    "./huginn-jarvis/scripts/knowledge_graph",
    "./huginn-nav/scripts/knowledge_graph",
    "./scripts/knowledge_graph",
)
_AUTO_GLOB_PATTERN = "*_llm_graph.json"
_ROUTING_CONFIG_NAME = "graph_routing.json"


def discover_graph_paths(extra_paths=None):
    """Resolve graph JSON paths from env vars, auto-glob dirs, and caller-supplied extras.

    Order: env vars in declaration order, then auto-globbed LLM graphs from the
    private sub-repos and local scripts dir, then any ``extra_paths``.
    Non-existent paths are skipped silently. Duplicates removed, order preserved.
    """
    paths: list[Path] = []

    def _add(p: Path):
        if p not in paths:
            paths.append(p)

    for env_var in _ENV_PATH_VARS:
        raw = os.environ.get(env_var, "")
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            _add(p)
        else:
            # An explicitly-set env path that doesn't resolve is almost always a
            # misconfiguration — surface it rather than silently disabling the
            # graph it was meant to load (mirrors the extra_paths handling below).
            logger.warning(f"{env_var} is set but path does not exist, skipping: {raw}")

    for search_dir in _AUTO_GLOB_DIRS:
        for p in Path(search_dir).glob(_AUTO_GLOB_PATTERN):
            _add(p)

    for raw in extra_paths or ():
        p = Path(raw)
        if p.exists():
            _add(p)
        else:
            logger.warning(f"Knowledge graph path not found, skipping: {raw}")

    return paths


def resolve_graph_output_path(collection, output=None):
    """Resolve where a collection's ``*_llm_graph.json`` extraction output belongs.

    Precedence:
      1. An explicit ``output`` (the ``--output`` flag) always wins.
      2. Otherwise a ``graph_routing.json`` discovered in one of the auto-glob
         dirs (the same private sub-repo dirs graph JSONs load from). A routing
         file either lists ``collection`` under ``"collections"`` — output then
         goes to that dir — or is marked ``"default": true`` to claim any
         collection no other file lists.
      3. Otherwise raise ``ValueError`` — the caller must pass ``--output``.

    Keeping collection ownership in the private routing files means no private
    collection names live in this public repo.
    """
    if output:
        return Path(output)

    default_dir = None
    for search_dir in _AUTO_GLOB_DIRS:
        routing_path = Path(search_dir) / _ROUTING_CONFIG_NAME
        if not routing_path.exists():
            continue
        try:
            cfg = json.loads(routing_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable routing config {routing_path}: {e}")
            continue
        if collection in cfg.get("collections", []):
            return Path(search_dir) / f"{collection}_llm_graph.json"
        if cfg.get("default") and default_dir is None:
            default_dir = Path(search_dir)

    if default_dir is not None:
        return default_dir / f"{collection}_llm_graph.json"

    raise ValueError(
        f"No output routing for collection '{collection}'. Pass --output <path>, "
        f"or list '{collection}' in a {_ROUTING_CONFIG_NAME} in one of: "
        + ", ".join(_AUTO_GLOB_DIRS)
    )


def check_graph_staleness(paths, data_path):
    """Warn when a stamped graph JSON diverges from its indexed collection.

    A graph produced by the extractor carries a ``source_stamp`` describing the
    collection it was built from. Here we compare it against the collection's
    current ``manifest.json``. Unstamped graphs (older extractions) and
    collections without a readable manifest are skipped silently, so a fresh
    clone or a legacy graph never warns. This is a signal only — nothing rebuilds.
    """
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stamp = data.get("source_stamp")
        if not isinstance(stamp, dict):
            continue
        collection = stamp.get("collection")
        if not collection:
            continue
        manifest_path = Path(data_path) / collection / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        reasons = []
        stamp_count = stamp.get("document_count")
        cur_count = manifest.get("numberOfDocuments")
        if stamp_count is not None and cur_count is not None and stamp_count != cur_count:
            reasons.append(f"document_count {stamp_count} → {cur_count}")
        stamp_updated = stamp.get("updated_time")
        cur_updated = manifest.get("updatedTime")
        if stamp_updated and cur_updated and stamp_updated != cur_updated:
            reasons.append(f"collection re-indexed ({stamp_updated} → {cur_updated})")

        if reasons:
            logger.warning(
                f"Knowledge graph '{Path(path).name}' may be stale relative to "
                f"collection '{collection}': {'; '.join(reasons)}. "
                f"Re-run entity extraction to refresh it."
            )


def load_default_knowledge_graph(extra_paths=None, data_path=None):
    """Load and return a ``KnowledgeGraph`` from the default discovery sources.

    Returns ``None`` (and logs an info line) when no graph paths resolve. When
    ``data_path`` is given, stamped graphs are checked against their collection
    manifests and a warning is logged on divergence.
    """
    paths = discover_graph_paths(extra_paths)
    if not paths:
        logger.info("No knowledge graph found — graph features disabled")
        return None
    if data_path:
        check_graph_staleness(paths, data_path)
    graph = KnowledgeGraph(paths)
    logger.info(
        f"Knowledge graph loaded from {len(paths)} file(s): "
        f"{graph.node_count()} nodes, {graph.edge_count()} edges"
    )
    return graph
