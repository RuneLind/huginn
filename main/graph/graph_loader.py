"""Default discovery of knowledge graph JSON files.

Centralizes the env-var + auto-glob convention used by both the HTTP API server
and the multi-collection MCP adapter so they pick up the same set of graphs
without re-implementing the resolution.
"""
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
        if raw and Path(raw).exists():
            _add(Path(raw))

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


def load_default_knowledge_graph(extra_paths=None):
    """Load and return a ``KnowledgeGraph`` from the default discovery sources.

    Returns ``None`` (and logs an info line) when no graph paths resolve.
    """
    paths = discover_graph_paths(extra_paths)
    if not paths:
        logger.info("No knowledge graph found — graph features disabled")
        return None
    graph = KnowledgeGraph(paths)
    logger.info(
        f"Knowledge graph loaded from {len(paths)} file(s): "
        f"{graph.node_count()} nodes, {graph.edge_count()} edges"
    )
    return graph
