#!/usr/bin/env python3
"""
Knowledge API MCP Adapter — thin MCP wrapper over the Knowledge API Server.

Exposes vector search, document retrieval, and Notion page access as MCP tools.
Requires the Knowledge API Server to be running separately.

Usage:
    uv run knowledge_api_mcp_adapter.py

Environment:
    KNOWLEDGE_API_URL       Base URL of the Knowledge API Server (default: http://localhost:8321)
    KNOWLEDGE_COLLECTIONS   Comma-separated list of allowed collections (default: all)
    KNOWLEDGE_DESCRIPTION   Human-readable description of what this knowledge base contains
    HUGINN_TRACE_DEFAULT    "1"/"true"/"yes" to embed a per-search trace in tool results.
                            Only enable when an orchestrator (e.g. Muninn) strips the trace
                            block before the LLM sees it. See docs/search-tracing-plan.md.
    HUGINN_TRACE_POINTER    "1"/"true"/"yes" to emit a `huginn-trace-url: <url>` pointer
                            line instead of inlining the full trace JSON. The orchestrator
                            fetches the trace via that URL (TTL ~5 min on the server side).
                            Avoids blowing past MCP-stdio output-size limits when traces are
                            large. URL is built from KNOWLEDGE_API_URL so the pointer is
                            self-contained — orchestrator does not need to know Huginn's
                            location separately.

This entry file stays the runnable stdio entry point (user MCP configs reference
it by path). The config/feature detection lives in ``mcp_adapter.config`` and the
markdown rendering in ``mcp_adapter.formatting``; both are re-exported here so
``import knowledge_api_mcp_adapter as adapter`` exposes the full symbol surface
(and ``patch.object`` targets) the test suite depends on.
"""
import json
import logging
import sys
from typing import Literal

import httpx
from mcp.server.fastmcp import FastMCP

from mcp_adapter import formatting
from mcp_adapter.config import (
    ALLOWED_COLLECTIONS,
    API_URL,
    AVAILABLE_TAGS_DOC,
    KNOWLEDGE_DESCRIPTION,
    TRACE_DEFAULT,
    _SEARCH_DOC,
    _build_search_description,
    _detect_feature,
    _has_graph,
    _has_notion,
    _has_sessions,
    _has_tags,
)
from mcp_adapter.formatting import (
    _format_date,
    _format_relevance,
    _format_relevance_band,
    _format_retry_hints,
    _INTERNAL_METADATA_KEYS,
    _is_wip,
)

# Redirect logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("knowledge", instructions=KNOWLEDGE_DESCRIPTION or None)


def _api_get(path: str, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
    """Make a GET request to the Knowledge API Server."""
    return httpx.get(f"{API_URL}{path}", params=params, timeout=timeout)


def _append_trace_marker(text: str, data: dict) -> str:
    if not TRACE_DEFAULT:
        return text
    trace_id = data.get("traceId")
    if trace_id:
        return text + f"\n\nhuginn-trace-url: {API_URL}/api/trace/{trace_id}\n"
    if data.get("trace") is not None:
        return text + f"\n\n```huginn-trace\n{json.dumps(data['trace'], ensure_ascii=False)}\n```"
    return text


def _format_graph_error(e: Exception, node_id: str) -> str:
    if isinstance(e, httpx.ConnectError):
        return f"Knowledge API server is not running at {API_URL}."
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 404:
            return f"Node '{node_id}' not found in graph."
        if e.response.status_code == 503:
            return "Knowledge graph not loaded on the server."
        return f"API returned {e.response.status_code}: {e.response.text}"
    return f"Error calling Knowledge API: {e}"


def _search_knowledge_impl(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    project: str | None = None,
    git_branch: str | None = None,
    tags: str | None = None,
    min_relevance: float | None = None,
    rerank: bool | None = None,
    corrective: Literal["auto", "off", "force"] | None = None,
) -> str:
    """Search indexed document collections using vector search.

    This is the shared implementation called by all signature variants.
    ``min_relevance`` (0.0-1.0) drops weak results; if everything is below it,
    the response says so and offers retry hints instead of low-quality filler.
    ``rerank`` forces cross-encoder reranking on (or off); ``None`` leaves the
    server-side default in place (which keys off ``brief``). Forcing
    ``rerank=True`` from a corrective-retrieval client makes ``bestScore`` a
    real confidence estimate even when ``brief=True``. ``corrective`` is an
    operator escape-hatch for huginn-side rescue retrieval — leave unset
    (defaults to server-side ``"auto"``); only set ``"off"`` to reproduce
    pre-corrective behaviour for testing.
    """
    try:
        params = {"q": query, "limit": limit, "brief": brief, "max_chunks_per_doc": 2}
        if collection:
            if ALLOWED_COLLECTIONS and collection not in ALLOWED_COLLECTIONS:
                return f"Collection '{collection}' is not available. Available: {', '.join(ALLOWED_COLLECTIONS)}"
            params["collection"] = collection
        elif ALLOWED_COLLECTIONS:
            params["collection"] = ALLOWED_COLLECTIONS
        if project:
            params["project"] = project
        if git_branch:
            params["git_branch"] = git_branch
        if tags:
            params["tags"] = tags
        if min_relevance is not None:
            params["min_relevance"] = min_relevance
        if rerank is not None:
            params["rerank"] = "true" if rerank else "false"
        if corrective is not None:
            params["corrective"] = corrective
        if TRACE_DEFAULT:
            params["trace"] = "true"
        resp = _api_get("/api/search", params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return f"Knowledge API server is not running at {API_URL}. Start it with: uv run knowledge_api_server.py --collections <name>"
    except httpx.HTTPStatusError as e:
        return f"API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error calling Knowledge API: {e}"

    results = data.get("results", [])
    if not results and not data.get("graph_answer"):
        low = " (low confidence)" if data.get("lowConfidence") else ""
        logger.info(f"No-hit search '{query}'{low}")
        text = f"No results found for '{query}'{low}." + _format_retry_hints(data)
        return _append_trace_marker(text, data)

    text = formatting.render_results(data, brief) + _format_retry_hints(data)
    return _append_trace_marker(text, data)


def get_document(collection: str, doc_id: str) -> str:
    """Fetch full document content by collection and document ID.

    Use this after a brief search to get full content for a specific result.
    The collection and doc_id are shown as labeled fields in search results.
    """
    if ALLOWED_COLLECTIONS and collection not in ALLOWED_COLLECTIONS:
        return f"Collection '{collection}' is not available. Available: {', '.join(ALLOWED_COLLECTIONS)}"
    try:
        resp = _api_get(f"/api/document/{collection}/{doc_id}")
        resp.raise_for_status()
        doc = resp.json()
    except httpx.ConnectError:
        return f"Knowledge API server is not running at {API_URL}."
    except httpx.HTTPStatusError as e:
        return f"API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error calling Knowledge API: {e}"
    return formatting.render_document(doc, doc_id)


def get_notion_page(notion_id: str, source: str = "auto") -> str:
    """Fetch a Notion page by its ID.

    Source options:
    - "auto" (default): tries live Notion API first, falls back to local index
    - "live": only Notion API (freshest content)
    - "local": only local index (fastest, no API call)
    """
    try:
        resp = _api_get(f"/api/notion/page/{notion_id}", params={"source": source}, timeout=60.0)
        resp.raise_for_status()
        page = resp.json()
    except httpx.ConnectError:
        return f"Knowledge API server is not running at {API_URL}."
    except httpx.HTTPStatusError as e:
        return f"API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error calling Knowledge API: {e}"
    return formatting.render_notion_page(page)


def list_collections() -> str:
    """List all loaded document collections with their stats."""
    try:
        resp = _api_get("/api/collections")
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return f"Knowledge API server is not running at {API_URL}."
    except httpx.HTTPStatusError as e:
        return f"API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error calling Knowledge API: {e}"

    collections = data.get("collections", [])
    if ALLOWED_COLLECTIONS:
        collections = [c for c in collections if c["name"] in ALLOWED_COLLECTIONS]
    if not collections:
        return "No collections loaded."
    return formatting.render_collections(collections)


def get_graph_node(node_id: str, edge_types: str | None = None) -> str:
    """Inspect a knowledge graph node and its relationships.

    Use this to explore BUC/SED/Article/Forordning/Epic/Issue entities and their connections.
    Node ID format: buc:LA_BUC_01, sed:A003, artikkel:13, forordning:883/2004, epic:PROJECT-6079, issue:PROJECT-1234

    edge_types: optional comma-separated edge type names to include (e.g. "tilhører_epic,er_subtask_av").
    When omitted, all edges are returned.

    Returns the node's type, label, properties, and incoming/outgoing edges.
    """
    params = {"edge_types": edge_types} if edge_types else None
    try:
        resp = _api_get(f"/api/graph/{node_id}", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return _format_graph_error(e, node_id)
    return formatting.render_graph_node(data)


def get_graph_subtree(
    node_id: str,
    depth: int = 2,
    direction: str = "incoming",
    edge_types: str | None = None,
) -> str:
    """Fetch a multi-hop subtree from a graph node in one call.

    Use this for full epic context: from an epic, default args walk stories + their subtasks
    (2 hops via tilhører_epic + er_subtask_av) in one response — no N+1.

    Node ID format: epic:PROJECT-6079, issue:PROJECT-1234, buc:LA_BUC_01, etc.
    depth: BFS levels to traverse (1-5, default 2).
    direction: "incoming" follows edges TO frontier (typical for epic→stories→subtasks),
        "outgoing" follows edges FROM frontier, "both" follows either.
    edge_types: optional comma-separated edge type names (e.g. "tilhører_epic,er_subtask_av")
        to restrict traversal — strongly recommended for hierarchical walks to skip refererer_til noise.

    Returns the root, full node list, edge list, and stats grouped by type.
    """
    params: dict[str, object] = {"depth": depth, "direction": direction}
    if edge_types:
        params["edge_types"] = edge_types
    try:
        resp = _api_get(f"/api/graph/{node_id}/subtree", params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return _format_graph_error(e, node_id)
    return formatting.render_graph_subtree(data)


def list_tags(collection: str | None = None) -> str:
    """List available tags and their document counts for a collection.

    Use this to discover which tags can be used with the `tags` parameter in search_knowledge().
    """
    try:
        params = {}
        if collection:
            params["collection"] = collection
        resp = _api_get("/api/tags", params=params, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return f"Knowledge API server is not running at {API_URL}."
    except httpx.HTTPStatusError as e:
        return f"API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error calling Knowledge API: {e}"

    if ALLOWED_COLLECTIONS:
        data = {k: v for k, v in data.items() if k in ALLOWED_COLLECTIONS}

    if not data:
        return "No tag data available."
    return formatting.render_tags(data)


# --- Signature variants for search_knowledge (controls which params the agent sees) ---
#
# The *signature* of the registered function is what the MCP framework
# introspects to build the tool's parameter schema, so these variants are how
# the schema varies by feature detection. Do not collapse them: the per-variant
# signatures + dispatch identity are pinned by tests.


def _search_with_sessions_and_tags(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    project: str | None = None,
    git_branch: str | None = None,
    tags: str | None = None,
    min_relevance: float | None = None,
    rerank: bool | None = None,
    corrective: Literal["auto", "off", "force"] | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, project, git_branch, tags, min_relevance, rerank, corrective)


def _search_with_sessions(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    project: str | None = None,
    git_branch: str | None = None,
    min_relevance: float | None = None,
    rerank: bool | None = None,
    corrective: Literal["auto", "off", "force"] | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, project, git_branch, min_relevance=min_relevance, rerank=rerank, corrective=corrective)


def _search_with_tags(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    tags: str | None = None,
    min_relevance: float | None = None,
    rerank: bool | None = None,
    corrective: Literal["auto", "off", "force"] | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, tags=tags, min_relevance=min_relevance, rerank=rerank, corrective=corrective)


def _search_basic(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    min_relevance: float | None = None,
    rerank: bool | None = None,
    corrective: Literal["auto", "off", "force"] | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, min_relevance=min_relevance, rerank=rerank, corrective=corrective)


def _pick_search_function():
    """Pick the search function variant with the right parameter signature."""
    if _has_sessions and _has_tags:
        return _search_with_sessions_and_tags
    if _has_sessions:
        return _search_with_sessions
    if _has_tags:
        return _search_with_tags
    return _search_basic


# Alias for backward compatibility (tests call adapter.search_knowledge directly)
search_knowledge = _search_knowledge_impl


# --- Tool registration (conditional based on detected features) ---

mcp.add_tool(_pick_search_function(), name="search_knowledge", description=_SEARCH_DOC)
mcp.add_tool(get_document)
mcp.add_tool(list_collections)
mcp.add_tool(list_tags)

if _has_notion:
    mcp.add_tool(get_notion_page)

if _has_graph:
    mcp.add_tool(get_graph_node)
    mcp.add_tool(get_graph_subtree)


if __name__ == "__main__":
    scope = f", collections: {ALLOWED_COLLECTIONS}" if ALLOWED_COLLECTIONS else ""
    features = [f for f, v in [("notion", _has_notion), ("graph", _has_graph), ("sessions", _has_sessions)] if v]
    logger.info(f"Starting Knowledge API MCP adapter (API: {API_URL}{scope}, features: {features})")
    mcp.run(transport="stdio")
