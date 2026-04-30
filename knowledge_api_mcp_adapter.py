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
"""
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from main.utils.env import env_bool

# Redirect logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

API_URL = os.environ.get("KNOWLEDGE_API_URL", "http://localhost:8321").rstrip("/")
ALLOWED_COLLECTIONS = [
    c.strip() for c in os.environ.get("KNOWLEDGE_COLLECTIONS", "").split(",") if c.strip()
] or None  # None means all collections

KNOWLEDGE_DESCRIPTION = os.environ.get("KNOWLEDGE_DESCRIPTION", "")

# Only enable when an orchestrator (e.g. Muninn) is wired to strip the trace
# block before the LLM sees it — otherwise the full trace lands in model context.
# See docs/search-tracing-plan.md.
TRACE_DEFAULT = env_bool("HUGINN_TRACE_DEFAULT")

def _detect_feature(allowed_collections: list[str] | None, keyword: str) -> bool:
    """Check if a feature keyword matches any allowed collection name (or all if None)."""
    return not allowed_collections or any(keyword in c for c in allowed_collections)


# Feature detection from collection names and local files
_has_notion = _detect_feature(ALLOWED_COLLECTIONS, "notion")
_has_sessions = _detect_feature(ALLOWED_COLLECTIONS, "session")
# Graph is enabled if at least one graph path env var is set (non-empty) and the file exists
_knowledge_graph_path = os.environ.get("KNOWLEDGE_GRAPH_PATH", "")
_jira_graph_path = os.environ.get("JIRA_GRAPH_PATH", "")
_has_graph = (
    (_knowledge_graph_path and Path(_knowledge_graph_path).exists())
    or (_jira_graph_path and Path(_jira_graph_path).exists())
)


def _load_available_tags() -> str:
    """Load tag taxonomies from scripts/tagging/*_taxonomy.json and format for tool description.

    Only includes taxonomies whose name matches an allowed collection (if KNOWLEDGE_COLLECTIONS is set).
    Matching is fuzzy: taxonomy 'notion' matches collection 'my-notion-v9', 'my-project' matches 'my-confluence'.
    """
    import json

    taxonomy_dir = Path(__file__).parent / "scripts" / "tagging"
    if not taxonomy_dir.exists():
        return ""

    parts = []
    for f in sorted(taxonomy_dir.glob("*_taxonomy.json")):
        try:
            data = json.loads(f.read_text())
            taxonomy_name = f.stem.replace("_taxonomy", "")
            # Skip if ALLOWED_COLLECTIONS is set and no collection matches this taxonomy
            if ALLOWED_COLLECTIONS:
                if not any(taxonomy_name in coll for coll in ALLOWED_COLLECTIONS):
                    continue
            all_tags = []
            for tags in data.get("tags", {}).values():
                all_tags.extend(tags)
            if all_tags:
                parts.append(f"{taxonomy_name}: {', '.join(all_tags)}")
        except Exception:
            continue
    if not parts:
        return ""
    return "\n\nAvailable tags per collection:\n" + "\n".join(parts)


AVAILABLE_TAGS_DOC = _load_available_tags()
_has_tags = bool(AVAILABLE_TAGS_DOC)


def _build_search_description(
    description: str | None = None,
    has_sessions: bool | None = None,
    has_graph: bool | None = None,
    has_tags: bool | None = None,
    tags_doc: str | None = None,
) -> str:
    """Assemble search tool description based on available features.

    Parameters accept overrides for testing; defaults to module-level globals.
    """
    if description is None:
        description = KNOWLEDGE_DESCRIPTION
    if has_sessions is None:
        has_sessions = _has_sessions
    if has_graph is None:
        has_graph = _has_graph
    if tags_doc is None:
        tags_doc = AVAILABLE_TAGS_DOC
    if has_tags is None:
        has_tags = bool(tags_doc)

    parts = ["Search indexed document collections using vector search."]

    if description:
        parts.append(f"\nThis knowledge base contains: {description}")

    parts.append(
        "\n\nUse brief=True for an initial overview (returns titles, URLs, and short snippets)."
        "\nThen use get_document(collection, doc_id) to fetch full content for specific results."
        "\n\nEach result includes collection and doc_id fields for use with get_document()."
        "\nEach document contains a 'url' field — always include it when citing information."
    )

    if has_sessions:
        parts.append(
            "\n\nOptional project and git_branch params filter results by metadata "
            "(useful for claude-sessions collection)."
        )

    if has_tags:
        parts.append(
            "\nOptional tags param filters by document tags (comma-separated, matches any)."
        )

    if has_graph:
        parts.append(
            '\n\nGraph-enhanced: queries mentioning BUCs (LA_BUC_01), SEDs (A003), '
            'articles (artikkel 13), or Jira issue keys (PROJECT-1234) automatically get '
            'entity detection, query expansion, and graph context annotations. Relational queries '
            '("hvilke SEDer inneholder LA_BUC_02?", "hvilke issues tilhører PROJECT-6079?") '
            'may return a direct graph_answer before the search results.'
        )

    if tags_doc:
        parts.append(tags_doc)

    return "".join(parts)


_SEARCH_DOC = _build_search_description()

mcp = FastMCP("knowledge", instructions=KNOWLEDGE_DESCRIPTION or None)


def _format_relevance(score):
    """Format relevance score as percentage string."""
    if score is None:
        return ""
    return f"{score * 100:.1f}% relevant"


def _format_date(date_str):
    """Extract YYYY-MM-DD from an ISO datetime string."""
    if not date_str:
        return ""
    return date_str[:10]


def _is_wip(result):
    """Check if a search result is flagged as work-in-progress."""
    meta = result.get("metadata") or {}
    return meta.get("wip") == "true"


# Metadata keys that are internal and should not be shown in MCP output
_INTERNAL_METADATA_KEYS = {"page_id", "space", "breadcrumb", "title", "wip"}


def _api_get(path: str, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
    """Make a GET request to the Knowledge API Server."""
    return httpx.get(f"{API_URL}{path}", params=params, timeout=timeout)


def _search_knowledge_impl(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    project: str | None = None,
    git_branch: str | None = None,
    tags: str | None = None,
) -> str:
    """Search indexed document collections using vector search.

    This is the shared implementation called by all signature variants.
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
        return f"No results found for '{query}'{low}."

    parts = []
    if data.get("graph_answer"):
        parts.append(f"**Graph:**\n{data['graph_answer']}\n")
    if data.get("lowConfidence"):
        parts.append("*Low confidence results — query may not match indexed content well.*\n")

    if brief:
        for i, r in enumerate(results, 1):
            heading = f" > {r['heading']}" if r.get("heading") else ""
            relevance = f" ({_format_relevance(r.get('relevance'))})" if r.get("relevance") is not None else ""
            date = f" | {_format_date(r['modifiedTime'])}" if r.get("modifiedTime") else ""
            breadcrumb = f"\n   {r['breadcrumb']}" if r.get("breadcrumb") else ""
            wip = " **[UNDER ARBEID]**" if _is_wip(r) else ""
            graph_ctx = f"\n   *{' | '.join(r['graph_context'])}*" if r.get("graph_context") else ""
            meta = r.get("metadata") or {}
            visible_meta = {k: v for k, v in meta.items() if k not in _INTERNAL_METADATA_KEYS and v}
            meta_line = f"\n   *{' | '.join(f'{k}: {v}' for k, v in visible_meta.items())}*" if visible_meta else ""
            parts.append(
                f"{i}. **{r['title']}**{heading}{wip}{relevance}{date}\n"
                f"   {r.get('url', '')}{breadcrumb}\n"
                f"   {r.get('snippet', '')}{graph_ctx}{meta_line}\n"
                f"   collection: `{r['collection']}` doc_id: `{r['id']}`"
            )
    else:
        for r in results:
            relevance = f" ({_format_relevance(r.get('relevance'))})" if r.get("relevance") is not None else ""
            date = f" | updated: {_format_date(r['modifiedTime'])}" if r.get("modifiedTime") else ""
            wip = " **[UNDER ARBEID]**" if _is_wip(r) else ""
            header = f"## {r['title']}{wip}{relevance}{date}"
            if r.get("url"):
                header += f"\n{r['url']}"
            if r.get("breadcrumb"):
                header += f"\n{r['breadcrumb']}"
            header += f"\ncollection: `{r['collection']}` doc_id: `{r['id']}`"
            if r.get("graph_context"):
                header += f"\n*{' | '.join(r['graph_context'])}*"
            chunks = r.get("matchedChunks", [])
            chunk_lines = []
            for chunk in chunks:
                if chunk.get("heading"):
                    chunk_lines.append(f"**{chunk['heading']}**")
                chunk_lines.append(chunk.get("content", ""))
                if chunk.get("metadata"):
                    visible_meta = {k: v for k, v in chunk["metadata"].items() if k not in _INTERNAL_METADATA_KEYS}
                    if visible_meta:
                        meta_str = " | ".join(f"{k}: {v}" for k, v in visible_meta.items())
                        chunk_lines.append(f"*{meta_str}*")
            parts.append(header + "\n\n" + "\n\n".join(chunk_lines))

    text = "\n\n".join(parts)
    if TRACE_DEFAULT and data.get("trace") is not None:
        text += f"\n\n```huginn-trace\n{json.dumps(data['trace'], ensure_ascii=False)}\n```"
    return text


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

    title = doc.get("title", doc_id)
    url = doc.get("url", "")
    text = doc.get("text", "")
    metadata = doc.get("metadata") or {}
    is_wip = metadata.get("wip") == "true"

    header = f"# {title}"
    if is_wip:
        header += "\n**[UNDER ARBEID]** Dette dokumentet er merket som under arbeid i Confluence."
    if url:
        header += f"\n{url}"
    visible_meta = {k: v for k, v in metadata.items() if k not in _INTERNAL_METADATA_KEYS and v}
    if visible_meta:
        header += "\n" + " | ".join(f"**{k}:** {v}" for k, v in visible_meta.items())
    return f"{header}\n\n{text}"


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

    title = page.get("title", "Untitled")
    url = page.get("url", "")
    content = page.get("content", "")
    source_label = f" (source: {page.get('source', 'live')})" if page.get("source") else ""

    header = f"# {title}{source_label}"
    if url:
        header += f"\n{url}"
    return f"{header}\n\n{content}"


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

    lines = ["**Loaded collections:**\n"]
    for c in collections:
        lines.append(
            f"- **{c['name']}**: {c.get('document_count', '?')} documents, "
            f"{c.get('embedding_count', '?')} embeddings"
            f" (updated: {c.get('updatedTime', 'unknown')})"
        )
    return "\n".join(lines)


def get_graph_node(node_id: str) -> str:
    """Inspect a knowledge graph node and its relationships.

    Use this to explore BUC/SED/Article/Forordning/Epic/Issue entities and their connections.
    Node ID format: buc:LA_BUC_01, sed:A003, artikkel:13, forordning:883/2004, epic:PROJECT-6079, issue:PROJECT-1234

    Returns the node's type, label, properties, and all incoming/outgoing edges.
    """
    try:
        resp = _api_get(f"/api/graph/{node_id}")
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return f"Knowledge API server is not running at {API_URL}."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Node '{node_id}' not found in graph."
        if e.response.status_code == 503:
            return "Knowledge graph not loaded on the server."
        return f"API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error calling Knowledge API: {e}"

    parts = [f"**{data['label']}** ({data['type']})"]

    props = data.get("properties", {})
    if props:
        prop_lines = [f"  {k}: {v}" for k, v in props.items()]
        parts.append("Properties:\n" + "\n".join(prop_lines))

    outgoing = data.get("outgoing", [])
    if outgoing:
        edge_lines = [f"  --{e['type']}--> {e['target_label']}" for e in outgoing]
        parts.append(f"Outgoing ({len(outgoing)}):\n" + "\n".join(edge_lines))

    incoming = data.get("incoming", [])
    if incoming:
        edge_lines = [f"  <--{e['type']}-- {e['source_label']}" for e in incoming]
        parts.append(f"Incoming ({len(incoming)}):\n" + "\n".join(edge_lines))

    return "\n\n".join(parts)


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

    parts = []
    for coll_name, info in data.items():
        tags = info.get("tags", {})
        if not tags:
            parts.append(f"**{coll_name}**: no tags")
            continue
        parts.append(f"**{coll_name}** ({info.get('unique_tags', 0)} tags):")
        for tag, count in tags.items():
            parts.append(f"  {tag}: {count} docs")
    return "\n".join(parts)


# --- Signature variants for search_knowledge (controls which params the agent sees) ---


def _search_with_sessions_and_tags(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    project: str | None = None,
    git_branch: str | None = None,
    tags: str | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, project, git_branch, tags)


def _search_with_sessions(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    project: str | None = None,
    git_branch: str | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, project, git_branch)


def _search_with_tags(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
    tags: str | None = None,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief, tags=tags)


def _search_basic(
    query: str,
    collection: str | None = None,
    limit: int = 10,
    brief: bool = False,
) -> str:
    return _search_knowledge_impl(query, collection, limit, brief)


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


if __name__ == "__main__":
    scope = f", collections: {ALLOWED_COLLECTIONS}" if ALLOWED_COLLECTIONS else ""
    features = [f for f, v in [("notion", _has_notion), ("graph", _has_graph), ("sessions", _has_sessions)] if v]
    logger.info(f"Starting Knowledge API MCP adapter (API: {API_URL}{scope}, features: {features})")
    mcp.run(transport="stdio")
