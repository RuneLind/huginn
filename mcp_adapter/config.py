"""Import-time configuration and feature detection for the MCP adapter.

All values here are derived from the environment (and local taxonomy files) at
import time. The entry file re-exports them as its own module-level attributes
so the test suite's ``patch.object(adapter, ...)`` continues to target them and
the ``_has_*`` flags keep attribute-assignment semantics.
"""
import json
import os
from pathlib import Path

from main.core.trace_store import any_trace_enabled

# Repo root — this file lives at <root>/mcp_adapter/config.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent

API_URL = os.environ.get("KNOWLEDGE_API_URL", "http://localhost:8321").rstrip("/")
ALLOWED_COLLECTIONS = [
    c.strip() for c in os.environ.get("KNOWLEDGE_COLLECTIONS", "").split(",") if c.strip()
] or None  # None means all collections

KNOWLEDGE_DESCRIPTION = os.environ.get("KNOWLEDGE_DESCRIPTION", "")

# Only enable when an orchestrator (e.g. Muninn) is wired to strip the trace
# block (or pointer URL) before the LLM sees it — otherwise the full trace
# lands in model context. See docs/search-tracing-plan.md.
TRACE_DEFAULT = any_trace_enabled()


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
    taxonomy_dir = _REPO_ROOT / "scripts" / "tagging"
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
