"""Shared CLI args for the stdio MCP search adapters.

Both ``collection_search_mcp_stdio_adapter.py`` (single collection) and
``multi_collection_search_mcp_adapter.py`` (multi collection) accept the same
five flags that drive the shared search pipeline. Each adapter still defines
its own ``-collection`` / ``-collections`` arg; this helper adds the rest.
"""
from argparse import ArgumentParser


def add_search_tool_args(parser: ArgumentParser) -> None:
    """Register the five shared search-tool flags on ``parser``."""
    parser.add_argument(
        "-index", "--index",
        required=False, default=None,
        help="Index that will be used for search (auto-detected if omitted)",
    )
    parser.add_argument(
        "-maxNumberOfChunks", "--maxNumberOfChunks",
        required=False, type=int, default=100,
        help="Max number of text chunks in result",
    )
    parser.add_argument(
        "-maxNumberOfDocuments", "--maxNumberOfDocuments",
        required=False, type=int, default=10,
        help="Max number of documents in result (default: 10)",
    )
    parser.add_argument(
        "-includeFullText", "--includeFullText",
        action="store_true", required=False, default=False,
        help=(
            "Include full text content in search results. If passed, reduce "
            "--maxNumberOfChunks or set a small --maxNumberOfDocuments to "
            "avoid blowing model context."
        ),
    )
    parser.add_argument(
        "-graphPaths", "--graphPaths",
        required=False, nargs="*", default=None,
        help=(
            "Optional knowledge graph JSON paths. Combined with "
            "KNOWLEDGE_GRAPH_PATH/JIRA_GRAPH_PATH/LLM_GRAPH_PATH and "
            "*_llm_graph.json files auto-detected in the private sub-repo dirs."
        ),
    )
