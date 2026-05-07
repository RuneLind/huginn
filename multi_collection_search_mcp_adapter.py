#!/usr/bin/env python3
"""
Optimized multi-collection MCP adapter that explicitly shares the embedding model.
This saves ~180 MB RAM compared to running 3 separate MCP servers.

Each registered tool runs the shared search pipeline: knowledge-graph query
augmentation → searcher.search → response shaping → graph context enrichment.
The output shape matches knowledge_api_server.py /api/search.

Usage:
    uv run multi_collection_search_mcp_adapter.py \
        --collections my-confluence my-jira my-docs \
        --maxNumberOfChunks 50

Environment:
    HUGINN_TRACE_DEFAULT    "1"/"true"/"yes" to attach a per-search trace under
                            result.trace in the JSON tool result. Only enable when
                            an orchestrator (e.g. Muninn) strips the field before
                            the LLM sees it. See docs/search-tracing-plan.md.
    KNOWLEDGE_GRAPH_PATH    Optional graph JSON path; combined with --graphPaths
    JIRA_GRAPH_PATH         and any auto-detected *_llm_graph.json files in the
    LLM_GRAPH_PATH          private sub-repo dirs (huginn-{nav,jarvis}/scripts/...).
"""
import argparse
import logging

from mcp.server.fastmcp import FastMCP

from main.core.mcp_search_tool import build_search_tool_fn
from main.core.trace_store import TRACE_DEFAULT_ENV
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.runtime.knowledge_store import get_store
from main.utils.env import env_bool
from main.utils.logger import add_file_handler, route_handlers_to_stderr, setup_root_logger


setup_root_logger()

TRACE_DEFAULT = env_bool(TRACE_DEFAULT_ENV)


def register_collection_tools(mcp, searchers, augmenter, **tool_kwargs):
    """Register one MCP search tool per collection on the given FastMCP instance."""
    for collection_name, searcher in searchers.items():
        tool_description = (
            f"Search in {collection_name} collection using vector search. "
            "Each document contains 'url' field - always include it in responses when citing information."
        )
        search_fn = build_search_tool_fn(searcher, collection_name, augmenter, **tool_kwargs)
        mcp.tool(name=f"search_{collection_name}", description=tool_description)(search_fn)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-collections", "--collections", required=True, nargs='+',
                    help="List of collection names (e.g., my-confluence my-jira)")
    ap.add_argument("-index", "--index", required=False, default=None,
                    help="Index that will be used for search (auto-detected if omitted)")
    ap.add_argument("-maxNumberOfChunks", "--maxNumberOfChunks", required=False, type=int, default=100,
                    help="Max number of text chunks in result")
    ap.add_argument("-maxNumberOfDocuments", "--maxNumberOfDocuments", required=False, type=int, default=10,
                    help="Max number of documents in result (default: 10)")
    ap.add_argument("-includeFullText", "--includeFullText", action="store_true", required=False, default=False,
                    help="Include full text content in search results")
    ap.add_argument("-graphPaths", "--graphPaths", required=False, nargs='*', default=None,
                    help="Optional knowledge graph JSON paths. Combined with KNOWLEDGE_GRAPH_PATH/JIRA_GRAPH_PATH/LLM_GRAPH_PATH and *_llm_graph.json files auto-detected in the private sub-repo dirs.")
    return vars(ap.parse_args(argv))


def main():
    route_handlers_to_stderr()
    add_file_handler("huginn-multi-mcp.log")
    args = _parse_args()

    store = get_store()
    store.load_collections(
        args['collections'],
        faiss_index_name=args['index'],
        extra_graph_paths=args['graphPaths'],
        build_aux_indexes=False,
    )
    augmenter = GraphSearchAugmenter(store.graph)
    searchers = store.get_searchers()

    mcp = FastMCP("huginn-search")
    register_collection_tools(
        mcp,
        searchers,
        augmenter,
        max_number_of_chunks=args['maxNumberOfChunks'],
        max_number_of_documents=args['maxNumberOfDocuments'],
        include_full_text=args['includeFullText'],
        trace_default=TRACE_DEFAULT,
    )
    logging.info(f"🚀 MCP server ready with {len(searchers)} collections (shared embedding model)")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
