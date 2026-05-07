#!/usr/bin/env python3
"""Single-collection MCP stdio adapter.

Registers one ``search_<collection>`` tool whose handler runs the shared
graph-aware search pipeline (knowledge_api_server.py /api/search-equivalent).

Environment:
    HUGINN_TRACE_DEFAULT    "1"/"true"/"yes" to attach a per-search trace
                            under result.trace. Only enable when an
                            orchestrator strips the field before the LLM
                            sees it.
    KNOWLEDGE_GRAPH_PATH    Optional graph JSON path; combined with
    JIRA_GRAPH_PATH         --graphPaths and any auto-detected
    LLM_GRAPH_PATH          *_llm_graph.json files in the private sub-repo
                            dirs (huginn-{nav,jarvis}/scripts/...).
"""
import argparse

from mcp.server.fastmcp import FastMCP

from main.core.mcp_search_tool import build_search_tool_fn
from main.core.trace_store import TRACE_DEFAULT_ENV
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.runtime.knowledge_store import get_store
from main.utils.env import env_bool
from main.utils.logger import add_file_handler, route_handlers_to_stderr, setup_root_logger


setup_root_logger()

TRACE_DEFAULT = env_bool(TRACE_DEFAULT_ENV)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-collection", "--collection", required=True,
                    help="Collection name (will be used as root folder name)")
    ap.add_argument("-index", "--index", required=False, default=None,
                    help="Index that will be used for search (auto-detected if omitted)")
    ap.add_argument("-maxNumberOfChunks", "--maxNumberOfChunks", required=False, type=int, default=100,
                    help="Max number of text chunks in result")
    ap.add_argument("-maxNumberOfDocuments", "--maxNumberOfDocuments", required=False, type=int, default=10,
                    help="Max number of documents in result (default: 10)")
    ap.add_argument("-includeFullText", "--includeFullText", action="store_true", required=False, default=False,
                    help="Include full text content in search results. If passed, reduce --maxNumberOfChunks or set a small --maxNumberOfDocuments to avoid blowing model context.")
    ap.add_argument("-graphPaths", "--graphPaths", required=False, nargs='*', default=None,
                    help="Optional knowledge graph JSON paths. Combined with KNOWLEDGE_GRAPH_PATH/JIRA_GRAPH_PATH/LLM_GRAPH_PATH and *_llm_graph.json files auto-detected in the private sub-repo dirs.")
    return vars(ap.parse_args(argv))


def main():
    args = _parse_args()
    route_handlers_to_stderr()
    add_file_handler(f"huginn-{args['collection']}-mcp.log")

    store = get_store()
    store.load_collections(
        [args['collection']],
        faiss_index_name=args['index'],
        extra_graph_paths=args['graphPaths'],
    )
    searcher = store.get_searchers()[args['collection']]
    augmenter = GraphSearchAugmenter(store.graph)

    mcp = FastMCP("documents-search")
    tool_description = (
        "The tool allows searching in collection of documents by vector search. "
        "Each document contains 'url' field; if you consider a document relevant to the query, "
        "always include the 'url' field in the response, near the cited information."
    )
    search_fn = build_search_tool_fn(
        searcher,
        args['collection'],
        augmenter,
        max_number_of_chunks=args['maxNumberOfChunks'],
        max_number_of_documents=args['maxNumberOfDocuments'],
        include_full_text=args['includeFullText'],
        trace_default=TRACE_DEFAULT,
    )
    mcp.tool(name=f"search_{args['collection']}", description=tool_description)(search_fn)

    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
