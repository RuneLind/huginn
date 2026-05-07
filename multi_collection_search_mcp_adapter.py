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

from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.mcp_search_tool import build_search_tool_fn
from main.core.trace_store import TRACE_DEFAULT_ENV
from main.graph.graph_loader import load_default_knowledge_graph
from main.graph.graph_search_augmenter import GraphSearchAugmenter
from main.indexes.indexer_factory import (
    create_embedder,
    create_reranker,
    detect_faiss_index,
    load_search_indexer,
)
from main.persisters.disk_persister import DiskPersister
from main.utils.env import env_bool
from main.utils.logger import add_file_handler, route_handlers_to_stderr, setup_root_logger


setup_root_logger()

TRACE_DEFAULT = env_bool(TRACE_DEFAULT_ENV)


def build_searchers(collection_names, index_name=None, base_path="./data/collections"):
    """Build per-collection searchers backed by a single shared embedder + reranker."""
    disk_persister = DiskPersister(base_path=base_path)
    if index_name is None:
        index_name = detect_faiss_index(collection_names[0], disk_persister)

    logging.info(f"Loading shared embedding model for index: {index_name}")
    shared_embedder = create_embedder(index_name)
    logging.info(f"Embedding model loaded: {shared_embedder.model_name}")

    shared_reranker = create_reranker()
    logging.info(f"Reranker loaded: {shared_reranker.model_name}")

    searchers = {}
    for name in collection_names:
        logging.info(f"Loading collection: {name}")
        indexer = load_search_indexer(name, disk_persister, shared_embedder=shared_embedder)
        searchers[name] = DocumentCollectionSearcher(
            collection_name=name,
            indexer=indexer,
            persister=disk_persister,
            reranker=shared_reranker,
        )
        logging.info(f"Collection {name} loaded: {indexer.get_name()} ({indexer.get_size()} embeddings)")
    return searchers


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

    searchers = build_searchers(args['collections'], index_name=args['index'])
    graph = load_default_knowledge_graph(extra_paths=args['graphPaths'])
    augmenter = GraphSearchAugmenter(graph)

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
