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
import json
import logging
import sys
from pathlib import Path
from typing import Callable

from mcp.server.fastmcp import FastMCP

from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.search_response_formatter import shape_search_results
from main.core.search_trace import create_trace
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
from main.utils.logger import setup_root_logger


setup_root_logger()

# Only enable when an orchestrator (e.g. Muninn) is wired to strip the trace
# field before the LLM sees it — otherwise the full trace lands in model context.
TRACE_DEFAULT = env_bool("HUGINN_TRACE_DEFAULT")


def _redirect_logging_to_stderr():
    try:
        for handler in logging.getLogger().handlers:
            handler.setStream(sys.stderr)
    except Exception:
        pass


def _add_file_log_handler(name: str):
    try:
        log_file = Path.home() / "logs" / name
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    except Exception:
        pass


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


def build_search_tool_fn(
    searcher,
    collection_name: str,
    augmenter: GraphSearchAugmenter,
    *,
    max_number_of_chunks: int,
    max_number_of_documents: int,
    include_full_text: bool,
    trace_default: bool = False,
) -> Callable[[str], str]:
    """Build the per-tool search function for one collection.

    Mirrors knowledge_api_server.py /api/search: graph-aware query expansion,
    raw search via DocumentCollectionSearcher, shape_search_results post-
    processing, per-result graph context enrichment.
    """
    def search_fn(query: str) -> str:
        logging.info(f"Searching in {collection_name}: {query}")
        trace = create_trace(trace_default)
        trace.set_query_raw(query)

        search_q, graph_answer, detected_entities = augmenter.augment_query(query, trace)

        raw = searcher.search(
            search_q,
            max_number_of_chunks=max_number_of_chunks,
            max_number_of_documents=max_number_of_documents,
            include_text_content=include_full_text,
            include_matched_chunks_content=not include_full_text,
            trace=trace,
            title_boost_query=query,
        )
        results, any_low_confidence = shape_search_results(
            [(collection_name, raw)],
            limit=max_number_of_documents,
        )
        augmenter.enrich_results(results, detected_entities)

        response = {"results": results}
        if graph_answer:
            response["graph_answer"] = graph_answer
        if any_low_confidence:
            response["lowConfidence"] = True
        if trace_default:
            response["trace"] = trace.to_dict()
        return json.dumps(response, indent=2, ensure_ascii=False)

    return search_fn


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
    _redirect_logging_to_stderr()
    _add_file_log_handler("huginn-multi-mcp.log")
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
