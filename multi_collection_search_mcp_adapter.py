#!/usr/bin/env python3
"""
Optimized multi-collection MCP adapter that explicitly shares the embedding model.
This saves ~180 MB RAM compared to running 3 separate MCP servers.

Usage:
    uv run multi_collection_search_mcp_adapter.py \
        --collections my-confluence my-jira my-docs \
        --maxNumberOfChunks 50
"""
import json
import argparse
import os
import sys
import logging

from mcp.server.fastmcp import FastMCP

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import detect_faiss_index, create_embedder, load_search_indexer, create_reranker
from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.core.search_trace import create_trace
from main.utils.logger import setup_root_logger

setup_root_logger()

# When set, attach a per-search trace to the JSON tool result. Orchestrators (e.g.
# Muninn) read result.trace, store it for the inspector UI, and strip it before
# showing the result to the model.
TRACE_DEFAULT = os.environ.get("HUGINN_TRACE_DEFAULT", "").lower() in ("1", "true", "yes")

# Redirect logging to stderr
try:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setStream(sys.stderr)
    
    try:
        from pathlib import Path
        log_file = Path.home() / "logs" / "huginn-multi-mcp.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    except Exception:
        pass
except Exception:
    pass

mcp = FastMCP("huginn-search")

ap = argparse.ArgumentParser()
ap.add_argument("-collections", "--collections", required=True, nargs='+', 
                help="List of collection names (e.g., my-confluence my-jira)")
ap.add_argument("-index", "--index", required=False,
                default=None,
                help="Index that will be used for search (auto-detected if omitted)")
ap.add_argument("-maxNumberOfChunks", "--maxNumberOfChunks", required=False, type=int, default=100, 
                help="Max number of text chunks in result")
ap.add_argument("-maxNumberOfDocuments", "--maxNumberOfDocuments", required=False, type=int, default=None, 
                help="Max number of documents in result")
ap.add_argument("-includeFullText", "--includeFullText", action="store_true", required=False, default=False, 
                help="Include full text content in search results")

args = vars(ap.parse_args())

# Create searchers for all collections using a shared embedder
disk_persister = DiskPersister(base_path="./data/collections")
searchers = {}

# Auto-detect index from first collection if not specified
index_name = args['index']
if index_name is None:
    index_name = detect_faiss_index(args['collections'][0], disk_persister)

logging.info(f"Loading shared embedding model for index: {index_name}")
shared_embedder = create_embedder(index_name)
logging.info(f"Embedding model loaded: {shared_embedder.model_name}")

shared_reranker = create_reranker()
logging.info(f"Reranker loaded: {shared_reranker.model_name}")

for collection_name in args['collections']:
    logging.info(f"Loading collection: {collection_name}")

    indexer = load_search_indexer(collection_name, disk_persister, shared_embedder=shared_embedder)

    searchers[collection_name] = DocumentCollectionSearcher(
        collection_name=collection_name,
        indexer=indexer,
        persister=disk_persister,
        reranker=shared_reranker,
    )

    logging.info(f"Collection {collection_name} loaded: {indexer.get_name()} ({indexer.get_size()} embeddings)")

# Register one tool per collection
for collection_name, searcher in searchers.items():
    tool_description = f"""Search in {collection_name} collection using vector search. 
Each document contains 'url' field - always include it in responses when citing information."""
    
    # Create a closure to capture the correct searcher
    def create_search_function(searcher_instance, coll_name):
        def search_fn(query: str) -> str:
            logging.info(f"Searching in {coll_name}: {query}")
            trace = create_trace(TRACE_DEFAULT)
            trace.set_query_raw(query)
            search_results = searcher_instance.search(
                query,
                max_number_of_chunks=args['maxNumberOfChunks'],
                max_number_of_documents=args['maxNumberOfDocuments'],
                include_text_content=args['includeFullText'],
                include_matched_chunks_content=not args['includeFullText'],
                trace=trace,
            )
            if TRACE_DEFAULT:
                search_results["trace"] = trace.to_dict()
            return json.dumps(search_results, indent=2, ensure_ascii=False)
        return search_fn
    
    search_function = create_search_function(searcher, collection_name)
    
    # Register the tool with FastMCP
    mcp.tool(name=f"search_{collection_name}", description=tool_description)(search_function)

logging.info(f"🚀 MCP server ready with {len(searchers)} collections (shared embedding model)")

if __name__ == "__main__":
    mcp.run(transport='stdio')
