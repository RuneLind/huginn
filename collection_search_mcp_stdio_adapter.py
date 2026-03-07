#!/usr/bin/env python3
import json
import argparse
import sys
import logging

from mcp.server.fastmcp import FastMCP

from main.factories.search_collection_factory import create_collection_searcher
from main.utils.logger import setup_root_logger

setup_root_logger()

# Redirect logging to stderr to avoid interfering with MCP protocol on stdout
try:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setStream(sys.stderr)
    
    # Optionally add file logging (won't break if it fails)
    try:
        from pathlib import Path
        log_file = Path.home() / "logs" / "teessi-confluence-mcp.log"
        log_file.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    except Exception:
        pass  # If file logging fails, continue without it
except Exception:
    pass  # If anything fails, continue without stderr redirect

mcp = FastMCP("documents-search")

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=True, help="Collection name (will be used as root folder name)")
ap.add_argument("-index", "--index", required=False, default=None, help="Index that will be used for search (auto-detected if omitted)")

ap.add_argument("-maxNumberOfChunks", "--maxNumberOfChunks", required=False, type=int, default=100, help="Max number of text chunks in result")
ap.add_argument("-maxNumberOfDocuments", "--maxNumberOfDocuments", required=False, type=int, default=None, help="Max number of documents in result")

ap.add_argument("-includeFullText", "--includeFullText", action="store_true", required=False, default=False, help="If passed - full text content will be included in the search result. By default only matched chunks content is included. If passed, it's better to reduce --maxNumberOfChunks or set small --maxNumberOfDocuments like 10-30 to avoid too big response and breaking AI agent.")
args = vars(ap.parse_args())

searcher = create_collection_searcher(collection_name=args['collection'], index_name=args['index'])

tool_description = """The tool allows searching in collection of documents by vector search. 
Each document contains 'url' field, if you consider a document as relevant to the query, always include the 'url' field in the response, put it close to the information used from the document"""

@mcp.tool(name=f"search_{args['collection']}", description=tool_description)
def search_documents(query: str) -> str:
    search_results = searcher.search(query, 
                                     max_number_of_chunks=args['maxNumberOfChunks'], 
                                     max_number_of_documents=args['maxNumberOfDocuments'],
                                     include_text_content=args['includeFullText'],
                                     include_matched_chunks_content=not args['includeFullText'])

    return json.dumps(search_results, indent=2, ensure_ascii=False)
    

if __name__ == "__main__":
    mcp.run(transport='stdio')