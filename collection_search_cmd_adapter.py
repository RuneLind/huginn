"""CLI search against one built collection.

The query goes through the same de-alias seam the HTTP route and the MCP tools
use (``main/privacy/query_privacy.py``): a collection whose index was built
aliased cannot be searched by a real name, so the CLI would silently return
nothing for exactly the queries the aliasing was introduced for.
"""
import json
import argparse
import logging
from pathlib import Path

from main.utils.logger import setup_root_logger
from main.utils.performance import log_execution_duration
from main.factories.search_collection_factory import create_collection_searcher
from main.privacy.query_privacy import dealias_query

COLLECTIONS_DIR = Path("./data/collections")


def dealiased_query(collection_name: str, query: str) -> str:
    """``query`` as this collection's index spells it.

    Never fatal, for the same reason serving is not: the index on disk is
    already clean, so a missing map or manifest costs de-aliasing, not privacy.
    """
    manifest_path = COLLECTIONS_DIR / collection_name / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return query
    try:
        from main.privacy.alias_registry import resolve_registry_for_manifest

        registry = resolve_registry_for_manifest(manifest, collection_name)
    except Exception as e:
        logging.error("Privacy: searching %s WITHOUT query de-aliasing: %s",
                      collection_name, e)
        return query
    return dealias_query(query, registry)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-collection", "--collection", required=True, help="Collection name (will be used as root folder name)")
    ap.add_argument("-query", "--query", required=True, help="Text query for search")

    ap.add_argument("-index", "--index", required=False, default=None, help="Index that will be used for search (auto-detected if omitted)")

    ap.add_argument("-maxNumberOfChunks", "--maxNumberOfChunks", required=False, type=int, default=None, help="Max number of text chunks in result")
    ap.add_argument("-maxNumberOfDocuments", "--maxNumberOfDocuments", required=False, type=int, default=10, help="Max number of documents in result")

    ap.add_argument("-includeFullText", "--includeFullText", action="store_true", required=False, default=False, help="If passed - full text content will be included in the search result.")
    ap.add_argument("-includeAllChunksText", "--includeAllChunksText", action="store_true", required=False, default=False, help="If passed - all chunks text content will be included in the search result.")
    ap.add_argument("-includeMatchedChunksText", "--includeMatchedChunksText", action="store_true", required=False, default=False, help="If passed - matched chunks text content will be included in the search result.")
    return vars(ap.parse_args(argv))


def main(argv=None):
    setup_root_logger()
    args = _parse_args(argv)

    searcher = create_collection_searcher(collection_name=args['collection'], index_name=args['index'])
    query = dealiased_query(args['collection'], args['query'])

    max_number_of_chunks = args['maxNumberOfChunks'] if args['maxNumberOfChunks'] is not None else args['maxNumberOfDocuments'] * 3
    search_result = log_execution_duration(lambda: searcher.search(query,
                                                                   max_number_of_chunks=max_number_of_chunks,
                                                                   max_number_of_documents=args['maxNumberOfDocuments'],
                                                                   include_text_content=args['includeFullText'],
                                                                   include_matched_chunks_content=args['includeMatchedChunksText'],
                                                                   include_all_chunks_content=args['includeAllChunksText']),
                                           identifier=f"Searching collection: \"{args['collection']}\" by query: \"{query}\"")

    logging.info(f"Search results:\n{json.dumps(search_result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
