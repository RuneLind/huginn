import argparse
import os

from main.utils.logger import setup_root_logger
from main.sources.files.files_document_reader import FilesDocumentReader
from main.sources.files.files_document_converter import FilesDocumentConverter
from main.factories.create_collection_factory import create_collection_creator

setup_root_logger()

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=False, help="Collection name (will be used as root folder name). If not provided, it will be derived from the basePath folder name.")

ap.add_argument("-basePath", "--basePath", required=True, help="Path to the root folder from which files will be read.")
ap.add_argument("-includePatterns", "--includePatterns", required=False, default=[".*"], help="List of file patterns to include into collection", nargs='+')
ap.add_argument("-excludePatterns", "--excludePatterns", required=False, default=[], help="List of file patterns to NOT include into collection", nargs='+')

ap.add_argument("-indexers", "--indexers", required=False, default=["indexer_FAISS_IndexFlatL2__embeddings_multilingual-e5-base", "indexer_BM25"], help="List on indexer names", nargs='+')

ap.add_argument("-failFast", "--failFast", action="store_true", required=False, default=False, help="If passed - the process will stop on the first error. Otherwise, it will try to process all files and log errors for those that failed.")

ap.add_argument("--contextual-model", required=False, default="none",
                help="Contextual-prefix backend spec, e.g. 'none', 'echo', 'ollama:qwen3.6:35b-a3b-nvfp4', 'claude-code:claude-haiku-4-5'.")
ap.add_argument("--contextual-cache", required=False, default=None,
                help="Path to the contextual-prefix cache JSON (defaults to data/contextual_caches/<name>.json — outside the collection folder so it survives re-creates).")
ap.add_argument("--contextual-workers", required=False, type=int, default=1,
                help="How many documents to prefix concurrently (default 1 = sequential). 4 is a sensible value for claude-code:claude-haiku-4-5 and cuts wall time roughly N×; watch rate limits.")

args = vars(ap.parse_args())

files_document_reader = FilesDocumentReader(base_path=args['basePath'],
                                            include_patterns=args['includePatterns'],
                                            exclude_patterns=args['excludePatterns'],
                                            fail_fast=args['failFast'])
files_document_converter = FilesDocumentConverter()

collection_name = args['collection'] if args['collection'] else os.path.basename(args['basePath'])
files_collection_creator = create_collection_creator(collection_name=collection_name,
                                                     indexers=args['indexers'],
                                                     document_reader=files_document_reader,
                                                     document_converter=files_document_converter,
                                                     use_cache=False,
                                                     contextual_backend_spec=args['contextual_model'],
                                                     contextual_cache_path=args['contextual_cache'],
                                                     contextual_workers=args['contextual_workers'])

files_collection_creator.run()


