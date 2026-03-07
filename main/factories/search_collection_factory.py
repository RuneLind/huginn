import logging

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import load_search_indexer, create_reranker
from main.core.documents_collection_searcher import DocumentCollectionSearcher

from main.utils.performance import log_execution_duration

logger = logging.getLogger(__name__)

def create_collection_searcher(collection_name, index_name=None):
    return log_execution_duration(
        lambda: __create_collection_searcher(collection_name, index_name),
        identifier=f"Preparing collection searcher"
    )

def __create_collection_searcher(collection_name, index_name):
    disk_persister = DiskPersister(base_path="./data/collections")

    indexer = load_search_indexer(collection_name, disk_persister, faiss_index_name=index_name)

    reranker = create_reranker()
    logger.info(f"Reranker loaded: {reranker.model_name}")

    return DocumentCollectionSearcher(collection_name=collection_name,
                                      indexer=indexer,
                                      persister=disk_persister,
                                      reranker=reranker)
