from main.sources.document_cache_reader_decorator import CacheReaderDecorator
from main.core.documents_collection_creator import DocumentCollectionCreator, OPERATION_TYPE
from main.core.contextual_prefix import ChunkPrefixer, ContextualCache, make_backend
from main.indexes.indexer_factory import create_indexer
from main.persisters.disk_persister import DiskPersister

from main.utils.performance import log_execution_duration

def create_collection_creator(collection_name, indexers, document_reader, document_converter, use_cache=True,
                              contextual_backend_spec="none", contextual_cache_path=None, contextual_workers=1):
    return log_execution_duration(
        lambda: __create_collection_creator(collection_name, indexers, document_reader, document_converter, use_cache,
                                            contextual_backend_spec, contextual_cache_path, contextual_workers),
        identifier=f"Preparing collection creator"
    )

def __create_collection_creator(collection_name, indexers, document_reader, document_converter, use_cache,
                                contextual_backend_spec, contextual_cache_path, contextual_workers):
    if use_cache:
        cache_disk_persister = DiskPersister(base_path="./data/caches")
        result_document_reader = CacheReaderDecorator(reader=document_reader,
                                                      persister=cache_disk_persister)
    else:
        result_document_reader = document_reader

    document_indexers = [create_indexer(indexer_name) for indexer_name in indexers]

    disk_persister = DiskPersister(base_path="./data/collections")

    chunk_prefixer = _build_chunk_prefixer(collection_name, contextual_backend_spec, contextual_cache_path)

    return DocumentCollectionCreator(collection_name=collection_name,
                                     document_reader=result_document_reader,
                                     document_converter=document_converter,
                                     document_indexers=document_indexers,
                                     persister=disk_persister,
                                     operation_type=OPERATION_TYPE.CREATE,
                                     chunk_prefixer=chunk_prefixer,
                                     contextual_workers=contextual_workers)


def _build_chunk_prefixer(collection_name, backend_spec, cache_path):
    generator = make_backend(backend_spec)
    if generator is None:
        return None
    # Cache lives outside the collection folder so it survives a full re-create
    # (which wipes data/collections/<name>/). Unchanged chunks stay cached across rebuilds.
    cache_path = cache_path or f"./data/contextual_caches/{collection_name}.json"
    return ChunkPrefixer(generator=generator, cache=ContextualCache(cache_path))
