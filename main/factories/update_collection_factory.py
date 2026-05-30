import os
from datetime import datetime, timedelta
import json

from main.persisters.disk_persister import DiskPersister
from main.sources.jira.jira_document_reader import JiraDocumentReader
from main.sources.jira.jira_document_converter import JiraDocumentConverter
from main.sources.jira.jira_cloud_document_reader import JiraCloudDocumentReader
from main.sources.jira.jira_cloud_document_converter import JiraCloudDocumentConverter
from main.sources.confluence.confluence_document_reader import ConfluenceDocumentReader
from main.sources.confluence.confluence_cloud_document_reader import ConfluenceCloudDocumentReader
from main.sources.confluence.confluence_document_converter import ConfluenceDocumentConverter
from main.sources.confluence.confluence_cloud_document_converter import ConfluenceCloudDocumentConverter
from main.sources.files.files_document_reader import FilesDocumentReader
from main.sources.files.files_document_converter import FilesDocumentConverter
from main.sources.notion.notion_document_reader import NotionDocumentReader
from main.sources.notion.notion_document_converter import NotionDocumentConverter
from main.indexes.indexer_factory import load_indexer
from main.core.documents_collection_creator import DocumentCollectionCreator, OPERATION_TYPE
from main.core.contextual_prefix import ChunkPrefixer, ContextualCache, make_backend

from main.utils.performance import log_execution_duration

def create_collection_updater(collection_name, contextual_backend_spec=None, contextual_cache_path=None,
                              contextual_workers=1):
    return log_execution_duration(
        lambda: __create_collection_updater(collection_name, contextual_backend_spec, contextual_cache_path,
                                            contextual_workers),
        identifier=f"Preparing collection updater"
    )

def __create_collection_updater(collection_name, contextual_backend_spec, contextual_cache_path, contextual_workers):
    disk_persister = DiskPersister(base_path="./data/collections")

    if not disk_persister.is_path_exists(collection_name):
        raise Exception(f"Collection {collection_name} does not exist")

    manifest = json.loads(disk_persister.read_text_file(f"{collection_name}/manifest.json"))

    document_reader, document_converter = _create_reader_and_converter(manifest)

    document_indexers = [load_indexer(indexer["name"], collection_name, disk_persister) for indexer in manifest['indexers']]

    chunk_prefixer = __build_chunk_prefixer_for_update(collection_name, manifest, contextual_backend_spec, contextual_cache_path)

    return DocumentCollectionCreator(collection_name=collection_name,
                                     document_reader=document_reader,
                                     document_converter=document_converter,
                                     document_indexers=document_indexers,
                                     persister=disk_persister,
                                     operation_type=OPERATION_TYPE.UPDATE,
                                     chunk_prefixer=chunk_prefixer,
                                     contextual_workers=contextual_workers)


def __build_chunk_prefixer_for_update(collection_name, manifest, override_spec, cache_path):
    spec = override_spec
    if not spec:
        existing = manifest.get('contextualPrefix') or {}
        spec = existing.get('model') or 'none'

    generator = make_backend(spec)
    if generator is None:
        return None

    cache_path = cache_path or f"./data/contextual_caches/{collection_name}.json"
    return ChunkPrefixer(generator=generator, cache=ContextualCache(cache_path))


def __calculate_update_time(manifest):
    return datetime.fromisoformat(manifest['lastModifiedDocumentTime']) - timedelta(days=1)

def __calculate_update_date(manifest):
    return __calculate_update_time(manifest).date()

def _create_reader_and_converter(manifest):
    """Dispatch to the builder registered for the manifest's reader type."""
    reader_type = manifest['reader']['type']
    builder = _READER_BUILDERS.get(reader_type)
    if builder is None:
        raise Exception(f"Unknown document reader type: {reader_type}")
    return builder(manifest)


# --- Auth resolvers (read + validate credentials from the environment) ---

def _jira_basic_auth():
    return {
        "token": os.environ.get('JIRA_TOKEN'),
        "login": os.environ.get('JIRA_LOGIN'),
        "password": os.environ.get('JIRA_PASSWORD'),
    }


def _confluence_basic_auth():
    token = os.environ.get('CONF_TOKEN')
    login = os.environ.get('CONF_LOGIN')
    password = os.environ.get('CONF_PASSWORD')
    if not token and (not login or not password):
        raise ValueError("Either 'token' ('CONF_TOKEN' env variable) or both 'login' ('CONF_LOGIN' env variable) and 'password' ('CONF_PASSWORD' env variable) must be provided.")
    return {"token": token, "login": login, "password": password}


def _atlassian_cloud_auth(product):
    def resolve():
        email = os.environ.get('ATLASSIAN_EMAIL')
        api_token = os.environ.get('ATLASSIAN_TOKEN')
        if not email or not api_token:
            raise ValueError(f"Both 'ATLASSIAN_EMAIL' and 'ATLASSIAN_TOKEN' environment variables must be provided for {product}.")
        return {"email": email, "api_token": api_token}
    return resolve


# --- Builders ---

def _build_query_reader(manifest, reader_cls, converter_cls, modified_field, auth, with_comments):
    """Build a CQL/JQL-style reader: an incremental query window plus credentials.

    modified_field is the source's "last changed" field name ("updated" for Jira,
    "lastModified" for Confluence); auth() returns the credential kwargs (and
    validates them); with_comments forwards the readAllComments flag.
    """
    cfg = manifest['reader']
    update_date = __calculate_update_date(manifest).isoformat()
    query_addition = f'AND (created >= "{update_date}" OR {modified_field} >= "{update_date}")'

    kwargs = dict(
        base_url=cfg['baseUrl'],
        query=f"{cfg['query']} {query_addition}",
        **auth(),
        batch_size=cfg['batchSize'],
    )
    if with_comments:
        kwargs['read_all_comments'] = cfg['readAllComments']
    return reader_cls(**kwargs), converter_cls()


def _build_jira(manifest):
    return _build_query_reader(manifest, JiraDocumentReader, JiraDocumentConverter,
                               "updated", _jira_basic_auth, with_comments=False)


def _build_jira_cloud(manifest):
    return _build_query_reader(manifest, JiraCloudDocumentReader, JiraCloudDocumentConverter,
                               "updated", _atlassian_cloud_auth("Jira Cloud"), with_comments=False)


def _build_confluence(manifest):
    return _build_query_reader(manifest, ConfluenceDocumentReader, ConfluenceDocumentConverter,
                               "lastModified", _confluence_basic_auth, with_comments=True)


def _build_confluence_cloud(manifest):
    return _build_query_reader(manifest, ConfluenceCloudDocumentReader, ConfluenceCloudDocumentConverter,
                               "lastModified", _atlassian_cloud_auth("Confluence Cloud"), with_comments=True)


def _build_notion(manifest):
    token = os.environ.get('NOTION_TOKEN')
    if not token:
        raise ValueError("NOTION_TOKEN environment variable must be set.")

    update_time = __calculate_update_time(manifest)
    reader = NotionDocumentReader(
        token=token,
        root_page_id=manifest['reader'].get('rootPageId'),
        request_delay=manifest['reader'].get('requestDelay', 0.35),
        start_from_time=update_time,
    )
    return reader, NotionDocumentConverter()


def _build_local_files(manifest):
    cfg = manifest['reader']
    update_time = __calculate_update_time(manifest)
    reader = FilesDocumentReader(
        base_path=cfg['basePath'],
        include_patterns=cfg.get('includePatterns', [".*"]),
        exclude_patterns=cfg.get('excludePatterns', []),
        fail_fast=cfg.get('failFast', False),
        start_from_time=update_time,
    )
    return reader, FilesDocumentConverter()


_READER_BUILDERS = {
    "jira": _build_jira,
    "jiraCloud": _build_jira_cloud,
    "confluence": _build_confluence,
    "confluenceCloud": _build_confluence_cloud,
    "notion": _build_notion,
    "localFiles": _build_local_files,
}