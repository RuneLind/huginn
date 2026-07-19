"""Registry of push-ingest sources — one config per HTTP ingest endpoint.

A single source of truth that drives two otherwise-duplicated things:

  1. the generic FastAPI route factory in ``main/routes/ingest.py``,
  2. the ``--*-sources-path`` / ``--*-collection`` argparse args and the matching
     ``app.state`` assignments in ``knowledge_api_server.py``.

Adding a new push source is now: write its ``*IngestRequest`` model + ingest
function, then append one ``IngestSource(...)`` entry here. No new route handler,
no new argparse block, no new state wiring.

HARD CONTRACT: ``route_path``, request/response JSON keys, ``path_arg`` /
``collection_arg`` flag names, and env var names below are called by Muninn, a
Chrome extension, and the daily update scripts in production. Do not rename them.
"""
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from main.ingest.anthropic_summaries import (
    AnthropicSummaryIngestRequest,
    ingest_anthropic_summary,
)
from main.ingest.articles import ArticleIngestRequest, ingest_article
from main.ingest.jira import JiraIngestRequest, ingest_jira
from main.ingest.tiktok import TikTokIngestRequest, ingest_tiktok
from main.ingest.x_articles import XArticleIngestRequest, ingest_x_article
from main.ingest.youtube import YouTubeIngestRequest, ingest_youtube


@dataclass
class IngestSource:
    """Everything the routes and the server need to know about one push source."""

    name: str  # short slug, used for operation ids / logging
    route_path: str  # exact HTTP path (production contract)
    request_model: type[BaseModel]
    ingest_fn: Callable[..., dict]
    path_kwarg: str  # keyword the ingest_fn takes the write-root as

    # argparse + app.state wiring (dest is derived like argparse: strip -- and s/-/_/)
    path_arg: str
    path_env: str
    path_help: str
    collection_arg: str
    collection_env: str
    collection_default: str
    collection_help: str

    # route behavior
    operation: str  # human label for _ingest_errors / 5xx wrapping
    not_configured_detail: str  # 503 body when the path is unset
    response_fields: tuple[str, ...]  # result keys copied into the JSON response, in order
    similar_query: Callable[[BaseModel, dict], str]  # (req, result) -> similarity query
    exclude_match: Callable[[BaseModel, dict], bool]  # (req, doc) -> skip this hit?
    do_reindex: bool = True  # enqueue a background reindex + emit "reindex" key?

    @property
    def path_attr(self) -> str:
        return self.path_arg.lstrip("-").replace("-", "_")

    @property
    def collection_attr(self) -> str:
        return self.collection_arg.lstrip("-").replace("-", "_")


INGEST_SOURCES: list[IngestSource] = [
    IngestSource(
        name="youtube",
        route_path="/api/youtube/ingest",
        request_model=YouTubeIngestRequest,
        ingest_fn=ingest_youtube,
        path_kwarg="transcripts_path",
        path_arg="--youtube-transcripts-path",
        path_env="YOUTUBE_TRANSCRIPTS_PATH",
        path_help="Path to youtube-transcripts markdown repo",
        collection_arg="--youtube-collection",
        collection_env="YOUTUBE_COLLECTION",
        collection_default="youtube-summaries",
        collection_help="Collection name for youtube transcripts",
        operation="YouTube ingest",
        not_configured_detail="YouTube transcripts path not configured",
        response_fields=("file_path", "category", "summary"),
        similar_query=lambda req, result: result["summary"][:2000],
        exclude_match=lambda req, doc: doc.get("url", "") == req.url,
    ),
    IngestSource(
        name="jira",
        route_path="/api/jira/ingest",
        request_model=JiraIngestRequest,
        ingest_fn=ingest_jira,
        path_kwarg="sources_path",
        path_arg="--jira-sources-path",
        path_env="JIRA_SOURCES_PATH",
        path_help="Path to save Jira issue markdown files",
        collection_arg="--jira-collection",
        collection_env="JIRA_COLLECTION",
        collection_default="jira-issues",
        collection_help="Collection name for Jira issues",
        operation="Jira ingest",
        not_configured_detail="Jira sources path not configured (--jira-sources-path)",
        response_fields=("issue_key", "file_path", "summary"),
        similar_query=lambda req, result: f"{req.issueKey} {result['summary']}",
        exclude_match=lambda req, doc: req.issueKey in doc.get("url", ""),
        # Reindex skipped — the daily update script rebuilds the index + graph in
        # one pass. Trigger POST /api/collections/{name}/update manually if needed.
        do_reindex=False,
    ),
    IngestSource(
        name="x_article",
        route_path="/api/x-articles/ingest",
        request_model=XArticleIngestRequest,
        ingest_fn=ingest_x_article,
        path_kwarg="sources_path",
        path_arg="--x-articles-sources-path",
        path_env="X_ARTICLES_SOURCES_PATH",
        path_help="Path to save X article summary markdown files",
        collection_arg="--x-articles-collection",
        collection_env="X_ARTICLES_COLLECTION",
        collection_default="x-articles",
        collection_help="Collection name for X article summaries",
        operation="X article ingest",
        not_configured_detail="X articles sources path not configured (--x-articles-sources-path)",
        response_fields=("file_path", "author", "category", "summary"),
        similar_query=lambda req, result: req.summary[:2000],
        exclude_match=lambda req, doc: doc.get("url", "") == req.url,
    ),
    IngestSource(
        name="tiktok",
        route_path="/api/tiktok/ingest",
        request_model=TikTokIngestRequest,
        ingest_fn=ingest_tiktok,
        path_kwarg="sources_path",
        path_arg="--tiktok-sources-path",
        path_env="TIKTOK_SOURCES_PATH",
        path_help="Path to save TikTok summary markdown files",
        collection_arg="--tiktok-collection",
        collection_env="TIKTOK_COLLECTION",
        collection_default="tiktok-summaries",
        collection_help="Collection name for TikTok summaries",
        operation="TikTok ingest",
        not_configured_detail="TikTok sources path not configured (--tiktok-sources-path)",
        response_fields=("file_path", "author", "category", "summary"),
        similar_query=lambda req, result: req.summary[:2000],
        exclude_match=lambda req, doc: doc.get("url", "") == req.url,
    ),
    IngestSource(
        name="anthropic_summary",
        route_path="/api/anthropic-summaries/ingest",
        request_model=AnthropicSummaryIngestRequest,
        ingest_fn=ingest_anthropic_summary,
        path_kwarg="sources_path",
        path_arg="--anthropic-summaries-sources-path",
        path_env="ANTHROPIC_SUMMARIES_SOURCES_PATH",
        path_help="Path to save Anthropic summary markdown files",
        collection_arg="--anthropic-summaries-collection",
        collection_env="ANTHROPIC_SUMMARIES_COLLECTION",
        collection_default="anthropic-summaries",
        collection_help="Collection name for Anthropic summaries",
        operation="Anthropic summary ingest",
        not_configured_detail="Anthropic summaries sources path not configured (--anthropic-summaries-sources-path)",
        response_fields=("file_path", "category", "summary"),
        similar_query=lambda req, result: req.summary[:2000],
        exclude_match=lambda req, doc: doc.get("url", "") == req.url,
    ),
    IngestSource(
        name="article",
        route_path="/api/articles/ingest",
        request_model=ArticleIngestRequest,
        ingest_fn=ingest_article,
        path_kwarg="sources_path",
        path_arg="--articles-sources-path",
        path_env="ARTICLES_SOURCES_PATH",
        path_help="Path to save pasted-article summary markdown files",
        collection_arg="--articles-collection",
        collection_env="ARTICLES_COLLECTION",
        collection_default="article-summaries",
        collection_help="Collection name for pasted-article summaries",
        operation="Article ingest",
        not_configured_detail="Articles sources path not configured (--articles-sources-path)",
        response_fields=("file_path", "author", "category", "summary"),
        similar_query=lambda req, result: req.summary[:2000],
        exclude_match=lambda req, doc: doc.get("url", "") == req.url,
    ),
]


def source_by_name(name: str) -> IngestSource:
    """Look up a registered source by its ``name`` slug (raises KeyError if absent)."""
    for src in INGEST_SOURCES:
        if src.name == name:
            return src
    raise KeyError(name)
