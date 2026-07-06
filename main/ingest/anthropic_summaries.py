"""Anthropic summary ingest: save Muninn-supplied summaries as categorized markdown.

Mirrors x_articles ingest but without an `author` field — Anthropic releases
(docs, blog posts, changelog/commit items) have no single human handle. The
summary is produced upstream by Muninn's anthropic vertical; this route just
files it under its `ai/*` category so it shows on the Summaries shelf badged
"Claude".
"""
import logging
from typing import Optional

from pydantic import BaseModel

from main.ingest._summary_ingest import write_summary

logger = logging.getLogger(__name__)


class AnthropicSummaryIngestRequest(BaseModel):
    """A finished Anthropic summary pushed by Muninn's anthropic vertical."""
    title: str
    url: str
    summary: str  # pre-made summary from the Muninn summarizer
    date: Optional[str] = None
    category: Optional[str] = None  # falls back to ai/general
    tags: Optional[list[str]] = None


def ingest_anthropic_summary(req: AnthropicSummaryIngestRequest, *, sources_path: str) -> dict:
    """Save an Anthropic summary to disk under its category. Returns {file_path, category, summary}."""
    result = write_summary(
        root=sources_path,
        title=req.title,
        url=req.url,
        summary=req.summary,
        category=req.category,
        date=req.date,
        tags=req.tags,
    )
    logger.info(f"Anthropic summary ingest: saved {result['file_path']} (category: {result['category']})")
    return result
