"""Anthropic summary ingest: save Muninn-supplied summaries as categorized markdown.

Mirrors x_articles ingest but without an `author` field — Anthropic releases
(docs, blog posts, changelog/commit items) have no single human handle. The
summary is produced upstream by Muninn's anthropic vertical; this route just
files it under its `ai/*` category so it shows on the Summaries shelf badged
"Claude".
"""
import logging
import datetime as dt
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from main.ingest.categories import CATEGORIES
from main.ingest._markdown_writer import write_categorized_markdown
from main.utils.frontmatter import escape_frontmatter_value

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
    date = req.date or dt.date.today().isoformat()

    category = req.category or "ai/general"
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}")

    # Tags: category parts + explicit tags, de-duped, order preserved
    tag_parts = list(category.split("/"))
    if req.tags:
        for t in req.tags:
            if t not in tag_parts:
                tag_parts.append(t)
    tags = ", ".join(tag_parts)

    frontmatter = (
        "---\n"
        f"date: {escape_frontmatter_value(date)}\n"
        f"url: {escape_frontmatter_value(req.url)}\n"
        f"category: {escape_frontmatter_value(category)}\n"
        f"tags: {escape_frontmatter_value(tags)}\n"
        "---\n\n"
    )
    md_content = frontmatter + req.summary

    file_rel_path = write_categorized_markdown(
        root=sources_path,
        category=category,
        title=req.title,
        url=req.url,
        content=md_content,
    )
    logger.info(f"Anthropic summary ingest: saved {file_rel_path} (category: {category})")

    return {
        "file_path": file_rel_path,
        "category": category,
        "summary": req.summary,
    }
