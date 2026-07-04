"""TikTok summary ingest: save Muninn-supplied TikTok summaries as categorized markdown.

Mirrors x_articles ingest. The summary (and optional visual-content notes) is
produced upstream by Muninn's tiktok vertical — yt-dlp download + whisper
transcript + scene-change keyframes fed to the summarizer. This route just
files the finished summary under its category so it shows on the Summaries
shelf badged "TikTok".

`author` is optional (mapped from yt-dlp's `uploader`) and defaults to
"unknown" — unlike x_articles where the @handle is always known.
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


class TikTokIngestRequest(BaseModel):
    """A finished TikTok summary pushed by Muninn's tiktok vertical."""
    title: str
    url: str
    summary: str  # pre-made summary from the Muninn summarizer
    author: Optional[str] = None  # yt-dlp uploader; defaults to "unknown"
    date: Optional[str] = None
    category: Optional[str] = None  # falls back to ai/general
    tags: Optional[list[str]] = None


def ingest_tiktok(req: TikTokIngestRequest, *, sources_path: str) -> dict:
    """Save a TikTok summary to disk under its category. Returns {file_path, author, category, summary}."""
    date = req.date or dt.date.today().isoformat()
    author = req.author or "unknown"

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
        f"author: {escape_frontmatter_value(author)}\n"
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
    logger.info(f"TikTok ingest: saved {file_rel_path} (author: {author}, category: {category})")

    return {
        "file_path": file_rel_path,
        "author": author,
        "category": category,
        "summary": req.summary,
    }
