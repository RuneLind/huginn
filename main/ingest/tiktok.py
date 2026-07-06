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
from typing import Optional

from pydantic import BaseModel

from main.ingest._summary_ingest import write_summary

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
    author = req.author or "unknown"
    result = write_summary(
        root=sources_path,
        title=req.title,
        url=req.url,
        summary=req.summary,
        category=req.category,
        date=req.date,
        tags=req.tags,
        extra_frontmatter={"author": author},
    )
    result["author"] = author
    logger.info(f"TikTok ingest: saved {result['file_path']} (author: {author}, category: {result['category']})")
    return result
