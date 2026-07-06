"""X/Twitter article ingest: save Chrome-extension-supplied summaries as categorized markdown."""
import logging
from typing import Optional

from pydantic import BaseModel

from main.ingest._summary_ingest import write_summary

logger = logging.getLogger(__name__)


class XArticleIngestRequest(BaseModel):
    """X/Twitter article content summarized by the Chrome extension."""
    title: str
    url: str
    author: str  # @handle of the article author
    summary: str  # pre-made summary from the extension
    date: Optional[str] = None
    category: Optional[str] = None  # auto-detected if not provided
    tags: Optional[list[str]] = None


def ingest_x_article(req: XArticleIngestRequest, *, sources_path: str) -> dict:
    """Save an X article summary to disk under its category. Returns {file_path, author, category, summary}."""
    result = write_summary(
        root=sources_path,
        title=req.title,
        url=req.url,
        summary=req.summary,
        category=req.category,
        date=req.date,
        tags=req.tags,
        extra_frontmatter={"author": req.author},
    )
    result["author"] = req.author
    logger.info(f"X article ingest: saved {result['file_path']} (author: {req.author}, category: {result['category']})")
    return result
