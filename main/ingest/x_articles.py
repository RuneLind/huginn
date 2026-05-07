"""X/Twitter article ingest: save Chrome-extension-supplied summaries as categorized markdown."""
import logging
import datetime as dt
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from main.ingest.categories import CATEGORIES
from main.ingest._markdown_writer import write_categorized_markdown

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
        f"---\n"
        f"date: {date}\n"
        f"url: {req.url}\n"
        f"author: {req.author}\n"
        f"category: {category}\n"
        f"tags: \"{tags}\"\n"
        f"---\n\n"
    )
    md_content = frontmatter + req.summary

    file_rel_path = write_categorized_markdown(
        root=sources_path,
        category=category,
        title=req.title,
        url=req.url,
        content=md_content,
    )
    logger.info(f"X article ingest: saved {file_rel_path} (author: {req.author}, category: {category})")

    return {
        "file_path": file_rel_path,
        "author": req.author,
        "category": category,
        "summary": req.summary,
    }
