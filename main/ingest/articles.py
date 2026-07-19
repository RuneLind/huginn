"""Pasted-article ingest: save Muninn-supplied summaries of pasted articles as categorized markdown.

Mirrors x_articles ingest, but for arbitrary pasted articles (LinkedIn posts,
newsletters, ...) rather than X/Twitter articles. Two deliberate divergences from
the X model:

  * ``url`` is OPTIONAL — a pasted newsletter or LinkedIn post often has no
    canonical URL. When absent, the shared writer treats the doc as never
    matching an existing file, so two distinct URL-less pastes fork
    (``Title (2).md``) instead of clobbering each other.
  * ``author`` is OPTIONAL free-text (a person's name or a publication), not a
    required X ``@handle``. It is emitted into the frontmatter only when present.

The summary is produced upstream by Muninn's ``article`` capture vertical; this
route just files it under its ``ai/*`` category so it shows on the Summaries shelf.
"""
import logging
from typing import Optional

from pydantic import BaseModel

from main.ingest._summary_ingest import write_summary

logger = logging.getLogger(__name__)


class ArticleIngestRequest(BaseModel):
    """A finished pasted-article summary pushed by Muninn's article vertical."""
    title: str
    summary: str  # pre-made summary from the Muninn summarizer
    url: Optional[str] = None  # pasted articles often have no canonical URL
    author: Optional[str] = None  # free-text name/publication, not an X @handle
    date: Optional[str] = None
    category: Optional[str] = None  # falls back to ai/general
    tags: Optional[list[str]] = None


def ingest_article(req: ArticleIngestRequest, *, sources_path: str) -> dict:
    """Save a pasted-article summary to disk under its category.

    Returns ``{file_path, author, category, summary}`` (``author`` is echoed back,
    possibly ``None``).
    """
    result = write_summary(
        root=sources_path,
        title=req.title,
        url=req.url,
        summary=req.summary,
        category=req.category,
        date=req.date,
        tags=req.tags,
        extra_frontmatter={"author": req.author} if req.author else None,
    )
    result["author"] = req.author
    logger.info(
        f"Article ingest: saved {result['file_path']} "
        f"(author: {req.author}, category: {result['category']})"
    )
    return result
