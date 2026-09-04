"""Shared body for category-organized summary ingest (YouTube, X, TikTok, Anthropic, article, Vimeo).

These push sources all do the same thing: default + validate a category,
build category-derived tags, assemble a small YAML frontmatter block, and write
the markdown via ``write_categorized_markdown``. The only real difference is a
handful of source-specific frontmatter fields (e.g. ``author`` for X/TikTok),
passed through ``extra_frontmatter``. Jira ingest is deliberately *not* built on
this — its frontmatter, metadata merge, and mtime handling are genuinely
different.
"""
import datetime as dt
from typing import Optional

from fastapi import HTTPException

from main.ingest.categories import CATEGORIES
from main.ingest._markdown_writer import write_categorized_markdown
from main.utils.frontmatter import escape_frontmatter_value


def build_summary_tags(category: str, tags: Optional[list[str]]) -> str:
    """Category parts + explicit tags, de-duped, order preserved, comma-joined."""
    tag_parts = list(category.split("/"))
    for t in tags or []:
        if t not in tag_parts:
            tag_parts.append(t)
    return ", ".join(tag_parts)


def write_summary(
    *,
    root: str,
    title: str,
    url: str,
    summary: str,
    category: Optional[str] = None,
    date: Optional[str] = None,
    tags: Optional[list[str]] = None,
    extra_frontmatter: Optional[dict[str, str]] = None,
    body_suffix: Optional[str] = None,
) -> dict:
    """Validate + write a summary as categorized markdown.

    ``category`` defaults to ``ai/general`` and must be one of ``CATEGORIES``
    (400 otherwise). ``extra_frontmatter`` keys are emitted between ``url`` and
    ``category`` in insertion order, so callers control field placement (e.g.
    ``author`` for X/TikTok). Returns ``{file_path, category, summary}``.

    ``body_suffix`` is appended to the FILE after the summary (one blank line
    between; the caller owns its heading) and is deliberately NOT part of the
    returned ``summary``. The Vimeo vertical uses it for the full timestamped
    transcript, which belongs in the indexed document — so a search hit can
    cite a cue — but not in the HTTP response (``response_fields``) nor in the
    similarity query built from the summary.
    """
    date = date or dt.date.today().isoformat()
    category = category or "ai/general"
    if category not in CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}",
        )

    tags_str = build_summary_tags(category, tags)

    lines = [
        "---",
        f"date: {escape_frontmatter_value(date)}",
        f"url: {escape_frontmatter_value(url)}",
    ]
    for key, value in (extra_frontmatter or {}).items():
        lines.append(f"{key}: {escape_frontmatter_value(value)}")
    lines.append(f"category: {escape_frontmatter_value(category)}")
    lines.append(f"tags: {escape_frontmatter_value(tags_str)}")
    lines.append("---")
    frontmatter = "\n".join(lines) + "\n\n"

    body = summary
    if body_suffix:
        body = body.rstrip("\n") + "\n\n" + body_suffix

    file_rel_path = write_categorized_markdown(
        root=root,
        category=category,
        title=title,
        url=url,
        content=frontmatter + body,
    )
    return {"file_path": file_rel_path, "category": category, "summary": summary}
