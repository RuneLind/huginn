"""Vimeo summary ingest: save Muninn-supplied Vimeo summaries as categorized markdown.

Mirrors the TikTok/X ingest shape, with one difference that is the whole point of
the vertical: the request may carry ``transcript_markdown`` — the talk's captions
grouped into absolute 2-minute windows, each headed ``[HH:MM:SS]`` — and that is
written into the document under a ``## Transcript`` heading AFTER the summary.
The summary alone is what the HTTP response and the similarity query see (see
``write_summary``'s ``body_suffix``); the transcript is there so a search hit can
be cited down to the minute of a 50-minute conference talk.

``caption_lang`` / ``caption_kind`` are frontmatter provenance: Vimeo's
auto-generated captions garble proper nouns, so a reader (or a later fact-check)
must be able to tell an ``auto`` transcript from a ``manual`` one without
re-deriving it. ``video_id`` is derived from the url rather than sent, so it can
never disagree with the url the document is keyed on.
"""
import logging
import re
from typing import Optional

from pydantic import BaseModel

from main.ingest._summary_ingest import write_summary

logger = logging.getLogger(__name__)

#: A Vimeo id is the numeric path segment of the canonical watch URL
#: (``https://vimeo.com/1223358361``). Anchored on a ``/`` so a query string or a
#: trailing unlisted-hash segment cannot be mistaken for the id.
_VIMEO_ID_RE = re.compile(r"/(\d+)(?:/|\?|#|$)")


def vimeo_video_id_from_url(url: Optional[str]) -> Optional[str]:
    """First numeric path segment of a Vimeo URL, or ``None`` when there is none."""
    if not url:
        return None
    match = _VIMEO_ID_RE.search(url)
    return match.group(1) if match else None


class VimeoIngestRequest(BaseModel):
    """A finished Vimeo summary pushed by Muninn's vimeo capture vertical."""
    title: str
    url: str
    summary: str  # pre-made summary from the Muninn summarizer
    category: Optional[str] = None  # falls back to ai/general
    date: Optional[str] = None  # oEmbed upload date
    tags: Optional[list[str]] = None
    transcript_markdown: Optional[str] = None  # windowed [HH:MM:SS] cues
    caption_lang: Optional[str] = None  # BCP-47 tag of the chosen track
    caption_kind: Optional[str] = None  # "manual" | "auto"
    duration_sec: Optional[int] = None  # from oEmbed, not the player


def ingest_vimeo(req: VimeoIngestRequest, *, sources_path: str) -> dict:
    """Save a Vimeo summary (+ optional transcript) under its category.

    Returns ``{file_path, category, summary}`` — ``summary`` without the
    transcript, so the response stays small and the similarity query stays about
    the summary rather than 50 minutes of captions.
    """
    extra: dict[str, str] = {}
    video_id = vimeo_video_id_from_url(req.url)
    if video_id:
        extra["video_id"] = video_id
    if req.caption_lang:
        extra["caption_lang"] = req.caption_lang
    if req.caption_kind:
        extra["caption_kind"] = req.caption_kind
    if req.duration_sec is not None:
        extra["duration_sec"] = str(req.duration_sec)

    transcript = (req.transcript_markdown or "").strip()
    body_suffix = f"## Transcript\n\n{transcript}\n" if transcript else None

    result = write_summary(
        root=sources_path,
        title=req.title,
        url=req.url,
        summary=req.summary,
        category=req.category,
        date=req.date,
        tags=req.tags,
        extra_frontmatter=extra or None,
        body_suffix=body_suffix,
    )
    logger.info(
        f"Vimeo ingest: saved {result['file_path']} "
        f"(video_id: {video_id}, category: {result['category']}, "
        f"transcript: {'yes' if body_suffix else 'no'})"
    )
    return result
