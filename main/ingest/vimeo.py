"""Vimeo summary ingest: save Muninn-supplied Vimeo summaries as categorized markdown.

Mirrors the TikTok/X ingest shape, with one difference that is the whole point of
the vertical: the request may carry ``transcript_markdown`` — the talk's captions
grouped into absolute 2-minute windows — and that is written into the document
under a ``## Transcript`` heading AFTER the summary. The summary alone is what
the HTTP response and the similarity query see (see ``write_summary``'s
``body_suffix``); the transcript is there so a search hit can be cited down to
the minute of a 50-minute conference talk.

EACH WINDOW IS ITS OWN ``### [HH:MM:SS]`` HEADING, and that shape is load-bearing
rather than cosmetic::

    ## Transcript

    ### [00:00:00]

    <the first two minutes of speech>

    ### [00:02:00]

    <the next two minutes>

``FilesDocumentConverter`` splits with ``MarkdownHeadingSplitter``, which cuts at
H1–H3 boundaries and carries the section's heading onto EVERY sub-chunk a long
section is cut into. Under a single ``## Transcript`` heading the whole transcript
is ONE section, so a 1000-char chunk taken from its middle is labelled
``Transcript`` and carries a cue only when a ``[HH:MM:SS]`` line happens to fall
inside its text — measured on the first real capture, 48 of 75 transcript chunks
carried none. With a heading per window, every chunk is labelled with the window
it was cut from. ``tests/test_knowledge_api_server.py``'s
``TestVimeoDocumentThroughTheConverter`` pins that against the real splitter.

``caption_lang`` / ``caption_kind`` are frontmatter provenance: Vimeo's
auto-generated captions garble proper nouns, so a reader (or a later fact-check)
must be able to tell an ``auto`` transcript from a ``manual`` one without
re-deriving it. They reach a consumer only because they are named in
``files_document_converter``'s ``_FRONTMATTER_METADATA_FIELDS`` allowlist — a
field this module writes and that list does not name is invisible over the API.
``video_id`` is derived from the url rather than sent, so it can never disagree
with the url the document is keyed on.
"""
import logging
import re
from typing import Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, field_validator

from main.ingest._summary_ingest import write_summary

logger = logging.getLogger(__name__)

#: Muninn caps the VTT it harvests at 2 MB (``VIMEO_VTT_MAX_BYTES``,
#: ``src/vimeo/captions.ts``). The same number here, so the receiving end refuses
#: a body that size on its own evidence rather than trusting the sender to have
#: capped it: this string is written to disk whole and then reindexed.
VIMEO_TRANSCRIPT_MAX_BYTES = 2 * 1024 * 1024

#: The hosts an id may be read from, matched EXACTLY (after dropping a ``www.``
#: prefix and a trailing root dot). A suffix test would also accept
#: ``evilvimeo.com``, and no host test at all accepts
#: ``https://evil.example/vimeo.com/999/redirect`` — a url a third party
#: controls, donating an id to a document keyed on it.
_VIMEO_HOSTS = frozenset({"vimeo.com", "player.vimeo.com"})

#: No leading zero: ``/0123`` and ``/123`` are the same video to Vimeo but two
#: different keys, and Vimeo never writes the first. Muninn's ``VIDEO_ID_RE``
#: refuses one, so deriving an id here would make the two sides disagree about
#: what the same document is called.
_ID = r"[1-9][0-9]*"

#: The url shapes an id may be read from, as WHOLE-PATH matches — Muninn's
#: ``src/vimeo/url.ts`` accepted forms, plus the showcase/group shapes its
#: capture entry point never produces but a stored document can carry.
#:
#: Enumerated rather than "take the last numeric path segment": ``/groups/12345``
#: is a group landing page whose last numeric segment names no video, and
#: enumerating is equally what refuses ``/showcase/7654321`` while accepting
#: ``/showcase/7654321/video/1223358361``. A first-numeric-segment ``search`` got
#: both of those backwards — it keyed the showcase and group urls on the
#: CONTAINER id, so two different talks in one showcase became one document key.
_VIMEO_PATH_RES = tuple(re.compile(pattern) for pattern in (
    rf"^/({_ID})/?$",                       # /<id>
    rf"^/({_ID})/[^/]+/?$",                 # /<id>/<unlisted hash>
    rf"^/video/({_ID})/?$",                 # player embed
    rf"^/video/({_ID})/[^/]+/?$",           # player embed + hash
    rf"^/channels/[^/]+/({_ID})/?$",        # channel page
    rf"^/showcase/[^/]+/video/({_ID})/?$",  # showcase page
    rf"^/groups/[^/]+/videos/({_ID})/?$",   # group page
))


def vimeo_video_id_from_url(url: Optional[str]) -> Optional[str]:
    """The video id a Vimeo URL names, or ``None`` when it names none.

    Host-gated first, then matched against the known path shapes — see
    ``_VIMEO_HOSTS`` and ``_VIMEO_PATH_RES`` for why neither half is optional.
    """
    if not url:
        return None
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").rstrip(".")
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    if host.startswith("www."):
        host = host[4:]
    if host not in _VIMEO_HOSTS:
        return None
    for pattern in _VIMEO_PATH_RES:
        match = pattern.match(parts.path)
        if match:
            return match.group(1)
    return None


def is_same_vimeo_video(url: Optional[str], other: Optional[str]) -> bool:
    """Do two urls address the same Vimeo video?

    The ingest route's self-link exclusion. A stored document's url and the
    incoming one are routinely different spellings of one video — ``www.``, the
    unlisted ``?h=`` hash, the player host — so an exact string compare lets the
    document that was just written come back as its own "related reading". When
    either side names no video (a re-categorized paste, a hand-written page) the
    string compare is still the answer.
    """
    left = vimeo_video_id_from_url(url)
    right = vimeo_video_id_from_url(other)
    if left and right:
        return left == right
    return (url or "") == (other or "")


class VimeoIngestRequest(BaseModel):
    """A finished Vimeo summary pushed by Muninn's vimeo capture vertical."""
    title: str
    url: str
    summary: str  # pre-made summary from the Muninn summarizer
    category: Optional[str] = None  # falls back to ai/general
    date: Optional[str] = None  # oEmbed upload date
    tags: Optional[list[str]] = None
    transcript_markdown: Optional[str] = None  # ### [HH:MM:SS] windows
    caption_lang: Optional[str] = None  # BCP-47 tag of the chosen track
    caption_kind: Optional[str] = None  # "manual" | "auto"
    duration_sec: Optional[int] = None  # from oEmbed, not the player

    @field_validator("transcript_markdown")
    @classmethod
    def _cap_transcript(cls, value: Optional[str]) -> Optional[str]:
        # Bytes, not characters: the cap bounds a whole-file write and the index
        # rebuild that follows it, and a transcript of Norwegian or Japanese
        # speech is well over one byte per character. Raised from the model, so
        # an oversized body is a 422 the route never has to handle.
        if value is not None and len(value.encode("utf-8")) > VIMEO_TRANSCRIPT_MAX_BYTES:
            raise ValueError(
                f"transcript_markdown exceeds {VIMEO_TRANSCRIPT_MAX_BYTES} bytes"
            )
        return value


def ingest_vimeo(req: VimeoIngestRequest, *, sources_path: str) -> dict:
    """Save a Vimeo summary (+ optional transcript) under its category.

    Returns ``{file_path, category, summary}`` — ``summary`` without the
    transcript, so the response stays small and the similarity query stays about
    the summary rather than 50 minutes of captions.
    """
    extra: dict[str, object] = {}
    video_id = vimeo_video_id_from_url(req.url)
    if video_id:
        extra["video_id"] = video_id
    if req.caption_lang:
        extra["caption_lang"] = req.caption_lang
    if req.caption_kind:
        extra["caption_kind"] = req.caption_kind
    if req.duration_sec is not None:
        # Written as an int, so the writer emits a bare `duration_sec: 3220` and
        # the converter can serve it as a number. A quoted "3220" compares and
        # sorts as text for every consumer that forgets to coerce it.
        extra["duration_sec"] = req.duration_sec

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
