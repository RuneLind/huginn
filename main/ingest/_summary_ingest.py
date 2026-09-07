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
from main.utils.frontmatter import (
    escape_frontmatter_value,
    frontmatter_scalar,
    normalize_frontmatter_string,
)

#: The rendered frontmatter's bound, in CHARACTERS: `read_frontmatter_from_path`
#: parses only the first 8192 characters of a file (`_MAX_HEAD_BYTES` is a
#: text-mode read), and `write_categorized_markdown`'s overwrite check reads the
#: url through it — a head that overruns it parses to nothing, the check sees
#: no url, and every re-ingest of the same url forks `Title (2).md`, silently.
#: Bounding the OUTPUT closes that for every field that reaches the head, on
#: every vertical that writes through here — url, a numeric field, any number
#: of tags — where a per-field cap on one request model kept missing one.
#: 75% of the head: real frontmatter is a few hundred characters. Two
#: residuals, knowingly outside this bound: `main/ingest/jira.py` renders its
#: own frontmatter and does not pass through here, and
#: `scripts/cross_collection_gap_analysis.py` reads a 2000-character head of
#: its own — a document between 2000 and 6144 is invisible to that script.
FRONTMATTER_MAX_CHARS = 6144

#: Cap on ONE frontmatter-bound string of a request model, in bytes. Well above
#: any real value (a Vimeo CDN thumbnail url is ~80 bytes). It exists so an
#: oversized field answers a 422 NAMING THE FIELD instead of the 413 above,
#: which names only the whole head. Shared rather than per-vertical: the Vimeo
#: and YouTube verticals write the same keys through the same writer, and two
#: numbers for one bound is how they drift apart. It bounds a VALUE — the head
#: itself is bounded by `FRONTMATTER_MAX_CHARS`, which is the real bound, since
#: `url`, a bare numeric field and a tags list of any length reach the head too.
FRONTMATTER_FIELD_MAX_BYTES = 512


def check_frontmatter_field(value: Optional[str]) -> Optional[str]:
    """Pydantic validator body for a string written verbatim into frontmatter.

    NORMALIZES then caps, and both halves are load-bearing.

    The normalization is ``normalize_frontmatter_string`` — a strip, with a
    stripped-empty value becoming ``None`` — the SAME helper the two disk
    readers of these keys apply (the documents listing and the files
    converter), so the three of them cannot drift into three rules. Every
    vertical omits such a field with ``if req.<field>:``, and ``"  "`` is
    TRUTHY — so without this the omit branch never fires and the document
    carries ``summary_kind: "  "``, which is neither a kind nor the "we do not
    know" that the MISSING key means. The whole design of these fields is that
    "key absent" is the one no-value signal, and a truthy blank is the one
    input that defeats it. A padded value is stored stripped for the other half
    of the same rule: ``" deep "`` compares unequal to ``"deep"`` for every
    consumer — the documents listing, the converter metadata, a shelf filter.
    Callers therefore have to read the value BACK off the model (pydantic
    replaces the field with what this returns) rather than trusting what they
    posted.

    LEADING AND TRAILING whitespace only, which is all this does. The writer
    transforms the value further: ``escape_frontmatter_value`` collapses runs
    of ``\\r``/``\\n`` to a single space, so ``"de\\nep"`` passes here unchanged
    and reaches disk as ``de ep``. A caller that must know the stored spelling
    of an interior-whitespace value has to read the document back, not the
    model.

    The cap is raised from the request MODEL, so an oversized value is a 422
    the route never has to handle. Why a per-field cap at all:
    ``read_frontmatter_from_path`` reads only the first 8192 characters of a
    file, so a value that pushes the closing ``---`` past that would make the
    overwrite check see no url and fork ``Title (2).md`` on every re-ingest,
    silently. ``FRONTMATTER_MAX_CHARS`` closes that for every field whatever
    carries it; this one names the culprit. It counts the STRIPPED value,
    which is the one that reaches the head.

    Bytes, not characters: the head is bounded in characters, but this is the
    conservative direction and text in Norwegian or Japanese is well over one
    byte per character.
    """
    value = normalize_frontmatter_string(value)
    if value is None:
        return None
    if len(value.encode("utf-8")) > FRONTMATTER_FIELD_MAX_BYTES:
        raise ValueError(f"field exceeds {FRONTMATTER_FIELD_MAX_BYTES} bytes")
    return value


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
    extra_frontmatter: Optional[dict[str, object]] = None,
    body_suffix: Optional[str] = None,
) -> dict:
    """Validate + write a summary as categorized markdown.

    ``category`` defaults to ``ai/general`` and must be one of ``CATEGORIES``
    (400 otherwise). A rendered frontmatter over ``FRONTMATTER_MAX_CHARS`` is
    refused with 413 before anything is written (see the constant). ``extra_frontmatter`` keys are emitted between ``url`` and
    ``category`` in insertion order, so callers control field placement (e.g.
    ``author`` for X/TikTok); an ``int`` value is written BARE (``duration_sec:
    3220``) so the reader can serve it as a number — see ``frontmatter_scalar``.
    Returns ``{file_path, category, summary}``.

    ``body_suffix`` is appended to the FILE after the summary (one blank line
    between; the caller owns its heading) and is deliberately NOT part of the
    returned ``summary``. ``None`` means "nothing to append"; an EMPTY STRING is
    appended like any other suffix (i.e. contributes only the blank line), so a
    caller that computes a suffix does not get a silent skip the moment the
    computation comes out empty. The Vimeo vertical uses it for the full timestamped
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
        lines.append(f"{key}: {frontmatter_scalar(value)}")
    lines.append(f"category: {escape_frontmatter_value(category)}")
    lines.append(f"tags: {escape_frontmatter_value(tags_str)}")
    lines.append("---")
    frontmatter = "\n".join(lines) + "\n\n"
    if len(frontmatter) > FRONTMATTER_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"frontmatter would be {len(frontmatter)} characters; the readers parse {FRONTMATTER_MAX_CHARS}",
        )

    body = summary
    if body_suffix is not None:
        body = body.rstrip("\n") + "\n\n" + body_suffix

    file_rel_path = write_categorized_markdown(
        root=root,
        category=category,
        title=title,
        url=url,
        content=frontmatter + body,
    )
    return {"file_path": file_rel_path, "category": category, "summary": summary}
