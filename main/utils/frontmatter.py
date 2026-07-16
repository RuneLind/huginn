"""YAML-frontmatter helpers for Huginn markdown sources.

Hand-rolled parser matching the conventions actually present across Huginn's
markdown ingestion pipelines (Jira, Confluence, sessions, wiki). All values
are returned as strings; YAML lists (``- item`` lines under an empty-value
key) are joined with commas. This is *not* a full YAML implementation —
deliberate, since downstream code expects ``dict[str, str]``.
"""
import logging
import re

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)

_MAX_HEAD_BYTES = 8192


def escape_frontmatter_value(value) -> str:
    """Quote a frontmatter scalar, escaping internal backslashes and quotes.

    Symmetric with the unescaping in ``_parse_block`` so a value containing a
    double-quote round-trips intact. Shared by every markdown ingest writer so
    they can't drift into writing unescaped (or differently-escaped) YAML.
    """
    text = "" if value is None else str(value)
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def parse_tags(value: str) -> list[str]:
    """Split a frontmatter ``tags`` scalar into a clean list of tag strings.

    Handles both canonical inline arrays (``[a, b, c]``) and bare comma lists
    (``a, b, c``) — huginn's frontmatter parser stores the value literally, so a
    bracketed doc arrives here as the string ``"[a, b, c]"``. Strips wrapping
    brackets and per-tag quotes, splits on commas, and drops empties. Returns
    ``[]`` for a falsy/whitespace value.

    This is the single normalization point for every doc-metadata tags consumer
    (``?tags=`` filter, indexed-text enrichment, graph node category/tags, the
    per-collection tag histogram) so a bracketed doc's first/last tag no longer
    leaks the ``[``/``]`` into a chip or a filter key.
    """
    if not value:
        return []
    inner = value.strip()
    if inner.startswith("["):
        inner = inner[1:]
    if inner.endswith("]"):
        inner = inner[:-1]
    tags = []
    for part in inner.split(","):
        tag = part.strip().strip('"').strip("'").strip()
        if tag:
            tags.append(tag)
    return tags


def read_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from markdown text into a ``dict[str, str]``.

    Returns ``{}`` if no frontmatter block is present.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    return _parse_block(match.group(1))


def read_frontmatter_from_path(filepath: str) -> dict[str, str]:
    """Open a file and parse its frontmatter. Returns ``{}`` on read/parse failure (logged)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            head = f.read(_MAX_HEAD_BYTES)
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read frontmatter from {filepath}: {e}")
        return {}
    return read_frontmatter(head)


def strip_frontmatter(text: str) -> str:
    """Remove the leading frontmatter block (including its trailing newline) from ``text``."""
    return FRONTMATTER_RE.sub('', text)


def read_frontmatter_and_body(filepath: str) -> tuple[dict[str, str], str]:
    """Read full file; return ``(frontmatter_dict, body_text)``.

    Returns ``({}, "")`` on read failure (logged). Unlike ``read_frontmatter_from_path``
    this reads the whole file because callers need the body too.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read frontmatter from {filepath}: {e}")
        return {}, ""
    return read_frontmatter(text), strip_frontmatter(text)


def _parse_block(fm_text: str) -> dict[str, str]:
    metadata = {}
    current_list_key = None
    for line in fm_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith("- ") and current_list_key:
            item = stripped[2:].strip()
            existing = metadata.get(current_list_key, "")
            metadata[current_list_key] = (existing + "," + item) if existing else item
            continue
        current_list_key = None
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                # Quoted scalar: drop the wrapping quotes and unescape only the
                # sequences escape_frontmatter_value emits (\\ and \"). Restricting
                # to those reverses the writer exactly while leaving an unrelated
                # backslash in a hand-written value (e.g. "C:\Users") untouched.
                # The regex consumes each \X pair left-to-right, so adjacent
                # escapes (\\\" → \") decode correctly, unlike chained .replace.
                value = re.sub(r'\\(["\\])', r'\1', value[1:-1])
            else:
                value = value.strip('"')
            if key and value:
                metadata[key] = value
            elif key and not value:
                current_list_key = key
    return metadata
