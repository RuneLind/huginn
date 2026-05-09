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
            value = value.strip().strip('"')
            if key and value:
                metadata[key] = value
            elif key and not value:
                current_list_key = key
    return metadata
