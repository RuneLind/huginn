"""Shared write-helper for category-organized markdown ingest (YouTube, X articles)."""
import logging
import os

from main.utils.filename import sanitize_filename

logger = logging.getLogger(__name__)


def write_categorized_markdown(
    *,
    root: str,
    category: str,
    title: str,
    url: str,
    content: str,
) -> str:
    """Write `content` to `<root>/<category>/<sanitize_filename(title)>.md`.

    If a file with the same title exists for a different URL, append a numeric
    suffix `(2)`, `(3)`, ... up to `(99)`. Same URL → overwrite.

    Returns the relative path under `root` (e.g. "ai/general/My Title.md").
    """
    category_dir = os.path.join(root, category)
    os.makedirs(category_dir, exist_ok=True)
    base_filename = sanitize_filename(title)
    filename = base_filename + ".md"
    filepath = os.path.join(category_dir, filename)

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing_head = f.read(500)
        except OSError as e:
            logger.warning(f"Could not read existing file {filepath} for URL check: {e}")
            existing_head = ""
        if f"url: {url}" not in existing_head:
            for i in range(2, 100):
                filename = f"{base_filename} ({i}).md"
                filepath = os.path.join(category_dir, filename)
                if not os.path.exists(filepath):
                    break

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.join(category, filename)
