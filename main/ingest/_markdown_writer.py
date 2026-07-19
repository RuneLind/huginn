"""Shared write-helper for category-organized markdown ingest (YouTube, X articles)."""
import os

from main.utils.filename import sanitize_filename
from main.utils.frontmatter import read_frontmatter_from_path


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
    suffix `(2)`, `(3)`, ... up to `(99)`. Same URL → overwrite. An absent/empty
    incoming URL never matches, so two distinct URL-less pastes fork instead of
    clobbering each other.

    Returns the relative path under `root` (e.g. "ai/general/My Title.md").
    """
    category_dir = os.path.join(root, category)
    os.makedirs(category_dir, exist_ok=True)
    base_filename = sanitize_filename(title)
    filename = base_filename + ".md"
    filepath = os.path.join(category_dir, filename)

    if os.path.exists(filepath):
        # Same URL → overwrite; same title but a different URL → fork a numbered
        # name. Compare the parsed frontmatter url: the writer quotes values
        # (url: "..."), so a raw `url: <value>` substring check never matches and
        # would fork a (2), (3) file on every same-URL re-ingest.
        #
        # An absent/empty incoming URL never "matches" an existing doc: empty
        # URLs would otherwise compare equal (both parse to None), so two
        # distinct URL-less pastes with the same title would silently clobber
        # each other. `not url` forces them down the numbered-suffix fork below.
        # Accepted tradeoff: re-ingesting the *same* URL-less paste forks a (2)
        # file instead of overwriting. All four url-bearing verticals send a
        # required non-empty url and keep their overwrite-on-same-url behavior.
        existing_url = read_frontmatter_from_path(filepath).get("url")
        if not url or existing_url != url:
            for i in range(2, 100):
                filename = f"{base_filename} ({i}).md"
                filepath = os.path.join(category_dir, filename)
                if not os.path.exists(filepath):
                    break

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.join(category, filename)
