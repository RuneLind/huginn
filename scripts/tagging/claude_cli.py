"""Shared helpers for calling Claude CLI headless and processing markdown documents."""
import json
import re

from main.utils.claude_cli import call_claude  # noqa: F401  (re-exported)
from main.utils.frontmatter import strip_frontmatter

# tag_documents.py uses this for tag-line manipulation inside the FM block. The inner
# capture group + no trailing-newline form is intentional — inject_tags relies on
# match.end() landing right after the second `---` so the body's leading newline
# is preserved when splicing.
FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---', re.DOTALL)


def get_content_excerpt(content: str, max_chars: int = 2000) -> str:
    """Get content without frontmatter, truncated to max_chars."""
    stripped = strip_frontmatter(content).strip()
    if len(stripped) > max_chars:
        return stripped[:max_chars] + "..."
    return stripped


def extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from text."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None
