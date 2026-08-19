"""Markdown / JSON text helpers used by the tagging scripts. Named for what it
holds: the Claude CLI wrapper lives in ``main.utils.claude_cli``, and this module
used to share that basename, which would have shadowed any unqualified
``import claude_cli`` from the tagging scripts, since they put this directory on
``sys.path``. Nothing was ever actually shadowed — every call site imports the
package-qualified path — so this was a latent hazard, not a live bug."""
import json
import re

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
