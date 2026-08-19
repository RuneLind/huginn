"""Tests for scripts/tagging/tagging_text — the markdown/JSON helpers the tagging
scripts import off sys.path (module renamed from claude_cli to stop shadowing
main.utils.claude_cli)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tagging"))

from tagging_text import extract_json_array, get_content_excerpt  # noqa: E402


class TestExtractJsonArray:
    def test_plain_array(self):
        assert extract_json_array('["a", "b"]') == ["a", "b"]

    def test_array_embedded_in_prose_or_fences(self):
        text = 'Here you go:\n```json\n["a", "b"]\n```\nHope that helps.'
        assert extract_json_array(text) == ["a", "b"]

    def test_object_falls_through_to_the_bracket_scan(self):
        # The whole-text parse yields a dict, not a list, so the []-scan runs and
        # recovers the inner array. Documented because callers get tags either way.
        assert extract_json_array('{"tags": ["a"]}') == ["a"]

    def test_unparseable_text_returns_none(self):
        assert extract_json_array("no array here") is None


class TestGetContentExcerpt:
    def test_strips_frontmatter(self):
        content = "---\ntitle: T\ntags: [a]\n---\n\nBody text.\n"
        assert get_content_excerpt(content) == "Body text."

    def test_truncates_with_ellipsis(self):
        assert get_content_excerpt("x" * 50, max_chars=10) == "x" * 10 + "..."
