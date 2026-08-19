"""Tests for scripts/tagging/tagging_text — the markdown/JSON helpers the tagging
scripts import off sys.path (module renamed from claude_cli to stop shadowing
main.utils.claude_cli)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tagging"))

from tagging_text import (  # noqa: E402
    FRONTMATTER_RE,
    extract_json_array,
    get_content_excerpt,
)


class TestExtractJsonArray:
    def test_plain_array(self):
        assert extract_json_array('["a", "b"]') == ["a", "b"]

    def test_array_embedded_in_prose_or_fences(self):
        text = 'Here you go:\n```json\n["a", "b"]\n```\nHope that helps.'
        assert extract_json_array(text) == ["a", "b"]

    def test_object_falls_through_to_the_bracket_scan(self):
        # This is the normal ollama path, not a fallback: main/utils/ollama_cli.py
        # sends "format": "json", and Ollama's JSON mode routinely answers with an
        # object, so the dict -> bracket-scan route is what recovers the tag list.
        assert extract_json_array('{"tags": ["a"]}') == ["a"]

    def test_two_arrays_return_none(self):
        # Known limitation: find('[')/rfind(']') span BOTH arrays, so the slice is
        # '["a"] and ["b"]' — not valid JSON. tag_document treats None as "no tags".
        assert extract_json_array('["a"] and ["b"]') is None

    def test_empty_array_is_a_list_not_none(self):
        # tag_document branches on `is None`, so [] must stay a list: an empty tag
        # list is a real answer (nothing matched), not an extraction failure.
        assert extract_json_array("[]") == []

    def test_unparseable_text_returns_none(self):
        assert extract_json_array("no array here") is None


class TestFrontmatterRe:
    def test_match_ends_immediately_after_closing_marker(self):
        # inject_tags splices at match.end(); the body's leading newline must survive.
        doc = "---\na: 1\n---\nBody"
        assert doc[FRONTMATTER_RE.match(doc).end():] == "\nBody"


class TestGetContentExcerpt:
    def test_strips_frontmatter(self):
        content = "---\ntitle: T\ntags: [a]\n---\n\nBody text.\n"
        assert get_content_excerpt(content) == "Body text."

    def test_truncates_with_ellipsis(self):
        assert get_content_excerpt("x" * 50, max_chars=10) == "x" * 10 + "..."
