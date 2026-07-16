"""Tests for scripts/tagging/tag_documents — inject_tags emit shape + merge,
and the format round-trip against the ?tags= metadata filter."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tagging"))

from tag_documents import build_prompt, call_backend, has_tags, inject_tags  # noqa: E402

from main.core.search_response_formatter import apply_metadata_filters  # noqa: E402


class TestInjectTags:
    def test_emits_bracketed_form_no_frontmatter(self):
        out = inject_tags("body text", ["muninn", "tracing"])
        assert out == "---\ntags: [muninn, tracing]\n---\nbody text"

    def test_emits_bracketed_form_into_existing_block(self):
        content = "---\ntitle: X\n---\nbody"
        out = inject_tags(content, ["a", "b"])
        assert "tags: [a, b]" in out

    def test_preserves_existing_frontmatter_fields(self):
        # mimir-style multi-field frontmatter (sister:, verified:) must survive.
        content = "---\ntitle: My Page\nsister: some/other.md\nverified: 2026-05-16\n---\nbody"
        out = inject_tags(content, ["muninn", "tracing"])
        assert "title: My Page" in out
        assert "sister: some/other.md" in out
        assert "verified: 2026-05-16" in out
        assert "tags: [muninn, tracing]" in out

    def test_replaces_existing_tags_line(self):
        content = "---\ntitle: X\ntags: old\n---\nbody"
        out = inject_tags(content, ["new1", "new2"])
        assert "tags: [new1, new2]" in out
        assert "old" not in out

    def test_body_horizontal_rule_not_confused_for_frontmatter_end(self):
        # Duplicate-`---` edge: a `---` horizontal rule in the body must not be
        # mistaken for the frontmatter terminator — tags land in the FM block,
        # the body rule is preserved.
        content = "---\ntitle: X\n---\nintro\n\n---\n\nmore body"
        out = inject_tags(content, ["a"])
        assert out.startswith("---\ntitle: X\ntags: [a]\n---\n")
        assert "\n---\n\nmore body" in out

    def test_has_tags_detects_existing(self):
        assert has_tags("---\ntags: [a, b]\n---\nbody") is True
        assert has_tags("---\ntitle: X\n---\nbody") is False


class TestFormatRoundTrip:
    """A page tagged by the job must be filterable on its FIRST and LAST tag."""

    def test_tagged_page_first_and_last_tag_match_filter(self):
        tagged = inject_tags("body", ["muninn", "tracing", "dashboard"])
        # Extract the value the parser would store (bracketed literal string).
        from main.utils.frontmatter import read_frontmatter
        stored_tags = read_frontmatter(tagged)["tags"]
        result = [{"metadata": {"tags": stored_tags}}]

        assert apply_metadata_filters(result, tags="muninn")   # first
        assert apply_metadata_filters(result, tags="dashboard")  # last
        assert apply_metadata_filters(result, tags="tracing")   # middle
        assert not apply_metadata_filters(result, tags="absent")


class TestBuildPromptAndBackend:
    def test_prompt_note_appended(self):
        taxonomy = {"categories": {"project": ["muninn"]}, "flat": ["muninn"],
                    "note": "First tag = owning project."}
        prompt = build_prompt("T", "b", "excerpt", taxonomy)
        assert "First tag = owning project." in prompt

    def test_call_backend_ollama_dispatch(self):
        with patch("tag_documents.call_ollama", return_value='["x"]') as mock_ollama:
            out = call_backend("p", "ollama", "claude-model", "qwen", 60)
        assert out == '["x"]'
        mock_ollama.assert_called_once()
        assert mock_ollama.call_args.kwargs["model"] == "qwen"

    def test_call_backend_claude_dispatch(self):
        with patch("tag_documents.call_claude", return_value='["y"]') as mock_claude:
            out = call_backend("p", "claude-cli", "claude-model", "qwen", 60)
        assert out == '["y"]'
        mock_claude.assert_called_once()
        assert mock_claude.call_args.kwargs["model"] == "claude-model"
