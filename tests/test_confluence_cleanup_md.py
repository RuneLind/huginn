import os
import pytest

from confluence_cleanup_md import (
    sanitize_content,
    detect_wip,
    match_title_noise_pattern,
    match_noise_pattern,
    classify_body,
)


def _write_md(tmp_path, filename, content):
    """Helper to write a markdown file and return its path."""
    filepath = tmp_path / filename
    filepath.write_text(content, encoding="utf-8")
    return filepath


class TestSanitizeContent:
    def test_removes_at_at_markers(self):
        text = "Some text\n@@mention to follow up\nMore text\n"
        cleaned, removed = sanitize_content(text)
        assert removed == 1
        assert "@@" not in cleaned
        assert "Some text\nMore text\n" == cleaned

    def test_removes_personal_action_items(self):
        text = "Discussion notes\nOla følger opp med team\nConclusion here\n"
        cleaned, removed = sanitize_content(text)
        assert removed == 1
        assert "følger opp" not in cleaned
        assert "Discussion notes\nConclusion here\n" == cleaned

    def test_removes_various_action_patterns(self):
        lines = [
            "Kari tar det videre med UX\n",
            "Per sjekker status\n",
            "Lisa undersøker alternativet\n",
        ]
        text = "Intro\n" + "".join(lines) + "End\n"
        cleaned, removed = sanitize_content(text)
        assert removed == 3
        assert cleaned == "Intro\nEnd\n"

    def test_no_changes_returns_zero(self):
        text = "Clean content\nNo markers here\n"
        cleaned, removed = sanitize_content(text)
        assert removed == 0
        assert cleaned == text

    def test_empty_input(self):
        cleaned, removed = sanitize_content("")
        assert removed == 0
        assert cleaned == ""

    def test_preserves_line_endings(self):
        text = "Line one\nLine two\n"
        cleaned, _ = sanitize_content(text)
        assert cleaned == text


class TestDetectWip:
    def test_title_under_arbeid(self):
        assert detect_wip("CDM 4.4 analyse - Under arbeid", "") is True

    def test_title_utkast(self):
        assert detect_wip("Utkast til ny arkitektur", "") is True

    def test_title_wip(self):
        assert detect_wip("WIP: new feature", "") is True

    def test_body_wip_heading(self):
        assert detect_wip("Normal title", "## UNDER ARBEID\nSome content") is True

    def test_body_standalone_wip_line(self):
        assert detect_wip("Normal title", "Intro\nUnder arbeid\nMore content") is True

    def test_no_wip_markers(self):
        assert detect_wip("Regular title", "Regular body content") is False

    def test_empty_title_and_body(self):
        assert detect_wip("", "") is False

    def test_title_with_dash_prefix(self):
        assert detect_wip("Analyse - under arbeid", "") is True

    def test_case_insensitive_title(self):
        assert detect_wip("UNDER ARBEID: draft", "") is True


class TestMatchTitleNoisePattern:
    @pytest.fixture
    def patterns(self):
        return [
            {"title_pattern": "møtereferat", "reason": "meeting minutes"},
            {"title_pattern": "statusrapport", "reason": "status report"},
            {"pattern": "*/meetings/*", "reason": "path-based (should be ignored)"},
        ]

    def test_matches_title(self, patterns):
        result = match_title_noise_pattern("Sprint 42 møtereferat", patterns)
        assert result == "meeting minutes"

    def test_case_insensitive(self, patterns):
        result = match_title_noise_pattern("Statusrapport uke 5", patterns)
        assert result == "status report"

    def test_no_match(self, patterns):
        result = match_title_noise_pattern("Architecture overview", patterns)
        assert result is None

    def test_empty_title(self, patterns):
        assert match_title_noise_pattern("", patterns) is None

    def test_none_title(self, patterns):
        assert match_title_noise_pattern(None, patterns) is None

    def test_ignores_path_patterns(self, patterns):
        result = match_title_noise_pattern("*/meetings/*", patterns)
        assert result is None


class TestMatchNoisePattern:
    @pytest.fixture
    def patterns(self):
        return [
            {"pattern": "*/Tekniske møter/*", "reason": "technical meetings"},
            {"title_pattern": "møtereferat", "reason": "should be ignored"},
        ]

    def test_matches_path(self, patterns):
        result = match_noise_pattern("MYSPACE/Tekniske møter/2025-01.md", patterns)
        assert result == "technical meetings"

    def test_ignores_title_patterns(self, patterns):
        result = match_noise_pattern("møtereferat.md", patterns)
        assert result is None


class TestClassifyBodyMinWordCount:
    def test_below_word_count_excluded(self):
        body = "Short text only five words"
        result = classify_body(body, min_content_length=10, min_word_count=30)
        assert result == "low_word_count"

    def test_above_word_count_passes(self):
        body = " ".join(["word"] * 40)
        result = classify_body(body, min_content_length=10, min_word_count=30)
        assert result is None

    def test_word_count_disabled_by_default(self):
        body = "Short text"
        result = classify_body(body, min_content_length=5)
        assert result is None

    def test_zero_word_count_disabled(self):
        body = "Short text"
        result = classify_body(body, min_content_length=5, min_word_count=0)
        assert result is None

    def test_content_length_checked_before_word_count(self):
        body = "Hi"
        result = classify_body(body, min_content_length=50, min_word_count=30)
        assert result == "minimal_content"


class TestSanitizeFrontmatterHandling:
    """Test the frontmatter boundary detection used in the sanitization pass."""

    def _run_sanitize_on_file(self, tmp_path, content):
        """Simulate the sanitization file-write logic from main()."""
        from confluence_cleanup_md import sanitize_content as _sanitize, detect_wip as _detect_wip
        from confluence_cleanup_md import parse_frontmatter_and_body

        filepath = _write_md(tmp_path, "test.md", content)
        metadata, body = parse_frontmatter_and_body(filepath)
        cleaned_body, removed = _sanitize(body)
        is_wip = _detect_wip(metadata.get("title", ""), body)
        needs_wip_flag = is_wip and metadata.get("wip") != "true"

        if removed > 0 or needs_wip_flag:
            raw = filepath.read_text(encoding="utf-8")
            fm_end = 0
            if raw.startswith("---"):
                close_idx = raw.find("\n---\n", 3)
                if close_idx == -1:
                    close_idx = raw.find("\n---", 3)
                if close_idx < 0:
                    # Malformed frontmatter — skip (mirrors main() continue)
                    return filepath.read_text(encoding="utf-8")
                second_sep = close_idx + 1
                if needs_wip_flag:
                    raw = raw[:second_sep] + "wip: true\n" + raw[second_sep:]
                    second_sep += len("wip: true\n")
                fm_end = second_sep + 3
                if fm_end < len(raw) and raw[fm_end] == "\n":
                    fm_end += 1
            filepath.write_text(raw[:fm_end] + cleaned_body, encoding="utf-8")

        return filepath.read_text(encoding="utf-8")

    def test_wip_injection_normal(self, tmp_path):
        content = "---\ntitle: WIP document\nurl: https://x.com\n---\nBody content.\n"
        result = self._run_sanitize_on_file(tmp_path, content)
        assert "wip: true" in result
        assert result.startswith("---\n")
        assert "Body content." in result

    def test_dashes_in_frontmatter_value_not_confused(self, tmp_path):
        content = "---\ntitle: 2025---analysis - Under arbeid\nurl: https://x.com\n---\nBody.\n"
        result = self._run_sanitize_on_file(tmp_path, content)
        assert "wip: true" in result
        # The title should remain intact
        assert "2025---analysis" in result
        # wip: true should be before the closing ---
        lines = result.split("\n")
        closing_idx = None
        for i, line in enumerate(lines):
            if i > 0 and line.strip() == "---":
                closing_idx = i
                break
        assert closing_idx is not None
        # wip: true should be right before the closing ---
        assert lines[closing_idx - 1] == "wip: true"

    def test_no_closing_frontmatter_preserves_file(self, tmp_path):
        content = "---\ntitle: WIP broken\nBody without closing.\n"
        result = self._run_sanitize_on_file(tmp_path, content)
        # Malformed frontmatter — file should be left untouched
        assert result == content

    def test_sanitize_removes_lines_with_frontmatter(self, tmp_path):
        content = "---\ntitle: Normal doc\n---\nGood content.\n@@mention to follow\nMore good.\n"
        result = self._run_sanitize_on_file(tmp_path, content)
        assert "@@" not in result
        assert "Good content." in result
        assert "More good." in result
