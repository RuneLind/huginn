"""Tests for scripts/tagging/tag_documents — inject_tags emit shape + merge,
the format round-trip against the ?tags= metadata filter, and the process_files
changed-files manifest + all-failed exit signal."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tagging"))

from tag_documents import (  # noqa: E402
    build_prompt,
    call_backend,
    has_tags,
    inject_tags,
    process_files,
)

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


def _make_args(source, taxonomy, **overrides):
    """Build the argparse.Namespace process_files() expects, with sane defaults."""
    args = argparse.Namespace(
        source=str(source), taxonomy=str(taxonomy), dry_run=False, force=False,
        limit=None, pattern=None, workers=1, model="m", backend="claude-cli",
        ollama_model="qwen", timeout=60, changed_files_out=None, exclude=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def taxonomy_file(tmp_path):
    p = tmp_path / "tax.json"
    p.write_text(json.dumps({"tags": {"project": ["muninn", "tracing"]}}), encoding="utf-8")
    return p


class TestChangedFilesManifest:
    """--changed-files-out lists exactly the files written, one absolute path per line."""

    def test_lists_only_files_actually_written(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("body about muninn", encoding="utf-8")
        (src / "b.md").write_text("body with no fitting tag", encoding="utf-8")
        out = tmp_path / "changed.txt"

        # a.md → real tags (written); b.md → empty tags (model responded, nothing written).
        def fake_tag(title, breadcrumb, excerpt, taxonomy, model, timeout,
                     rel_path="", backend="claude-cli", ollama_model="qwen"):
            return ["muninn"] if rel_path == "a.md" else []

        args = _make_args(src, taxonomy_file, changed_files_out=str(out))
        with patch("tag_documents.tag_document", side_effect=fake_tag):
            result = process_files(args)

        lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln]
        assert lines == [str((src / "a.md").resolve())]
        assert result["all_failed"] is False
        # b.md must have been left untouched (still no tags frontmatter).
        assert not has_tags((src / "b.md").read_text(encoding="utf-8"))

    def test_dry_run_writes_no_manifest(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("body about muninn", encoding="utf-8")
        out = tmp_path / "changed.txt"

        args = _make_args(src, taxonomy_file, dry_run=True, changed_files_out=str(out))
        with patch("tag_documents.tag_document", return_value=["muninn"]):
            process_files(args)

        assert not out.exists()

    def test_empty_manifest_when_nothing_written(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("body", encoding="utf-8")
        out = tmp_path / "changed.txt"

        args = _make_args(src, taxonomy_file, changed_files_out=str(out))
        with patch("tag_documents.tag_document", return_value=[]):  # no tags fit
            process_files(args)

        assert out.read_text(encoding="utf-8") == ""


class TestAllFailedExit:
    """A run where every candidate file errors signals failure; partials do not."""

    def test_all_errors_signals_failure(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("body about muninn", encoding="utf-8")
        (src / "b.md").write_text("more body", encoding="utf-8")

        args = _make_args(src, taxonomy_file)
        with patch("tag_documents.tag_document", side_effect=RuntimeError("model missing")):
            result = process_files(args)

        assert result["errors"] == 2
        assert result["success"] == 0
        assert result["all_failed"] is True

    def test_partial_failure_stays_success(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("body about muninn", encoding="utf-8")
        (src / "b.md").write_text("more body", encoding="utf-8")

        def fake_tag(title, breadcrumb, excerpt, taxonomy, model, timeout,
                     rel_path="", backend="claude-cli", ollama_model="qwen"):
            if rel_path == "a.md":
                raise RuntimeError("one failed")
            return ["muninn"]

        args = _make_args(src, taxonomy_file)
        with patch("tag_documents.tag_document", side_effect=fake_tag):
            result = process_files(args)

        assert result["errors"] == 1
        assert result["success"] == 1
        assert result["all_failed"] is False

    def test_all_skipped_is_not_failure(self, tmp_path, taxonomy_file):
        # Already-tagged files are skipped, not errors — must not trip all-failed.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("---\ntags: [muninn]\n---\nbody", encoding="utf-8")

        args = _make_args(src, taxonomy_file)
        with patch("tag_documents.tag_document", return_value=["muninn"]):
            result = process_files(args)

        assert result["errors"] == 0
        assert result["all_failed"] is False


class TestFileExclusion:
    """Hidden dirs are never candidates; --exclude globs drop files by relative path."""

    def test_hidden_directories_always_excluded(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        (src / ".claude" / "skills").mkdir(parents=True)
        (src / ".claude" / "skills" / "SKILL.md").write_text("body about muninn", encoding="utf-8")
        (src / "a.md").write_text("body about muninn", encoding="utf-8")

        args = _make_args(src, taxonomy_file, dry_run=True)
        seen = []

        def fake_tag(title, breadcrumb, excerpt, taxonomy, model, timeout,
                     rel_path="", backend="claude-cli", ollama_model="qwen"):
            seen.append(rel_path)
            return ["muninn"]

        with patch("tag_documents.tag_document", side_effect=fake_tag):
            process_files(args)

        assert seen == ["a.md"]

    def test_exclude_globs_filter_relative_paths(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        (src / "plans").mkdir(parents=True)
        (src / "log.md").write_text("body about muninn", encoding="utf-8")
        (src / "index.md").write_text("body about muninn", encoding="utf-8")
        (src / "plans" / "index.md").write_text("body about muninn", encoding="utf-8")
        (src / "plans" / "real.md").write_text("body about muninn", encoding="utf-8")

        args = _make_args(src, taxonomy_file, dry_run=True,
                          exclude=["log.md", "index.md", "*/index.md"])
        seen = []

        def fake_tag(title, breadcrumb, excerpt, taxonomy, model, timeout,
                     rel_path="", backend="claude-cli", ollama_model="qwen"):
            seen.append(rel_path)
            return ["muninn"]

        with patch("tag_documents.tag_document", side_effect=fake_tag):
            process_files(args)

        assert seen == ["plans/real.md"]
