"""Tests for scripts/tagging/tag_documents — inject_tags emit shape + merge,
the format round-trip against the ?tags= metadata filter, and the process_files
changed-files manifest + all-failed exit signal."""
import argparse
import builtins
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tagging"))

from tag_documents import (  # noqa: E402
    build_prompt,
    external_backend_refusal,
    call_backend,
    get_html_excerpt,
    has_html_tags,
    has_tags,
    html_title,
    inject_html_tags,
    inject_tags,
    process_files,
)
from tag_documents import main as tag_documents_main  # noqa: E402

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
        include_html=False,
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


class TestInjectHtmlTags:
    """HTML mode: keywords-meta injection, anchor fallback chain, byte-for-byte."""

    META = '<meta name="keywords" content="muninn, tracing">'

    def test_after_viewport(self):
        html = (
            '<!DOCTYPE html>\n<html>\n<head>\n'
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '    <title>X</title>\n</head>\n<body>hi</body>\n</html>\n'
        )
        out = inject_html_tags(html, ["muninn", "tracing"])
        assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n' \
               f'    {self.META}\n    <title>X</title>' in out
        # Everything except the inserted line is byte-identical.
        assert out.replace(f"    {self.META}\n", "", 1) == html

    def test_after_charset_when_no_viewport(self):
        html = ('<!DOCTYPE html>\n<html>\n<head>\n'
                '    <meta charset="UTF-8">\n    <title>X</title>\n</head>\n')
        out = inject_html_tags(html, ["muninn", "tracing"])
        assert f'    <meta charset="UTF-8">\n    {self.META}\n    <title>X</title>' in out
        assert out.replace(f"    {self.META}\n", "", 1) == html

    def test_after_head_when_no_charset(self):
        html = '<html>\n<head>\n    <title>X</title>\n</head>\n<body>hi</body>\n'
        out = inject_html_tags(html, ["muninn", "tracing"])
        assert f'<head>\n{self.META}\n    <title>X</title>' in out
        assert out.replace(f"{self.META}\n", "", 1) == html

    def test_prepend_for_headless_fragment(self):
        # No <head>, no charset — starts directly at <title>. Meta prepended.
        html = '<title>X</title>\n<p>body</p>\n'
        out = inject_html_tags(html, ["muninn", "tracing"])
        assert out == f"{self.META}\n{html}"

    def test_replace_existing_keywords_in_place(self):
        html = ('<head>\n    <meta name="viewport" content="w">\n'
                '    <meta name="keywords" content="old, stale">\n    <title>X</title>\n</head>\n')
        out = inject_html_tags(html, ["new1", "new2"])
        assert '<meta name="keywords" content="new1, new2">' in out
        assert "old, stale" not in out
        # Position unchanged (replaced in place, not re-anchored).
        assert out == html.replace('content="old, stale"', 'content="new1, new2"')

    def test_has_html_tags_detection(self):
        assert has_html_tags('<meta name="keywords" content="a, b">') is True
        assert has_html_tags('<meta name="keywords" content="">') is False
        assert has_html_tags('<meta name="viewport" content="w">') is False

    def test_injection_idempotent(self):
        html = '<head>\n    <meta charset="UTF-8">\n</head>\n'
        once = inject_html_tags(html, ["muninn"])
        assert has_html_tags(once) is True
        # Second inject replaces in place; the shape is stable.
        twice = inject_html_tags(once, ["muninn"])
        assert twice == once

    def test_html_title_and_excerpt(self):
        html = ('<head><title>My &amp; Page</title><style>.a{color:red}</style></head>'
                '<body><script>var x=1</script><p>Hello world</p></body>')
        assert html_title(html, "fallback") == "My & Page"
        assert html_title("<p>no title</p>", "fallback") == "fallback"
        excerpt = get_html_excerpt(html)
        assert "Hello world" in excerpt
        assert "color:red" not in excerpt
        assert "var x=1" not in excerpt


class TestHtmlMode:
    """process_files with/without --include-html: discovery, skip, manifest, exclude."""

    def _fake_tag(self, *a, **k):
        return ["muninn"]

    def test_md_only_default_leaves_html_untouched(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("body about muninn", encoding="utf-8")
        html = '<head>\n    <meta charset="UTF-8">\n</head>\n<body>muninn tracing</body>\n'
        (src / "page.html").write_text(html, encoding="utf-8")

        args = _make_args(src, taxonomy_file)  # include_html defaults False
        with patch("tag_documents.tag_document", side_effect=self._fake_tag):
            process_files(args)

        # html file untouched (no keywords meta), md file tagged.
        assert (src / "page.html").read_text(encoding="utf-8") == html
        assert has_tags((src / "a.md").read_text(encoding="utf-8"))

    def test_dry_run_include_html_touches_zero_html(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        html = '<head>\n    <meta charset="UTF-8">\n</head>\n<body>muninn</body>\n'
        (src / "page.html").write_text(html, encoding="utf-8")
        seen = []

        def fake(title, breadcrumb, excerpt, taxonomy, model, timeout,
                 rel_path="", backend="claude-cli", ollama_model="qwen"):
            seen.append(rel_path)
            return ["muninn"]

        args = _make_args(src, taxonomy_file, include_html=True, dry_run=True)
        with patch("tag_documents.tag_document", side_effect=fake):
            process_files(args)

        assert seen == ["page.html"]  # discovered
        assert (src / "page.html").read_text(encoding="utf-8") == html  # but not written

    def test_include_html_writes_and_manifest_lists_html(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        html = '<head>\n    <meta name="viewport" content="w">\n</head>\n<body>muninn</body>\n'
        (src / "page.html").write_text(html, encoding="utf-8")
        out = tmp_path / "changed.txt"

        args = _make_args(src, taxonomy_file, include_html=True, changed_files_out=str(out))
        with patch("tag_documents.tag_document", side_effect=self._fake_tag):
            process_files(args)

        written = (src / "page.html").read_text(encoding="utf-8")
        assert '<meta name="keywords" content="muninn">' in written
        # Byte-for-byte outside the inserted line.
        assert written.replace('    <meta name="keywords" content="muninn">\n', "", 1) == html
        # Second run skips (idempotent).
        with patch("tag_documents.tag_document", side_effect=self._fake_tag):
            result2 = process_files(_make_args(src, taxonomy_file, include_html=True))
        assert result2["tagged"] == 0

        lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln]
        assert lines == [str((src / "page.html").resolve())]

    def test_exclude_globs_apply_to_html(self, tmp_path, taxonomy_file):
        src = tmp_path / "src"
        src.mkdir()
        (src / "keep.html").write_text('<head></head><body>muninn</body>', encoding="utf-8")
        (src / "drop.html").write_text('<head></head><body>muninn</body>', encoding="utf-8")
        seen = []

        def fake(title, breadcrumb, excerpt, taxonomy, model, timeout,
                 rel_path="", backend="claude-cli", ollama_model="qwen"):
            seen.append(rel_path)
            return ["muninn"]

        args = _make_args(src, taxonomy_file, include_html=True, dry_run=True,
                          exclude=["drop.html"])
        with patch("tag_documents.tag_document", side_effect=fake):
            process_files(args)

        assert seen == ["keep.html"]


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


class TestExternalBackendGuard:
    """A privacy-scoped source tree may only be tagged by a local backend.

    This script reads RAW source markdown — the pre-alias text — and its default
    backend ships an excerpt of every file to a hosted model. The guard has to
    fire before a single file is opened; a refusal after the first excerpt has
    left the machine is not a refusal.
    """

    @pytest.fixture
    def scoped(self, tmp_path, monkeypatch):
        """A tmp repo root: one tree named by basePaths, one named only through
        an in-scope collection's built manifest, and one out of scope."""
        from main.privacy import alias_registry
        tree = tmp_path / "sources" / "in-scope"
        (tree / "sub").mkdir(parents=True)
        by_manifest = tmp_path / "sources" / "by-manifest"
        by_manifest.mkdir(parents=True)
        (tmp_path / "outside").mkdir()
        privacy = tmp_path / "huginn-x" / "privacy"
        privacy.mkdir(parents=True)
        (privacy / "scope.json").write_text(
            json.dumps({"collections": ["scoped-collection"],
                        "basePaths": ["./sources/in-scope"]}),
            encoding="utf-8")
        collection = tmp_path / "data" / "collections" / "scoped-collection"
        collection.mkdir(parents=True)
        (collection / "manifest.json").write_text(
            json.dumps({"collectionName": "scoped-collection",
                        "reader": {"type": "localFiles",
                                   "basePath": "./sources/by-manifest"}}),
            encoding="utf-8")
        monkeypatch.setattr(alias_registry, "REPO_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        return tmp_path, tree

    def test_an_in_scope_tree_refuses_an_external_backend(self, scoped):
        _, tree = scoped
        refusal = external_backend_refusal(str(tree), "claude-cli")
        assert refusal is not None
        assert "ollama" in refusal and "Nothing was read" in refusal

    def test_a_subdirectory_of_an_in_scope_tree_refuses_too(self, scoped):
        _, tree = scoped
        assert external_backend_refusal(str(tree / "sub"), "claude-cli") is not None

    def test_an_in_scope_tree_may_be_tagged_locally(self, scoped):
        _, tree = scoped
        assert external_backend_refusal(str(tree), "ollama") is None

    def test_an_out_of_scope_tree_is_unaffected(self, scoped):
        root, _ = scoped
        assert external_backend_refusal(str(root / "outside"), "claude-cli") is None

    def test_main_exits_before_reading_anything(self, scoped, monkeypatch, capsys):
        """End to end through main(): no file is opened, not even the taxonomy."""
        _, tree = scoped
        (tree / "a.md").write_text("body", encoding="utf-8")
        # `builtins.open`, not `Path.open`: `load_taxonomy` calls the builtin,
        # and so does every other read on this path, so spying on the Path
        # method watched a door nobody uses and the assertion was vacuous.
        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy)
        monkeypatch.setattr(sys, "argv", [
            "tag_documents.py", "--source", str(tree),
            "--taxonomy", "/nonexistent/taxonomy.json", "--dry-run",
        ])
        with pytest.raises(SystemExit) as excinfo:
            tag_documents_main()
        assert excinfo.value.code == 2
        assert "sends document text off this machine" in capsys.readouterr().err
        # The guard reads its own scope files; nothing from the CORPUS, and not
        # even the taxonomy, which `main()` loads on the very next line.
        assert [path for path in opened
                if str(tree) in path or "taxonomy" in path] == []


class TestGuardIsCwdIndependent:
    """The guard's refusal must not depend on where the operator is standing.

    `load_scope` resolved its relative basePaths against the process CWD, so from
    any directory other than the repo root the whole scope resolved to paths that
    do not exist, `path_in_scope` returned False for everything, and the guard
    silently permitted the hosted backend on the exact trees it exists to
    protect. A finder reproduced it from /tmp and the run reached the backend.
    """

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        from main.privacy import alias_registry
        tree = tmp_path / "data" / "sources" / "scoped-tree"
        (tree / "nested").mkdir(parents=True)
        privacy = tmp_path / "huginn-x" / "privacy"
        privacy.mkdir(parents=True)
        (privacy / "scope.json").write_text(
            json.dumps({"collections": [], "basePaths": ["./data/sources/scoped-tree"]}),
            encoding="utf-8")
        monkeypatch.setattr(alias_registry, "REPO_ROOT", str(tmp_path))
        return tmp_path, tree

    def test_the_guard_refuses_from_an_unrelated_working_directory(
            self, repo, tmp_path_factory, monkeypatch):
        root, tree = repo
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        monkeypatch.chdir(elsewhere)
        assert os.getcwd() != str(root)
        assert external_backend_refusal(str(tree), "claude-cli") is not None
        assert external_backend_refusal(str(tree / "nested"), "claude-cli") is not None

    def test_main_refuses_from_an_unrelated_working_directory(
            self, repo, tmp_path_factory, monkeypatch, capsys):
        root, tree = repo
        (tree / "a.md").write_text("body", encoding="utf-8")
        monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere2"))
        monkeypatch.setattr(sys, "argv", [
            "tag_documents.py", "--source", str(tree),
            "--taxonomy", "/nonexistent/taxonomy.json", "--dry-run",
            "--backend", "claude-cli",
        ])
        with pytest.raises(SystemExit) as excinfo:
            tag_documents_main()
        assert excinfo.value.code == 2
        assert "sends document text off this machine" in capsys.readouterr().err


def test_a_tree_named_only_by_an_in_scope_collections_manifest_is_in_scope(tmp_path,
                                                                          monkeypatch):
    """Scope is declared twice — by collection name and by basePath — and the two
    drift. The built manifest is the record of what the collection actually
    reads, so it is consulted rather than trusted to be mirrored by hand."""
    from main.privacy import alias_registry
    tree = tmp_path / "sources" / "by-manifest"
    (tree / "sub").mkdir(parents=True)
    privacy = tmp_path / "huginn-x" / "privacy"
    privacy.mkdir(parents=True)
    (privacy / "scope.json").write_text(
        json.dumps({"collections": ["scoped-collection"], "basePaths": []}),
        encoding="utf-8")
    collection = tmp_path / "data" / "collections" / "scoped-collection"
    collection.mkdir(parents=True)
    (collection / "manifest.json").write_text(
        json.dumps({"reader": {"type": "localFiles", "basePath": "./sources/by-manifest"}}),
        encoding="utf-8")
    monkeypatch.setattr(alias_registry, "REPO_ROOT", str(tmp_path))
    assert external_backend_refusal(str(tree), "claude-cli") is not None
    assert external_backend_refusal(str(tree / "sub"), "claude-cli") is not None
    assert external_backend_refusal(str(tree), "ollama") is None
    assert external_backend_refusal(str(tmp_path / "sources"), "claude-cli") is None
