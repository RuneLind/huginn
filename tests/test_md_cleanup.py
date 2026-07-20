"""Tests for the shared cleanup filesystem mechanics (H11)."""

import json
import os

from main.sources.cleanup import md_cleanup


def _touch(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class TestIterMarkdownFiles:
    def test_yields_md_files_with_rel_paths(self, tmp_path):
        _touch(str(tmp_path / "a.md"))
        _touch(str(tmp_path / "sub" / "b.md"))
        _touch(str(tmp_path / "c.txt"))
        found = dict(
            (rel, fp) for fp, rel in md_cleanup.iter_markdown_files(str(tmp_path))
        )
        assert set(found) == {"a.md", os.path.join("sub", "b.md")}

    def test_skips_excluded_dir(self, tmp_path):
        _touch(str(tmp_path / "keep.md"))
        _touch(str(tmp_path / md_cleanup.EXCLUDED_DIR / "gone.md"))
        rels = {rel for _, rel in md_cleanup.iter_markdown_files(str(tmp_path))}
        assert rels == {"keep.md"}


class TestMoveToExcluded:
    def test_moves_preserving_relative_path(self, tmp_path):
        src = str(tmp_path / "sub" / "doc.md")
        _touch(src, "body")
        excluded = str(tmp_path / md_cleanup.EXCLUDED_DIR)
        dest = md_cleanup.move_to_excluded(src, os.path.join("sub", "doc.md"), excluded)
        assert not os.path.exists(src)
        assert os.path.isfile(dest)
        assert dest == os.path.join(excluded, "sub", "doc.md")
        assert open(dest, encoding="utf-8").read() == "body"


class TestWriteExcludedManifest:
    def test_writes_entries(self, tmp_path):
        excluded = str(tmp_path / md_cleanup.EXCLUDED_DIR)
        entries = [{"page_id": "1", "reason": "stub"}]
        path = md_cleanup.write_excluded_manifest(excluded, entries, "page_id")
        assert json.loads(open(path, encoding="utf-8").read()) == entries

    def test_no_entries_returns_none_writes_nothing(self, tmp_path):
        excluded = str(tmp_path / md_cleanup.EXCLUDED_DIR)
        assert md_cleanup.write_excluded_manifest(excluded, [], "page_id") is None
        assert not os.path.exists(excluded)

    def test_merges_with_existing_deduped(self, tmp_path):
        excluded = str(tmp_path / md_cleanup.EXCLUDED_DIR)
        md_cleanup.write_excluded_manifest(excluded, [{"page_id": "1"}], "page_id")
        md_cleanup.write_excluded_manifest(
            excluded, [{"page_id": "1"}, {"page_id": "2"}], "page_id"
        )
        path = os.path.join(excluded, md_cleanup.MANIFEST_NAME)
        ids = [e["page_id"] for e in json.loads(open(path, encoding="utf-8").read())]
        assert ids == ["1", "2"]


class TestRemoveEmptyDirs:
    def test_removes_empty_keeps_nonempty_and_excluded(self, tmp_path):
        os.makedirs(str(tmp_path / "empty"))
        _touch(str(tmp_path / "full" / "a.md"))
        _touch(str(tmp_path / md_cleanup.EXCLUDED_DIR / "x.md"))
        md_cleanup.remove_empty_dirs(str(tmp_path))
        assert not os.path.exists(str(tmp_path / "empty"))
        assert os.path.isdir(str(tmp_path / "full"))
        assert os.path.isdir(str(tmp_path / md_cleanup.EXCLUDED_DIR))


# --- Shared classify_body skeleton (PR F) ---

import confluence_cleanup_md
import jira_cleanup_md
import notion_cleanup_md

_KEEP_BODY = "This is a genuine paragraph of real content well over the fifty character minimum length."


class TestClassifyBodySkeleton:
    """The shared strip/filter/threshold machinery in md_cleanup.classify_body.

    Policy (line_filters and the two *_reason knobs) is injected by callers;
    these tests exercise the machinery directly."""

    def test_empty_returns_empty_reason(self):
        assert md_cleanup.classify_body("", 50) == "empty_stub"

    def test_whitespace_only_returns_empty_reason(self):
        assert md_cleanup.classify_body("   \n\t\n ", 50) == "empty_stub"

    def test_empty_reason_is_parameterizable(self):
        assert md_cleanup.classify_body("", 50, empty_reason="blank") == "blank"

    def test_all_lines_filtered_returns_filtered_empty_reason_default(self):
        filters = [lambda line: line.startswith("SKIP")]
        assert md_cleanup.classify_body("SKIP a\nSKIP b", 50, filters) == "reference_only"

    def test_filtered_empty_reason_is_parameterizable(self):
        # This is exactly the jira divergence: all-filtered -> empty_stub.
        filters = [lambda line: line.startswith("#")]
        assert (
            md_cleanup.classify_body(
                "# heading", 50, filters, filtered_empty_reason="empty_stub"
            )
            == "empty_stub"
        )

    def test_minimal_content_below_length(self):
        assert md_cleanup.classify_body("short", 50) == "minimal_content"

    def test_kept_when_over_length(self):
        assert md_cleanup.classify_body(_KEEP_BODY, 50) is None

    def test_low_word_count_when_enabled(self):
        assert (
            md_cleanup.classify_body("one two three four five", 10, min_word_count=30)
            == "low_word_count"
        )

    def test_word_count_disabled_by_default(self):
        assert md_cleanup.classify_body("one two three four five", 10) is None

    def test_content_length_checked_before_word_count(self):
        assert (
            md_cleanup.classify_body("hi", 50, min_word_count=30) == "minimal_content"
        )

    def test_filters_drop_matching_lines_only(self):
        filters = [lambda line: line == "DROP"]
        # Only "DROP" removed; remaining real text kept.
        assert md_cleanup.classify_body("DROP\n" + _KEEP_BODY, 50, filters) is None


class TestPerSourcePredicateWiring:
    """Each script's thin classify_body wires its own policy into the skeleton."""

    def test_confluence_boilerplate_headings_only_is_reference_only(self):
        body = "## Aktivitet\n### Bidragsytere\n## Spaceeier"
        assert confluence_cleanup_md.classify_body(body, 50) == "reference_only"

    def test_confluence_links_only_is_reference_only(self):
        body = "https://example.com/x\n[Link](https://example.com/y)"
        assert confluence_cleanup_md.classify_body(body, 50) == "reference_only"

    def test_jira_headings_only_is_empty_stub_divergence(self):
        # KEPT divergence: jira reports empty_stub where confluence/notion say
        # reference_only for an all-filtered body.
        body = "# ABC-123: Title\n**Epic:** ABC-1\n## Description"
        assert jira_cleanup_md.classify_body(body, 50) == "empty_stub"

    def test_notion_reference_markers_only_is_reference_only(self):
        body = "[Child page: Foo]\n[Child database: Bar]\n[Unsupported: table]"
        assert notion_cleanup_md.classify_body(body, 50) == "reference_only"


class TestNotionMinWordCountRestored:
    """PR F restores min_word_count to notion (was silently absent).

    Default (0) keeps behavior byte-stable; passing it activates low_word_count."""

    def test_low_word_count_now_excluded(self):
        body = "just a handful of words here"
        assert (
            notion_cleanup_md.classify_body(body, min_content_length=10, min_word_count=30)
            == "low_word_count"
        )

    def test_disabled_by_default_keeps_thin_page(self):
        body = "just a handful of words here"
        assert notion_cleanup_md.classify_body(body, min_content_length=10) is None


# --- Integration: each script's main() end-to-end on a tmp tree (H11) ---

import sys


def _md(path, fm, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["---"] + [f"{k}: {v}" for k, v in fm.items()] + ["---", body]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _excluded(tmp_path, rel):
    return os.path.join(str(tmp_path), md_cleanup.EXCLUDED_DIR, rel)


def _run(main_fn, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    main_fn()


def _empty_noise_cfg(tmp_path):
    cfg = str(tmp_path / "noise.json")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write("[]")
    return cfg


class TestNotionMainIntegration:
    def test_stub_moved_real_kept_manifest_written(self, tmp_path, monkeypatch):
        _md(str(tmp_path / "stub.md"), {"title": "Stub", "notion_id": "n1"}, "")
        _md(str(tmp_path / "real.md"), {"title": "Real", "notion_id": "n2"}, _KEEP_BODY)
        _run(notion_cleanup_md.main, ["prog", "--saveMd", str(tmp_path)], monkeypatch)

        assert not os.path.exists(str(tmp_path / "stub.md"))
        assert os.path.isfile(_excluded(tmp_path, "stub.md"))
        assert os.path.isfile(str(tmp_path / "real.md"))
        manifest = json.loads(open(_excluded(tmp_path, md_cleanup.MANIFEST_NAME), encoding="utf-8").read())
        assert [e["notion_id"] for e in manifest] == ["n1"]

    def test_min_word_count_flag_excludes_thin_page(self, tmp_path, monkeypatch):
        # PR F: notion now honors --minWordCount (restored). A short-but-over-
        # minContentLength page is excluded as low_word_count only when the flag
        # is passed; the default run above keeps such pages.
        thin = "one two three four five"  # 5 words, >10 chars
        _md(str(tmp_path / "thin.md"), {"title": "Thin", "notion_id": "n1"}, thin)
        _md(str(tmp_path / "real.md"), {"title": "Real", "notion_id": "n2"}, _KEEP_BODY)
        _run(
            notion_cleanup_md.main,
            ["prog", "--saveMd", str(tmp_path), "--minContentLength", "10",
             "--minWordCount", "10"],
            monkeypatch,
        )

        assert not os.path.exists(str(tmp_path / "thin.md"))
        assert os.path.isfile(_excluded(tmp_path, "thin.md"))
        assert os.path.isfile(str(tmp_path / "real.md"))
        manifest = json.loads(open(_excluded(tmp_path, md_cleanup.MANIFEST_NAME), encoding="utf-8").read())
        assert [e["reason"] for e in manifest] == ["low_word_count"]


class TestConfluenceMainIntegration:
    def test_stub_moved_real_kept_manifest_written(self, tmp_path, monkeypatch):
        _md(str(tmp_path / "stub.md"), {"title": "Stub", "page_id": "p1"}, "")
        _md(str(tmp_path / "real.md"), {"title": "Real", "page_id": "p2"}, _KEEP_BODY)
        cfg = _empty_noise_cfg(tmp_path)
        _run(confluence_cleanup_md.main,
             ["prog", "--saveMd", str(tmp_path), "--noiseConfig", cfg], monkeypatch)

        assert not os.path.exists(str(tmp_path / "stub.md"))
        assert os.path.isfile(_excluded(tmp_path, "stub.md"))
        assert os.path.isfile(str(tmp_path / "real.md"))
        manifest = json.loads(open(_excluded(tmp_path, md_cleanup.MANIFEST_NAME), encoding="utf-8").read())
        assert [e["page_id"] for e in manifest] == ["p1"]


class TestJiraMainIntegration:
    def test_stub_moved_real_kept_manifest_written(self, tmp_path, monkeypatch):
        _md(str(tmp_path / "stub.md"), {"title": "Stub", "issue_key": "J-1"}, "")
        _md(str(tmp_path / "real.md"), {"title": "Real", "issue_key": "J-2"}, _KEEP_BODY)
        cfg = _empty_noise_cfg(tmp_path)
        _run(jira_cleanup_md.main,
             ["prog", "--saveMd", str(tmp_path), "--noiseConfig", cfg], monkeypatch)

        assert not os.path.exists(str(tmp_path / "stub.md"))
        assert os.path.isfile(_excluded(tmp_path, "stub.md"))
        assert os.path.isfile(str(tmp_path / "real.md"))
        manifest = json.loads(open(_excluded(tmp_path, md_cleanup.MANIFEST_NAME), encoding="utf-8").read())
        assert [e["issue_key"] for e in manifest] == ["J-1"]
