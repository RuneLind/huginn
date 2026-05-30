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


# --- Integration: each script's main() end-to-end on a tmp tree (H11) ---

import sys

import confluence_cleanup_md
import jira_cleanup_md
import notion_cleanup_md

_KEEP_BODY = "This is a genuine paragraph of real content well over the fifty character minimum length."


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
