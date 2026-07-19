"""Tests for the shared category-organized markdown writer's overwrite/fork logic.

The write path is keyed on the sanitized title and decides overwrite-vs-fork by
comparing the incoming URL against the existing file's frontmatter URL. The
article vertical introduced URL-less pastes, which must never clobber each other.
"""
import os

from main.ingest._markdown_writer import write_categorized_markdown
from main.ingest._summary_ingest import write_summary


def _frontmatter(url: str) -> str:
    return f'---\nurl: "{url}"\n---\n\n'


class TestWriteCategorizedMarkdown:
    def test_same_url_overwrites(self, tmp_path):
        root = str(tmp_path)
        p1 = write_categorized_markdown(
            root=root, category="ai/general", title="My Title",
            url="https://x.com/a", content=_frontmatter("https://x.com/a") + "first",
        )
        p2 = write_categorized_markdown(
            root=root, category="ai/general", title="My Title",
            url="https://x.com/a", content=_frontmatter("https://x.com/a") + "second",
        )
        assert p1 == p2
        with open(os.path.join(root, p2)) as f:
            assert "second" in f.read()

    def test_same_title_different_url_forks(self, tmp_path):
        root = str(tmp_path)
        p1 = write_categorized_markdown(
            root=root, category="ai/general", title="My Title",
            url="https://x.com/a", content=_frontmatter("https://x.com/a") + "a",
        )
        p2 = write_categorized_markdown(
            root=root, category="ai/general", title="My Title",
            url="https://x.com/b", content=_frontmatter("https://x.com/b") + "b",
        )
        assert p1 != p2
        assert "(2)" in p2

    def test_two_urlless_pastes_same_title_fork(self, tmp_path):
        """Absent URL must never match an existing doc — the core clobber guard."""
        root = str(tmp_path)
        p1 = write_categorized_markdown(
            root=root, category="ai/general", title="Untitled Paste",
            url=None, content='---\nurl: ""\n---\n\nfirst paste',
        )
        p2 = write_categorized_markdown(
            root=root, category="ai/general", title="Untitled Paste",
            url=None, content='---\nurl: ""\n---\n\nsecond paste',
        )
        assert p1 != p2
        assert "(2)" in p2
        # Both files exist and retain their distinct bodies.
        with open(os.path.join(root, p1)) as f:
            assert "first paste" in f.read()
        with open(os.path.join(root, p2)) as f:
            assert "second paste" in f.read()

    def test_empty_string_url_also_forks(self, tmp_path):
        """`not url` covers the empty-string case too, not just None."""
        root = str(tmp_path)
        p1 = write_categorized_markdown(
            root=root, category="ai/general", title="Same",
            url="", content='---\nurl: ""\n---\n\none',
        )
        p2 = write_categorized_markdown(
            root=root, category="ai/general", title="Same",
            url="", content='---\nurl: ""\n---\n\ntwo',
        )
        assert p1 != p2


class TestWriteSummaryUrlless:
    def test_two_urlless_summaries_same_title_fork(self, tmp_path):
        """End-to-end through write_summary (the article vertical's entry point)."""
        root = str(tmp_path)
        r1 = write_summary(
            root=root, title="Newsletter Digest", url=None,
            summary="body one", category="ai/general",
        )
        r2 = write_summary(
            root=root, title="Newsletter Digest", url=None,
            summary="body two", category="ai/general",
        )
        assert r1["file_path"] != r2["file_path"]
        assert os.path.exists(os.path.join(root, r1["file_path"]))
        assert os.path.exists(os.path.join(root, r2["file_path"]))
