"""Tests for FilesDocumentReader UTF-8 reading + exclude-pattern validation (Phase 7a / H9, H10)."""

import logging

from main.sources.files.files_document_reader import FilesDocumentReader


class TestUtf8Reading:
    def test_reads_norwegian_multibyte_content(self, tmp_path):
        (tmp_path / "doc.md").write_text("hælló æøå — naïve", encoding="utf-8")
        reader = FilesDocumentReader(str(tmp_path), include_patterns=[r".*\.md"])

        docs = list(reader.read_all_documents())
        assert len(docs) == 1
        assert docs[0]["content"][0]["text"] == "hælló æøå — naïve"


class TestExcludePatternValidation:
    """The double-backslash foot-gun: a pattern that compiles but matches nothing
    must not silently include files, and should be flagged (H10)."""

    def _tree(self, tmp_path):
        (tmp_path / "keep.md").write_text("keep", encoding="utf-8")
        excluded = tmp_path / ".excluded"
        excluded.mkdir()
        (excluded / "skip.md").write_text("skip", encoding="utf-8")

    def test_correct_pattern_excludes_and_does_not_warn(self, tmp_path, caplog):
        self._tree(tmp_path)
        reader = FilesDocumentReader(
            str(tmp_path), include_patterns=[r".*\.md"], exclude_patterns=[r"^\.excluded/.*"]
        )
        with caplog.at_level(logging.WARNING):
            count = reader.get_number_of_documents()
        assert count == 1  # skip.md excluded
        assert not any("matched 0 files" in r.message for r in caplog.records)

    def test_double_backslash_pattern_excludes_nothing_and_warns(self, tmp_path, caplog):
        self._tree(tmp_path)
        # The CLAUDE.md foot-gun: '\\.excluded' compiles but never matches a real path.
        reader = FilesDocumentReader(
            str(tmp_path), include_patterns=[r".*\.md"], exclude_patterns=[r"^\\.excluded/.*"]
        )
        with caplog.at_level(logging.WARNING):
            count = reader.get_number_of_documents()
        assert count == 2  # nothing excluded — the bug being surfaced
        assert any("matched 0 files" in r.message for r in caplog.records)
