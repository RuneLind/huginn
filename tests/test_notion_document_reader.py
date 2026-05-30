"""Tests for NotionDocumentReader failure handling in read_all_documents (M11)."""

import pytest

from main.sources.notion.notion_document_reader import NotionDocumentReader


def _make_reader(pages, failing_ids=None, max_failures_in_row=5):
    """Build a reader whose page traversal and block fetch are stubbed.

    pages: list of {"id": ...} dicts to iterate.
    failing_ids: ids for which _fetch_all_blocks raises.
    """
    failing_ids = failing_ids or set()
    reader = NotionDocumentReader(token="x", max_failures_in_row=max_failures_in_row)

    reader._iterate_pages = lambda: iter(pages)
    reader.get_page_title = lambda page: page["id"]
    reader.build_breadcrumb = lambda page: ""
    reader._resolve_relation_titles = lambda page: None

    def _fetch(page_id):
        if page_id in failing_ids:
            raise RuntimeError(f"boom {page_id}")
        return [{"block": page_id}]

    reader._fetch_all_blocks = _fetch
    return reader


class TestReadAllDocuments:
    def test_isolated_failure_is_skipped_not_fatal(self):
        pages = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        reader = _make_reader(pages, failing_ids={"2"})
        out = list(reader.read_all_documents())
        assert [d["page"]["id"] for d in out] == ["1", "3"]

    def test_consecutive_failures_abort(self):
        pages = [{"id": str(i)} for i in range(10)]
        reader = _make_reader(pages, failing_ids={str(i) for i in range(10)}, max_failures_in_row=3)
        with pytest.raises(RuntimeError):
            list(reader.read_all_documents())

    def test_failure_streak_resets_on_success(self):
        # 2 fail, 1 succeed, 2 fail — with threshold 3, never 3-in-a-row, so no abort.
        pages = [{"id": "a"}, {"id": "b"}, {"id": "ok"}, {"id": "c"}, {"id": "d"}]
        reader = _make_reader(pages, failing_ids={"a", "b", "c", "d"}, max_failures_in_row=3)
        out = list(reader.read_all_documents())
        assert [d["page"]["id"] for d in out] == ["ok"]

    def test_all_success_yields_all(self):
        pages = [{"id": "1"}, {"id": "2"}]
        reader = _make_reader(pages)
        out = list(reader.read_all_documents())
        assert len(out) == 2
        assert out[0]["blocks"] == [{"block": "1"}]
