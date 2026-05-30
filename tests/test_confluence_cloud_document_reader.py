"""Tests for ConfluenceCloudDocumentReader comment resolution (M8)."""

import pytest

from main.sources.confluence.confluence_cloud_document_reader import (
    ConfluenceCloudDocumentReader,
)


@pytest.fixture
def reader():
    # __init__ performs no network I/O; valid-looking credentials are enough.
    return ConfluenceCloudDocumentReader(
        base_url="https://example.atlassian.net",
        query="",
        email="user@example.com",
        api_token="token",
    )


def _read_comments(reader, page):
    # __read_comments is name-mangled (private).
    return reader._ConfluenceCloudDocumentReader__read_comments(page)


class TestReadComments:
    def test_page_missing_comment_structure_returns_empty(self, reader):
        # A page with no children/comment expansion must not KeyError.
        assert _read_comments(reader, {"content": {"id": "1"}}) == []

    def test_page_missing_content_returns_empty(self, reader):
        assert _read_comments(reader, {}) == []

    def test_zero_comments_returns_empty(self, reader):
        page = {"content": {"id": "1", "children": {"comment": {"size": 0, "results": []}}}}
        assert _read_comments(reader, page) == []

    def test_returns_inline_comment_results(self, reader):
        page = {
            "content": {
                "id": "1",
                "children": {"comment": {"size": 2, "results": [{"a": 1}, {"a": 2}]}},
            }
        }
        assert _read_comments(reader, page) == [{"a": 1}, {"a": 2}]

    def test_size_present_but_results_missing_returns_empty(self, reader):
        # Defensive: size>0 but the results key absent should not KeyError.
        page = {"content": {"id": "1", "children": {"comment": {"size": 3}}}}
        assert _read_comments(reader, page) == []
