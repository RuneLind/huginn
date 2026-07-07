"""Unit tests for the document dedup helper extracted from
``DocumentCollectionSearcher.__build_results``.

``deduplicate_document`` collapses documents that share a source URL or an
identical body-text MD5, keeping the first-seen document as canonical.
"""
import hashlib

from main.core.documents_collection_searcher import deduplicate_document


def _text_hash(text):
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


class TestDeduplicateDocument:

    def test_first_document_is_kept_and_registered(self):
        seen_urls, seen_hashes = {}, {}
        is_dup = deduplicate_document(
            "doc-1", "https://example.com/a", lambda: "body text",
            seen_urls, seen_hashes,
        )
        assert is_dup is False
        assert seen_urls == {"https://example.com/a": "doc-1"}
        assert seen_hashes == {_text_hash("body text"): "doc-1"}

    def test_same_url_is_duplicate_and_skips_text_read(self):
        seen_urls = {"https://example.com/a": "doc-1"}
        seen_hashes = {}
        read_calls = []

        def _provider():
            read_calls.append(True)
            return "whatever"

        is_dup = deduplicate_document(
            "doc-2", "https://example.com/a", _provider, seen_urls, seen_hashes,
        )
        assert is_dup is True
        # URL check short-circuits before any document read.
        assert read_calls == []
        # No new registration.
        assert seen_urls == {"https://example.com/a": "doc-1"}
        assert seen_hashes == {}

    def test_same_text_different_url_is_duplicate(self):
        seen_urls, seen_hashes = {}, {}
        deduplicate_document("doc-1", "https://example.com/a", lambda: "identical body",
                             seen_urls, seen_hashes)
        is_dup = deduplicate_document("doc-2", "https://example.com/b", lambda: "identical body",
                                      seen_urls, seen_hashes)
        assert is_dup is True
        # doc-2 was not registered as canonical for its URL.
        assert "https://example.com/b" not in seen_urls

    def test_empty_text_never_dedups_on_content(self):
        # Two empty-body docs with distinct URLs both survive: the ``text_content
        # and ...`` guard means empty bodies never collapse together.
        seen_urls, seen_hashes = {}, {}
        assert deduplicate_document("doc-1", "https://example.com/a", lambda: "",
                                    seen_urls, seen_hashes) is False
        assert deduplicate_document("doc-2", "https://example.com/b", lambda: "",
                                    seen_urls, seen_hashes) is False

    def test_missing_url_still_dedups_on_text(self):
        seen_urls, seen_hashes = {}, {}
        assert deduplicate_document("doc-1", "", lambda: "shared", seen_urls, seen_hashes) is False
        assert deduplicate_document("doc-2", "", lambda: "shared", seen_urls, seen_hashes) is True
        # No URL was recorded for either doc.
        assert seen_urls == {}

    def test_distinct_url_and_text_are_all_kept(self):
        seen_urls, seen_hashes = {}, {}
        assert deduplicate_document("doc-1", "https://example.com/a", lambda: "alpha",
                                    seen_urls, seen_hashes) is False
        assert deduplicate_document("doc-2", "https://example.com/b", lambda: "beta",
                                    seen_urls, seen_hashes) is False
        assert len(seen_urls) == 2
        assert len(seen_hashes) == 2
