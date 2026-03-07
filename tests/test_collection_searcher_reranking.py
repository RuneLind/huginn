import json
import numpy as np
from unittest.mock import MagicMock

from main.core.documents_collection_searcher import DocumentCollectionSearcher


def _make_mapping(chunk_id, doc_id, chunk_number, doc_url=None, doc_path=None):
    if doc_url is None:
        doc_url = f"http://example.com/{doc_id}"
    if doc_path is None:
        doc_path = f"col/documents/{doc_id}.json"
    return {
        str(chunk_id): {
            "documentId": doc_id,
            "documentUrl": doc_url,
            "documentPath": doc_path,
            "chunkNumber": chunk_number,
        }
    }


def _make_document(doc_id, chunks):
    return {
        "id": doc_id,
        "text": f"Full text of {doc_id}",
        "chunks": chunks,
    }


class TestSearcherWithReranking:
    def _setup(self, reranker=None):
        # Two documents, three chunks total
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))
        mapping.update(_make_mapping(1, "doc-A", 1))
        mapping.update(_make_mapping(2, "doc-B", 0))

        doc_a = _make_document("doc-A", [
            {"indexedData": "chunk A0 text"},
            {"indexedData": "chunk A1 text"},
        ])
        doc_b = _make_document("doc-B", [
            {"indexedData": "chunk B0 text"},
        ])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)

        persister.read_text_file.side_effect = read_text

        # Indexer returns chunks in order: 0, 1, 2
        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        return DocumentCollectionSearcher(
            collection_name="col",
            indexer=indexer,
            persister=persister,
            reranker=reranker,
        )

    def test_search_without_reranker_unchanged(self):
        """Without reranker, behavior is identical to original."""
        searcher = self._setup(reranker=None)
        result = searcher.search("test query", max_number_of_chunks=3)

        # Should have both documents
        assert len(result["results"]) == 2
        # doc-A should come first (lower score = better)
        assert result["results"][0]["id"] == "doc-A"
        assert result["reranked"] is False

    def test_search_with_reranker_reorders_results(self):
        """Reranker promotes doc-B to top."""
        reranker = MagicMock()
        # Reranker says chunk 2 (doc-B) is best, chunk 0 next, chunk 1 worst
        reranker.rerank.return_value = (
            np.array([[-0.9, -0.5, -0.1]], dtype=np.float32),
            np.array([[2, 0, 1]], dtype=np.int64),
        )

        searcher = self._setup(reranker=reranker)
        result = searcher.search("test query", max_number_of_chunks=3)

        # doc-B should come first after reranking
        assert result["results"][0]["id"] == "doc-B"
        assert result["reranked"] is True

    def test_reranker_receives_chunk_texts(self):
        """Verify the reranker is called with correct chunk texts."""
        reranker = MagicMock()
        reranker.rerank.return_value = (
            np.array([[-0.9, -0.5, -0.1]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        searcher = self._setup(reranker=reranker)
        searcher.search("test query", max_number_of_chunks=3)

        call_args = reranker.rerank.call_args
        chunk_texts = call_args[0][3]  # 4th positional arg
        assert chunk_texts == ["chunk A0 text", "chunk A1 text", "chunk B0 text"]

    def test_skip_reranker_flag_bypasses_reranker(self):
        """skip_reranker=True skips reranker even when one is configured."""
        reranker = MagicMock()
        searcher = self._setup(reranker=reranker)
        result = searcher.search("test query", max_number_of_chunks=3, skip_reranker=True)

        reranker.rerank.assert_not_called()
        assert result["reranked"] is False

    def test_reranker_overfetch_with_dedup_buffer(self):
        """Verify indexer overfetches with dedup buffer when reranker is present."""
        reranker = MagicMock()
        reranker.rerank.return_value = (
            np.array([[-0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        searcher = self._setup(reranker=reranker)
        searcher.search("test query", max_number_of_chunks=5)

        # effective_chunks = 5 + max(3, 5//3) = 5 + 3 = 8, fetch_k = int(8 * 1.5) = 12
        searcher.indexer.search.assert_called_once_with("test query", 12)


class TestConfidenceFiltering:
    def _setup(self, reranker_scores):
        """Setup searcher with reranker returning given scores."""
        # Three documents, one chunk each
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))
        mapping.update(_make_mapping(1, "doc-B", 0))
        mapping.update(_make_mapping(2, "doc-C", 0))

        doc_a = _make_document("doc-A", [{"indexedData": "relevant content"}])
        doc_b = _make_document("doc-B", [{"indexedData": "somewhat relevant"}])
        doc_c = _make_document("doc-C", [{"indexedData": "noise content"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            if "doc-C" in path:
                return json.dumps(doc_c)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        reranker = MagicMock()
        reranker.rerank.return_value = (
            np.array([reranker_scores], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        return DocumentCollectionSearcher(
            collection_name="col",
            indexer=indexer,
            persister=persister,
            reranker=reranker,
        )

    def test_high_confidence_no_flag(self):
        """Strong results: no lowConfidence flag."""
        searcher = self._setup([-0.9, -0.5, -0.3])
        result = searcher.search("test", max_number_of_chunks=3)
        assert "lowConfidence" not in result
        assert len(result["results"]) == 3

    def test_noise_results_filtered_out(self):
        """Results above noise threshold (-0.01) are filtered."""
        searcher = self._setup([-0.5, -0.008, -0.002])
        result = searcher.search("test", max_number_of_chunks=3)
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "doc-A"
        assert "lowConfidence" not in result

    def test_low_confidence_flagged(self):
        """Best result above low-confidence threshold → flag set."""
        searcher = self._setup([-0.05, -0.008, -0.002])
        result = searcher.search("test", max_number_of_chunks=3)
        assert result["lowConfidence"] is True
        # doc-A kept (-0.05 < -0.01), doc-B and doc-C filtered
        assert len(result["results"]) == 1

    def test_all_noise_returns_empty_with_flag(self):
        """All results are noise → empty results with lowConfidence."""
        searcher = self._setup([-0.005, -0.003, -0.001])
        result = searcher.search("test", max_number_of_chunks=3)
        assert result["lowConfidence"] is True
        assert len(result["results"]) == 0

    def test_no_filtering_without_reranker(self):
        """Without reranker, no confidence filtering is applied."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))
        doc_a = _make_document("doc-A", [{"indexedData": "content"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            return json.dumps(doc_a)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister, reranker=None)
        result = searcher.search("test", max_number_of_chunks=1)
        assert "lowConfidence" not in result
        assert len(result["results"]) == 1


class TestResultDeduplication:
    def _setup_with_duplicate(self, reranker=None):
        """Setup with doc-A and doc-C having identical text content."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))
        mapping.update(_make_mapping(1, "doc-B", 0))
        mapping.update(_make_mapping(2, "doc-C", 0))  # duplicate of doc-A

        doc_a = _make_document("doc-A", [{"indexedData": "chunk A0"}])
        doc_b = _make_document("doc-B", [{"indexedData": "chunk B0"}])
        doc_c = {"id": "doc-C", "text": doc_a["text"], "chunks": [{"indexedData": "chunk C0"}]}  # same text as doc-A

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            if "doc-C" in path:
                return json.dumps(doc_c)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        return DocumentCollectionSearcher("col", indexer, persister, reranker=reranker)

    def test_duplicate_content_is_deduplicated(self):
        """Documents with identical text are deduplicated, keeping first."""
        searcher = self._setup_with_duplicate()
        result = searcher.search("test", max_number_of_chunks=3)

        doc_ids = [r["id"] for r in result["results"]]
        assert doc_ids == ["doc-A", "doc-B"]  # doc-C removed (same text as doc-A)

    def test_empty_text_documents_not_deduplicated(self):
        """Documents with empty/missing text should NOT be deduplicated against each other."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))
        mapping.update(_make_mapping(1, "doc-B", 0))

        doc_a = {"id": "doc-A", "text": "", "chunks": [{"indexedData": "chunk A"}]}
        doc_b = {"id": "doc-B", "text": "", "chunks": [{"indexedData": "chunk B"}]}

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister)
        result = searcher.search("test", max_number_of_chunks=2)
        assert len(result["results"]) == 2  # both kept despite identical empty text

    def test_same_url_different_text_is_deduplicated(self):
        """Documents with same URL but different text/paths are deduplicated (Notion page ID bug)."""
        notion_url = "https://www.notion.so/304cce311db4804bb8cfeac0b80ec816"
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0, doc_url=notion_url,
                                     doc_path="col/documents/Team Lønn - få det i gang.json"))
        mapping.update(_make_mapping(1, "doc-B", 0, doc_url="http://other.com"))
        mapping.update(_make_mapping(2, "doc-C", 0, doc_url=notion_url,
                                     doc_path="col/documents/Teams/Team Geiter/Team Lønn - få det i gang.json"))

        doc_a = {"id": "doc-A", "text": "Team Lønn content v1", "chunks": [{"indexedData": "chunk A"}]}
        doc_b = _make_document("doc-B", [{"indexedData": "chunk B"}])
        doc_c = {"id": "doc-C", "text": "Team Lønn content v2 with breadcrumbs", "chunks": [{"indexedData": "chunk C"}]}

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            # Check most-specific path first: doc-C's path contains both
            # "Team Geiter" and "Team Lønn", so match "Team Geiter" first
            if "Team Geiter" in path:
                return json.dumps(doc_c)
            if "Team Lønn" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister)
        result = searcher.search("test", max_number_of_chunks=3)

        doc_ids = [r["id"] for r in result["results"]]
        assert doc_ids == ["doc-A", "doc-B"]  # doc-C removed (same URL as doc-A)

    def test_different_urls_not_deduplicated(self):
        """Documents with different URLs are kept even if they have similar content."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0, doc_url="http://example.com/page1"))
        mapping.update(_make_mapping(1, "doc-B", 0, doc_url="http://example.com/page2"))

        doc_a = _make_document("doc-A", [{"indexedData": "chunk A"}])
        doc_b = _make_document("doc-B", [{"indexedData": "chunk B"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister)
        result = searcher.search("test", max_number_of_chunks=2)
        assert len(result["results"]) == 2

    def test_empty_url_not_deduplicated(self):
        """Documents with empty URLs should not be deduplicated by URL."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0, doc_url=""))
        mapping.update(_make_mapping(1, "doc-B", 0, doc_url=""))

        doc_a = _make_document("doc-A", [{"indexedData": "chunk A"}])
        doc_b = _make_document("doc-B", [{"indexedData": "chunk B"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister)
        result = searcher.search("test", max_number_of_chunks=2)
        assert len(result["results"]) == 2  # both kept despite same empty URL

    def test_unique_documents_not_deduplicated(self):
        """Documents with different text are all kept."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))
        mapping.update(_make_mapping(1, "doc-B", 0))

        doc_a = _make_document("doc-A", [{"indexedData": "unique A"}])
        doc_b = _make_document("doc-B", [{"indexedData": "unique B"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "doc-A" in path:
                return json.dumps(doc_a)
            if "doc-B" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister)
        result = searcher.search("test", max_number_of_chunks=2)
        assert len(result["results"]) == 2


class TestTitleBoost:
    def _setup_with_title_match(self):
        """Setup where doc-B's filename matches query terms but is ranked lower by reranker."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0, doc_path="col/documents/general-info.json"))
        mapping.update(_make_mapping(1, "doc-B", 0, doc_path="col/documents/slack-kanal-onboarding.json"))

        doc_a = _make_document("doc-A", [{"indexedData": "some info about channels"}])
        doc_b = _make_document("doc-B", [{"indexedData": "slack channel setup"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "general-info" in path:
                return json.dumps(doc_a)
            if "slack-kanal" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        reranker = MagicMock()
        # Reranker ranks doc-A first (lower score = better)
        reranker.rerank.return_value = (
            np.array([[-0.5, -0.3]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        return DocumentCollectionSearcher("col", indexer, persister, reranker=reranker)

    def test_title_match_boosts_document(self):
        """Document with query terms in title gets boosted above others."""
        searcher = self._setup_with_title_match()
        result = searcher.search("slack kanal onboarding", max_number_of_chunks=2)

        # doc-B ("slack-kanal-onboarding.json") should be boosted to #1
        # boost: 3 matching tokens * -0.5 = -1.5, capped at -1.5
        # doc-B score: -0.3 + (-1.5) = -1.8
        # doc-A score: -0.5 (no boost)
        assert result["results"][0]["id"] == "doc-B"

    def test_no_boost_without_title_match(self):
        """No boost applied when query doesn't match any document title."""
        searcher = self._setup_with_title_match()
        result = searcher.search("completely unrelated query", max_number_of_chunks=2)

        # doc-A should stay first (no title match for either)
        assert result["results"][0]["id"] == "doc-A"

    def test_title_boost_applied_without_reranker(self):
        """Title boost is applied even without reranker (FAISS L2 scores)."""
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0, doc_path="col/documents/general-info.json"))
        mapping.update(_make_mapping(1, "doc-B", 0, doc_path="col/documents/slack-kanal-onboarding.json"))

        doc_a = _make_document("doc-A", [{"indexedData": "some info"}])
        doc_b = _make_document("doc-B", [{"indexedData": "slack info"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            if "general-info" in path:
                return json.dumps(doc_a)
            if "slack-kanal" in path:
                return json.dumps(doc_b)
            raise FileNotFoundError(path)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        # doc-A slightly better by FAISS L2, but doc-B title matches 3 query terms
        indexer.search.return_value = (
            np.array([[0.5, 1.5]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        searcher = DocumentCollectionSearcher("col", indexer, persister, reranker=None)
        result = searcher.search("slack kanal onboarding", max_number_of_chunks=2)

        # Title boost (3 matching terms) should promote doc-B to #1
        assert result["results"][0]["id"] == "doc-B"


class TestCrossLingualSkip:
    def _setup(self):
        mapping = {}
        mapping.update(_make_mapping(0, "doc-A", 0))

        doc_a = _make_document("doc-A", [{"indexedData": "norwegian content"}])

        persister = MagicMock()
        def read_text(path):
            if "index_document_mapping" in path:
                return json.dumps(mapping)
            return json.dumps(doc_a)
        persister.read_text_file.side_effect = read_text

        indexer = MagicMock()
        indexer.get_name.return_value = "test_indexer"
        indexer.search.return_value = (
            np.array([[0.5]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        reranker = MagicMock()
        reranker.rerank.return_value = (
            np.array([[-0.5]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        return DocumentCollectionSearcher("col", indexer, persister, reranker=reranker)

    def test_english_query_skips_reranker(self):
        """English queries (3+ words) skip the reranker."""
        searcher = self._setup()
        result = searcher.search("employee benefits overview", max_number_of_chunks=1)

        # Reranker should NOT have been called
        searcher.reranker.rerank.assert_not_called()
        # Should still return results
        assert len(result["results"]) == 1
        # No confidence filtering when reranker is skipped
        assert "lowConfidence" not in result

    def test_norwegian_query_uses_reranker(self):
        """Norwegian queries still use the reranker."""
        searcher = self._setup()
        result = searcher.search("oversikt over goder og fordeler", max_number_of_chunks=1)

        # Reranker SHOULD have been called
        searcher.reranker.rerank.assert_called_once()

    def test_short_query_uses_reranker(self):
        """Short queries (< 3 words) always use reranker regardless of language."""
        searcher = self._setup()
        result = searcher.search("benefits overview", max_number_of_chunks=1)

        # Reranker should be called (query too short for reliable detection)
        searcher.reranker.rerank.assert_called_once()

    def test_no_confidence_filtering_when_reranker_skipped(self):
        """Confidence filtering is skipped when reranker is skipped."""
        searcher = self._setup()
        # Use a score that would normally be flagged as low confidence
        searcher.indexer.search.return_value = (
            np.array([[0.5]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        result = searcher.search("what are the employee benefits here", max_number_of_chunks=1)

        # No confidence filtering applied
        assert "lowConfidence" not in result
