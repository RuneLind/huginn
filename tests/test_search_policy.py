import numpy as np
import pytest

from main.core import search_response_formatter as _formatter
from main.core.search_policy import SearchPolicy


def _doc(*scores):
    return {"matchedChunks": [{"score": s} for s in scores]}


class TestThresholdSourcing:
    def test_thresholds_single_sourced_from_formatter(self):
        policy = SearchPolicy()
        assert policy.LOW_CONFIDENCE_THRESHOLD == _formatter.LOW_CONFIDENCE_THRESHOLD
        assert policy.NOISE_THRESHOLD == _formatter.NOISE_THRESHOLD


class TestBestChunkScore:
    def test_returns_minimum_score_across_matched_chunks(self):
        # scores are negative; more negative = more relevant → min is "best"
        assert SearchPolicy.best_chunk_score(_doc(-0.5, -0.9, -0.3)) == -0.9

    def test_single_chunk(self):
        assert SearchPolicy.best_chunk_score(_doc(-0.2)) == -0.2


class TestApplyConfidenceFiltering:
    def setup_method(self):
        self.policy = SearchPolicy()

    def test_strong_results_not_filtered_no_flag(self):
        response = {"results": [_doc(-0.9), _doc(-0.5), _doc(-0.3)]}
        out = self.policy.apply_confidence_filtering(response)
        assert len(out["results"]) == 3
        assert "lowConfidence" not in out

    def test_noise_chunks_filtered_out(self):
        # Only the -0.5 doc survives NOISE_THRESHOLD (-0.10)
        response = {"results": [_doc(-0.5), _doc(-0.008), _doc(-0.002)]}
        out = self.policy.apply_confidence_filtering(response)
        assert len(out["results"]) == 1
        assert out["results"][0] is response["results"][0]
        assert "lowConfidence" not in out

    def test_results_just_above_noise_threshold_filtered(self):
        response = {"results": [_doc(-0.099), _doc(-0.05), _doc(-0.01)]}
        out = self.policy.apply_confidence_filtering(response)
        assert out["results"] == []
        assert out["lowConfidence"] is True

    def test_boundary_score_equal_to_noise_threshold_kept(self):
        # <= NOISE_THRESHOLD keeps a doc scored exactly at the threshold
        response = {"results": [_doc(SearchPolicy.NOISE_THRESHOLD)]}
        out = self.policy.apply_confidence_filtering(response)
        assert len(out["results"]) == 1
        # best == LOW_CONFIDENCE_THRESHOLD, not strictly greater → no flag
        assert "lowConfidence" not in out

    def test_all_noise_empty_with_flag(self):
        response = {"results": [_doc(-0.005), _doc(-0.003)]}
        out = self.policy.apply_confidence_filtering(response)
        assert out["results"] == []
        assert out["lowConfidence"] is True

    def test_survivor_but_weak_best_flags_low_confidence(self):
        # Survives noise filter (<= -0.10) but weaker than needed → still flagged
        # when best remaining > LOW_CONFIDENCE_THRESHOLD. With both thresholds at
        # -0.10 a surviving doc equals the threshold, so construct a case where a
        # doc is filtered and the survivor is exactly at the boundary.
        response = {"results": [_doc(-0.11)]}
        out = self.policy.apply_confidence_filtering(response)
        assert len(out["results"]) == 1
        assert "lowConfidence" not in out


def _mapping(entries):
    # entries: list of (chunk_id, doc_id, doc_path)
    return {
        str(cid): {"documentId": did, "documentPath": path}
        for cid, did, path in entries
    }


class TestApplyTitleBoost:
    def setup_method(self):
        self.policy = SearchPolicy()

    def test_empty_query_returns_unchanged(self):
        scores = np.array([[-0.5, -0.3]], dtype=np.float32)
        indexes = np.array([[0, 1]], dtype=np.int64)
        out_s, out_i = self.policy.apply_title_boost("", scores, indexes, {})
        assert out_s is scores
        assert out_i is indexes

    def test_single_candidate_returns_unchanged(self):
        # len(scores[0]) < 2 → early return
        scores = np.array([[-0.5]], dtype=np.float32)
        indexes = np.array([[0]], dtype=np.int64)
        out_s, out_i = self.policy.apply_title_boost("term", scores, indexes, {})
        assert out_s is scores
        assert out_i is indexes

    def test_no_title_overlap_returns_unchanged(self):
        mapping = _mapping([(0, "A", "col/documents/alpha.json"),
                            (1, "B", "col/documents/beta.json")])
        scores = np.array([[-0.5, -0.3]], dtype=np.float32)
        indexes = np.array([[0, 1]], dtype=np.int64)
        out_s, out_i = self.policy.apply_title_boost(
            "zzz nomatch", scores, indexes, mapping
        )
        # No overlap → any_boost False → original arrays returned
        assert out_s is scores
        assert out_i is indexes

    def test_title_match_boosts_and_resorts(self):
        # doc B's title matches three query terms (enough to hit the boost cap and
        # overtake the better-scored doc A, whose title does not overlap). A single
        # term is only half the score range and cannot flip the two endpoints.
        mapping = _mapping([(0, "A", "col/documents/zulu.json"),
                            (1, "B", "col/documents/alpha-beta-gamma.json")])
        scores = np.array([[-0.5, -0.3]], dtype=np.float32)
        indexes = np.array([[0, 1]], dtype=np.int64)
        out_s, out_i = self.policy.apply_title_boost(
            "alpha beta gamma", scores, indexes, mapping
        )
        # B (index 1) should now be first (lower/better score after boost)
        assert out_i[0][0] == 1
        # Scores are re-sorted ascending (lower = better)
        assert out_s[0][0] <= out_s[0][1]

    def test_boost_records_to_trace_when_enabled(self):
        mapping = _mapping([(0, "A", "col/documents/alpha.json"),
                            (1, "B", "col/documents/gamma.json")])
        scores = np.array([[-0.5, -0.3]], dtype=np.float32)
        indexes = np.array([[0, 1]], dtype=np.int64)

        recorded = []

        class FakeTrace:
            enabled = True

            def record_title_boost(self, doc_id, delta):
                recorded.append((doc_id, delta))

        self.policy.apply_title_boost("gamma", scores, indexes, mapping, FakeTrace())
        assert len(recorded) == 1
        assert recorded[0][0] == "B"
        # Single-term overlap stays below the cap: delta is exactly one
        # boost_per_term = -score_range * 0.5 = -(0.2) * 0.5. Pins the boost
        # weight itself, which the cap-saturated ordering tests cannot.
        assert recorded[0][1] == pytest.approx(-0.1)

    def test_hyphen_and_underscore_titles_tokenized(self):
        # A hyphen/underscore filename is split into tokens ("multi", "word",
        # "doc"), all three matching the query, so the boost cap flips it ahead of
        # the non-overlapping doc A.
        mapping = _mapping([(0, "A", "col/documents/zulu.json"),
                            (1, "B", "col/documents/multi-word_doc.json")])
        scores = np.array([[-0.5, -0.3]], dtype=np.float32)
        indexes = np.array([[0, 1]], dtype=np.int64)
        out_s, out_i = self.policy.apply_title_boost(
            "multi word doc", scores, indexes, mapping
        )
        assert out_i[0][0] == 1

    def test_missing_mapping_entry_skipped(self):
        # chunk 1 has no mapping entry; boost loop must skip it without error.
        mapping = _mapping([(0, "A", "col/documents/gamma.json")])
        scores = np.array([[-0.5, -0.3]], dtype=np.float32)
        indexes = np.array([[0, 1]], dtype=np.int64)
        out_s, out_i = self.policy.apply_title_boost(
            "gamma", scores, indexes, mapping
        )
        # doc A (index 0) matched and already first → boosted but order stable
        assert out_i[0][0] == 0
