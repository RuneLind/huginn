import re

import numpy as np

from main.core import search_response_formatter as _formatter
from main.utils.filename import title_from_doc_path


class SearchPolicy:
    """Confidence-filtering and title-boost policy for the search hot path.

    Extracted verbatim from DocumentCollectionSearcher — same thresholds, same
    arithmetic, zero behaviour change. The thresholds are single-sourced from
    search_response_formatter, which derives the relevance-space confidence bands
    from the same values.

    The policy carries no searcher instance state: apply_title_boost takes the
    index→document ``mapping`` as a parameter rather than owning it. The mapping
    is paired with a frozen in-memory index for a searcher's lifetime and swapped
    atomically on reload, so it belongs to the searcher, not the policy.
    """

    # Cross-encoder reranker score thresholds (negative scores, more negative =
    # more relevant). Defined in search_response_formatter, which derives the
    # relevance-space confidence bands from the same values.
    LOW_CONFIDENCE_THRESHOLD = _formatter.LOW_CONFIDENCE_THRESHOLD
    NOISE_THRESHOLD = _formatter.NOISE_THRESHOLD

    @staticmethod
    def best_chunk_score(doc):
        return min(chunk["score"] for chunk in doc["matchedChunks"])

    def apply_confidence_filtering(self, response):
        results = response["results"]

        # Filter out documents where all matched chunks are noise
        filtered = [
            doc for doc in results
            if self.best_chunk_score(doc) <= self.NOISE_THRESHOLD
        ]
        response["results"] = filtered

        # Flag response as low confidence if best remaining result is weak
        if not filtered or self.best_chunk_score(filtered[0]) > self.LOW_CONFIDENCE_THRESHOLD:
            response["lowConfidence"] = True

        return response

    def apply_title_boost(self, query, scores, indexes, mapping, coll_trace=None):
        """Boost scores for documents whose title matches query terms.

        Boost magnitude scales with the score spread so it works across
        different score types (cross-encoder, hybrid RRF, FAISS L2).

        ``mapping`` is the searcher's index→document mapping, passed in rather
        than owned (see class docstring).
        """
        query_tokens = set(re.findall(r'\w+', query.lower()))
        if not query_tokens or len(scores[0]) < 2:
            return scores, indexes

        # Scale boost to score range (scores sorted ascending, lower = better)
        score_range = float(scores[0][-1] - scores[0][0])
        if score_range < 1e-6:
            score_range = max(abs(float(scores[0][0])) * 0.1, 0.01)
        boost_per_term = -score_range * 0.5
        boost_cap = -score_range * 1.5

        # Calculate and apply boosts in a single pass
        doc_boosts = {}
        boosted_scores = scores[0].copy()
        any_boost = False

        for i, chunk_id in enumerate(indexes[0]):
            entry = mapping.get(str(int(chunk_id)))
            if not entry:
                continue
            doc_id = entry["documentId"]
            if doc_id not in doc_boosts:
                title = title_from_doc_path(entry.get("documentPath", "")).replace("-", " ").replace("_", " ")
                title_tokens = set(re.findall(r'\w+', title.lower()))
                overlap = len(query_tokens & title_tokens)
                doc_boosts[doc_id] = max(boost_per_term * overlap, boost_cap) if overlap > 0 else 0.0
            if doc_boosts[doc_id] != 0.0:
                boosted_scores[i] += doc_boosts[doc_id]
                any_boost = True

        if coll_trace is not None and coll_trace.enabled:
            for doc_id, delta in doc_boosts.items():
                if delta != 0.0:
                    coll_trace.record_title_boost(doc_id, delta)

        if not any_boost:
            return scores, indexes

        # Re-sort by boosted score (lower = better)
        order = np.argsort(boosted_scores)
        return (
            np.array([boosted_scores[order]], dtype=scores.dtype),
            np.array([indexes[0][order]], dtype=indexes.dtype),
        )
