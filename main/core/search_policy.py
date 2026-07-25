"""Search relevance policy — noise filtering, low-confidence flagging, title boost.

Two decisions the searcher makes about *relevance* rather than retrieval, kept
here so they can be tuned and tested without standing up a searcher:

- ``apply_confidence_filtering`` drops documents whose best chunk is noise and
  flags the response ``lowConfidence`` when the best survivor is still weak.
  Operates on raw cross-encoder scores, so it only applies to reranked results.
- ``apply_title_boost`` rewards documents whose title tokens overlap the query,
  scaling the boost to the score spread so it behaves the same across score
  types (cross-encoder, hybrid RRF, FAISS L2) and capping it at three terms'
  worth so a long title cannot dominate genuine relevance.

Stateless functions plus thresholds, like ``search_response_formatter``. Nothing
here owns searcher state: ``apply_title_boost`` takes the index→document
``mapping`` as a parameter, because that mapping is paired with a frozen
in-memory index for a searcher's lifetime and swapped atomically on reload — it
belongs to the searcher, not to the policy.

Thresholds are imported from ``search_response_formatter``, which derives the
relevance-space confidence bands from the same values, so filtering and the
reported bands stay in sync by construction.
"""
import re

import numpy as np

from main.core.search_response_formatter import (
    LOW_CONFIDENCE_THRESHOLD,
    NOISE_THRESHOLD,
)
from main.utils.filename import title_from_doc_path


def best_chunk_score(doc):
    """Best (lowest) score among a document's matched chunks."""
    return min(chunk["score"] for chunk in doc["matchedChunks"])


def apply_confidence_filtering(response):
    """Drop noise documents and flag a weak response, in place."""
    results = response["results"]

    # Filter out documents where all matched chunks are noise
    filtered = [doc for doc in results if best_chunk_score(doc) <= NOISE_THRESHOLD]
    response["results"] = filtered

    # Flag response as low confidence if best remaining result is weak
    if not filtered or best_chunk_score(filtered[0]) > LOW_CONFIDENCE_THRESHOLD:
        response["lowConfidence"] = True

    return response


def apply_title_boost(query, scores, indexes, mapping, coll_trace=None):
    """Boost scores for documents whose title matches query terms.

    Boost magnitude scales with the score spread so it works across
    different score types (cross-encoder, hybrid RRF, FAISS L2).

    ``mapping`` is the searcher's index→document mapping, passed in rather
    than owned (see module docstring).
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
