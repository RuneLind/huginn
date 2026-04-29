"""Per-search trace recorder.

Captures stage-by-stage data (graph expansion, FAISS / BM25 ranks, RRF, cross-encoder,
title boost, confidence filtering, timings) so callers can see *why* a result ranked
where it did.

Designed for zero overhead when disabled: callers either hold a real `SearchTrace` or
a `NullSearchTrace`. Both expose the same surface; the null variant short-circuits
every method to a no-op and returns sibling null collectors so the caller never branches.

See docs/search-tracing-plan.md for the schema this produces.
"""

import time

SCHEMA_VERSION = 1

VALID_STAGES = ("faiss", "bm25", "rrf", "ce", "final")
VALID_DROP_REASONS = ("noise", "dedup", "missingDoc", "perDocCap", "metadataFilter")


def create_trace(enabled):
    """Factory: returns a real recorder if `enabled`, else a null recorder."""
    return SearchTrace() if enabled else NULL_TRACE


class SearchTrace:
    """Top-level trace, owned by the API server. Holds query info + per-collection traces."""

    enabled = True

    def __init__(self):
        self._t_start = time.monotonic()
        self._query = {
            "raw": None,
            "expanded": None,
            "detectedEntities": [],
            "expansionTerms": [],
            "graphAnswered": False,
            "rerankerSkipped": False,
            "rerankerSkipReason": None,
        }
        self._collections = []

    def set_query_raw(self, text):
        self._query["raw"] = text
        # `expanded` defaults to raw until set_expansion() overrides it.
        self._query["expanded"] = text

    def set_expansion(self, expanded_text, expansion_terms):
        self._query["expanded"] = expanded_text
        self._query["expansionTerms"] = list(expansion_terms)

    def add_detected_entity(self, entity_id, entity_type, label, matched_span):
        self._query["detectedEntities"].append({
            "id": entity_id,
            "type": entity_type,
            "label": label,
            "matchedSpan": matched_span,
        })

    def set_graph_answered(self, answered):
        self._query["graphAnswered"] = bool(answered)

    def set_reranker_skipped(self, skipped, reason=None):
        self._query["rerankerSkipped"] = bool(skipped)
        self._query["rerankerSkipReason"] = reason

    def start_collection(self, name, indexer, fetch_k):
        coll = CollectionTrace(name=name, indexer=indexer, fetch_k=fetch_k)
        self._collections.append(coll)
        return coll

    def to_dict(self):
        return {
            "query": dict(self._query),
            "collections": [c.to_dict() for c in self._collections],
            "totalMs": int((time.monotonic() - self._t_start) * 1000),
            "schemaVersion": SCHEMA_VERSION,
        }


class CollectionTrace:
    """Per-collection trace. Owned by `DocumentCollectionSearcher.search()`."""

    enabled = True

    def __init__(self, name, indexer, fetch_k):
        self._name = name
        self._indexer = indexer
        self._fetch_k = fetch_k
        # Keyed by chunk_id; lazily created on first stage record.
        self._candidates = {}
        # Title boost is recorded per-document (one delta affects all chunks of a doc).
        self._title_boosts = {}
        self._confidence = None
        self._timings = {}

    def _ensure(self, chunk_id):
        c = self._candidates.get(chunk_id)
        if c is None:
            c = {
                "chunkId": chunk_id,
                "documentId": None,
                "docTitle": None,
                "headings": None,
                "stages": {},
                "kept": True,
                "dropReason": None,
            }
            self._candidates[chunk_id] = c
        return c

    def record_stage(self, stage, chunk_id, rank, score):
        if stage not in VALID_STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {VALID_STAGES}")
        c = self._ensure(chunk_id)
        c["stages"][stage] = {"rank": int(rank), "score": float(score)}

    def annotate_candidate(self, chunk_id, document_id=None, doc_title=None, headings=None):
        """Attach human-readable identifiers once the chunk → doc mapping is loaded."""
        c = self._ensure(chunk_id)
        if document_id is not None:
            c["documentId"] = document_id
        if doc_title is not None:
            c["docTitle"] = doc_title
        if headings is not None:
            c["headings"] = list(headings)

    def record_title_boost(self, document_id, delta):
        self._title_boosts[document_id] = float(delta)

    def mark_dropped(self, chunk_id, reason):
        if reason not in VALID_DROP_REASONS:
            raise ValueError(f"unknown drop reason {reason!r}; expected one of {VALID_DROP_REASONS}")
        c = self._ensure(chunk_id)
        c["kept"] = False
        c["dropReason"] = reason

    def set_confidence(self, low_confidence, best_score, low_confidence_threshold,
                       noise_threshold, filtered_count):
        self._confidence = {
            "lowConfidence": bool(low_confidence),
            "bestScore": float(best_score) if best_score is not None else None,
            "lowConfidenceThreshold": float(low_confidence_threshold),
            "noiseThreshold": float(noise_threshold),
            "filteredCount": int(filtered_count),
        }

    def set_timings(self, **timings_ms):
        for k, v in timings_ms.items():
            self._timings[k] = int(v)

    def to_dict(self):
        # Apply per-doc title boosts onto each candidate's stage dict for output.
        candidates = []
        for c in self._candidates.values():
            out = dict(c)
            out["stages"] = dict(c["stages"])
            doc_id = c.get("documentId")
            if doc_id is not None and doc_id in self._title_boosts:
                out["stages"]["titleBoost"] = {
                    "applied": True,
                    "delta": self._title_boosts[doc_id],
                }
            candidates.append(out)

        return {
            "name": self._name,
            "indexer": self._indexer,
            "fetchK": self._fetch_k,
            "candidates": candidates,
            "confidence": self._confidence,
            "timingsMs": dict(self._timings),
        }


class NullSearchTrace:
    """No-op trace. Every method is a noop; `start_collection` returns the null collection.

    A single shared instance (`NULL_TRACE`) is fine because nothing mutates state.
    """

    enabled = False

    def set_query_raw(self, text): pass
    def set_expansion(self, expanded_text, expansion_terms): pass
    def add_detected_entity(self, entity_id, entity_type, label, matched_span): pass
    def set_graph_answered(self, answered): pass
    def set_reranker_skipped(self, skipped, reason=None): pass

    def start_collection(self, name, indexer, fetch_k):
        return NULL_COLLECTION_TRACE

    def to_dict(self):
        return None


class NullCollectionTrace:
    enabled = False

    def record_stage(self, stage, chunk_id, rank, score): pass
    def annotate_candidate(self, chunk_id, document_id=None, doc_title=None, headings=None): pass
    def record_title_boost(self, document_id, delta): pass
    def mark_dropped(self, chunk_id, reason): pass
    def set_confidence(self, low_confidence, best_score, low_confidence_threshold,
                       noise_threshold, filtered_count): pass
    def set_timings(self, **timings_ms): pass

    def to_dict(self):
        return None


NULL_TRACE = NullSearchTrace()
NULL_COLLECTION_TRACE = NullCollectionTrace()
