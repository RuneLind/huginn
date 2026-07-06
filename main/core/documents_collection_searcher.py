import hashlib
import json
import logging
import re
import time

import numpy as np

from main.core import search_response_formatter as _formatter
from main.core.search_trace import NULL_TRACE
from main.utils.filename import title_from_doc_path
from main.utils.performance import delta_ms

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _langdetect_available = True
except ImportError:
    _langdetect_available = False

logger = logging.getLogger(__name__)


def deduplicate_document(doc_id, doc_url, text_provider, seen_urls, seen_text_hashes):
    """Decide whether a newly-encountered document duplicates one already kept.

    Two dedup signals, checked in order:

    1. **URL** — same source ``documentUrl`` means the same page regardless of
       path or text. Checked first so a URL-duplicate never triggers a document
       read (``text_provider`` stays uncalled).
    2. **Content** — MD5 hash of the document body text; identical non-empty
       bodies collapse to the first-seen document.

    On a non-duplicate the document's URL and text hash are registered into
    ``seen_urls`` / ``seen_text_hashes`` (both mutated in place) so later
    documents dedup against it.

    ``text_provider`` is a zero-arg callable returning the document body text;
    it is invoked lazily (only past the URL check) to preserve the original
    lazy-read behaviour.

    Returns ``True`` if ``doc_id`` is a duplicate and should be skipped,
    ``False`` if it was registered as the canonical document.
    """
    if doc_url and doc_url in seen_urls:
        logger.debug(f"Dedup: skipping {doc_id}, same URL as {seen_urls[doc_url]}")
        return True

    text_content = text_provider()
    text_hash = hashlib.md5(text_content.encode(), usedforsecurity=False).hexdigest()

    if text_content and text_hash in seen_text_hashes:
        logger.debug(f"Dedup: skipping {doc_id}, same content as {seen_text_hashes[text_hash]}")
        return True

    if doc_url:
        seen_urls[doc_url] = doc_id
    seen_text_hashes[text_hash] = doc_id
    return False


class DocumentCollectionSearcher:
    # Cross-encoder reranker score thresholds (negative scores, more negative =
    # more relevant). Defined in search_response_formatter, which derives the
    # relevance-space confidence bands from the same values.
    LOW_CONFIDENCE_THRESHOLD = _formatter.LOW_CONFIDENCE_THRESHOLD
    NOISE_THRESHOLD = _formatter.NOISE_THRESHOLD

    def __init__(self, collection_name, indexer, persister, reranker=None):
        self.collection_name = collection_name
        self.indexer = indexer
        self.persister = persister
        self.reranker = reranker
        self._doc_cache = {}
        # Load the index→document mapping once and pair it with the in-memory
        # index for this searcher's lifetime. A background update rewrites the
        # mapping on disk with a new chunk-id range; re-reading it per search
        # would desync it from the frozen in-memory index and yield stale or
        # blanked-out results mid-update. reload_collection builds a fresh
        # searcher, so a new (index, mapping) pair is swapped in atomically.
        self._mapping = self._load_mapping()

    def search(self, text,
               max_number_of_chunks=15,
               max_number_of_documents=None,
               include_text_content=False,
               include_all_chunks_content=False,
               include_matched_chunks_content=False,
               skip_reranker=False,
               trace=None,
               title_boost_query=None):
        """Search the collection.

        title_boost_query: query string used for title-boost token matching.
            Defaults to `text`. Pass the raw user query when `text` is graph-expanded
            so title-boost doesn't reward documents whose titles happen to overlap
            with expansion terms instead of the user's actual intent.
        """
        t_start = time.monotonic()
        self._doc_cache = {}

        if trace is None:
            trace = NULL_TRACE
        if title_boost_query is None:
            title_boost_query = text

        skip_reason = self._reranker_skip_reason(text, skip_reranker)
        use_reranker = skip_reason is None
        if skip_reason:
            trace.set_reranker_skipped(True, reason=skip_reason)

        # Overfetch to compensate for dedup/confidence filtering reducing result count
        dedup_buffer = max(3, max_number_of_chunks // 3)
        effective_chunks = max_number_of_chunks + dedup_buffer

        fetch_k = int(effective_chunks * 1.5) if use_reranker else effective_chunks
        coll_trace = trace.start_collection(
            name=self.collection_name,
            indexer=self.indexer.get_name(),
            fetch_k=fetch_k,
        )

        capture_breakdown = coll_trace.enabled and getattr(self.indexer, "supports_breakdown", False)

        t0 = time.monotonic()
        if capture_breakdown:
            scores, indexes, breakdown = self.indexer.search(text, fetch_k, return_breakdown=True)
            self._record_index_breakdown(coll_trace, breakdown)
        else:
            scores, indexes = self.indexer.search(text, fetch_k)
        t_index = time.monotonic()

        if use_reranker:
            chunk_texts = self._get_chunk_texts(indexes)
            t_chunks = time.monotonic()
            if coll_trace.enabled:
                scores, indexes, ce_breakdown = self.reranker.rerank(
                    text, scores, indexes, chunk_texts, effective_chunks, return_ce_scores=True
                )
                for rank, (chunk_id, ce_score) in enumerate(ce_breakdown):
                    coll_trace.record_stage("ce", chunk_id=chunk_id, rank=rank, score=ce_score)
            else:
                scores, indexes = self.reranker.rerank(text, scores, indexes, chunk_texts, effective_chunks)
            t_rerank = time.monotonic()
            logger.info(
                f"Search '{self.collection_name}' ({len(chunk_texts)} candidates): "
                f"index={delta_ms(t0, t_index)}ms, chunks={delta_ms(t_index, t_chunks)}ms, "
                f"rerank={delta_ms(t_chunks, t_rerank)}ms"
            )
        else:
            t_chunks = t_index
            t_rerank = t_index
            logger.info(f"Search '{self.collection_name}' (no rerank): index={delta_ms(t0, t_index)}ms")

        scores, indexes = self._apply_title_boost(title_boost_query, scores, indexes, coll_trace)
        t_boost = time.monotonic()

        if coll_trace.enabled:
            self._record_final_and_annotate(coll_trace, scores, indexes)

        results = self.__build_results(scores, indexes, include_text_content, include_all_chunks_content, include_matched_chunks_content)
        if max_number_of_documents:
            results = results[:max_number_of_documents]

        response = {
            "collectionName": self.collection_name,
            "indexerName": self.indexer.get_name(),
            "results": results,
            "reranked": use_reranker,
        }

        results_before_filter = len(results)
        if use_reranker and results:
            response = self._apply_confidence_filtering(response)
        if coll_trace.enabled:
            filtered = response["results"]
            best = self._best_chunk_score(filtered[0]) if filtered else None
            coll_trace.set_confidence(
                low_confidence=response.get("lowConfidence", False),
                best_score=best,
                low_confidence_threshold=self.LOW_CONFIDENCE_THRESHOLD,
                noise_threshold=self.NOISE_THRESHOLD,
                filtered_count=results_before_filter - len(filtered),
            )

        t_end = time.monotonic()
        coll_trace.set_timings(
            indexFetch=delta_ms(t0, t_index),
            chunkLoad=delta_ms(t_index, t_chunks),
            rerank=delta_ms(t_chunks, t_rerank),
            titleBoost=delta_ms(t_rerank, t_boost),
            assembly=delta_ms(t_boost, t_end),
            total=delta_ms(t_start, t_end),
        )

        logger.info(f"Search '{self.collection_name}' total: {delta_ms(t_start, t_end)}ms")
        return response

    def _reranker_skip_reason(self, text, caller_skip_reranker):
        if not self.reranker:
            return "no_reranker"
        if caller_skip_reranker:
            return "caller_opted_out"
        if self._should_skip_reranker(text):
            return "english_query"
        return None

    @staticmethod
    def _record_index_breakdown(coll_trace, breakdown):
        for chunk_id, rank, score in breakdown.get("faiss", []):
            coll_trace.record_stage("faiss", chunk_id=chunk_id, rank=rank, score=score)
        for chunk_id, rank, score in breakdown.get("bm25", []):
            coll_trace.record_stage("bm25", chunk_id=chunk_id, rank=rank, score=score)
        for chunk_id, rank, score in breakdown.get("rrf", []):
            coll_trace.record_stage("rrf", chunk_id=chunk_id, rank=rank, score=score)

    def _record_final_and_annotate(self, coll_trace, scores, indexes):
        mapping = self._mapping
        for rank, chunk_id in enumerate(indexes[0]):
            cid = int(chunk_id)
            coll_trace.record_stage("final", chunk_id=cid, rank=rank, score=float(scores[0][rank]))
            entry = mapping.get(str(cid))
            if entry:
                coll_trace.annotate_candidate(
                    cid,
                    document_id=entry["documentId"],
                    doc_title=title_from_doc_path(entry.get("documentPath", "")),
                )

    def _apply_confidence_filtering(self, response):
        results = response["results"]

        # Filter out documents where all matched chunks are noise
        filtered = [
            doc for doc in results
            if self._best_chunk_score(doc) <= self.NOISE_THRESHOLD
        ]
        response["results"] = filtered

        # Flag response as low confidence if best remaining result is weak
        if not filtered or self._best_chunk_score(filtered[0]) > self.LOW_CONFIDENCE_THRESHOLD:
            response["lowConfidence"] = True

        return response

    @staticmethod
    def _best_chunk_score(doc):
        return min(chunk["score"] for chunk in doc["matchedChunks"])

    def _apply_title_boost(self, query, scores, indexes, coll_trace=None):
        """Boost scores for documents whose title matches query terms.

        Boost magnitude scales with the score spread so it works across
        different score types (cross-encoder, hybrid RRF, FAISS L2).
        """
        mapping = self._mapping
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

    def _should_skip_reranker(self, query):
        """Skip reranker for English queries (cross-lingual score collapse)."""
        if not _langdetect_available:
            return False
        words = query.split()
        if len(words) < 3:
            return False
        try:
            lang = detect(query)
            if lang == 'en':
                logger.info(f"Skipping reranker for English query: {query[:50]}")
                return True
        except Exception:
            pass
        return False

    def _load_mapping(self):
        """Read the index-to-document mapping from disk.

        Called once from __init__; the result is held in self._mapping for the
        searcher's lifetime so it stays consistent with the in-memory index.
        """
        indexes_base_path = f"{self.collection_name}/indexes"
        return json.loads(
            self.persister.read_text_file(f"{indexes_base_path}/index_document_mapping.json")
        )

    def _get_chunk_texts(self, indexes):
        """Look up chunk text for each candidate index."""
        mapping = self._mapping

        chunk_texts = []

        for chunk_id in indexes[0]:
            chunk_id_str = str(int(chunk_id))
            entry = mapping.get(chunk_id_str)
            if not entry:
                chunk_texts.append("")
                continue

            doc = self._get_document_cached(entry["documentPath"])
            chunk_number = entry["chunkNumber"]
            if doc and "chunks" in doc and chunk_number < len(doc["chunks"]):
                chunk = doc["chunks"][chunk_number]
                if isinstance(chunk, dict):
                    chunk_texts.append(chunk.get("indexedData", str(chunk)))
                else:
                    chunk_texts.append(str(chunk))
            else:
                chunk_texts.append("")

        return chunk_texts

    def _get_document_cached(self, document_path):
        """Read and cache a document JSON file. Cache lives for one search call."""
        if document_path not in self._doc_cache:
            try:
                self._doc_cache[document_path] = json.loads(self.persister.read_text_file(document_path))
            except Exception as e:
                logger.warning(f"Failed to read document {document_path}: {e}")
                self._doc_cache[document_path] = None

        return self._doc_cache[document_path]

    def __build_results(self, scores, indexes, include_text_content, include_all_chunks_content, include_matched_chunks_content):
        index_document_mapping = self._mapping

        result = {}
        seen_text_hashes = {}  # text_hash -> documentId (first seen)
        seen_urls = {}  # url -> documentId (first seen)
        skipped_doc_ids = set()

        for result_number in range(0, len(indexes[0])):
            chunk_id_str = str(int(indexes[0][result_number]))
            mapping = index_document_mapping.get(chunk_id_str)
            if not mapping:
                logger.warning(f"Missing mapping for chunk index {chunk_id_str}, skipping")
                continue
            doc_id = mapping["documentId"]

            # Skip chunks from already-deduplicated documents
            if doc_id in skipped_doc_ids:
                continue

            if doc_id not in result:
                doc_url = mapping.get("documentUrl", "")

                def _load_text():
                    document = self._get_document_cached(mapping["documentPath"])
                    return document.get("text", "") if document else ""

                if deduplicate_document(doc_id, doc_url, _load_text, seen_urls, seen_text_hashes):
                    skipped_doc_ids.add(doc_id)
                    continue

                # Cached from ``_load_text`` when it ran; a URL-only match never
                # reads the document, so fetch it here (still cached) for the
                # fields below.
                document = self._get_document_cached(mapping["documentPath"])

                doc_result = {
                    "id": doc_id,
                    "url": mapping["documentUrl"],
                    "path": mapping["documentPath"],
                    "matchedChunks": [self.__build_chunk_result(mapping, scores, result_number, include_matched_chunks_content)]
                }

                if document and document.get("modifiedTime"):
                    doc_result["modifiedTime"] = document["modifiedTime"]

                result[doc_id] = doc_result

                if document and (include_all_chunks_content or include_text_content):
                    if include_all_chunks_content:
                        result[doc_id]["allChunks"] = document.get("chunks", [])

                    if include_text_content:
                        result[doc_id]["text"] = document.get("text", "")

            else:
                result[doc_id]["matchedChunks"].append(self.__build_chunk_result(mapping, scores, result_number, include_matched_chunks_content))

        return list(result.values())

    def __build_chunk_result(self, mapping, scores, result_number, include_matched_chunks_content):
        return {
            "chunkNumber": mapping["chunkNumber"],
            "score":  float(scores[0][result_number]),
            **({ "content": self._get_chunk_content(mapping) } if include_matched_chunks_content else {})
        }

    def _get_chunk_content(self, mapping):
        """Return the matched chunk's content, or "" if the document is unreadable.

        Guards against a missing/corrupt/locked document JSON (cached as None),
        an absent "chunks" array, or a chunkNumber out of range — any of which
        would otherwise crash the whole search response on the default
        include_matched_chunks_content path.
        """
        document = self._get_document_cached(mapping["documentPath"])
        if not document:
            return ""
        chunks = document.get("chunks")
        chunk_number = mapping["chunkNumber"]
        if not chunks or chunk_number >= len(chunks):
            logger.warning(
                f"Chunk {chunk_number} unavailable for {mapping['documentPath']} "
                f"({len(chunks) if chunks else 0} chunks); returning empty content"
            )
            return ""
        return chunks[chunk_number]
