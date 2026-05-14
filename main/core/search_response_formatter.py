"""Search response shaping — extract chunk fields, score normalization, snippet truncation, metadata filters.

Used by knowledge_api_server.py /api/search to convert raw DocumentCollectionSearcher
output into the public response format. Stateless functions; safe to share across runtimes.
"""
import math
import re

from main.utils.filename import title_from_doc_path


# Cap for non-reranked results: without cross-encoder validation we can't
# claim high confidence, so rank-based relevance is bounded.
NON_RERANKED_MAX_RELEVANCE = 0.75

# Relevance-space confidence bands. 0.40 ≈ normalize_score(-0.10) — the
# LOW_CONFIDENCE_THRESHOLD the searcher uses to flag weak responses, so a "low"
# band is the reranker's own noise zone. 0.65 ≈ normalize_score(-0.23),
# comfortably past that floor.
HIGH_CONFIDENCE_RELEVANCE = 0.65
MEDIUM_CONFIDENCE_RELEVANCE = 0.40

# A response whose best result is below this is "weak": callers should treat it
# as a corrective-retry signal even when some results came back.
WEAK_RESULT_RELEVANCE = 0.45

_METADATA_LINE_RE = re.compile(r'^\*\*([^*]+?):\*\*\s*(.+)$')


def confidence_band(relevance, is_reranked=True):
    """Bucket a 0.0–1.0 relevance into ``'high'`` | ``'medium'`` | ``'low'``.

    Non-reranked results carry rank-based relevance — an ordering hint, not a
    confidence estimate — so they never earn a ``'high'`` band.
    """
    if not is_reranked:
        return "medium" if relevance >= 0.5 else "low"
    if relevance >= HIGH_CONFIDENCE_RELEVANCE:
        return "high"
    if relevance >= MEDIUM_CONFIDENCE_RELEVANCE:
        return "medium"
    return "low"


def extract_chunk_text(content):
    """Extract plain text from chunk content (may be dict with indexedData or plain string)."""
    if isinstance(content, dict):
        return content.get("indexedData", str(content))
    return str(content) if content else ""


def extract_chunk_heading(content):
    """Extract heading from chunk content if available."""
    if isinstance(content, dict):
        return content.get("heading")
    return None


def extract_chunk_metadata(content):
    """Extract metadata dict from chunk content if available."""
    if isinstance(content, dict):
        return content.get("metadata")
    return None


def truncate_snippet(text, target=200):
    """Truncate text at a sentence boundary near target length, falling back to word boundary."""
    if not text or len(text) <= target:
        return text
    window_start = max(target - 40, 0)
    window_end = min(target + 40, len(text))
    window = text[window_start:window_end]
    best = -1
    for m in re.finditer(r'[.!?]\s', window):
        best = m.start() + 1
    if best >= 0:
        cut = window_start + best
        return text[:cut].rstrip()
    cut = text.rfind(' ', 0, target + 20)
    if cut > target - 40:
        return text[:cut].rstrip() + "…"
    return text[:target] + "…"


def separate_metadata(text):
    """Parse **Key:** Value lines from start of text into a metadata dict.

    Also extracts [Breadcrumb > Path] line for navigation context.
    Returns (clean_content, metadata_dict, breadcrumb_or_None).
    """
    if not text:
        return "", {}, None
    # Fast path: most chunks (markdown body text) have no metadata prefix.
    # Skip the per-line scan when the first non-blank char rules it out.
    first_non_blank = text.lstrip()
    if not first_non_blank.startswith(("**", "[")):
        return text.strip(), {}, None
    lines = text.split('\n')
    metadata = {}
    breadcrumb = None
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            content_start = i + 1
            continue
        if stripped.startswith('[') and '>' in stripped and stripped.endswith(']'):
            breadcrumb = stripped[1:-1].strip()
            content_start = i + 1
            continue
        m = _METADATA_LINE_RE.match(stripped)
        if m:
            metadata[m.group(1).strip()] = m.group(2).strip()
            content_start = i + 1
        else:
            break
    clean = '\n'.join(lines[content_start:]).strip()
    return clean, metadata, breadcrumb


def apply_metadata_filters(results, project=None, git_branch=None, tags=None):
    """Filter results by metadata fields. Checks document-level and chunk-level metadata."""
    filtered = []
    requested_tags = {t.strip() for t in tags.split(",") if t.strip()} if tags else None
    for r in results:
        doc_meta = r.get("metadata") or {}
        chunk_meta = {}
        for chunk in r.get("matchedChunks", []):
            if chunk.get("metadata"):
                chunk_meta.update(chunk["metadata"])
        merged = {**doc_meta, **chunk_meta}

        if project and merged.get("project") != project:
            continue
        if git_branch and merged.get("gitBranch") != git_branch:
            continue
        if requested_tags:
            doc_tags = {t.strip() for t in merged.get("tags", "").split(",") if t.strip()}
            if not requested_tags & doc_tags:
                continue
        filtered.append(r)
    return filtered


def normalize_score(raw_score, is_reranked=True):
    """Convert internal score (lower=better) to 0.0-1.0 relevance (higher=better).

    For reranked results: shifted sigmoid calibrated to cross-encoder score range.
    Maps score -1.0 → ~0.999, -0.5 → ~0.94, -0.15 → ~0.50, -0.01 → ~0.25.

    For non-reranked results: placeholder — search handler overrides with rank-based
    relevance (NON_RERANKED_MAX_RELEVANCE-bounded).
    """
    if not is_reranked:
        return 0.5

    shifted = (raw_score + 0.15) * 8
    clamped = max(min(shifted, 500), -500)
    return 1.0 / (1.0 + math.exp(clamped))


def _shape_doc(doc, coll_name, is_reranked, brief, max_chunk_chars, max_chunks_per_doc):
    """Shape a single searcher.search() document into the public response format.

    Returns the result dict (with internal _score/_reranked fields), or None if
    the document has no matched chunks.
    """
    matched_chunks = []
    for chunk in doc.get("matchedChunks", []):
        raw = chunk.get("content", "")
        entry = {
            "content": extract_chunk_text(raw),
            "score": chunk.get("score", 0),
            "heading": extract_chunk_heading(raw),
        }
        chunk_meta = extract_chunk_metadata(raw)
        if chunk_meta:
            entry["metadata"] = chunk_meta
        matched_chunks.append(entry)
    if not matched_chunks:
        return None

    matched_chunks.sort(key=lambda c: c["score"])
    matched_chunks = matched_chunks[:max_chunks_per_doc]

    title = title_from_doc_path(doc.get("path", ""))
    url = doc.get("url", "")
    modified_time = doc.get("modifiedTime")
    best_score = matched_chunks[0]["score"]
    relevance = normalize_score(best_score, is_reranked)

    doc_breadcrumb = None
    for chunk in matched_chunks:
        clean_content, text_metadata, breadcrumb = separate_metadata(chunk["content"])
        chunk["content"] = clean_content
        chunk_existing = chunk.get("metadata")
        if text_metadata and chunk_existing:
            chunk["metadata"] = {**chunk_existing, **text_metadata}
        elif text_metadata:
            chunk["metadata"] = text_metadata
        if breadcrumb and not doc_breadcrumb:
            doc_breadcrumb = breadcrumb

    if brief:
        best_chunk = matched_chunks[0]
        snippet = truncate_snippet(best_chunk["content"])
        if not snippet and best_chunk.get("metadata"):
            snippet = " | ".join(f"{k}: {v}" for k, v in best_chunk["metadata"].items())
        result = {
            "collection": coll_name,
            "id": doc.get("id"),
            "title": title,
            "url": url,
            "snippet": snippet,
            "relevance": round(relevance, 3),
            "_score": best_score,
            "_reranked": is_reranked,
        }
        if modified_time:
            result["modifiedTime"] = modified_time
        if doc_breadcrumb:
            result["breadcrumb"] = doc_breadcrumb
        if best_chunk.get("heading"):
            result["heading"] = best_chunk["heading"]
        if best_chunk.get("metadata"):
            result["metadata"] = best_chunk["metadata"]
        return result

    if max_chunk_chars is not None:
        for chunk in matched_chunks:
            if len(chunk["content"]) > max_chunk_chars:
                chunk["content"] = chunk["content"][:max_chunk_chars] + "…"
    for chunk in matched_chunks:
        chunk["relevance"] = round(normalize_score(chunk["score"], is_reranked), 3)
    result = {
        "collection": coll_name,
        "id": doc.get("id"),
        "title": title,
        "url": url,
        "relevance": round(relevance, 3),
        "matchedChunks": matched_chunks,
        "_score": best_score,
        "_reranked": is_reranked,
    }
    if modified_time:
        result["modifiedTime"] = modified_time
    if doc_breadcrumb:
        result["breadcrumb"] = doc_breadcrumb
    best_meta = matched_chunks[0].get("metadata") if matched_chunks else None
    if best_meta:
        result["metadata"] = best_meta
    return result


def shape_search_results(
    per_collection_results,
    *,
    limit,
    brief=False,
    max_chunk_chars=None,
    max_chunks_per_doc=3,
    project=None,
    git_branch=None,
    tags=None,
):
    """Shape (collection_name, search_result) pairs into the public API results list.

    Performs per-document chunk shaping, metadata filtering, score sorting,
    rank-relevance override for non-reranked results, ``confidenceBand``
    tagging, and internal-field cleanup. Returns (results_capped_at_limit,
    any_low_confidence).

    Graph augmentation (query expansion before, per-result context after) is
    the caller's concern — this function only shapes raw search output.
    """
    all_results = []
    any_low_confidence = False
    for coll_name, search_result in per_collection_results:
        if search_result.get("lowConfidence"):
            any_low_confidence = True
        is_reranked = search_result.get("reranked", True)
        for doc in search_result.get("results", []):
            shaped = _shape_doc(doc, coll_name, is_reranked, brief, max_chunk_chars, max_chunks_per_doc)
            if shaped is not None:
                all_results.append(shaped)

    if project or git_branch or tags:
        all_results = apply_metadata_filters(all_results, project=project, git_branch=git_branch, tags=tags)

    # Sort by best chunk score (lower = better: L2 distance for FAISS, negated RRF for hybrid)
    all_results.sort(key=lambda r: r["_score"])

    # Override relevance for non-reranked results with rank-based scoring
    # (absolute hybrid/FAISS scores aren't meaningful as relevance values)
    for i, r in enumerate(all_results[:limit]):
        if not r.get("_reranked"):
            rank_relevance = round(NON_RERANKED_MAX_RELEVANCE / (1.0 + 0.12 * i), 3)
            r["relevance"] = rank_relevance
            for j, chunk in enumerate(r.get("matchedChunks", [])):
                chunk["relevance"] = round(max(0.1, rank_relevance * (1.0 - 0.1 * j)), 3)

    top = all_results[:limit]
    for r in top:
        r["confidenceBand"] = confidence_band(r["relevance"], r.get("_reranked", True))
        r.pop("_score", None)
        r.pop("_reranked", None)
        for chunk in r.get("matchedChunks", []):
            chunk.pop("score", None)
    return top, any_low_confidence


def _compute_corrective_signal(
    results,
    *,
    query,
    augmenter,
    detected_entities,
    min_relevance,
):
    """Pure signal computation — no trace writes. Returns the underlying state
    both ``apply_corrective_signal`` (for response shaping + trace recording)
    and ``run_corrective_search`` (for the rescue decision) work from.

    ``bestScore`` is captured **before** the ``min_relevance`` filter so callers
    can tell "found something below your bar" from "found nothing".
    """
    best_score = results[0]["relevance"] if results else 0.0
    dropped_by_min_relevance = 0
    if min_relevance is not None:
        kept = [r for r in results if r["relevance"] >= min_relevance]
        dropped_by_min_relevance = len(results) - len(kept)
        results = kept

    no_confident_results = not results
    weak = no_confident_results or best_score < WEAK_RESULT_RELEVANCE
    retry_hints = augmenter.get_retry_hints(query, detected_entities) if weak else None

    return {
        "results": results,
        "best_score": best_score,
        "weak": weak,
        "no_confident_results": no_confident_results,
        "retry_hints": retry_hints,
        "dropped_by_min_relevance": dropped_by_min_relevance,
    }


def _finalize_signal(sig, *, trace, reranked):
    """Build the response dict + write the trace from an already-computed
    ``_compute_corrective_signal`` result. ``run_corrective_search`` calls this
    directly to avoid recomputing the signal it already has."""
    response = {
        "results": sig["results"],
        "bestScore": round(sig["best_score"], 3),
        "reranked": bool(reranked),
    }
    if sig["no_confident_results"]:
        response["noConfidentResults"] = True
    if sig["retry_hints"]:
        response["retryHints"] = sig["retry_hints"]

    trace.set_response_meta(
        best_score=round(sig["best_score"], 3),
        no_confident_results=sig["no_confident_results"],
        retry_hints=sig["retry_hints"],
        dropped_by_min_relevance=sig["dropped_by_min_relevance"],
        reranked=bool(reranked),
    )
    return sig["results"], response


def apply_corrective_signal(
    results,
    *,
    query,
    augmenter,
    detected_entities,
    min_relevance,
    trace,
    reranked=True,
):
    """Filter by ``min_relevance``, compute the corrective-signal fields, record on the trace.

    Returns ``(kept_results, response)`` — the caller merges any additional
    response keys (``graph_answer``, ``lowConfidence``, trace) into the dict.
    ``augmenter`` is duck-typed to anything exposing ``get_retry_hints``;
    ``trace`` to anything exposing ``set_response_meta`` (the null trace is
    fine). ``reranked`` is the top-level honest signal for whether the reranker
    ran; when False, ``bestScore`` is rank-based (top hit = 0.75) and not a
    real confidence estimate — consumers should lean on ``noConfidentResults``
    or inspect snippets instead of gating on ``bestScore`` alone.
    """
    sig = _compute_corrective_signal(
        results,
        query=query,
        augmenter=augmenter,
        detected_entities=detected_entities,
        min_relevance=min_relevance,
    )
    return _finalize_signal(sig, trace=trace, reranked=reranked)


def merge_search_results(originals, rescue, *, limit=None):
    """Merge two shaped-result lists. Dedupe by ``(collection, id)``;
    rescue wins on the dedupe key (it's the "more confident" set by
    construction). Re-sort by ``relevance`` descending — the relevance scale is
    already calibrated and ``_score`` was stripped by ``shape_search_results``.
    """
    seen: set = set()
    merged: list = []
    for r in rescue:
        key = (r.get("collection"), r.get("id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)
    for r in originals:
        key = (r.get("collection"), r.get("id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)
    merged.sort(key=lambda r: r.get("relevance", 0.0), reverse=True)
    if limit is not None:
        merged = merged[:limit]
    return merged


def run_corrective_search(
    initial_results,
    *,
    query,
    augmenter,
    detected_entities,
    min_relevance,
    trace,
    reranked,
    mode="auto",
    rerun_search_fn=None,
    limit=None,
):
    """Run the corrective signal; when weak, optionally re-search with a hint
    query and merge results.

    ``mode``:
        - ``"off"`` — pure delegation to ``apply_corrective_signal``; response
          shape and trace contents are byte-identical to the pre-Path-D world.
        - ``"auto"`` — rescue iff the signal is weak *and* a usable hint exists.
        - ``"force"`` — rescue iff a hint exists, regardless of weakness
          (test/debug knob).

    ``rerun_search_fn(rescue_query)`` returns a list of shaped + enriched
    result dicts in the same shape ``shape_search_results`` produces. Required
    when ``mode != "off"``; when omitted, ``run_corrective_search`` falls back
    to no-rescue behaviour (verdict ``"weak_skipped"``).

    On rescue, the response gains a ``corrective`` dict with
    ``{mode, retries, rescueFired, verdict, queriesTried}``; the same dict is
    recorded on the trace. No ``corrective`` field is added to the response in
    the no-rescue case (keeps the response shape unchanged for confident
    queries — backward-compatible), but the trace still records it for
    dashboards.
    """
    if mode == "off":
        return apply_corrective_signal(
            initial_results,
            query=query,
            augmenter=augmenter,
            detected_entities=detected_entities,
            min_relevance=min_relevance,
            trace=trace,
            reranked=reranked,
        )

    sig = _compute_corrective_signal(
        initial_results,
        query=query,
        augmenter=augmenter,
        detected_entities=detected_entities,
        min_relevance=min_relevance,
    )
    hints = sig["retry_hints"] or {}
    weak = sig["weak"]
    # Force mode rescues even on confident results, so ``_compute_corrective_signal``'s
    # weak-only hint shortcut won't have populated them — fetch directly.
    if mode == "force" and not hints:
        hints = augmenter.get_retry_hints(query, detected_entities) or {}
    rescue_query = hints.get("broaderQuery") or hints.get("narrowerQuery")

    will_rescue = (
        rerun_search_fn is not None
        and bool(rescue_query)
        and (mode == "force" or (mode == "auto" and weak))
    )

    queries_tried = [query]

    if not will_rescue:
        if weak and not rescue_query:
            verdict = "weak_no_hint"
        elif weak:
            verdict = "weak_skipped"
        else:
            verdict = "confident"
        corrective_meta = {
            "mode": mode,
            "retries": 0,
            "rescueFired": False,
            "verdict": verdict,
            "queriesTried": queries_tried,
        }
        results, response = _finalize_signal(sig, trace=trace, reranked=reranked)
        trace.set_corrective(corrective_meta)
        return results, response

    queries_tried.append(rescue_query)
    rescue_shaped = rerun_search_fn(rescue_query)
    merged = merge_search_results(initial_results, rescue_shaped, limit=limit)

    sig2 = _compute_corrective_signal(
        merged,
        query=query,
        augmenter=augmenter,
        detected_entities=detected_entities,
        min_relevance=min_relevance,
    )
    verdict = "rescued" if not sig2["weak"] else "still_weak"

    corrective_meta = {
        "mode": mode,
        "retries": 1,
        "rescueFired": True,
        "verdict": verdict,
        "queriesTried": queries_tried,
    }
    results, response = _finalize_signal(sig2, trace=trace, reranked=reranked)
    trace.set_corrective(corrective_meta)
    response["corrective"] = corrective_meta
    return results, response
