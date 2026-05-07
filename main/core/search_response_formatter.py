"""Search response shaping — extract chunk fields, score normalization, snippet truncation, metadata filters.

Used by knowledge_api_server.py /api/search to convert raw DocumentCollectionSearcher
output into the public response format. Stateless functions; safe to share across runtimes.
"""
import math
import re


# Cap for non-reranked results: without cross-encoder validation we can't
# claim high confidence, so rank-based relevance is bounded.
NON_RERANKED_MAX_RELEVANCE = 0.75

_METADATA_LINE_RE = re.compile(r'^\*\*([^*]+?):\*\*\s*(.+)$')


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
