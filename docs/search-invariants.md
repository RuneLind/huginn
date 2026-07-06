# Search-path invariants

Load-bearing conventions in the search pipeline. None of these are enforced by
types — they hold by convention across several files, so know them before
changing scoring, response shaping, or collection reloading.

## 1. Lower score = better, everywhere inside the pipeline

L2 distances, negated RRF scores, and cross-encoder scores are all treated as
ascending-sort throughout the internal path (`DocumentCollectionSearcher`,
result merging, chunk sorting). Conversion to a 0.0–1.0 relevance
(higher = better) happens **only** at the formatter boundary
(`normalize_score` in `main/core/search_response_formatter.py`). A single sign
error anywhere in the chain reverses ranking silently — there is no assertion
that catches it.

## 2. `reranked` is a calibration honesty contract, not a flag

When the cross-encoder didn't run, results carry rank-based relevance — an
ordering hint, not a confidence estimate. The formatter therefore caps
non-reranked relevance at `NON_RERANKED_MAX_RELEVANCE` (0.75), overrides
per-result relevance by rank, and `confidence_band` never returns `"high"` for
non-reranked results. Consumers (notably Muninn) depend on this: `bestScore`
from a non-reranked response must not be read as a real confidence estimate.
Don't "fix" the cap.

## 3. Confidence thresholds have one source of truth

The raw cross-encoder thresholds (`LOW_CONFIDENCE_THRESHOLD`,
`NOISE_THRESHOLD`, `HIGH_CONFIDENCE_RAW_SCORE`) live in
`search_response_formatter.py`; the relevance-space band constants
(`HIGH_CONFIDENCE_RELEVANCE`, `MEDIUM_CONFIDENCE_RELEVANCE`) are **derived**
from them via `normalize_score` at import time, and
`DocumentCollectionSearcher` imports the raw thresholds for its filtering
policy. Change the thresholds or the sigmoid in one place and both layers move
together — don't reintroduce hardcoded copies.

## 4. Searcher mappings are frozen per lifetime, swapped atomically

`DocumentCollectionSearcher` loads the index→document mapping once in
`__init__` and pairs it with the in-memory index for the searcher's lifetime.
A background collection update rewrites the mapping on disk with a new
chunk-id range; re-reading it per search would desync it from the frozen
in-memory index mid-update. `KnowledgeStore.reload_collection` builds a fresh
searcher and swaps the `(index, mapping)` pair in atomically under the store
lock. This is the decision that makes background rebuilds safe against
in-flight searches — never add a "refresh mapping" shortcut to a live
searcher.

## 5. The four search entry points are intentionally not one path

- **HTTP API** (`main/routes/search.py`) and **in-process MCP**
  (`main/core/mcp_search_tool.py`, used by the stdio adapters) share the
  `search_and_shape` orchestration in `main/core/search_pipeline.py`.
- **`knowledge_api_mcp_adapter.py`** is a thin httpx *client* over
  `/api/search` — it renders, it doesn't search.
- **`collection_search_cmd_adapter.py`** deliberately bypasses shaping, graph
  augmentation, and corrective retry: it dumps raw `searcher.search()` output
  as a debug path.

"Consolidate all entry points onto one pipeline" is a non-goal; only the two
in-process paths are meant to converge.
