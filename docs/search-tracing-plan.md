# Search Tracing — Plan

Status: Phase 1 implemented (opt-in flat trace blob); see "Rollout" for next steps
Author: Rune (drafted with Claude)
Last updated: 2026-04-29

## Why

Today a Huginn search is a black box from the caller's side. We log stage timings server-side, but callers (Muninn, CLI users, evals) only see the final ranked list. That makes it hard to:

- Understand *why* a chunk ranked where it did (FAISS vs BM25 vs RRF vs cross-encoder vs title boost contributions).
- Tell whether graph expansion helped, hurt, or did nothing on a given query.
- Debug "why didn't this obvious doc come back" without re-running with extra logging.
- Compare runs across changes (new model, tuned RRF k, new graph) with apples-to-apples per-query data.

Endpoint goal: when Muninn calls Huginn, the operator opens the tool span in Muninn's waterfall UI and sees the search internals — the same way they see Claude's tool calls today. Longer term, this is the ground truth for measuring whether changes to graph expansion / reranking / weighting actually improve EØS-task analysis.

## Scope

**In scope (Phase 1):** flat trace blob in the search response, off by default, opt-in per request. No new tables, no streaming, no UI work in Huginn. Muninn renders it via its existing span-detail JSON viewer.

**Out of scope for now:** nested span emission, persistent trace store, eval harness integration, automatic regression detection. Listed under "Future improvements" below.

## Trace schema (Phase 1)

A single optional `trace` object on the `/api/search` response. Field names use camelCase to match the rest of the API envelope.

```jsonc
{
  "results": [...],
  "graph_answer": "...",
  "lowConfidence": false,
  "trace": {
    "query": {
      "raw": "hva er LA_BUC_02",
      "expanded": "hva er LA_BUC_02 A003 A004 A009 A010",
      "detectedEntities": [
        { "id": "BUC:LA_BUC_02", "type": "BUC", "label": "LA_BUC_02", "matchedSpan": "LA_BUC_02" }
      ],
      "expansionTerms": ["A003", "A004", "A009", "A010"],
      "graphAnswered": false,
      "rerankerSkipped": false,
      "rerankerSkipReason": null
    },
    "collections": [
      {
        "name": "melosys-confluence-v3",
        "indexer": "hybrid",
        "fetchK": 22,
        "candidates": [
          {
            "chunkId": 18472,
            "documentId": "doc-…",
            "docTitle": "LA_BUC_02 oversikt",
            "headings": ["…"],
            "stages": {
              "faiss":  { "rank": 3,  "score": 0.412 },
              "bm25":   { "rank": 1,  "score": 8.91 },
              "rrf":    { "rank": 1,  "score": -0.0331 },
              "ce":     { "rank": 2,  "score": 4.21 },
              "titleBoost": { "applied": true, "delta": -0.18 },
              "final":  { "rank": 1,  "score": -0.20 }
            },
            "kept": true,
            "dropReason": null
          }
        ],
        "confidence": {
          "lowConfidence": false,
          "bestScore": -0.20,
          "lowConfidenceThreshold": -0.10,
          "noiseThreshold": -0.01,
          "filteredCount": 2
        },
        "timingsMs": {
          "indexFetch": 14,
          "chunkLoad":  3,
          "rerank":    41,
          "titleBoost": 1,
          "assembly":   2,
          "total":     63
        }
      }
    ],
    "totalMs": 71,
    "schemaVersion": 1
  }
}
```

Notes:

- `candidates` is bounded — we only emit chunks that survived to at least the RRF stage, capped at `fetchK` (~25 for default queries). Worst case ~3 KB per collection.
- `stages.<x>.score` is the raw value from that stage in the convention each stage uses (FAISS L2: lower better, BM25: higher better, RRF: negated lower better, CE: cross-encoder logit higher better). `stages.final.score` is the post-title-boost value going into result assembly.
- `dropReason` is one of: `"noise"`, `"dedup"`, `"missingDoc"`, `"perDocCap"`, `"metadataFilter"`, `null`.
- `schemaVersion` so Muninn can tolerate field changes without breaking.

### Opt-in surface

- HTTP: new query param `trace=true` (default false).
- MCP: new optional bool argument `trace` on the `search_knowledge` tool.
- Env: `HUGINN_TRACE_DEFAULT=true` to flip the default for a dev server (so local debugging doesn't need to set it every call).

When `trace=false`, zero overhead — the candidate-tracking dict isn't allocated.

## Insertion points

Concrete edits, organized by file. Line numbers are from the working copy on `main` at the time of writing; treat as approximate.

### 1. `knowledge_api_server.py:336` — accept the param + own the top-level trace

- Add `trace: bool = Query(False, ...)` to the `search()` signature.
- Initialize `trace_obj = {"query": {...}, "collections": [], "schemaVersion": 1}` if `trace` is true; otherwise pass `None` down.
- After the graph block at `:360–371`, fill `trace_obj["query"]` (raw, expanded, detectedEntities, expansionTerms, graphAnswered).
- Pass `trace=trace_obj` (or `None`) into `searcher.search(...)` at `:376`. Each searcher appends its own collection-level dict.
- At the end (around `:517`), if `trace_obj` is set, attach it to the response and stamp `totalMs`.

### 2. `main/core/documents_collection_searcher.py:36` — collect per-stage data

The existing `search()` is the right seam. Add an optional `trace=None` parameter and thread a `_TraceRecorder` helper through the call.

- At `:56` after `self.indexer.search(...)`, if the indexer is the hybrid one, ask it for its per-stage breakdown (see #3). Populate `candidate.stages.faiss/bm25/rrf` using the chunk ids returned.
- At `:60` after `reranker.rerank(...)`, capture CE scores keyed by chunk id; populate `candidate.stages.ce`.
- At `:72` `_apply_title_boost`, change the helper to also return per-doc deltas (`{doc_id: delta}`); fold into `candidate.stages.titleBoost`.
- At `:91–105` `_apply_confidence_filtering`, record `confidence.lowConfidence`, both thresholds, count of filtered docs.
- In `__build_results` (`:157+`), set `kept` / `dropReason` for each candidate as dedup, missing-doc, and per-doc-cap decisions are made.
- Final timing dict written at `:88` next to the existing log line.

`_TraceRecorder` should be a tiny class living next to the searcher (~50 lines): holds a dict keyed by chunk_id, has `record_stage(name, chunk_id, rank, score)`, `mark_dropped(chunk_id, reason)`, `to_dict()`. When `trace is None`, every method is a no-op via a separate `_NullRecorder`.

### 3. `main/indexes/hybrid_search_indexer.py:15` — expose intermediate ranks

Today `search()` only returns post-fusion `(scores, indexes)`. Add an optional `return_breakdown=False` flag; when true, also return `{"faiss": [(chunk_id, rank, score), ...], "bm25": [...], "rrf": [...]}` so the searcher can populate `candidate.stages` without re-running.

If `return_breakdown` is unset, the function returns its old 2-tuple unchanged — zero blast radius for non-trace callers.

### 4. `main/core/cross_encoder_reranker.py:25` — return CE scores keyed by chunk id

`rerank()` already has the scores; just optionally return a third element `ce_score_by_chunk_id: dict[int, float]`. Same compatibility approach as #3.

### 5. `main/graph/knowledge_graph.py` — surface entity span info

`detect_entities()` at `:62` currently returns entity ids. Add a `with_spans=False` flag; when true, return `[(entity_id, matched_span_text)]` so `trace.query.detectedEntities[].matchedSpan` is populated. Caller in `knowledge_api_server.py` is the only spot using the new variant.

## Muninn integration (Phase 1)

The MCP adapters embed the trace differently depending on transport:

- **HTTP-wrapper MCP** (`knowledge_api_mcp_adapter.py`) — appends a fenced `\`\`\`huginn-trace\n{json}\n\`\`\`` block at the end of the markdown tool result.
- **In-process multi-collection MCP** (`multi_collection_search_mcp_adapter.py`) — already returns JSON, so attaches the trace under a top-level `trace` field in the result dict.

Both adapters check the same env var `HUGINN_TRACE_DEFAULT=1`. Set it on the MCP server's environment when Muninn launches the stdio process.

**Recommended Muninn behavior**: parse out the trace blob from the tool result, store it on the tool span (`attributes.searchTrace` is the convention) so the existing waterfall span-detail viewer renders it, and **strip the trace from the text the LLM sees** so it doesn't pollute context.

The existing span-detail viewer at `src/dashboard/views/components/traces-waterfall.ts` already dumps attributes as JSON, so once the trace is in `attributes.searchTrace`, operators can inspect it immediately. A custom renderer keyed on `searchTrace.schemaVersion === 1` is a Phase 2 polish.

## Rollout

1. ✅ **Land the recorder plumbing + schema, gated off by default.** Done — `main/core/search_trace.py`, threaded through `documents_collection_searcher.py`, `hybrid_search_indexer.py`, `cross_encoder_reranker.py`, `knowledge_graph.py`, exposed via `?trace=true` on `/api/search`. 506 tests pass; null path is a no-op singleton. *Open: per-chunk drop reasons (`noise`, `dedup`, `metadataFilter`) not yet recorded — `kept` defaults true. Listed under future improvements.*
2. Flip `HUGINN_TRACE_DEFAULT=true` on the local dev API server only.
3. Use it for two weeks on real ERA / EØS queries; iterate on the schema (it's cheap, schemaVersion=1).
4. Once stable, point Muninn's Huginn tool wrapper at `trace=true` so every Muninn → Huginn search is inspectable in the waterfall.

## Validation

- Unit test: a recorded trace for a known query has the expected `detectedEntities` and `expansionTerms`.
- Unit test: with `trace=None`, the recorder is the null variant and `searcher.search()` returns identical output to today (compare against a snapshot).
- Manual: run a query with a strong BM25 hit and a strong vector hit; trace should clearly show RRF combining them and the CE re-ordering.
- Manual: run a query that triggers `lowConfidence`; trace's `confidence` block should explain which threshold tripped.

---

## Future improvements

Listed roughly in order of value. Not committed; we'll pull from this list as the basic trace gets used.

### Trace-driven

0. **Per-chunk drop reasons.** The recorder supports `mark_dropped(chunk_id, reason)` but the searcher only records stage scores, not which chunks were filtered (URL/text-hash dedup in `__build_results`, all-noise filtering in `_apply_confidence_filtering`). Wire these in so candidates that disappeared can be inspected. ~1 hour of work, high value for "why isn't doc X in results" debugging.
1. **Nested span emission to Muninn.** Replace the flat `trace` blob with child spans (`graph.expand`, `index.faiss`, `index.bm25`, `fuse.rrf`, `rerank.ce`, `boost.title`, `filter.confidence`, `assemble`) so they show up as bars in Muninn's waterfall. Requires Muninn's `message-processor.ts` to ingest non-Claude spans, or a small Huginn-side translator that emits OTel-compatible JSON.
2. **Persistent trace store.** Append every traced query to a SQLite file (`data/traces.db`) keyed by query + timestamp + collection. Lets us replay regressions and build dashboards without re-running searches. Pair with a `/api/traces` browse endpoint.
3. **Trace diff tool.** CLI: `huginn trace diff <query>` runs the same query against two configurations (e.g. with/without a graph file, two embedding models) and prints a side-by-side of which docs moved in ranking and why. This is what we actually want when tuning.
4. **Ground-truth annotation loop.** Operator clicks "this result was good / bad / missing" in Muninn's UI, the verdict is written to `data/judgements.jsonl` keyed to the trace. Drives the eval harness.

### Search quality (independent of tracing, but tracing makes them measurable)

5. **Learned RRF weights / score combination.** Today RRF k=60 is uniform across collections and query types. With trace data we can fit per-collection or per-query-class weights. Trace already has the per-stage ranks needed to do this offline.
6. **Adaptive `fetchK`.** When BM25 and FAISS agree strongly on the top-k, fetch fewer; when they disagree, fetch more. The trace's stage agreement is the signal.
7. **Graph expansion as alternates, not concatenation.** Today expansion terms are appended to the query string, which dilutes the original query embedding. Better: run expansion terms as a separate retrieval, fuse via RRF, weight lower than the raw query.
8. **Negative graph signals.** When the user's query contains "NOT X" or excludes a BUC/SED, graph traversal currently still expands toward neighbors of X. Detect negation and exclude.
9. **Title boost rework.** The current adaptive boost is opaque (depends on score range of the top-k). With trace data showing exactly when it flips a result, we can replace it with a calibrated boost or kill it.
10. **Per-collection rerank thresholds.** `LOW_CONFIDENCE_THRESHOLD = -0.10` is a global guess. Trace data + judgements lets us tune per collection.
11. **Query-class router.** Use detected entities + query shape to route: pure BUC/SED lookup → graph_answer only; "explain X" → search-heavy; "compare X and Y" → multi-query fanout. Trace shows when routing would have helped.

### Muninn-side (depends on Huginn trace)

12. **Trace-aware tool result renderer in Muninn.** Custom React/Lit component for `attributes.output.trace.schemaVersion === 1` that renders a stage-by-stage table instead of raw JSON. Dramatically lowers the cost of inspecting a search.
13. **Search-quality dashboard.** Aggregate over many traced searches: average `lowConfidence` rate per collection, p95 timings per stage, distribution of where the top-1 came from (BM25 vs FAISS dominant). Becomes the canary signal when we touch search code.
14. **Auto-flag suspicious searches.** A Muninn-side check that, after each Huginn call, looks at the trace and flags conditions like "graph expansion contributed nothing", "all top-5 came from BM25 only", "CE disagreed strongly with RRF" — surfaced inline in the chat as a small badge. Helps operators learn when to trust results.

### Eval & regression

15. **Golden-query regression suite.** Curate ~50 EØS / ERA queries with expected top-3 docs. CI replays them with trace on, fails if any gold doc drops out of top-k. The trace is what makes the failure debuggable.
16. **A/B trace harness.** Run two configurations against the same query batch in parallel, store both traces, render a side-by-side report. The single highest-leverage tool for actually improving the search, once we have traces.
