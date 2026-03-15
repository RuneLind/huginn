# Benchmark Findings — 2026-03-14

Baseline run on jira-issues (2,132 docs, 8,688 chunks) and melosys-confluence-v3 (294 docs, 2,336 chunks).

## Finding 1: Reranker is valuable but dominates latency

**Status:** Investigated — reranker is net positive, latency is the real concern

**Latency data:**
| Collection | With reranker p50 | Without reranker p50 | Overhead |
|------------|------------------|---------------------|----------|
| jira-issues | 643ms | 24ms | 509ms |
| melosys-confluence-v3 | 878ms | 13ms | 686ms |

Search scaling with reranker (Confluence): 5 chunks=398ms, 15=882ms, 30=1842ms, 50=3924ms

**Quality assessment:**

Self-retrieval (exact title queries) initially suggested the reranker was neutral/negative.
However, this was a MISLEADING benchmark — title queries are trivially easy for embeddings.

Testing with realistic natural-language queries (the kind developers actually ask via MCP)
showed the reranker reorders results on EVERY query for both collections. Qualitative review
shows it consistently surfaces more relevant documents:
- "valideringsregler for fødselsnummer" → surfaces "Valideringen av ulike opplysninger" (better match)
- "forskjellen mellom pliktig og frivillig" → surfaces "Vilkår for artikkel 14 - Frivillig trygd" (directly answers)
- "hva skjer når vi mottar en SED" → surfaces processing/control docs over generic "Motta SED"

**Conclusion:** The reranker provides genuine quality improvement on real queries.
The self-retrieval benchmark underestimates its value because it only tests trivial queries.

**Realistic query benchmark (13 natural-language queries per collection):**
| Collection | Metric | With reranker | Without reranker | Lift |
|------------|--------|--------------|-----------------|------|
| Confluence | Hit rate | 100% | 92.3% | +7.7% |
| Confluence | MRR | 0.923 | 0.782 | +14.1% |
| Jira | Hit rate | 100% | 100% | 0% |
| Jira | MRR | 0.962 | 0.865 | +9.6% |

The reranker provides substantial quality improvement on real queries for BOTH collections.
On Confluence it rescues a query that fails completely without it (RINA integration).

**Conclusion:** Reranker is clearly valuable. Keep it enabled. Latency (500-700ms) is acceptable.

**Possible future improvements if latency becomes a concern:**
- Reduce candidate count to speed up (currently 1.5x effective_chunks)
- Consider lighter reranker models
- Batch multiple MCP queries to amortize model overhead


## Finding 2: BM25 adds nothing for Jira but helps Confluence

**Status:** Informational

**Data:**
| Collection | FAISS recall@5 | Hybrid recall@5 | BM25 lift |
|------------|---------------|-----------------|-----------|
| jira-issues | 0.967 | 0.967 | +0.000 |
| melosys-confluence-v3 | 0.967 | 1.000 | +0.033 |

Jira issues are short and keyword-dense — embeddings already capture them well. Confluence pages have more varied vocabulary where lexical matching helps. BM25 adds ~5ms so no cost concern, but worth knowing the asymmetry.


## Finding 3: 12% of Jira docs not top result for own title

**Status:** Informational

**Data:**
- jira-issues: recall@1=0.880, recall@3=0.980, MRR=0.933
- melosys-confluence-v3: recall@1=0.960, recall@3=1.000, MRR=0.977

6 of 50 Jira documents are not the #1 result when searching for their own title. They almost always appear in top 3 (recall@3=0.98). This may be due to similar issues with overlapping titles or the title boost not being strong enough.


## Finding 4: 75 documents were untagged — FIXED

**Status:** Fixed (2026-03-14)

**Before:**
- jira-issues: 64/2,132 untagged (3.0%)
- melosys-confluence-v3: 11/294 untagged (3.7%)

**After:**
- jira-issues: 0/2,132 untagged (100% tagged)
- melosys-confluence-v3: 1/294 untagged (99.7% tagged)

Re-ran tag_documents.py on both sources, then collection_update_cmd_adapter.py to re-index.
The 1 remaining untagged Confluence doc is "Godterirullering" (candy rotation schedule) —
content too minimal for tagger excerpt extraction.


## Finding 5: No EESSI knowledge graph — FIXED

**Status:** Fixed (2026-03-15)

The system has entity detection patterns for BUC, SED, Artikkel, and Forordning entities, but no EESSI graph file existed. Only the Jira graph was loaded. EESSI entity detection was dormant.

**Fix:** The EESSI graph extractor in huginn-nav (`extract_melosys_graph.py`) already existed and produces a rich graph (71 nodes, 135 edges) from Confluence EESSI documentation. The graph includes BUC, SED, Artikkel, Forordning, AD_BUC, and H_BUC nodes with relationship edges (inneholder_sed, hjemlet_i, del_av, underprosess, refererer_til, implementerer). The server loads it via `KNOWLEDGE_GRAPH_PATH` env var set in huginn-nav's `start.sh`.

Entity detection benchmark passes at F1=1.000 with the EESSI graph loaded.


## Finding 6: Search latency baseline

**Status:** Reference

| Metric | jira-issues | melosys-confluence-v3 |
|--------|------------|----------------------|
| With reranker p50 | 643ms | 878ms |
| With reranker p90 | 863ms | 1,020ms |
| Without reranker p50 | 24ms | 13ms |
| Without reranker p90 | 26ms | 16ms |
| FAISS p50 | 10ms | — |
| BM25 p50 | 5ms | — |
| Embedding throughput | 1,453/s (batch 100) | — |


## Finding 7: Trace replay reveals real search gaps

**Status:** Investigated — session-level replay confirms remaining misses are multi-step search artifacts

**Data (from real MCP session traces):**
| Collection | Doc Recall | Query Hit Rate | MRR | Unique Queries |
|------------|-----------|----------------|-----|----------------|
| jira-issues (initial) | 61.5% | 78.9% | 0.47 | 38 |
| jira-issues (after fix) | **66.7%** | **84.2%** | **0.54** | 38 |
| melosys-confluence-v3 | 80.6% | 95.0% | 0.68 | 20 |

Trace replay uses actual query-document pairs captured from Jira analysis sessions.
These are the queries the MCP agent tried during real work, and the documents it
actually used. This is the highest-fidelity quality signal we have.

**Key gaps:**

### 7a. Initial "18 zero-result queries" was a stale concern
Investigation showed these queries all return results now (5-20 per query).
The trace data was captured on an older index before retagging/reindexing.

### 7b. 2 misses fixed by improving doc matching — FIXED
The trace replay benchmark had strict filename matching that failed on:
- Spaces vs underscores in filenames (MELOSYS-7937)
- Different filename variants for the same issue key

Fixed by normalizing filenames and adding issue-key matching fallback.
This improved Jira doc recall from 61.5% to 66.7% (+5.2%).

### 7c. 6 remaining Jira misses are multi-step search artifacts — CONFIRMED
The remaining misses are queries where the expected doc was found via
multi-step search (the MCP agent tried 3-4 different query variations
across a session). Testing each query independently is stricter than
how the system actually works. These docs rank outside top 50 even
with wider search, so they're genuinely cross-query discoveries.

Testing with max_chunks=50 only gains +4% recall (64→68 docs).

**Session-level replay confirms this (2026-03-15):**
| Collection | Per-query doc recall | Session doc recall | Lift |
|---|---|---|---|
| jira-issues | 66.7% | **94.4%** | +27.7% |
| melosys-confluence-v3 | 80.6% | **89.3%** | +8.7% |

Session replay groups queries by trace_id and checks if the union of all
results across a session covers all expected documents. Jira jumps from
66.7% to 94.4% doc recall, with 18/20 sessions achieving full recall.
This confirms the per-query "misses" are artifacts of single-query testing.

### 7d. 29 queries hit stale `jira` collection (not `jira-issues`)
The trace data contains queries against a collection named `jira` which
no longer exists. These are not real search failures.

### 7e. Confluence is strong at 81% recall
Only 1 missed query. 7 queries returned empty, mostly very specific.

**Remaining improvement opportunities:**
1. ~~Build EESSI knowledge graph for entity-rich queries (Finding 5)~~ — DONE
2. Expand trace dataset with more sessions for better coverage
3. ~~Consider session-level replay (group queries by trace_id)~~ — DONE, see 7c above
