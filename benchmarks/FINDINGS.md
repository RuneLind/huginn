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


## Finding 5: No EESSI knowledge graph

**Status:** Informational

The system has entity detection patterns for BUC, SED, Artikkel, and Forordning entities, but no EESSI graph file exists. Only the Jira graph (2,175 nodes, 2,004 edges) is loaded. EESSI entity detection is dormant.


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
