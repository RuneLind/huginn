# Building a Local RAG Search Engine From Scratch

How we went from zero to a production-quality search system over 7 commits, indexing thousands of documents from Notion and Confluence with sub-50ms query latency.

## The Problem

We had thousands of pages scattered across Notion and Confluence. Finding anything meant remembering which tool it was in, what it was called, and hoping the built-in search would surface it. We wanted a single search interface that understood Norwegian content, worked across all sources, and could power AI agent workflows via MCP.

The constraint: everything runs locally. No cloud APIs for indexing, no third-party vector databases. Just Python, open-source models, and FAISS.

## The Journey: 7 Commits, 7 Layers of Improvement

### Commit 1: Notion Source Adapter + Knowledge API Server

The foundation. A Notion source adapter following a Reader-Converter pattern:
- **Reader** fetches pages from the Notion API (search or tree traversal, incremental updates)
- **Converter** transforms pages to markdown with breadcrumb hierarchy

Plus a FastAPI Knowledge API Server that loads the embedding model and FAISS indexes once at startup, then serves search results over HTTP in <50ms. The existing MCP servers restarted per invocation (~800ms overhead) — too slow for a search dashboard.

*95 unit tests from day one.*

### Commit 2: Hybrid Search + Multilingual Embeddings

Two big changes that transformed search quality overnight.

**The multilingual problem.** Our initial embedding model (`all-MiniLM-L6-v2`) was English-only. It scored **0.018** cosine similarity between "Rammeavtaler" and "Framework agreements" — it literally couldn't see that these mean the same thing. Switching to `intfloat/multilingual-e5-base` brought this to **0.834**.

| Pair | English-only model | Multilingual model |
|------|---:|---:|
| "Rammeavtaler" <-> "Framework agreements" | 0.018 | **0.834** |
| "Rammeavtaler" <-> "Kontrakter og avtaler" | 0.522 | **0.903** |
| "feriepenger" <-> "holiday pay" | 0.168 | **0.888** |
| "ansettelsesforhold" <-> "employment relationship" | 0.125 | **0.848** |

**Hybrid search.** FAISS captures semantic meaning but misses exact terms. BM25 finds exact keyword matches but misses paraphrases. We added both and merged results with Reciprocal Rank Fusion (RRF). Neither alone is sufficient; together they cover each other's blind spots.

### Commit 3: Cross-Encoder Reranking

Hybrid search retrieves good candidates, but the top-k ordering is imprecise. A cross-encoder (`BAAI/bge-reranker-v2-m3`) reads query and document *together* through the full transformer — fundamentally different from comparing pre-computed vectors.

We ran a 5-query baseline comparison. Results:

| Query | Before | After |
|-------|--------|-------|
| "sykefravær regler" | `Ferie.md` in top-10, noise | `Sykmelding.md` #1, noise removed |
| "onboarding new employees" | Process docs ranked high | `Onboarding av nyansatte.md` surfaced |
| "teknologiledelse strategi" | Generic `Strategi.md` #1 | Service offering page promoted to #1 |
| "how to submit expenses" | All garbage, no signal | Honest near-zero scores (content gap) |

The reranker adds ~1,500ms latency but produces clearly separated scores: relevant results score >0.5, irrelevant score <0.05. Without reranking, RRF scores cluster tightly (0.014-0.032), making it impossible to tell good from bad.

### Commit 4: Chunk Quality Cleanup

We diagnosed what was actually going into the index. The findings were bad:

| Problem | Before | After |
|---------|--------|-------|
| Breadcrumb-only empty chunks | 8,223 | 0 |
| YAML frontmatter in chunks | 8,226 | 1 |
| S3 presigned URLs in chunks | 64 | 3 (legit) |
| Markdown images in chunks | 441 | 0 |
| Chunks under 100 chars | 3,983 | 151 |
| **Total chunks** | **22,325** | **14,481** |

35% fewer chunks, 22% less text, near-zero noise. Every query improved after reindex because the embedding space was no longer polluted with garbage.

Key changes:
- Eliminated breadcrumb-only chunk #0 (each document was wasting its first index slot)
- Stripped YAML frontmatter (URL preserved in document metadata instead)
- Read `.md` files as plain text (the Unstructured library was mangling frontmatter delimiters)
- Stripped markdown images and replaced S3 URLs with `[file]`
- Capped breadcrumbs at 4 levels: `[First > ... > Parent > Page]`

### Commit 5: Confidence Filtering + Code Block Stripping

A 15-query diagnostic across 5 categories revealed two more issues:

**Code blocks cause false positives.** Mermaid diagrams and ER diagrams containing keywords like `HEALTH_EXAMINATION` surfaced for "parental leave" queries. We strip all fenced code blocks from chunks before indexing.

**No confidence signal.** When nothing matches, the system returned garbage with no indication it was garbage. We added two-tier confidence filtering based on reranker scores:

| Score range | Meaning | Action |
|---|---|---|
| < -0.50 | Strongly relevant | Return normally |
| -0.10 to -0.50 | Likely relevant | Return normally |
| -0.01 to -0.10 | Uncertain | Flag as low confidence |
| > -0.01 | Almost certainly noise | Filter out |

The diagnostic also confirmed a cross-lingual gap: English queries for Norwegian content get correct FAISS retrieval but the reranker crushes the scores to near-zero. "oversikt over goder og fordeler" scores -0.368, but the equivalent "employee benefits overview" scores -0.018. This remains an open issue.

### Commit 6: Structure-Aware Heading-Based Chunking

Fixed-size chunking splits documents at arbitrary character boundaries. A chunk might start mid-sentence in one section and end mid-sentence in another. We built a `MarkdownHeadingSplitter` that respects document structure:

**Stage 1:** Regex splits markdown at H1-H3 heading boundaries into logical sections.
**Stage 2:** Sections exceeding the chunk size are sub-split with `RecursiveCharacterTextSplitter`.
**Fallback:** Content without headings uses the previous fixed-size behavior.

Each chunk now carries a `heading` key and includes `## Heading` in the indexed text. Preamble text before the first heading gets no heading key. The Knowledge API exposes the heading field so the search GUI can display section context.

Before: `...ompetanse\n\nViktige arkitekturdokumenter:\n\n- Saksgang - fra journalfø...`
After: `[My Team > Målbilde og arkitektur]\n## Viktige arkitekturdokumenter\nSaksgang - fra journalføring til vedtak...`

*28 new tests including Unicode headings (Norwegian characters), emoji, ATX closing hashes, and heading level normalization.*

### Commit 7: Multi-Collection Support + API Improvements

With the pipeline solid, we expanded to multiple document sources. Created a `my-confluence-v2` collection from 319 Confluence pages (73% with heading-based chunks, 2,439 total chunks).

Fixed the Knowledge API to support the search dashboard:
- `/api/collections` returns wrapped response with snake_case keys
- Search results include `heading` field for section-aware display
- Server loads multiple collections at startup with shared embedding model

## The Final Architecture

```
Query
  |
  v
[Hybrid Search: FAISS (multilingual-e5-base) + BM25]
  |
  | Top 45 candidates (3x over-fetch)
  v
[Cross-Encoder Reranker (bge-reranker-v2-m3)]
  |
  | Re-scored top 15, noise filtered, confidence flagged
  v
Results with headings, breadcrumbs, scores
```

**Two collections running:**
- my-notion: 8,311 docs, 14,481 chunks
- my-confluence-v2: 319 docs, 2,439 chunks

**Content pipeline:**
1. Fetch from source (Notion API / Confluence / local files)
2. Convert to markdown with breadcrumb hierarchy
3. Strip frontmatter, code blocks, images, S3 URLs
4. Split at heading boundaries (H1-H3), sub-split oversized sections
5. Embed with multilingual-e5-base, index in FAISS + BM25
6. Serve via FastAPI (<50ms per query after model warmup)

## What We Learned

**Clean your data before optimizing retrieval.** Commit 4 (chunk cleanup) improved every single query just by removing garbage from the index. 35% of our chunks were noise. No amount of reranking fixes bad input data.

**Multilingual embeddings are non-negotiable for non-English content.** The English-only model scored 0.018 on Norwegian semantic pairs. Switching models was the single highest-impact change.

**Hybrid search is strictly better than either method alone.** FAISS misses exact terms, BM25 misses paraphrases. RRF fusion is simple to implement and covers both.

**Cross-encoder reranking produces honest scores.** Without reranking, all scores cluster together and you can't tell relevant from irrelevant. With reranking, the score distribution is bimodal — you can confidently set thresholds.

**Structure-aware chunking preserves meaning.** Splitting at heading boundaries instead of character counts produces chunks that are coherent logical sections. Each chunk has a clear topic, which helps both embedding quality and result display.

**Start with the pipeline, not the model.** We built the full pipeline (reader, converter, indexer, searcher, API) first, then iteratively improved each layer. Each commit could be tested against real queries immediately.

## By The Numbers

| Metric | Start | Now |
|--------|-------|-----|
| Documents indexed | 0 | 8,630 |
| Total chunks | 0 | 16,920 |
| Embedding model | English-only (384-dim) | Multilingual (768-dim) |
| Search methods | FAISS only | FAISS + BM25 + cross-encoder |
| Chunk noise (empty/garbage) | ~35% | <1% |
| Heading-aware chunks | 0% | 73% (Confluence), growing |
| Query latency (API) | N/A | <50ms (+ ~1,500ms reranking) |
| Test coverage | 0 | 213 tests |
| Collections | 0 | 2 (Notion + Confluence) |

## What's Next

Open items from the roadmap, roughly in priority order:
1. **Cross-lingual scoring** — the reranker crushes EN->NO scores; quick fix is to skip reranking for detected English queries
2. **Result deduplication** — template pages duplicated across parents waste result slots
3. **Title/path boost** — direct title matches should rank higher
4. **Metadata filtering** — scope searches by collection, team, or date
5. **Parent-child retrieval** — match on small chunks, return parent sections for context
