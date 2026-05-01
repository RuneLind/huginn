# Search Quality Report: thread-arc queries on x-feed

**Date:** 2026-05-01
**Trigger:** Wiki maintainer (huginn-jarvis) querying x-feed for the Anthropic Max-subscription / Agent SDK / OAuth policy thread surfaced no high-signal results, despite x-feed containing the canonical posts (Boris Cherny's Apr 3 announcement, Tariq's Apr 27 verbatim apology, Pawel Huryn's Apr 30 reproduction, the Apr 26 HERMES.md billing-bug story). Fell back to date-filtered grep. The relevant story arc was reconstructed only after grep — the search API surfaced none of the load-bearing posts in the top 10 for any of three reasonably-phrased queries.

This is a **distinct failure mode** from the [autoresearch keyword-precision case](./search-quality-report-x-feed-autoresearch.md) — the queries here were intentionally semantic ("policy timeline", "subscription enforcement", "harness classifier"), not keyword-matched ("HERMES.md"). So the prior report's RRF-equal-weighting diagnosis doesn't apply directly. A different mechanism is in play.

## Summary

For a *thread-arc* query — multiple posts across 5+ days from 5+ authors that together tell one coherent story — the x-feed collection's hybrid search returns top-10 results dominated by **boilerplate-heavy short tweets** (e.g. trailing-only-a-link tweets, tag-heavy promotional posts, single-photo posts) that semantic-match the query's *vocabulary* but are not actually about the topic. The actual high-signal posts in the corpus rank nowhere visible in the top 10.

Three contributing causes (in priority order, all hypothesized — verification work suggested below):

1. **Engagement-footer boilerplate dominates the chunk content** for short tweets, swamping the body in cosine similarity
2. **Frontmatter tags get embedded as text**, creating a high-density "ai, developer-tools, open-source" cluster that any AI-related query matches
3. **Quote-tweet and reply chains are not traversable from a single search** — the HERMES.md story is a 5-author quote-tweet graph; chunk-level retrieval can find one node, never the arc

## Architecture context

x-feed manifest (current):

- 8,679 documents on disk (`huginn/data/sources/x-feed/`)
- Collection indexed with same hybrid pipeline as autoresearch report (FAISS + BM25 + RRF + cross-encoder rerank)
- Each x-feed file is a Twitter-post markdown with a stable structure:

```
---
title: "@author — first ~80 chars of post text..."
url: https://x.com/author/status/...
date: YYYY-MM-DD
category: x-feed
tags: "ai, developer-tools, open-source"     # 0–5 tags from a fixed vocabulary
engagement_score: NN.NN
relevance_score: 0.NN
author_score: 0.NN
combined_score: 0.NN
---
# @author — Display Name

<actual tweet body — can be 5 chars or 5000 chars>

(optional) > **Quoted @other:** quoted-tweet body

(optional) - [photo](...) / - [video](...)

---

- **Engagement:** N likes · N retweets · N views · N bookmarks
- **Engagement Score:** NN.NN
- **Date:** Day Mon DD HH:MM:SS +0000 YYYY
- **Type:** tweet | note
- **Link:** https://x.com/author/status/...
```

The standardized footer is **6+ lines of repeating boilerplate** at the bottom of every file. For a tweet whose body is 200 characters, the footer is a substantial fraction of the chunk content.

## Findings

### 1. Engagement-footer boilerplate floods short-tweet chunks

For a typical x-feed file with a body of 200–500 chars, the chunk that includes the body also includes the trailing footer (`---\n\n- **Engagement:** ... · ... · ... · ...\n- **Engagement Score:** ...\n- **Date:** ...\n- **Type:** ...\n- **Link:** ...`). After tokenization, this footer is ~40-60 tokens of recurring high-frequency vocabulary: *engagement, likes, retweets, views, bookmarks, score, date, type, link, tweet, note*.

For a tweet with a 100-char body, the footer can be more than half the embedded text. The semantic embedder then computes a vector that is partly about the tweet and partly about *Twitter post metadata vocabulary*. When two short tweets are compared, their footers cosine-align on this metadata vocabulary regardless of whether their bodies are about the same topic.

**Symptom we observed:** semantic-search top-10 for *"Claude Max subscription Agent SDK OAuth ban"* included:

| Rank | Hit | Body content |
|---|---|---|
| 1 | `2026-04-20_0xJeff_2046164193326628880.md` | Single line: `http://x.com/i/article/2046162070274789377` (just a link) |
| 2 | `2026-04-27_lovart_ai_2048612965297930730.md` | "Meet the new standard of AI-powered creativity..." (LovartAI ad) |
| 3 | `2026-04-28_brilliantlabsAR_2049253182316966349.md` | "The open source AI smart glasses built for play..." (smart glasses ad) |

All three are **short tweets where the footer is a large fraction of the chunk**, and all three have *zero* connection to the query topic. Meanwhile the actual canonical post — `2026-04-03_bcherny_2040206440556826908.md` ("Starting tomorrow at 12pm PT, Claude subscriptions will no longer cover usage on third-party tools like OpenClaw...") — did not appear in the top 10 for any of the three queries we ran.

### 2. Frontmatter tags get embedded as plaintext

The x-feed YAML frontmatter contains a `tags:` field with a small fixed vocabulary: *ai, developer-tools, open-source, cloud-infrastructure, business-startups, news-releases, web-frontend*. If the chunker includes frontmatter in the embedded text (likely, given the prior autoresearch report's observation that BM25 sees the literal string), then every "ai, developer-tools" tweet contributes to a vocabulary cluster around those tokens.

For an AI-themed query, the embedder will high-cosine any tweet that has those tags in its frontmatter — regardless of whether the body is on-topic. Combined with finding 1, this means: *short tweets in popular tag categories beat long, on-topic, body-rich posts*.

**Quick verification path:** dump 3 chunks for `2026-04-20_0xJeff_2046164193326628880.md` and `2026-04-03_bcherny_2040206440556826908.md` — compare which is text-dominated by body vs by frontmatter+footer. If the bcherny chunk's body-to-boilerplate ratio is ~5-10× higher than the 0xJeff one, finding 1+2 are confirmed.

### 3. Thread/quote-tweet arcs are invisible to chunk search

The HERMES.md story is structurally a **5-author quote-graph** spanning 5 days:

```
@om_patel5 (Apr 26, original report)
   ↑ quote-tweet
@theo (Apr 26, amplification, 117K views)
   ↓
   ─── @trq212 reply (Apr 27, verbatim apology) ─→ quote-tweeted by @WesRoth
                                                ─→ quote-tweeted by @Teknium (Apr 27)
                                                ─→ quote-tweeted by @PawelHuryn (Apr 30)
```

A user searching "Anthropic harness classifier enforcement" wants to land on *the arc*, not just one post. But chunk-level retrieval — even with perfect ranking on body content — returns one post per result; the user has to manually traverse `@bcherny → who quoted? → who replied?` to reconstruct the thread. Each x-feed file already encodes the quote relation as `> **Quoted @other:** ...` in its body, but the search pipeline doesn't expose it as a graph.

The **author-graph endpoint** at `/api/collection/{name}/author-graph` already exists in the API. The relationships needed for thread-arc traversal (author → quoted-author, author → replied-to, post-id → reply-chain) are *latent in the file content* but not in the indexed graph, AFAICT.

## Reproducing the issue

```bash
# Three reasonably-phrased queries that should have surfaced the @bcherny canonical post:

curl -s -G http://localhost:8321/api/search \
  --data-urlencode "q=Claude Max subscription Agent SDK OAuth ban third-party tools API key authentication" \
  --data "collection=x-feed" --data "limit=10" --data "brief=true"

curl -s -G http://localhost:8321/api/search \
  --data-urlencode "q=Anthropic banned OpenClaw Claude subscription OAuth third-party harness Boris Cherny" \
  --data "collection=x-feed" --data "limit=10" --data "brief=true"

curl -s -G http://localhost:8321/api/search \
  --data-urlencode "q=use Max plan without paying API tokens personal Claude Code subscription harness" \
  --data "collection=x-feed" --data "limit=10" --data "brief=true"
```

**Expected top hits (any of these would have been a save):**

- `2026-04-03_bcherny_2040206440556826908.md` — *"Starting tomorrow at 12pm PT, Claude subscriptions will no longer cover usage on third-party tools like OpenClaw..."* — 5,573 likes, 2.06M views, the canonical event
- `2026-04-26_om_patel5_2048204411986469232.md` — *"$200 in extra usage because HERMES.md was in his git commits..."* — the bug report
- `2026-04-26_theo_2048456227538231751.md` — Theo's amplification quote-tweet, 117K views
- `2026-04-27_WesRoth_2048643304913518628.md` — Tariq's verbatim apology, quoted in the body
- `2026-04-30_PawelHuryn_2049763687330689396.md` — `openclaw.inbound_meta.v1` reproduction

**Actual top hits we got:** none of the above. Top results were single-link tweets and AI-ad posts (LovartAI, Brilliant Labs, etc.) that share tag vocabulary but no topical content.

## What worked instead

Date-filtered grep on April 25–30 files with manually-chosen markers:

```bash
cd huginn/data/sources/x-feed/
grep -l -i -E "agent.sdk|subscription.*ban|api.key.*subscription|oauth.*token.*claude|max.plan|claude.code.subscription" 2026-04-2[5-9]*.md 2026-04-30_*.md
```

This surfaced the HERMES.md story in 3 shell calls. The semantic search pipeline, in contrast, would have required 10+ queries with no guarantee of ever surfacing the relevant posts.

The pattern that emerged: **date-bounded grep with topic-specific markers consistently beat semantic search for a story-arc question on this corpus.** That's a strong signal the indexer is being run against text that doesn't help it.

## Recommended fixes

### Fix 1: Strip standardized footer + frontmatter from embedded text on x-feed (high impact, low risk)

The cheapest, most-targeted fix. For x-feed (and any future short-tweet-style collection), pre-process documents before chunking to strip:

1. The YAML frontmatter (`---\n...---\n`) — keep it as metadata, don't embed it
2. The closing footer (everything after the final `---\n\n- **Engagement:**` line)
3. (Optional) the inline `[video](...)` and `[photo](...)` markdown links — they're just URLs, no semantic content

What remains for each file is the actual tweet body + any quoted body. The embedder then computes vectors that are dominated by tweet content, not metadata vocabulary.

Suggested implementation: a per-collection `text_extractor` hook, similar to per-collection ingest options elsewhere in the pipeline. Default = identity; x-feed = strip frontmatter + footer.

**Validation:** rerun the three queries above; expect the bcherny canonical post to enter the top 5.

### Fix 2: Use `combined_score` from frontmatter as a recency/quality boost in result ranking (medium impact, low risk)

Each x-feed file already has `combined_score: 0.NN` in YAML — the ingest pipeline computed a relevance × engagement × author score offline. This is *exactly the prior* the search ranker is missing.

Suggested: parse `combined_score` at index time, store as document-level metadata, and apply a multiplier in the final ranking pass (similar to `_apply_title_boost` in `documents_collection_searcher.py`):

```python
# pseudocode
final_score[doc_id] *= (1.0 + 0.3 * combined_score)  # combined_score in [0, 1]
```

This is collection-aware; only x-feed and similar collections would carry the field. The default `combined_score=0` is a no-op.

**Validation:** the bcherny canonical post has `combined_score: 0.5083`; the LovartAI ad has no combined_score (or a much lower one). Boost should reorder the top 10 toward bcherny.

### Fix 3: Add date-range filter to `/api/search` (low impact, low risk)

For story-arc queries, the user often knows the rough date window. Adding `date_from` and `date_to` query parameters would let the wiki maintainer (or any thread-tracing query) restrict the search to a known window before the boilerplate-floods kick in. This complements but does not replace Fix 1.

Suggested: parse the YAML `date:` field at index time, store as a doc-level filter dimension.

### Fix 4: Index quote-graph and reply-graph for x-feed (medium-high impact, moderate effort)

The author-graph endpoint exists; it should be extended (or a sibling `quote-graph` endpoint added) to expose the latent quote/reply structure already present in x-feed file bodies (the `> **Quoted @other:** ...` markers).

Story-arc traversal then becomes: search returns one node → user (or LLM agent) follows quote/reply edges to reconstruct the arc. This is exactly the [graph-enhanced-rag](./graph-enhanced-rag.html) pattern Huginn already supports for code/document graphs — applied to social-media threads.

**Implementation sketch:** an offline pass over x-feed extracting the quote-tweet IDs from `> **Quoted @user:** ...` blocks; storing them as edges in a per-collection graph file (matching the existing `*_llm_graph.json` pattern). The API server already auto-loads graph files at startup.

### Fix 5: Document the "tweet-shaped collection" anti-pattern (zero impact code-wise, high impact for users)

The autoresearch report and this report are both about x-feed. There's a pattern emerging: **collections of short, footer-heavy, tag-categorized documents need different search ergonomics than collections of long-form documents.** Worth a short addendum to `wiki-collection-pattern.md` or a sibling guide noting:

- Strip boilerplate before embedding
- Use ingest-time relevance scores as ranking priors
- Expect to need a graph layer for thread-shaped content
- BM25 + date filter may outperform pure semantic search for narrow story-arc queries

## Priority

**Fix 1 (strip footer/frontmatter) is the highest impact with lowest risk** — it directly addresses the dominant failure mode (boilerplate-flooded chunks) and is a single per-collection preprocessing step. Validate by rerunning the three queries above.

**Fix 2 (combined_score boost)** is a strong second — the data is already in the files; the search just isn't using it.

**Fix 4 (quote-graph)** is the highest *ceiling* but the most work. Worth scoping after Fix 1+2 are in.

**Fix 3 (date filter)** is a nice-to-have that helps callers (including future-me writing this kind of query) work around the underlying issue while Fix 1+2+4 are designed.

## Cross-references

- [Search Quality Report: autoresearch in x-feed](./search-quality-report-x-feed-autoresearch.md) — prior report, complementary failure mode (BM25/RRF for keyword queries; this report covers semantic queries on short documents)
- [graph-enhanced-rag.html](./graph-enhanced-rag.html) — existing graph layer pattern that Fix 4 would apply to x-feed
- [wiki-collection-pattern.md](./wiki-collection-pattern.md) — natural home for a short-document-collection addendum (Fix 5)
- huginn-jarvis wiki: [[HERMES.md Billing Routing Incident]] — the source page that motivated this report, and a worked example of the kind of thread-arc query the search pipeline should support
