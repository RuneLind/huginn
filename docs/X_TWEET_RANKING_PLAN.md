# X/Twitter Tweet Ranking Pipeline — Work Plan

## Problem

Muninn's X watcher sends up to 80 unranked tweets to Haiku for digest creation.
Haiku must do ALL the ranking with no quality signals — engagement data is partially
stripped, there's no relevance scoring, and no author quality signal. The result is
that interesting content gets lost in noise, and the digest quality depends entirely
on Haiku's ability to guess what matters.

## Goal

Add a pre-ranking pipeline between tweet fetching and summarization so that:
1. Haiku receives a pre-filtered, scored top-30 instead of 80 raw tweets
2. Engagement metrics, topic relevance, and author quality inform the ranking
3. The digest consistently surfaces high-signal content

## Architecture Overview

```
Fetch (x_fetcher.py)
  │
  ▼
Layer 1: Engagement Scoring          ← score each tweet by likes/RT/replies/views
  │
  ▼
Layer 2: Embedding Relevance         ← cosine similarity to user interest profiles
  │
  ▼
Layer 3: Author Signal Scoring       ← PageRank / community detection on author graph
  │
  ▼
Layer 4: LLM Pre-Triage (optional)   ← Ollama/Haiku quick relevance rating
  │
  ▼
Top-N Selection + Score Annotation
  │
  ▼
Muninn Digest (Haiku summarization)
```

Each layer is independently useful and deployable. They compose additively —
a tweet's final score blends all available layer scores.

---

## Layer 1: Engagement Scoring

**Status:** Not started
**Difficulty:** Easy (~50 lines across 2 files)
**Impact:** High — filters out low-engagement noise, surfaces viral/resonant content

### What

Compute a composite engagement score per tweet using X's own open-sourced
signal weights (from `twitter/the-algorithm` Heavy Ranker):

| Signal     | Weight | Rationale                                    |
|------------|--------|----------------------------------------------|
| Retweets   | 20x    | Strongest amplification signal               |
| Replies    | 13.5x  | Indicates discussion/controversy              |
| Bookmarks  | 10x    | High-intent save signal (private engagement) |
| Likes      | 1x     | Baseline engagement                          |

Normalize by `sqrt(views)` to produce an engagement *rate* rather than raw count.
This avoids bias toward mega-accounts with millions of impressions but mediocre
engagement ratios, and surfaces niche tweets with high engagement density.

```python
def engagement_score(tweet: dict) -> float:
    raw = (tweet['likes'] * 1.0 +
           tweet['retweets'] * 20.0 +
           tweet['replies'] * 13.5 +
           tweet['bookmarks'] * 10.0)
    if tweet.get('views', 0) > 0:
        return raw / (tweet['views'] ** 0.5)
    return raw
```

Bonus signals (binary boosts, not weighted):
- `tweet_type == "note"` → +20% boost (long-form = higher effort content)
- `quoted_tweet != null` → +10% boost (commentary = more context)
- `media != null` → +5% boost (visual content tends to be higher quality)

### Where to change

1. **`huginn-jarvis/scripts/x/fetchers/x_fetcher.py`**
   - Add `engagement_score()` function
   - Add `engagement_score` field to tweet output JSON in `_parse_tweet_result()`
   - Add `engagement_score` to YAML frontmatter in `_build_tweet_frontmatter()`
   - Sort tweets by score before JSON output (descending)

2. **`muninn/src/watchers/x.ts`**
   - In `fetchFromCollection()`: sort documents by engagement score from frontmatter
   - In `fetchFromPython()`: sort tweets by `engagement_score` field
   - Take top-N (configurable, default 30) after sorting
   - Include score in `compactTweetText()` output so Haiku can see it
   - Update `DEFAULT_X_PROMPT` to mention that tweets are pre-ranked

### Testing

- Run `x_fetcher.py --pages 3` and verify `engagement_score` in JSON output
- Check score distribution — top tweets should intuitively be "better"
- Compare before/after digest quality with same tweet set

---

## Layer 2: Embedding-Based Interest Relevance

**Status:** Not started
**Difficulty:** Medium (new script, uses existing embedding infra)
**Impact:** Medium-high — filters by *what you care about*, not just popularity
**Depends on:** Layer 1 (for score blending)

### What

Define a set of "interest profile" text anchors representing what the user
wants to see more of. Embed these using the existing `multilingual-e5-base`
model. For each fetched tweet, compute cosine similarity against the interest
profiles and produce a relevance score (0.0–1.0).

```python
INTEREST_PROFILES = [
    "Latest developments in AI agents, LLMs, and coding assistants",
    "Open source projects, developer tools, and engineering culture",
    "Cloud infrastructure, distributed systems, and platform engineering",
    "Tech industry news, product launches, and startup funding",
]
```

Blend with engagement: `final_score = 0.5 * norm(engagement) + 0.5 * relevance`
(weights configurable).

### Libraries to evaluate

| Library | Pros | Cons |
|---------|------|------|
| `multilingual-e5-base` (existing) | Already loaded in API server, no new deps | Not tweet-optimized |
| `VinAIResearch/BERTweet` | Pre-trained on 850M tweets, handles hashtags/emoji | English only, extra model to load |
| `cardiffnlp/tweetnlp` | 19-topic classifier built-in, tweet-native | Adds dependency, English-focused |
| `digio/Twitter4SSE` | Fine-tuned for tweet similarity | Less maintained |

**Recommendation:** Start with `multilingual-e5-base` since it's already loaded.
Evaluate BERTweet only if e5's short-text performance is poor.

### Where to change

1. **New file: `huginn-jarvis/scripts/x/scoring/relevance_scorer.py`**
   - Load interest profiles from config file
   - Embed profiles once, cache embeddings
   - Score tweets by max cosine similarity to any profile
   - Can run standalone or be imported by `x_fetcher.py`

2. **`huginn-jarvis/scripts/x/scoring/interest_profiles.json`**
   - User-editable interest definitions
   - Each profile: `{ "name": "AI & Agents", "text": "...", "weight": 1.0 }`

3. **`x_fetcher.py`** — add `relevance_score` to output when scoring is available

4. **`muninn/src/watchers/x.ts`** — blend engagement + relevance in sort

### Alternative: TweetNLP Topic Classification

Instead of (or in addition to) embedding similarity, use TweetNLP's built-in
19-topic classifier to auto-tag tweets, then boost/dampen by topic:

```python
import tweetnlp
model = tweetnlp.load_model('topic_classification')
result = model.topic("New Claude Code release with MCP support!")
# → {'label': ['science_&_technology'], 'probability': {'science_&_technology': 0.95, ...}}
```

This gives structured topic labels that could also improve Muninn's digest grouping.

---

## Layer 3: Author Signal Scoring

**Status:** Not started
**Difficulty:** Medium-hard (new graph construction, uses existing graph infra)
**Impact:** High over time — identifies consistently high-signal accounts
**Depends on:** Accumulated tweet data (needs history)

### What

Build an author-interaction graph from tweet metadata and score authors
by structural importance (PageRank) and community membership (Louvain).

**Graph construction:**
- **Nodes:** Authors (@handles seen in timeline)
- **Edges:** quote_tweet_of, reply_to, mentions (weighted by frequency)
- **Node attributes:** avg engagement rate, tweet frequency, follower/following if available

**Scoring:**
- Run `nx.pagerank()` on the interaction graph
- Run `louvain_communities()` to discover author clusters
- Compute per-author quality score: `author_score = pagerank * avg_engagement_rate`
- Store as JSON, update incrementally as new tweets arrive

### Inspiration

**X's SimClusters** (open-sourced, KDD 2020): Represents users and tweets as
sparse vectors in "community space" — cosine similarity between a user's interest
vector and a tweet's community vector drives recommendations.

**"Finding High Signal People" (LessWrong):**
1. Start with known-good seed accounts
2. Build follow/interaction graph from their connections
3. Run PageRank to find structurally important accounts
4. Score: high PageRank + low follower count = underrated/high-signal
5. LLM-evaluate recent tweets from top candidates

### Where to change

1. **New file: `huginn-jarvis/scripts/x/scoring/author_graph.py`**
   - Build graph from accumulated tweet data in `data/sources/x-feed/`
   - Extract interactions from quoted tweets, reply patterns, co-mentions
   - Run PageRank + Louvain
   - Output: `data/x-feed-author-scores.json`

2. **Extend `main/graph/knowledge_graph.py`**
   - Already supports arbitrary node/edge types
   - Could use existing LLM entity extraction pipeline for richer signals

3. **`x_fetcher.py`** — look up author score and add to tweet metadata

4. **Score blending:**
   ```
   final = 0.4 * norm(engagement) + 0.3 * relevance + 0.3 * author_signal
   ```

### Bootstrap strategy

Until enough history accumulates for meaningful PageRank:
- Manually seed 10-20 known high-signal accounts with a boost multiplier
- Use engagement rate averaged across all seen tweets per author as initial proxy
- Transition to graph-based scoring after ~2 weeks of data collection

---

## Layer 4: LLM Pre-Triage

**Status:** Not started
**Difficulty:** Easy (prompt engineering + existing Ollama/Haiku infra)
**Impact:** Highest quality, highest cost per run
**Depends on:** Layers 1-3 (runs on pre-filtered candidates)

### What

Send the top-N candidates (pre-filtered by layers 1-3) to an LLM for
a quick relevance score before the final digest generation.

**Approach A — Batch scoring with Ollama (free, local):**
```
Rate each tweet 1-5 for relevance to a tech professional interested in
AI, developer tools, and open source. Return JSON only.
Tweets: [...]
```

**Approach B — Anthropic Batch API (50% cheaper than real-time):**
Submit scoring requests via `POST /v1/messages/batches`. Results available
within 24h, suitable for scheduled digests.

**Approach C — Two-pass Haiku:**
First pass: score 40 tweets (cheap, fast). Second pass: generate digest
from top 20 (higher quality input → better output).

### Where to change

1. **New file: `huginn-jarvis/scripts/x/scoring/llm_triage.py`**
   - Accept pre-ranked tweets, score with Ollama or Haiku
   - Output scored + filtered list

2. **`muninn/src/watchers/x.ts`** — optionally call triage before digest

### Cost estimate

- 40 tweets × ~100 tokens each = ~4K input tokens per scoring call
- Haiku: ~$0.001 per run. Ollama: free.
- Running hourly: ~$0.024/day with Haiku, $0 with Ollama

---

## Implementation Order

| Phase | Layer | Effort | Shipped in |
|-------|-------|--------|------------|
| 1     | Layer 1: Engagement scoring | ~1 hour | huginn + muninn |
| 2     | Layer 2: Interest profiles | ~2-3 hours | huginn |
| 3     | Layer 3: Author graph | ~4-6 hours | huginn |
| 4     | Layer 4: LLM triage | ~1-2 hours | huginn or muninn |

Phase 1 can ship today. Each subsequent phase builds on the previous but
is independently valuable.

## Score Blending Strategy

As layers are added, the final score combines them with configurable weights:

```python
# Phase 1 (engagement only)
final = engagement_score

# Phase 2 (+ relevance)
final = 0.5 * norm(engagement) + 0.5 * relevance

# Phase 3 (+ author signal)
final = 0.4 * norm(engagement) + 0.3 * relevance + 0.3 * author_signal

# Phase 4 (+ LLM triage as reranker)
# LLM reranks the top-N from phase 3, doesn't blend numerically
```

Normalization: min-max scale each score to [0, 1] within the current batch
before blending, so different score magnitudes don't dominate.

## Open Questions

- **Interest profiles:** Should these be manually defined or learned from
  bookmarked/liked tweets over time?
- **Author seed list:** Which accounts should get initial high-signal boost?
- **Ollama model:** Which local model for triage — llama3, mistral, phi-3?
- **Feedback loop:** Should the user be able to thumbs-up/down digest items
  to tune the ranking over time?

## References

- [X's Recommendation Algorithm (open source)](https://github.com/twitter/the-algorithm)
- [SimClusters paper (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403370)
- [Finding High Signal People — PageRank on Twitter (LessWrong)](https://www.lesswrong.com/posts/s5PwfyRFrGFaZFevW/finding-high-signal-people-applying-pagerank-to-twitter-1)
- [BERTweet — Pre-trained on 850M tweets](https://github.com/VinAIResearch/BERTweet)
- [TweetNLP — Topic classification + embeddings](https://github.com/cardiffnlp/tweetnlp)
- [SemanTweet Search — Embedding-based tweet search](https://github.com/sankalp1999/semantweet-search)
- [RankLLM — Listwise reranking with LLMs](https://github.com/castorini/rank_llm)
- [Anthropic Batch API](https://docs.anthropic.com/en/api/creating-message-batches)
