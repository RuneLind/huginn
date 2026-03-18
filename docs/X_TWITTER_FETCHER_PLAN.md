# X/Twitter Feed Fetcher — Work Plan

## Goal

Fetch the user's X/Twitter home timeline using Playwright (real browser, regular user login — no developer account needed). Output structured JSON that muninn can consume for a daily morning digest.

## Why Playwright?

- No X developer account required — logs in as a regular user
- Same pattern as Confluence and Jira fetchers (proven in this project)
- Real browser bypasses Cloudflare and bot detection
- Cookie persistence via `storage_state` — only need to log in once

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Huginn (Python + Playwright)                       │
│                                                     │
│  scripts/x/fetchers/x_fetcher.py                    │
│    1. Load auth cookies from x_auth.json            │
│    2. Navigate to x.com/home                        │
│    3. Intercept GraphQL API responses (timeline)    │
│    4. Output tweets as JSON to stdout               │
│                                                     │
│  scripts/x/auth/x_auth.json  ← saved browser state │
└───────────────────┬─────────────────────────────────┘
                    │  stdout: JSON array of tweets
                    ▼
┌─────────────────────────────────────────────────────┐
│  Muninn (TypeScript/Bun)                            │
│                                                     │
│  src/watchers/x.ts                                  │
│    1. Shell out: uv run scripts/x/.../x_fetcher.py  │
│    2. Parse JSON output                             │
│    3. Haiku summarizes into morning digest           │
│    4. Return as WatcherAlert[]                      │
│                                                     │
│  Scheduler runs watcher once per day (morning)      │
│  → Sends digest via Telegram                        │
└─────────────────────────────────────────────────────┘
```

## Phase 1: Auth Setup (huginn)

**File:** `scripts/x/auth_setup.py`

Same pattern as Confluence/Jira:
1. Launch Chromium with `headless=False`
2. Navigate to `https://x.com/login`
3. User logs in manually (handles 2FA, captcha, whatever X throws)
4. Wait for redirect to `x.com/home`
5. Save `storage_state` → `scripts/x/auth/x_auth.json`
6. Print confirmation, close browser

```bash
uv run scripts/x/auth_setup.py
# Opens browser → user logs in → cookies saved
```

Re-run when session expires (probably every few weeks).

## Phase 2: Timeline Fetcher (huginn)

**File:** `scripts/x/fetchers/x_fetcher.py`

### Approach: GraphQL API Interception

X's frontend makes GraphQL calls to load the timeline. Instead of scraping the DOM (fragile), we intercept these API responses which contain clean structured data.

```python
async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state="scripts/x/auth/x_auth.json"
    )

    tweets = []

    # Intercept timeline GraphQL responses
    async def handle_response(response):
        url = response.url
        if "HomeTimeline" in url or "HomeLatestTimeline" in url:
            try:
                data = await response.json()
                tweets.extend(extract_tweets(data))
            except:
                pass

    page = await context.new_page()
    page.on("response", handle_response)

    await page.goto("https://x.com/home")
    await page.wait_for_timeout(5000)  # Let timeline load

    # Optional: scroll to load more
    for _ in range(scroll_count):
        await page.keyboard.press("End")
        await page.wait_for_timeout(2000)

    # Output as JSON
    print(json.dumps(tweets))
    await browser.close()
```

### Tweet Extraction

X's GraphQL response has nested structure. Extract:

```python
def extract_tweets(graphql_data) -> list[dict]:
    """Walk GraphQL response, extract tweet objects."""
    # Navigate: data.home.home_timeline_urt.instructions[].entries[]
    # Each entry has content.itemContent.tweet_results.result
    # Extract:
    return [{
        "id": tweet_id,
        "author": display_name,
        "handle": screen_name,
        "text": full_text,
        "created_at": timestamp,
        "url": f"https://x.com/{screen_name}/status/{tweet_id}",
        "likes": favorite_count,
        "retweets": retweet_count,
        "replies": reply_count,
        "is_retweet": bool,
        "quoted_tweet": { ... } or None,
        "media": [{ "type": "image|video", "url": ... }],
    }]
```

### CLI Interface

```bash
# Fetch latest ~50 tweets from home timeline
uv run scripts/x/fetchers/x_fetcher.py

# Fetch more (scroll N times)
uv run scripts/x/fetchers/x_fetcher.py --scrolls 5

# Output to file instead of stdout
uv run scripts/x/fetchers/x_fetcher.py --output data/x/timeline.json

# Save as markdown files (for indexing into huginn collections)
uv run scripts/x/fetchers/x_fetcher.py --saveMd data/sources/x-timeline
```

### Fallback: DOM Scraping

If GraphQL interception proves unstable (X changes response format), fall back to DOM scraping:

```python
# Each tweet is an <article> element
articles = await page.query_selector_all('article[data-testid="tweet"]')
for article in articles:
    text = await article.query_selector('[data-testid="tweetText"]')
    author = await article.query_selector('[data-testid="User-Name"]')
    # ...extract content...
```

DOM scraping is more fragile but simpler to debug.

## Phase 3: Muninn Watcher Integration

**File:** `src/watchers/x.ts` (in muninn)

```typescript
export async function checkX(watcher: Watcher): Promise<WatcherAlert[]> {
    const huginnPath = path.resolve("../huginn");
    const result = await Bun.$`uv run ${huginnPath}/scripts/x/fetchers/x_fetcher.py --scrolls 3`
        .cwd(huginnPath)
        .text();

    const tweets = JSON.parse(result);
    if (tweets.length === 0) return [];

    // Summarize with Haiku
    const prompt = `Summarize these ${tweets.length} tweets from my X timeline into a morning digest.
Group by topic/theme. Highlight anything important or trending.
Keep it concise — max 10-15 bullet points.

Tweets:
${JSON.stringify(tweets, null, 2)}`;

    const { result: summary } = await spawnHaiku(prompt, "watcher-x", botName);

    return [{
        id: `x-digest-${Date.now()}`,
        source: "x",
        summary: summary,
        urgency: "low",
    }];
}
```

### Watcher Type Registration

1. Add `"x"` to `WatcherType` in `src/types.ts`
2. Add case in `runChecker()` in `src/watchers/runner.ts`
3. DB migration to add `'x'` to the CHECK constraint
4. Update `/watch` command help text

### Usage

```
/watch x                    ← creates watcher, default 24h interval
/quiet 22-08               ← quiet hours (already exists)
```

Morning digest arrives as a single Telegram message.

## File Checklist

### Huginn (new files)

```
scripts/x/
├── __init__.py
├── auth_setup.py              ← interactive browser login + cookie save
├── auth/
│   ├── .gitkeep
│   └── x_auth.json            ← saved browser state (gitignored)
└── fetchers/
    ├── __init__.py
    └── x_fetcher.py            ← Playwright timeline fetcher
```

### Muninn (changes)

```
src/watchers/x.ts               ← new: X watcher checker
src/watchers/runner.ts           ← modify: add "x" case to runChecker
src/types.ts                     ← modify: add "x" to WatcherType
src/bot/watcher-commands.ts      ← modify: update help text
db/migrations/NNN_add_x_watcher.sql  ← new: add 'x' to CHECK constraint
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| X changes GraphQL schema | Fetcher breaks | Fall back to DOM scraping; both approaches in codebase |
| Session expires | No data fetched | Watcher logs warning, user re-runs auth_setup.py |
| X detects automation | Account flagged | headless=True + realistic viewport/UA; read-only usage is low risk |
| Rate limiting / slow loads | Incomplete timeline | Configurable scroll count + timeouts |
| Playwright startup overhead | Slow watcher tick | Acceptable for once-daily runs |

## Build Order

1. `scripts/x/auth_setup.py` — get login working, verify cookies persist
2. `scripts/x/fetchers/x_fetcher.py` — fetch timeline, figure out GraphQL structure
3. Test manually: `uv run scripts/x/fetchers/x_fetcher.py | jq .`
4. `src/watchers/x.ts` + runner wiring in muninn
5. DB migration + type updates
6. End-to-end test: trigger watcher → digest arrives on Telegram

## Optional Later: Index into Huginn Collections

Once fetching works, tweets can also be saved as markdown and indexed:

```bash
# Save as markdown
uv run scripts/x/fetchers/x_fetcher.py --saveMd data/sources/x-timeline

# Index into searchable collection
uv run files_collection_create_cmd_adapter.py -basePath data/sources/x-timeline -collection x-timeline
```

This lets you search your X history alongside Confluence/Jira/Notion via huginn's API. But that's a separate initiative — the morning digest doesn't need it.
