# X/Twitter Feed Fetcher — Work Plan

> **Status**: Huginn fetcher DONE. Muninn watcher DONE. Needs migration + end-to-end test.

## Goal

Fetch the user's X/Twitter home timeline and deliver a daily morning digest via Telegram.

## How It Works (Implemented)

No developer account needed. The fetcher uses cookie-based auth (auth_token + ct0 copied from Chrome DevTools) and makes direct HTTP requests to X's GraphQL API via httpx. No browser automation — just HTTP. Query IDs are dynamically extracted from X's JS bundles on each run.

### Key decisions during implementation

1. **Started with Playwright** (browser automation intercepting GraphQL responses), but X detects and blocks Playwright's bundled Chromium.
2. **Switched to direct HTTP** with cookies from the real browser. Undetectable since it makes identical requests to the web client, and faster since no browser process is needed.
3. **Auth verification** uses the notifications API (`/2/notifications/all.json`) because X removed the v1.1 account/verify endpoint. User data is read from the `user.core` path (not `legacy`, which X emptied out).
4. **JS bundle discovery** uses a separate unauthenticated httpx client because X rejects auth headers on the main page.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Huginn (Python, pure HTTP)                         │
│                                                     │
│  scripts/x/fetchers/x_fetcher.py                    │
│    1. Load cookies from x_auth.json                 │
│    2. Discover GraphQL query ID from JS bundles     │
│    3. Call HomeTimeline GraphQL API directly         │
│    4. Cursor-based pagination for multiple pages    │
│    5. Output tweets as JSON to stdout               │
│                                                     │
│  scripts/x/auth/x_auth.json  ← browser cookies     │
└───────────────────┬─────────────────────────────────┘
                    │  stdout: JSON array of tweets
                    ▼
┌─────────────────────────────────────────────────────┐
│  Muninn (TypeScript/Bun)                            │
│                                                     │
│  src/watchers/x.ts                                  │
│    1. Shell out: uv run x_fetcher.py --pages 3      │
│    2. Parse JSON output                             │
│    3. Haiku summarizes into morning digest           │
│    4. Return as WatcherAlert[]                      │
│                                                     │
│  Scheduler runs watcher every 24h                   │
│  → Sends digest via Telegram                        │
└─────────────────────────────────────────────────────┘
```

## Phase 1: Auth Setup (DONE)

**File:** `scripts/x/auth_setup.py`

Interactive CLI that guides the user through copying cookies from Chrome DevTools:

1. User opens Chrome, navigates to x.com, opens DevTools → Application → Cookies
2. User copies `auth_token` and `ct0` cookie values
3. Script verifies cookies work by calling X's notifications API
4. Saves `{ auth_token, ct0 }` as JSON to `scripts/x/auth/x_auth.json`

```bash
uv run scripts/x/auth_setup.py
# Prompts for auth_token and ct0 → verifies → saves
```

Re-run when session expires (probably every few weeks).

## Phase 2: Timeline Fetcher (DONE)

**File:** `scripts/x/fetchers/x_fetcher.py`

### Approach: Direct GraphQL API via HTTP

Pure HTTP with httpx — no Playwright, no browser process. Calls the same GraphQL endpoints that X's web client uses.

**Query ID discovery:** Fetches x.com's HTML (unauthenticated), finds JS bundle URLs, scans bundles for `queryId:"...",operationName:"HomeTimeline"`. Prioritizes api/endpoints bundles.

**Pagination:** Cursor-based. Each response contains a `Bottom` cursor used for the next page request.

**Tweet extraction:** Walks the GraphQL response structure (`data.home.home_timeline_urt.instructions[].entries[]`), handles `TweetWithVisibilityResults` wrappers, tombstones, retweets, quoted tweets, and media (images + video with best-bitrate selection).

**URL expansion:** Replaces t.co short URLs with their expanded form using entity data.

### CLI Interface

```bash
# Fetch latest ~20 tweets from home timeline
uv run scripts/x/fetchers/x_fetcher.py

# Fetch more (multiple pages, ~20 tweets each)
uv run scripts/x/fetchers/x_fetcher.py --pages 3

# Output to file instead of stdout
uv run scripts/x/fetchers/x_fetcher.py --output data/x/timeline.json

# Save as markdown files (for indexing into huginn collections)
uv run scripts/x/fetchers/x_fetcher.py --saveMd data/sources/x-timeline
```

### Error Handling

- **401 Unauthorized:** Session expired → tells user to re-run auth_setup.py
- **429 Rate limited:** Tells user to try again later
- **Duplicate tweets:** Deduplicates by tweet ID across pages
- **Empty page:** Stops pagination if no new tweets are found

## Phase 3: Muninn Watcher Integration (DONE)

**File:** `src/watchers/x.ts` (in muninn)

```typescript
export async function checkX(watcher: Watcher): Promise<WatcherAlert[]> {
    const huginnPath = path.resolve("../huginn");
    const result = await Bun.$`uv run ${huginnPath}/scripts/x/fetchers/x_fetcher.py --pages 3`
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

## Remaining Work

- [ ] DB migration to add `'x'` to watcher type CHECK constraint
- [ ] End-to-end test: trigger watcher → digest arrives on Telegram

## File Checklist

### Huginn (implemented)

```
scripts/x/
├── __init__.py
├── auth_setup.py              ← interactive cookie setup (no Playwright)
├── auth/
│   ├── .gitkeep
│   ├── README.md              ← instructions for cookie setup
│   └── x_auth.json            ← saved cookies (gitignored)
└── fetchers/
    ├── __init__.py
    └── x_fetcher.py            ← direct HTTP timeline fetcher
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
| X changes GraphQL schema | Fetcher breaks | Query ID auto-discovery adapts; tweet extraction may need updating |
| X changes JS bundle structure | Can't find query ID | Multiple regex patterns + fallback to broader bundle search |
| Session expires | No data fetched | Clear error message telling user to re-run auth_setup.py |
| Rate limiting | Incomplete timeline | Stops pagination gracefully; user can retry later |
| X removes notifications API | Auth verification breaks | Can try alternative lightweight endpoints |

## Optional Later: Index into Huginn Collections

Once fetching works, tweets can also be saved as markdown and indexed:

```bash
# Save as markdown
uv run scripts/x/fetchers/x_fetcher.py --saveMd data/sources/x-timeline

# Index into searchable collection
uv run files_collection_create_cmd_adapter.py -basePath data/sources/x-timeline -collection x-timeline
```

This lets you search your X history alongside Confluence/Jira/Notion via huginn's API. But that's a separate initiative — the morning digest doesn't need it.
