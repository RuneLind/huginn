---
type: analysis
title: "Huginn-Muninn-Wiki Integration"
created: 2026-04-09
updated: 2026-04-09
tags: [integration, huginn, muninn, architecture]
sources: ["[[Sources — Claude Code]]", "[[Sources — AI General]]"]
---

# Huginn-Muninn-Wiki Integration

Architecture and implementation plan for the LLM Wiki system integrated with Huginn and Muninn. The wiki lives inside Huginn at `data/wiki/` with an Obsidian vault at `data/`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Huginn (knowledge engine)                                      │
│  ├── data/sources/youtube-articles/  (raw transcripts)          │
│  ├── data/sources/x-feed/           (twitter posts)             │
│  ├── data/wiki/                     (LLM-maintained wiki)       │
│  ├── Collections: youtube-summaries, x-feed, wiki (NEW)         │
│  ├── MCP + HTTP API                                             │
│  └── Vector + BM25 hybrid search                                │
│       │                                                         │
│       ├──── MCP ──── Muninn bots (wiki first, raw second)       │
│       └──── MCP ──── Claude Code wiki sessions                  │
│                                                                 │
│  Muninn (AI assistant)                                          │
│  ├── Bots query wiki collection for synthesized answers         │
│  ├── YouTube extension → sources/ → triggers wiki ingest        │
│  ├── Scheduled task: periodic wiki lint/update                  │
│  └── Wiki as cross-conversation long-term memory                │
│                                                                 │
│  Obsidian vault = huginn/data/                                  │
│  ├── sources/youtube-articles/  (browseable raw sources)        │
│  ├── sources/x-feed/            (browseable raw sources)        │
│  └── wiki/                      (the compiled knowledge graph)  │
│                                                                 │
│  Three wiki vaults (independent):                               │
│  1. Personal: huginn/data/wiki/ (YouTube + X/Twitter)           │
│  2. Work: separate vault (Jira + Confluence)                    │
│  3. Notion: separate vault (Notion sources)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Index wiki pages in Huginn (priority: high)

**What:** Create a new `wiki` collection in Huginn that indexes the `wiki/` directory from this vault.

**Why:** Muninn bots currently search raw YouTube transcripts via Huginn. Raw chunks lack synthesis — the bot has to piece together fragments. Wiki pages are pre-synthesized concept pages, entity profiles, and cross-referenced analyses. Searching wiki pages first gives dramatically better answers.

**How:**
- Use Huginn's `files_collection_create_cmd_adapter.py` to create a `wiki` collection pointing at `/Users/rune/private/youtube-transcripts/wiki/`
- Include all `.md` files in `wiki/concepts/`, `wiki/entities/`, `wiki/sources/`, `wiki/analyses/`
- Exclude `wiki/index.md` and `wiki/log.md` (navigation files, not content)
- Set up periodic re-index (daily or on-demand) to pick up wiki changes
- Tag documents by their frontmatter `type` field: `source`, `entity`, `concept`, `analysis`

**Search strategy for Muninn bots:**
```
1. Search wiki collection first (compiled knowledge)
2. If wiki result is sufficient → answer from wiki
3. If wiki result is thin → supplement with youtube-summaries collection (raw detail)
4. Always cite which wiki page or source was used
```

### 2. Auto-ingest pipeline (priority: medium)

**What:** When Muninn's YouTube Chrome extension processes a new video, automatically save the summary to this vault's `raw/` directory and optionally trigger a wiki ingest.

**Why:** Currently, new YouTube summaries go to Huginn's data directory but not to this vault. The wiki falls behind as new sources arrive. Automating the pipeline keeps the wiki current.

**How:**
- **Step A — Copy to raw/:** After Muninn's Chrome extension saves a YouTube summary, also write it to `/Users/rune/private/youtube-transcripts/raw/` in the appropriate subdirectory (based on existing category tags: `ai/claude-code`, `ai/general`, `tech`, `health`, etc.)
- **Step B — Trigger ingest (optional):** After new files land in `raw/`, either:
  - A Muninn scheduled task runs `claude -p "Ingest new sources in raw/ that aren't in the wiki yet"` on a schedule (e.g., daily)
  - A file watcher detects new files and triggers ingest
  - Manual: user tells Claude Code to ingest when ready
- **Step C — Re-index wiki in Huginn:** After wiki pages are updated, trigger Huginn collection update for the `wiki` collection

**File routing logic:**
```
YouTube summary → Muninn Chrome extension
  → Huginn youtube-summaries collection (existing, for raw search)
  → raw/{category}/ in this vault (NEW, for wiki ingest)
  → Claude Code ingest → wiki/ pages updated
  → Huginn wiki collection re-indexed
```

### 3. Muninn bot wiki query mode (priority: medium)

**What:** Configure Muninn bots to use wiki knowledge for answering questions about topics covered by the wiki.

**Why:** A user asking "How do Claude Code skills work?" in Telegram should get the synthesized [[Skills System]] page content, not fragments from 25 different raw transcripts.

**How:**
- Add wiki collection to the bot's Huginn MCP tool configuration
- Update bot system prompt to prefer wiki results over raw results
- Suggested prompt addition:
  ```
  When answering questions about AI tools, coding workflows, or topics
  in the knowledge base, search the "wiki" collection first. Wiki pages
  contain synthesized, cross-referenced knowledge. Only fall back to
  "youtube-summaries" for specific quotes, recent content not yet in
  the wiki, or when wiki results are insufficient.
  ```

### 4. Wiki as cross-conversation memory (priority: low)

**What:** Use wiki analysis pages as persistent memory that survives across all Muninn bot conversations.

**Why:** Muninn's semantic memory is per-conversation. Important insights, decisions, and analyses that should persist across all bots and chats can be filed as wiki analysis pages.

**How:**
- When a Muninn bot conversation produces a valuable insight or analysis, the bot (or user) can trigger filing it to the wiki
- This could be a Muninn bot command: `/file-to-wiki <topic>` → writes to `wiki/analyses/` via Claude Code
- The analysis then becomes searchable via Huginn's wiki collection for all future conversations

## Implementation Order

| Phase | Task | Effort | Dependency |
|-------|------|--------|------------|
| 1 | Create `wiki` collection in Huginn | Small | None |
| 2 | Configure Muninn bot to search wiki collection | Small | Phase 1 |
| 3 | Route YouTube extension output to `raw/` | Medium | None |
| 4 | Scheduled Claude Code ingest for new sources | Medium | Phase 3 |
| 5 | Auto re-index wiki collection after ingest | Small | Phase 1, 4 |
| 6 | `/file-to-wiki` bot command | Medium | Phase 1 |

Phase 1-2 can be done immediately and give the biggest value. Phase 3-5 build the auto-pipeline. Phase 6 is a nice-to-have.

## Key Paths

| Component | Path |
|-----------|------|
| Huginn root | `/Users/rune/source/private/huginn/` |
| Obsidian vault | `/Users/rune/source/private/huginn/data/` |
| Wiki pages | `/Users/rune/source/private/huginn/data/wiki/` |
| Wiki schema | `/Users/rune/source/private/huginn/data/wiki/CLAUDE.md` |
| YouTube sources | `/Users/rune/source/private/huginn/data/sources/youtube-articles/` |
| X/Twitter sources | `/Users/rune/source/private/huginn/data/sources/x-feed/` |
| Muninn | `/Users/rune/source/private/muninn/` |

## Wiki Structure (for Huginn indexing)

```
wiki/
├── index.md          # Skip — navigation only
├── log.md            # Skip — activity log only
├── concepts/         # Index — synthesized topic pages (15 pages)
├── entities/         # Index — people, products, orgs (11 pages)
├── sources/          # Index — themed source catalogs (11 pages)
└── analyses/         # Index — filed query results (0 pages, will grow)
```

Each wiki page has YAML frontmatter with `type`, `title`, `tags`, and `sources` fields. Huginn can use these for filtering and faceted search.

## See also

- [[Claude Code]]
- [[Context Management]]
- [[Skills System]]
- [[AI Agent Design Patterns]]
