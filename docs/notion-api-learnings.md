# Notion API Learnings

Lessons learned from building a Notion source adapter that downloads an entire workspace as Markdown and indexes it for vector search.

## Page Discovery

### Search API (`POST /v1/search`)

The primary way to discover pages. Supports filtering by object type and sorting.

```python
client.search(
    filter={"value": "page", "property": "object"},
    sort={"direction": "descending", "timestamp": "last_edited_time"},
    page_size=100,
)
```

**Limitations:**
- No total count — you can't know how many pages exist upfront
- No filtering by parent or workspace section — returns everything the integration can see
- Pagination via `start_cursor` / `has_more`
- Also supports `filter.value = "data_source"` (discovered during our work, not well documented)

### Alternative: Tree Traversal via `--rootPageId`

Fetch a specific page, then recursively walk its `child_page` and `child_database` blocks:

```python
client.pages.retrieve(page_id=root_id)
client.blocks.children.list(block_id=page_id)
```

This gives you a subtree but requires more API calls per page.

## Page Hierarchy (The Hard Part)

### Parent Types

Every Notion page has a `parent` field with a `type`. We encountered four types:

| `parent.type` | What it means | How to resolve |
|---|---|---|
| `page_id` | Page is a child of another page | `client.pages.retrieve(page_id=...)` |
| `database_id` | Page is a row in a database | `client.databases.retrieve(database_id=...)` |
| `data_source_id` | Page is a row in a data source (new concept) | Bulk load via search API (see below) |
| `block_id` | Page is embedded inside a block (e.g. column, synced block) | `client.blocks.retrieve(block_id=...)` |
| `workspace` | Page is at workspace root | Terminal — stop walking |

### The `data_source_id` Problem

In our our workspace, **92% of pages** had `parent.type = "data_source_id"`. This is a newer Notion concept where databases can be "data sources" — essentially inline databases within teamspaces.

**Key discovery:** Data sources are NOT retrievable by direct API call (not a page, not a database). You must use the search API with `filter.value = "data_source"`:

```python
client.search(
    filter={"value": "data_source", "property": "object"},
    page_size=100,
)
```

Each data source has:
- `title` — the database name (e.g., "Team Meetings", "Tasks", "Conference Tickets 2025")
- `parent` — usually `{"type": "database_id", "database_id": "..."}`
- `database_parent` — the parent in the original hierarchy (can be `page_id` or `block_id`)

**Our approach:** Bulk-load all data sources on first encounter (~609 in our workspace, 8 API calls at 100/page) and cache them. This is much faster than trying to resolve individually.

### Walking the Full Parent Chain

To build a correct breadcrumb like `Teams -> Team HR -> Processes -> ... -> Page Title`, you need to walk up through multiple parent types:

```
page (data_source_id)
  → data source "Conference Tickets 2025" (database_parent: block_id)
    → block (parent: page_id)
      → page "2025" (parent: data_source_id)
        → data source "År" (database_parent: page_id)
          → page "Conferences" (parent: page_id)
            → page "Marketing" (parent: database_id)
              → database "HR Processes" (parent: page_id)
                → page "Team HR" (parent: data_source_id)
                  → data source "Teams" (database_parent: workspace)
                    → workspace (stop)
```

A typical chain is 5-12 levels deep. We cap at 15 to prevent infinite loops.

### Caching is Essential

Parent chain lookups are expensive (1 API call per level). But siblings share the same chain, so caching makes subsequent pages nearly free. We use a dict keyed by `page:{id}`, `db:{id}`, `ds:{id}`, `block:{id}`.

In practice, processing 8254 pages required ~2000 unique parent lookups (the rest were cache hits).

## Block Fetching

### Recursive Block Tree

Each page's content is a tree of blocks. Blocks can have children (toggles, columns, callouts, synced blocks, etc.):

```python
def fetch_all_blocks(page_id, depth=0, max_depth=10):
    blocks = []
    result = client.blocks.children.list(block_id=page_id, page_size=100)
    for block in result["results"]:
        if block.get("has_children"):
            block["children"] = fetch_all_blocks(block["id"], depth + 1)
        blocks.append(block)
    return blocks
```

This is the most expensive operation — a page with 50 blocks and nested toggles can require 5-10 API calls.

### Block Types We Handle

~20 block types converted to Markdown:

- **Text:** `paragraph`, `heading_1/2/3`, `bulleted_list_item`, `numbered_list_item`, `to_do`
- **Code:** `code` (fenced with language tag)
- **Formatting:** `quote`, `callout`, `divider`, `toggle`
- **Media:** `image`, `bookmark`, `embed`, `video`, `pdf`, `file`
- **Structure:** `table`/`table_row`, `column_list`/`column`, `synced_block`
- **References:** `child_page`, `child_database`, `equation`, `link_preview`

### Gotchas

- **Callout `icon` can be `null`** — not just an empty dict. Use `data.get("icon") or {}` instead of `data.get("icon", {})`.
- **Rich text annotations** — each text segment has `bold`, `italic`, `code`, `strikethrough`, and `href`. Apply in the right order to avoid nested markdown issues.
- **Table blocks** — children are `table_row` blocks with `cells` arrays. Each cell is itself a rich text array.
- **Synced blocks, columns** — just render their children recursively.

## Rate Limiting

### Notion's Limits

- 3 requests per second per integration
- The `notion-client` SDK handles 429 retries automatically with exponential backoff

### Our Approach

- Configurable `request_delay` (default 0.35s between calls)
- For the re-org script that only fetches metadata: reduced to 0.1s since calls are lighter
- With caching, the effective rate is much higher than raw API calls suggest

### Performance Numbers (our workspace, 8254 pages)

- Initial download (all blocks): ~45 minutes
- Re-org (metadata only, no blocks): ~15 minutes (heavy caching of parent chains)
- Data source bulk load: ~3 seconds (609 sources, 8 API calls)

## File Organization

### Markdown Output Format

Each page saved as `.md` with YAML frontmatter:

```yaml
---
title: "Page Title"
url: https://www.notion.so/abc123def456
last_edited_time: 2025-08-18T07:54:00.000Z
notion_id: 1bbcce31-1db4-80c5-a228-c5dbff210beb
---

# Page content as markdown...
```

### Folder Structure

Files are organized by breadcrumb hierarchy:
```
my-notion/
  Teams/
    Team HR/
      HR Processes/
        Marketing/
          Conferences/
            page.md
    Team Engineering/
      OKRs/
        page.md
```

### Filename Sanitization

- Replace `<>:"/\|?*` with `_`
- Collapse multiple spaces/underscores
- Truncate to 200 characters
- Handle collisions by appending `(notion_id[:8])` suffix

## Incremental Updates

### Strategy

Use `last_edited_time` from the manifest to only process pages modified since last sync:

```python
# Search API sorts by last_edited_time desc
# Stop when we hit pages older than cutoff
if page_time < start_from_time:
    return  # No more new pages
```

### `--skipExisting` for Resumability

On crash/interrupt, existing `.md` files are scanned for `notion_id` in frontmatter. Those IDs are passed to the reader as `skip_page_ids`, skipping expensive block fetching entirely.

## Architecture Decisions

### Why Not `notion2md`?

- No workspace traversal — you'd need to write all discovery code yourself
- Becomes two disconnected systems (discovery + conversion)
- Our adapter does both in one pass

### Why Not `notion-backup`?

- Uses undocumented internal Notion APIs (HTML export endpoint)
- Can break without notice
- No incremental update support

### Why `notion-client` SDK?

- Official Python SDK from Notion
- Built-in pagination helpers
- Automatic 429 retry with backoff
- Good typing support
- Actively maintained

### Reader-Converter Pattern

Follows the existing project architecture:
- **Reader** handles API communication, page discovery, block fetching
- **Converter** transforms to standard `{id, url, modifiedTime, text, chunks}` format
- **CLI adapter** adds `--saveMd` file writing as a side effect

This means the same reader/converter works for both vector indexing and Markdown export.
