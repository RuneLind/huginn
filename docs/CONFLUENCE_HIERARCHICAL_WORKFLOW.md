# Confluence Hierarchical Fetcher Workflow

This document describes how to fetch, update, and search Confluence pages using the hierarchical fetcher, which preserves the page structure from Confluence.

## Overview

The hierarchical fetcher downloads all pages from a Confluence space while preserving:
- Parent-child page relationships (folder structure)
- Page metadata (title, URL, breadcrumbs)
- Content in both JSON and Markdown formats

## Quick Start

```bash
# 1. Fetch pages from Confluence (opens browser for authentication)
uv run scripts/confluence/fetchers/confluence_fetcher_hierarchical.py --space MYSPACE

# 2. Check if there are updates available
uv run scripts/confluence/fetchers/confluence_check_updates.py --space MYSPACE

# 3. Index the downloaded pages for vector search
uv run files_collection_create_cmd_adapter.py \
  --basePath "./data/downloaded/confluence_hierarchical/markdown" \
  --collection "my-confluence"

# 4. Search the collection
uv run collection_search_cmd_adapter.py --collection "my-confluence" --query "your search query"
```

## Workflow Steps

### Step 1: Fetch Pages from Confluence

The fetcher uses Playwright to authenticate with Confluence and downloads all pages from the specified space.

```bash
uv run scripts/confluence/fetchers/confluence_fetcher_hierarchical.py --space MYSPACE
```

**Options:**
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--space` | `-s` | MYSPACE | Confluence space key |
| `--output` | `-o` | `./data/downloaded/confluence_hierarchical` | Output directory |
| `--base-url` | `-u` | `https://confluence.example.com` | Confluence base URL |
| `--format` | `-f` | both | Output format: `json`, `markdown`, or `both` |

**First-time authentication:**
1. A browser window will open
2. Log in to Confluence using your credentials
3. The script will save your session for future runs

**Output structure:**
```
data/downloaded/confluence_hierarchical/
├── fetch_metadata.json      # Fetch timestamp and stats
├── json/                    # Raw JSON with HTML content
│   └── My Team/
│       ├── Page A.json
│       └── Subfolder/
│           └── Page B.json
└── markdown/                # Clean markdown with frontmatter
    └── My Team/
        ├── Page A.md
        └── Subfolder/
            └── Page B.md
```

### Step 2: Check for Updates

Before re-fetching, you can check if there are any updates in Confluence:

```bash
uv run scripts/confluence/fetchers/confluence_check_updates.py --space MYSPACE
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--space` | `-s` | Confluence space key |
| `--output` | `-o` | Directory containing fetch_metadata.json |
| `--show-pages` | `-p` | List modified pages |

**Example output:**
```
📋 Previous fetch:
   Time: 2025-10-06 16:29:00 UTC
   Pages: 970

🔎 Checking for updates since 2025-10-06...

==================================================
📊 Updates found:
   Modified pages: 15
   Total in space: 985
   Page count change: +15 (was 970)

💡 To update, run:
   uv run confluence_fetcher_hierarchical.py --space MYSPACE
==================================================
```

### Step 3: Index for Vector Search

Index the markdown files to enable semantic search:

```bash
uv run files_collection_create_cmd_adapter.py \
  --basePath "./data/downloaded/confluence_hierarchical/markdown" \
  --collection "my-confluence"
```

This creates a vector database in `./data/collections/my-confluence/`.

### Step 4: Search the Collection

```bash
uv run collection_search_cmd_adapter.py \
  --collection "my-confluence" \
  --query "how does the membership period work"
```

## Re-fetching and Updates

The fetcher performs a **full refresh** each time - it downloads all pages from the space. This ensures:
- No pages are missed due to edge cases
- Deleted pages are handled correctly
- The local copy exactly matches Confluence

**Recommended workflow:**
1. Run `confluence_check_updates.py` to see if updates exist
2. If updates found, run `confluence_fetcher_hierarchical.py`
3. Re-index with `files_collection_create_cmd_adapter.py`

## Markdown Format

Each markdown file includes YAML frontmatter:

```markdown
---
title: Page Title
page_id: 12345678
space: MYSPACE
breadcrumb: My Team > Subfolder
url: https://confluence.example.com/spaces/MYSPACE/pages/12345678
---

# Page content here...
```

The frontmatter is preserved during indexing, making page URLs available in search results.

## Authentication

Session state is stored in:
```
scripts/confluence/auth/confluence_auth.json
```

This file is git-ignored. If authentication fails:
1. Delete the auth file
2. Run the fetcher again
3. Log in when the browser opens

## Troubleshooting

### "Authentication timeout"
- Your saved session may have expired
- Delete `scripts/confluence/auth/confluence_auth.json` and try again

### "No pages were fetched"
- Verify the space key is correct
- Check you have access to the space
- Ensure the base URL is correct

### Missing pages in search results
- Re-run the indexer after fetching new pages
- Check that the markdown files exist in the output directory

## File Locations

| Item | Location |
|------|----------|
| Fetcher script | `scripts/confluence/fetchers/confluence_fetcher_hierarchical.py` |
| Update checker | `scripts/confluence/fetchers/confluence_check_updates.py` |
| Auth state | `scripts/confluence/auth/confluence_auth.json` |
| Downloaded pages | `data/downloaded/confluence_hierarchical/` |
| Fetch metadata | `data/downloaded/confluence_hierarchical/fetch_metadata.json` |
| Vector index | `data/collections/my-confluence/` |
