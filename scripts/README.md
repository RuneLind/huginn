# Scripts Directory

This directory contains data source-specific scripts for fetching, processing, and indexing content.

## Structure

```
scripts/
├── confluence/
│   ├── fetchers/          # Scripts for downloading Confluence pages
│   ├── processing/        # Conversion and reprocessing scripts
│   └── auth/             # Authentication files
├── jira/
│   ├── fetchers/          # Scripts for downloading Jira issues
│   ├── sanitizers/        # PII detection and redaction
│   └── auth/             # Authentication files
├── traces/
│   └── extract_query_doc_pairs.py  # Extract search traces from Claude sessions
└── daily_notion_update.sh # Daily automated Notion index update
```

## Confluence Scripts

### Fetchers
- **confluence_fetcher_hierarchical.py** - Main crawler with hierarchical structure (RECOMMENDED)
  - Preserves parent-child page relationships
  - Outputs both JSON and Markdown formats
  - Saves `fetch_metadata.json` for update detection
  - CLI options: `--space`, `--output`, `--base-url`, `--format`
- **confluence_check_updates.py** - Check for updates since last fetch
  - Reads `fetch_metadata.json` to determine last fetch time
  - Queries Confluence for modified pages
  - CLI options: `--space`, `--output`, `--show-pages`
- **confluence_fetcher_complete.py** - Alternative fetcher
- **confluence_playwright_fetcher.py** - Playwright-based authentication
- **deprecated_confluence_fetcher_fixed.py** - Old version (deprecated)

### Processing
- **reprocess_json_to_md.py** - Convert existing JSON to Markdown
- **reprocess_json_to_markdown.py** - Alternative conversion script
- **reprocess_single_file.py** - Process single file
- **test_layout_fix.py** - Test layout parsing fixes
- **confluence_collection_create_cmd_adapter.py** - Create vector collection

## Jira Scripts

### Fetchers
- **jira_fetcher.py** - Main Jira issue fetcher
- **jira_discover_fields.py** - Discover available Jira fields
- **jira_collection_create_cmd_adapter.py** - Create vector collection

### Sanitizers
- **pii_sanitizer.py** - PII detection and redaction (personnummer, emails, passwords)
- **sanitize_jira_files.py** - Batch scan/redact PII from Jira markdown files (`--dryRun` / `--apply`)

## Usage

Scripts are organized by data source. Navigate to the appropriate directory:

```bash
# Confluence
cd scripts/confluence/fetchers
uv run confluence_fetcher_hierarchical.py

# Jira
cd scripts/jira/fetchers
uv run jira_fetcher.py
```

## Authentication

Authentication files (`.json`) are stored in `auth/` subdirectories and are git-ignored for security.

## Deprecated Scripts

Old scripts are kept in `deprecated/` folders for reference but should not be used for new work.
