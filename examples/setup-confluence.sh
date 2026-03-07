#!/usr/bin/env bash
# Set up a Confluence space collection (Playwright-based).
#
# Usage:
#   CONF_TOKEN="Bearer ..." ./examples/setup-confluence.sh MYSPACE my-confluence
#
# This will:
#   1. Download all pages from a Confluence space (opens browser for auth on first run)
#   2. Clean up noise pages (meeting notes, archives, etc.)
#   3. Index them into a searchable collection
#
# Prerequisites:
#   - uv sync && uv run playwright install chromium
#   - CONF_TOKEN env var (Bearer token), or CONF_LOGIN + CONF_PASSWORD
#   - For Cloud: ATLASSIAN_EMAIL + ATLASSIAN_TOKEN
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE="${1:?Usage: setup-confluence.sh <space-key> <collection-name>}"
COLLECTION="${2:?Usage: setup-confluence.sh <space-key> <collection-name>}"
DATA_PATH="${3:-./data/sources/$COLLECTION}"

echo "==> Fetching Confluence space $SPACE to $DATA_PATH"
uv run scripts/confluence/fetchers/confluence_fetcher_hierarchical.py \
  --space "$SPACE" \
  --saveMd "$DATA_PATH"

echo "==> Cleaning up noise pages"
uv run confluence_cleanup_md.py \
  --saveMd "$DATA_PATH" \
  --minWordCount 30 \
  --sanitize

echo "==> Indexing collection: $COLLECTION"
uv run files_collection_create_cmd_adapter.py \
  --basePath "$DATA_PATH" \
  --collection "$COLLECTION" \
  --excludePatterns "^\.excluded/.*" "^fetch_metadata\.json$"

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
