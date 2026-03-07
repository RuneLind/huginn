#!/usr/bin/env bash
# Set up a Jira project collection (Playwright-based).
#
# Usage:
#   JIRA_TOKEN="Bearer ..." ./examples/setup-jira.sh MYPROJECT my-jira
#
# This will:
#   1. Download all issues from a Jira project (opens browser for auth on first run)
#   2. Clean up empty/noise issues
#   3. Index them into a searchable collection
#
# Prerequisites:
#   - uv sync && uv run playwright install chromium
#   - JIRA_TOKEN env var (Bearer token), or JIRA_LOGIN + JIRA_PASSWORD
#   - For Cloud: ATLASSIAN_EMAIL + ATLASSIAN_TOKEN
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${1:?Usage: setup-jira.sh <project-key> <collection-name>}"
COLLECTION="${2:?Usage: setup-jira.sh <project-key> <collection-name>}"
DATA_PATH="${3:-./data/sources/$COLLECTION}"

echo "==> Fetching Jira issues for project $PROJECT to $DATA_PATH"
uv run scripts/jira/fetchers/jira_fetcher.py \
  --saveMd "$DATA_PATH" \
  --project "$PROJECT"

echo "==> Cleaning up noise issues"
uv run jira_cleanup_md.py \
  --saveMd "$DATA_PATH" \
  --minWordCount 30 \
  --maxAgeYears 2 \
  --dryRun

echo "==> Indexing collection: $COLLECTION"
uv run files_collection_create_cmd_adapter.py \
  --basePath "$DATA_PATH" \
  --collection "$COLLECTION" \
  --excludePatterns "^\.excluded/.*"

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
