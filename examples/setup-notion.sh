#!/usr/bin/env bash
# Set up a Notion workspace collection.
#
# Usage:
#   NOTION_TOKEN="secret_..." ./examples/setup-notion.sh my-notion
#
# This will:
#   1. Download all pages from your Notion workspace as markdown
#   2. Clean up empty/stub pages
#   3. Index them into a searchable collection
#
# Prerequisites:
#   - uv sync
#   - NOTION_TOKEN env var (create at https://www.notion.so/my-integrations)
#   - Share your Notion pages/databases with the integration
set -euo pipefail
cd "$(dirname "$0")/.."

COLLECTION="${1:?Usage: setup-notion.sh <collection-name>}"
DATA_PATH="${2:-./data/sources/$COLLECTION}"

if [ -z "${NOTION_TOKEN:-}" ]; then
  echo "ERROR: NOTION_TOKEN environment variable must be set"
  echo "  Create an integration at https://www.notion.so/my-integrations"
  exit 1
fi

echo "==> Downloading Notion pages to $DATA_PATH"
uv run notion_collection_create_cmd_adapter.py \
  --downloadOnly \
  --saveMd "$DATA_PATH"

echo "==> Cleaning up empty/stub pages"
uv run notion_cleanup_md.py --saveMd "$DATA_PATH"

echo "==> Indexing collection: $COLLECTION"
uv run files_collection_create_cmd_adapter.py \
  --basePath "$DATA_PATH" \
  --collection "$COLLECTION" \
  --excludePatterns "^\.excluded/.*"

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
