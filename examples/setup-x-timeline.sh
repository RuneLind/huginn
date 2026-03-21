#!/usr/bin/env bash
# Set up an X/Twitter timeline collection.
#
# Usage:
#   ./examples/setup-x-timeline.sh [pages] [update]
#
# Examples:
#   ./examples/setup-x-timeline.sh 5           # Full: fetch 5 pages, create collection
#   ./examples/setup-x-timeline.sh 3 update    # Incremental: fetch 3 pages, update index
#
# Prerequisites:
#   - uv sync
#   - X auth set up: uv run scripts/x/auth_setup.py
set -euo pipefail
cd "$(dirname "$0")/.."

PAGES="${1:-5}"
MODE="${2:-create}"
COLLECTION="x-timeline"
DATA_PATH="./data/sources/$COLLECTION"

echo "==> Fetching X timeline ($PAGES pages) to $DATA_PATH"
uv run scripts/x/fetchers/x_fetcher.py \
  --pages "$PAGES" \
  --saveMd "$DATA_PATH" \
  --skipExisting

if [ "$MODE" = "update" ]; then
  echo "==> Updating collection: $COLLECTION"
  uv run collection_update_cmd_adapter.py \
    --collection "$COLLECTION"
else
  echo "==> Creating collection: $COLLECTION"
  uv run files_collection_create_cmd_adapter.py \
    --basePath "$DATA_PATH" \
    --collection "$COLLECTION"
fi

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
