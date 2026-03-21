#!/usr/bin/env bash
# Set up and maintain an X/Twitter timeline collection.
#
# Usage:
#   ./examples/setup-x-timeline.sh [pages] [mode]
#
# Modes:
#   create (default)  — fetch + build full index
#   update            — fetch + incremental index update
#   watch             — fetch + update in a loop (default: every 60 min)
#
# Examples:
#   ./examples/setup-x-timeline.sh 5               # Full: fetch 5 pages, create collection
#   ./examples/setup-x-timeline.sh 3 update         # Incremental: fetch 3 pages, update index
#   ./examples/setup-x-timeline.sh 3 watch          # Loop: fetch 3 pages every 60 min
#   WATCH_INTERVAL=30 ./examples/setup-x-timeline.sh 3 watch  # Every 30 min
#
# Prerequisites:
#   - uv sync
#   - X auth set up: uv run scripts/x/auth_setup.py
set -euo pipefail
cd "$(dirname "$0")/.."

PAGES="${1:-5}"
MODE="${2:-create}"
COLLECTION="x-feed"
DATA_PATH="./data/sources/$COLLECTION"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"  # minutes

fetch_and_update() {
  echo "==> [$(date '+%H:%M:%S')] Fetching X timeline ($PAGES pages)"
  uv run scripts/x/fetchers/x_fetcher.py \
    --pages "$PAGES" \
    --saveMd "$DATA_PATH" \
    --skipExisting

  echo "==> Updating collection: $COLLECTION"
  uv run collection_update_cmd_adapter.py \
    --collection "$COLLECTION"
}

case "$MODE" in
  watch)
    echo "==> Watch mode: fetching every ${WATCH_INTERVAL}m (Ctrl+C to stop)"
    # First run — create collection if it doesn't exist
    if [ ! -d "data/collections/$COLLECTION" ]; then
      echo "==> Collection not found, creating initial index..."
      uv run scripts/x/fetchers/x_fetcher.py \
        --pages "$PAGES" \
        --saveMd "$DATA_PATH" \
        --skipExisting
      uv run files_collection_create_cmd_adapter.py \
        --basePath "$DATA_PATH" \
        --collection "$COLLECTION"
    else
      fetch_and_update
    fi
    while true; do
      echo "==> Sleeping ${WATCH_INTERVAL}m until next update..."
      sleep "$((WATCH_INTERVAL * 60))"
      fetch_and_update
    done
    ;;
  update)
    fetch_and_update
    ;;
  *)
    echo "==> Fetching X timeline ($PAGES pages) to $DATA_PATH"
    uv run scripts/x/fetchers/x_fetcher.py \
      --pages "$PAGES" \
      --saveMd "$DATA_PATH" \
      --skipExisting

    echo "==> Creating collection: $COLLECTION"
    uv run files_collection_create_cmd_adapter.py \
      --basePath "$DATA_PATH" \
      --collection "$COLLECTION"
    ;;
esac

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
