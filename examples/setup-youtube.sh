#!/usr/bin/env bash
# Set up a YouTube transcript collection from a channel.
#
# Usage:
#   ./examples/setup-youtube.sh "https://www.youtube.com/@ChannelName/videos" "channel-name"
#
# This will:
#   1. Fetch all video transcripts from the channel
#   2. Preprocess markdown for better search quality
#   3. Index them into a searchable collection
#
# Prerequisites: uv sync
set -euo pipefail
cd "$(dirname "$0")/.."

CHANNEL_URL="${1:?Usage: setup-youtube.sh <channel-url> <collection-name>}"
COLLECTION="${2:?Usage: setup-youtube.sh <channel-url> <collection-name>}"
DATA_PATH="${3:-./data/sources/$COLLECTION}"

echo "==> Fetching transcripts from $CHANNEL_URL"
uv run youtube_fetch_cmd_adapter.py \
  --channelUrl "$CHANNEL_URL" \
  --channelName "$COLLECTION" \
  --outputPath "$DATA_PATH"

MD_PATH="$DATA_PATH/markdown/$COLLECTION"

if [ -d "$MD_PATH" ]; then
  echo "==> Preprocessing markdown files"
  uv run youtube_preprocess_md.py --saveMd "$MD_PATH"

  echo "==> Indexing collection: $COLLECTION"
  uv run files_collection_create_cmd_adapter.py \
    --basePath "$MD_PATH" \
    --collection "$COLLECTION"

  echo "==> Done! Search with:"
  echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
else
  echo "ERROR: Expected markdown at $MD_PATH but not found"
  exit 1
fi
