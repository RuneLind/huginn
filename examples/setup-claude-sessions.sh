#!/usr/bin/env bash
# Set up a collection from Claude Code session transcripts.
#
# Usage:
#   ./examples/setup-claude-sessions.sh
#   ./examples/setup-claude-sessions.sh --projects myproject1 myproject2
#
# This will:
#   1. Convert Claude Code JSONL sessions to markdown
#   2. Index them for searching past conversations
#
# Prerequisites: uv sync
set -euo pipefail
cd "$(dirname "$0")/.."

COLLECTION="${1:-claude-sessions}"
DATA_PATH="./data/sources/$COLLECTION"
shift 2>/dev/null || true

echo "==> Converting Claude Code sessions to markdown"
uv run scripts/claude_sessions/claude_sessions_to_markdown.py \
  --saveMd "$DATA_PATH" \
  "$@"

echo "==> Indexing collection: $COLLECTION"
uv run files_collection_create_cmd_adapter.py \
  --basePath "$DATA_PATH" \
  --collection "$COLLECTION" \
  --excludePatterns "^\.excluded/.*"

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
