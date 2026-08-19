#!/usr/bin/env bash
# Set up the wiki collection from a wiki directory.
#
# Usage:
#   ./examples/setup-wiki.sh <wiki-path> [collection-name]
#
# Indexes all .md files in wiki/concepts/, wiki/entities/, wiki/sources/, and
# wiki/analyses/. Excludes index.md, log.md, and CLAUDE.md (navigation/schema
# files, not content).
#
# Prerequisites: uv sync
set -euo pipefail
cd "$(dirname "$0")/.."

WIKI_PATH="${1:?Usage: setup-wiki.sh <wiki-path> [collection-name]}"
COLLECTION="${2:-wiki}"

echo "==> Indexing wiki pages as collection: $COLLECTION"
uv run files_collection_create_cmd_adapter.py \
  --basePath "$WIKI_PATH" \
  --collection "$COLLECTION" \
  --includePatterns ".*\.md" \
  --excludePatterns "index\.md" "log\.md" "CLAUDE\.md"

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
