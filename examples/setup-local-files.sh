#!/usr/bin/env bash
# Set up a collection from local files (markdown, PDF, DOCX, etc.).
#
# Usage:
#   ./examples/setup-local-files.sh /path/to/docs my-docs
#
# Supports: .md, .txt, .pdf, .docx, .html, .rst, and more via Unstructured.
#
# Prerequisites: uv sync
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE_PATH="${1:?Usage: setup-local-files.sh <source-path> <collection-name>}"
COLLECTION="${2:?Usage: setup-local-files.sh <source-path> <collection-name>}"

echo "==> Indexing $SOURCE_PATH as collection: $COLLECTION"
uv run files_collection_create_cmd_adapter.py \
  --basePath "$SOURCE_PATH" \
  --collection "$COLLECTION"

echo "==> Done! Search with:"
echo "    uv run collection_search_cmd_adapter.py --collection $COLLECTION --query \"your search\""
