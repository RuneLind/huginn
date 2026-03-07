# Document Tagging Guide

How to add LLM-generated tags to document collections using Claude Code subagents.

## Overview

Documents in `data/sources/` have YAML frontmatter. We add a `tags:` field with 1-5 tags from a constrained taxonomy. Tags are:
- Injected into `indexedData` for FAISS embedding + BM25 keyword enrichment
- Stored as chunk metadata for `?tags=` filtering in the search API
- Picked from a fixed taxonomy JSON to prevent vocabulary explosion

## What's Already Done

- **my-confluence**: 289 files tagged with 50-tag taxonomy (`scripts/tagging/my_taxonomy.json`)
- **jira-issues**: 2778 files tagged with 57-tag taxonomy (`scripts/tagging/jira_taxonomy.json`)
- Pipeline support: `FilesDocumentConverter` reads `tags` from frontmatter, injects into indexed text
- Search API: `?tags=lovvalg,eessi` filter on `/api/search` endpoint
- MCP adapter: `tags` parameter on `search_knowledge()` tool

## How to Tag a New Collection (e.g., Notion)

### Step 1: Create Taxonomy via Discovery

Don't guess tags from folder names. Run free-form discovery on all files first.

**Option A: CLI script** (simpler, uses 10 parallel workers):
```bash
uv run scripts/tagging/discover_tags.py \
    --source data/sources/<collection> \
    --description "a Norwegian consulting company's internal Notion workspace" \
    --sample 200 --output discovery_results.json
```

**Option B: Claude Code subagents** (faster for large collections, uses 15+ parallel agents):

Split the source files into batches of ~20:
```bash
find data/sources/my-notion -name "*.md" -not -path "*/.excluded/*" | sort > /tmp/notion_files.txt
split -l 20 /tmp/notion_files.txt /tmp/notion_batch_
```

Create full-path batch files:
```bash
for f in /tmp/notion_batch_*; do
  sed "s|^|$(pwd)/|" "$f" > "${f}_full.txt"
done
```

Then in Claude Code, spawn parallel haiku subagents for discovery:
```
For each batch file, launch a Task agent (model: haiku, run_in_background: true):

Prompt:
"You are a document tagger for [DOMAIN DESCRIPTION]. For each file, read it
and suggest 3-5 FREE-FORM topic tags (lowercase, 1-3 words, specific).
Read file list from /tmp/notion_batch_XX_full.txt. Output TAGS: filename.md: tag1, tag2
for each. Then JSON: {"file_tags": {"path.md": ["tag1"], ...}}. Do NOT edit files."
```

After all agents complete, aggregate tags from the JSONL transcripts:
```python
# Parse from subagent transcripts in
# ~/.claude/projects/.../subagents/agent-<id>.jsonl
# Look for last assistant message containing {"file_tags": ...} in ```json blocks
# Count tag frequencies across all files
```

Review the frequency distribution. Tags appearing 3+ times are taxonomy candidates. Deduplicate synonyms (e.g., "self-employed" = "selvstendig-næringsdrivende").

### Step 2: Write Taxonomy JSON

Create `scripts/tagging/<collection>_taxonomy.json`:
```json
{
  "description": "Taxonomy for <collection>. Data-driven from discovery on N docs.",
  "tags": {
    "category1": ["tag1", "tag2", "tag3"],
    "category2": ["tag4", "tag5"]
  }
}
```

Aim for 30-60 tags across 2-4 categories. Too few = tags too broad, too many = inconsistent assignment.

### Step 3: Tag All Files with Constrained Taxonomy

Split files into batches of 20 (same as discovery). Spawn parallel haiku subagents:

```
Prompt:
"You are a document tagger. Read each markdown file, then add/replace the `tags:`
line in its YAML frontmatter.

TAXONOMY (only use these tags):
category1: tag1, tag2, tag3
category2: tag4, tag5

RULES:
- Pick 1-5 tags per file based on title, breadcrumb, and content
- ONLY use tags from the list above
- Add tags as comma-separated: `tags: tag1, tag2`
- If frontmatter exists, add `tags:` line before closing `---`
- If `tags:` already exists, replace it
- Use the Edit tool

Read file list from /tmp/notion_batch_XX_full.txt. Process ALL. Report at end."
```

### Step 4: Verify

Check coverage and distribution:
```bash
# Count tagged files
total=$(find data/sources/<collection> -name "*.md" -not -path "*/.excluded/*" | wc -l)
tagged=$(grep -rl "^tags:" data/sources/<collection>/ | wc -l)
echo "Tagged: $tagged / $total"

# Tag distribution
grep -rh "^tags:" data/sources/<collection>/ | sed 's/^tags: //' | \
  tr ',' '\n' | sed 's/^ *//' | sed 's/ *$//' | sort | uniq -c | sort -rn

# Check for taxonomy leaks (tags not in taxonomy)
# Compare distribution output against taxonomy JSON
```

Fix any untagged files manually. Fix any leaked tags by re-running the agent on those specific files.

### Step 5: Re-index

```bash
uv run files_collection_create_cmd_adapter.py \
  --basePath "./data/sources/<collection>" \
  --collection "<collection-name>" \
  --excludePatterns "^\.excluded/.*"
```

### Step 6: Test

```bash
# Search without filter
uv run collection_search_cmd_adapter.py --collection <name> --query "test query"

# Search with tag filter (via API)
curl "http://localhost:8321/api/search?q=test&tags=tag1&collection=<name>&brief=true"
```

## Architecture Notes

### How Tags Flow Through the Pipeline

1. **Frontmatter** (`tags: lovvalg, eessi`) in markdown files
2. **FilesDocumentConverter** extracts `tags` from `_FRONTMATTER_METADATA_FIELDS` set
3. **Chunk metadata**: tags propagated to every chunk's `metadata.tags`
4. **indexedData**: tags prepended as `tags: lovvalg, eessi\n` before chunk text (enriches both FAISS embeddings and BM25 tokens)
5. **Search API**: `_apply_metadata_filters()` checks `?tags=` parameter (comma-separated, match-any)
6. **MCP adapter**: `search_knowledge(tags="lovvalg")` passes through to API

### Key Files

| File | Role |
|------|------|
| `scripts/tagging/my_taxonomy.json` | Confluence tag taxonomy (50 tags) |
| `scripts/tagging/jira_taxonomy.json` | Jira issues tag taxonomy (57 tags) |
| `scripts/tagging/discover_tags.py` | Free-form discovery via Claude CLI |
| `scripts/tagging/tag_documents.py` | Constrained tagging via Claude CLI |
| `scripts/tagging/claude_cli.py` | Shared CLI helper (timeout, env, frontmatter parsing) |
| `main/sources/files/files_document_converter.py` | Reads tags from frontmatter, injects into indexedData |
| `knowledge_api_server.py` | `?tags=` filter on search endpoint |
| `knowledge_api_mcp_adapter.py` | `tags` param on MCP search tool |

### Performance

- 15 parallel haiku subagents process 289 files in ~2-3 minutes
- Each agent handles 20 files (read + edit frontmatter)
- No API costs — uses Max subscription via Claude Code subagents
- CLI-based scripts (`discover_tags.py`, `tag_documents.py`) also work but are ~10x slower due to process spawn overhead
