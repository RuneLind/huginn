# Huginn

Local RAG/knowledge search system. Python, FAISS + BM25 hybrid search, MCP integration.

Public repo: https://github.com/RuneLind/huginn

## Repo structure

```
huginn/                          # Public repo (this)
├── main/                        # Core library: indexing, search, graph, sources
├── *.py                         # Entry points (CLI adapters, MCP adapters, API server)
├── data/                        # Gitignored. Collections, sources, caches
│   ├── collections/             # Indexed collections (FAISS + BM25 indexes)
│   └── sources/                 # Raw source documents (confluence, jira, etc.)
├── docs/                        # Design docs, guides, plans
├── scripts/                     # Fetch/processing scripts (confluence, jira, etc.)
├── tests/                       # Pytest tests
└── huginn-*/                    # Private domain sub-repos (gitignored)
```

## Private sub-repos

Expect gitignored `huginn-*/` directories — these are private repos with their own `.git`, containing domain-specific collections, wikis, scripts, and credentials. See the "Advanced: Private Domain Collections" section in README.md for the pattern. Each may have its own CLAUDE.md with domain-specific instructions.

## Key entry points

- `knowledge_api_server.py` — HTTP API server
- `knowledge_api_mcp_adapter.py` — MCP server (single collection)
- `multi_collection_search_mcp_adapter.py` — MCP server (multi-collection)
- `files_collection_create_cmd_adapter.py` — Index local files into a collection
- `collection_update_cmd_adapter.py` — Update existing collection
- `collection_search_cmd_adapter.py` — CLI search

## Re-indexing a collection

Collections live in `data/collections/`. Source documents live in `data/sources/`. To re-index a collection after curating its source files:

```sh
.venv/bin/python files_collection_create_cmd_adapter.py \
  -collection <collection-name> \
  -basePath <source-path> \
  -excludePatterns '^\.excluded/.*' '^fetch_metadata\.json$'
```

**Escaping matters:** use single backslash in patterns (`^\.excluded/.*`), not double. Double backslashes produce broken regexes that silently index everything.

### Common collections

| Collection | Source path | Exclude patterns |
|---|---|---|
| `melosys-confluence-v3` | `./data/sources/melosys-confluence` | `^\.excluded/.*` `^fetch_metadata\.json$` |
| `jira-issues` | `./data/sources/jira-issues` | `^\.excluded/.*` |
| `work-wiki` | `./huginn-nav/wiki` | `index\.md` `log\.md` `CLAUDE\.md` |

### Verify after re-indexing

Check `data/collections/<name>/manifest.json` — confirm `numberOfDocuments` matches expectations and `excludePatterns` show single backslashes in JSON (e.g. `"^\\.excluded/.*"`, not `"^\\\\.excluded/.*"`).

## Development

- Python venv at `.venv/` — always use `.venv/bin/python`
- Tests: `.venv/bin/python -m pytest tests/`
- Detailed docs in `docs/` — check there for design decisions, architecture, and plans
