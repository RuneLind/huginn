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
| `melosys-confluence-v3` | `./data/sources/melosys-confluence` | `^\.excluded/.*` `^fetch_metadata\.json$` (+ a path-specific exclude — see manifest) |
| `jira-issues` | `./data/sources/jira-issues` | `^\.excluded/.*` |
| `capra-notion-v9` | `./data/sources/capra-notion` | — |
| `nav-wiki` | `./huginn-nav/wiki` | `index\.md` `log\.md` `CLAUDE\.md` `^\.obsidian/.*` |
| `wiki` | `./huginn-jarvis/data/wiki` | `^index\.md$` `^log\.md$` `^CLAUDE\.md$` `^plans/.*` |

### Verify after re-indexing

Check `data/collections/<name>/manifest.json` — confirm `numberOfDocuments` matches expectations and `excludePatterns` show single backslashes in JSON (e.g. `"^\\.excluded/.*"`, not `"^\\\\.excluded/.*"`).

## LLM entity extraction (knowledge graph)

Extract entities and relationships from a collection using a local Ollama model. Outputs a `*_llm_graph.json` used for query expansion and graph context enrichment at search time.

```sh
uv run scripts/knowledge_graph/extract_entities_llm.py --collection <collection-name>
uv run scripts/knowledge_graph/extract_entities_llm.py --collection <collection-name> --limit 20  # test run
```

- Requires Ollama running locally with `qwen3.6:35b-a3b-coding-nvfp4` (or pass `--model`)
- Incremental: uses a `.cache.json` file, safe to stop and resume
- Output auto-routes to private sub-repos: NAV collections → `huginn-nav/scripts/knowledge_graph/`, others → `huginn-jarvis/scripts/knowledge_graph/`
- The API server auto-loads all `*_llm_graph.json` files from those paths at startup
- See `docs/graph-enhanced-rag.html` for full architecture documentation

## Development

- Python venv at `.venv/` — always use `.venv/bin/python` for entry points
- `uv` is also configured (`pyproject.toml` + `uv.lock`); `uv run <script>` works for ad-hoc scripts (e.g. the LLM extractor above)
- Tests: `.venv/bin/python -m pytest tests/`
- Detailed docs in `docs/` — check there for design decisions, architecture, and plans

## Running the API server

Local dev uses a personal `start.sh` (gitignored) that launches `knowledge_api_server.py` with the user's full set of collections and `KNOWLEDGE_GRAPH_PATH` / `JIRA_GRAPH_PATH` env vars. It's the canonical record of which collections are live and which graph JSONs auto-load. To run a slimmer subset manually:

```sh
uv run knowledge_api_server.py --collections <name> [<name> ...] --port 8321
```
