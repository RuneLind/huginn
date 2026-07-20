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
| `nav-wiki` | `./huginn-nav/wiki` | `index\.md` `log\.md` `CLAUDE\.md` `^\.obsidian/.*` `^\.claude/.*` |
| `wiki` | `./huginn-jarvis/data/wiki` | `^index\.md$` `^log\.md$` `^CLAUDE\.md$` `^plans/.*` `^Clippings/.*` `^\..*` (dot-dirs: `.obsidian/`, `.smart-env/`, `.understand-anything/`) |

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
- Output routing (no private collection names live in this public repo):
  1. `--output <path>` always wins.
  2. Else a `graph_routing.json` in one of the private sub-repo dirs (`huginn-*/scripts/knowledge_graph/`) or `./scripts/knowledge_graph/`. Each routing file either lists owned collections (`{"collections": [...]}`) or is the catch-all (`{"default": true}`). A listed collection writes into that file's dir; unlisted collections go to the `default` dir.
  3. Else the run fails and asks for `--output`.
- The output graph is stamped with a `source_stamp` (`collection`, `document_count` from the manifest's `numberOfDocuments`, `last_modified_document_time` from `lastModifiedDocumentTime` — chosen because `updatedTime` moves on every reindex run, even no-ops). A `--limit` run stamps the truncated count so partial graphs report stale. At load time the server compares the stamp against the collection's current `manifest.json` and logs a warning on divergence — a staleness signal, nothing rebuilds. Old unstamped graphs load unchanged.
- `extract_jira_graph.py` routes its `jira_graph.json` output the same way, keyed by the `--source` directory name.
- The API server auto-loads all `*_llm_graph.json` files from those paths at startup
- See `docs/graph-enhanced-rag.html` for full architecture documentation

## Development

- Python venv at `.venv/` — always use `.venv/bin/python` for entry points
- `uv` is also configured (`pyproject.toml` + `uv.lock`); `uv run <script>` works for ad-hoc scripts (e.g. the LLM extractor above)
- Tests: `.venv/bin/python -m pytest tests/`
- Detailed docs in `docs/` — check there for design decisions, architecture, and plans

## Indexing run ledger

Durable per-collection history of indexing runs, so a dashboard can show when each
job last ran, how long it took, and whether it failed. Backed by JSONL files at
`data/state/runs/<collection>.jsonl` (gitignored), written by
`main/runtime/indexing_run_ledger.py`.

- Read it over HTTP: `GET /api/indexing/jobs` — per collection returns `current`
  (live status + elapsed), `lastRun`, `history`, `medianDurationSeconds` split by
  variant, `schedule` (from the installed `~/Library/LaunchAgents/com.huginn.*.plist`),
  `nextRunAt`, and `loaded`. Rows are the union of ledger files and served
  collections; a collection this server does not serve appears with
  `loaded: false` rather than being hidden. The response is a pinned contract
  (the muninn dashboard couples to it): `lastRun` is the fixed `LAST_RUN_FIELDS`
  projection, never the raw folded record; `current` is the single running
  channel merging in-memory reindex state and ledger-side script runs
  (`source`: `reindex`/`script`/`both` — API-triggered reindexes report `both`
  because `try_begin_update` writes the ledger's opening partial); `nextRunAt`
  is UTC while the raw `schedule` dict stays launchd machine-local wall-clock,
  tagged `timezone: "local"`; the median window is fixed (`MEDIAN_WINDOW_RUNS`),
  independent of the `history` param.
- **Schedule routing:** `main/runtime/indexing_schedule.py` maps job → collections
  by script basename. That table is **empty in this public repo by design** —
  most of the scheduled collection names were never public and one is
  customer-adjacent, which `CLAUDE.local.md` bans outright. The names live in
  each private sub-repo's `scripts/schedule_routing.json`, discovered under
  `huginn-*/scripts/`, mirroring the `graph_routing.json` precedent. No routing
  file ⇒ `schedule: null`, the designed degradation. A plist whose
  `StartCalendarInterval` is 24 entries at one minute reports
  `{kind: "hourly"}` rather than the first entry's wall-clock time.
- Writers: `KnowledgeStore.__finish_update` (the API path) and
  `collection_update_cmd_adapter.py` (the CLI fallback). Both emit a `reindex`
  phase. `try_begin_update` also appends an *opening* partial, so a server
  restarted mid-reindex leaves a trace instead of nothing.
- **Script phases:** all seven shell jobs report their own phases via
  `scripts/lib/indexing_run.sh` — `run_begin` / `run_variant` / `phase_begin` /
  `phase_end` / `run_end`. This is what makes the non-reindex work visible: the
  fetch-then-index jobs fold to a whole-job duration several times their
  reindex (one measured 110s against 19s), and the hourly feed job's
  fetch/score phases were previously outside any record at all. Each converted
  step is classified fatal or non-fatal explicitly — the scripts genuinely
  differ, and wrapping them mechanically would silently change which failures
  are fatal.
  `run_variant` reclassifies a run already in flight, for the hourly feed job:
  it only learns whether it is an incremental update or a full rebuild after cleanup
  reports what it deleted, and the two differ by an order of magnitude. The
  closing record outranks the opening partial's guess.
  `run_end` POSTs to `POST /api/indexing/runs`, falling back to
  `python -m main.runtime.indexing_run_ledger append --file -` when the API is
  down. Never `>>` the JSONL from shell: macOS has no `flock(1)`, so a redirect
  cannot take the `LOCK_EX` every other writer holds.
  Three rules the helper exists to enforce, all of which otherwise abort an
  unattended job under `set -euo pipefail` — trading "no observability" for
  "no indexing", which is worse. `tests/test_indexing_run_helper.py` asserts all
  three, so read that before editing the helper:
  1. Every exported helper returns 0, and every call site adds `|| true`.
  2. `RUN_ID` is defaulted in the stub block; call sites use `${RUN_ID:-}`.
  3. `indexing_run.sh` **ends with an explicit `return 0`** — `.` exits with the
     status of the sourced file's last command, and neither the `&&`/`||` nor
     the `if/else` sourcing form fixes that.
  Everything the helper exports is **observational**, which is what makes the
  no-op stub guard sound. `poll_update_status` stays duplicated in each script
  on purpose: it is functional, so stubbing it to `:` would make a script treat
  every reindex as instantly complete.
- Phases carry a per-phase `startedAt` (all three writers: `__record_run`, the
  CLI adapter, and `phase_begin` in the shell helper — identical fixed-width UTC
  format, so lexicographic sort is chronological). The fold sorts phases by it;
  legacy phases without the field keep their arrival position — do not "fix"
  that fallback, a naive sort scrambles backfilled history.
- A run whose writer appended a `stage: "begin"` but never a matching
  `stage: "end"` folds to `running`, then `incomplete` past a threshold that is
  cadence-aware: the jobs endpoint derives `max(2 × schedule cadence, 2h)` per
  collection, falling back to the flat `INCOMPLETE_AFTER_SECONDS` (6h) when no
  schedule is known. The ledger itself never imports the schedule module — the
  caller passes `incomplete_after`; keep that layering.
- `POST /api/indexing/runs` bounds the request body (256 KiB, Content-Length
  check plus a bounded streamed read for chunked bodies). `load_schedules()` is
  cached on an mtime signature over the plists and routing files; it returns
  the shared cached dict — treat it as read-only.
- `POST /api/collections/{name}/reload` swaps in a rebuilt on-disk index
  without a server restart (404 for unserved collections; a failed reload keeps
  the previous searcher). The x-feed full rebuild uses it after building into a
  temp collection name and two-rename-swapping into place — the collection dir
  is no longer deleted in place, and the running server picks up the new index
  immediately.
- **`skipped` is not `succeeded`.** A reindex skipped because the API answered
  409 exits 0, and huginn writes no record at all on that path (`try_begin_update`
  returns False before the opening partial), so recording the phase as
  `succeeded` would assert an index freshness the run never delivered — every
  hour, for the hourly job where 409 is the likeliest outcome. Call sites pass
  the literal `skipped` to `phase_end`. It is deliberately NOT a degradation
  (another process is doing that exact work; alarming would train the reader to
  ignore `degraded`): neutral beside real work, `skipped` when every phase was,
  loses to any genuine failure, and excluded from `medianDurationSeconds`.
  A phase with no status at all degrades the run — absence of an outcome is not
  evidence of a good one.
- **Correlation:** `POST /api/collections/{name}/update` takes an optional body
  `{runId, job, trigger}`, and the CLI adapter takes `--run-id/--job/--trigger`.
  Records sharing a `runId` are folded at read time, which is how a wrapping
  script's tagging phase and huginn's reindex phase become one run. Passing no
  body is still valid and unchanged.
- `HUGINN_RUNS_DIR` overrides the ledger directory (the test suite points it at a
  tmp dir).
- Locking is load-bearing: take the flock on `<collection>.lock` BEFORE opening
  the JSONL, never cache the data fd. Compaction swaps the inode via `os.replace`,
  so an fd opened before the lock writes into an unlinked inode and the record is
  lost silently. `tests/test_indexing_run_ledger.py` has a test that fails if this
  ordering is inverted.

One-off backfill from the existing `logs/daily_*.log` files (already run; it is
idempotent, keyed on `runId`):

```sh
.venv/bin/python scripts/backfill_indexing_runs.py --dry-run   # summary only
.venv/bin/python scripts/backfill_indexing_runs.py
```

## Running the API server

Local dev uses a personal `start.sh` (gitignored) that launches `knowledge_api_server.py` with the user's full set of collections and `KNOWLEDGE_GRAPH_PATH` / `JIRA_GRAPH_PATH` env vars. It's the canonical record of which collections are live and which graph JSONs auto-load — see `start.sh.example` for the template. To run a slimmer subset manually:

```sh
uv run knowledge_api_server.py --collections <name> [<name> ...] --port 8321
```
