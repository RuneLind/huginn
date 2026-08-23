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

> **Private collection names stay out of this public repo.** The table below lists
> only collections whose names carry nothing private. A collection tied to an
> employer, customer, or person is configured through the gitignored
> `huginn-*/scripts/` routing files (`graph_routing.json`,
> `schedule_routing.json` — the latter's `labelCollections` key also feeds
> `scripts/backfill_indexing_runs.py`), never named here, in `docs/`, in
> docstrings, or in test comments. Use a neutral placeholder such as
> `notion-workspace` when an example needs a name.


| Collection | Source path | Exclude patterns |
|---|---|---|
| `melosys-confluence-v3` | `./data/sources/melosys-confluence` | `^\.excluded/.*` `^fetch_metadata\.json$` (+ a path-specific exclude — see manifest) |
| `jira-issues` | `./data/sources/jira-issues` | `^\.excluded/.*` |
| `nav-wiki` | `./<private-sub-repo-a>/wiki` | `index\.md` `log\.md` `CLAUDE\.md` `^\.obsidian/.*` `^\.claude/.*` |
| `wiki` | `./<private-sub-repo-b>/data/wiki` | `^index\.md$` `^log\.md$` `^CLAUDE\.md$` `^plans/.*` `^Clippings/.*` `^\..*` (dot-dirs: `.obsidian/`, `.smart-env/`, `.understand-anything/`) `^life/.*` `^trails\.json$` |
| `wiki-life` | `./<private-sub-repo-b>/data/wiki` | same, minus `^life/.*` — plus `includePatterns: ["^life/.*"]` |

> Exact `basePath`s are in each collection's `manifest.json` (private sub-repo names stay out of this file).
>
> The two wiki collections share a `basePath` and are **mutually exclusive**: `wiki` is the
> AI/tech corpus and excludes `^life/.*`; `wiki-life` includes only `^life/.*`. That exclude
> was missing until the 2026-08-16 lint, so every `life/` page was indexed into both and the
> AI corpus returned health/parenting hits. If a `life/` result ever shows up in a `wiki`
> search again, check this pattern first.

### Verify after re-indexing

Check `data/collections/<name>/manifest.json` — confirm `numberOfDocuments` matches expectations and `excludePatterns` show single backslashes in JSON (e.g. `"^\\.excluded/.*"`, not `"^\\\\.excluded/.*"`).

## Deleting a document

`DELETE /api/document/{collection}/{doc_id}` (localFiles collections only) removes a
junk document — smoke-test residue, a bad capture — from a collection.

```sh
curl -X DELETE "http://127.0.0.1:8321/api/document/x-articles/some-doc.md"
# {"status":"deleted","collection":"x-articles","doc_id":"some-doc.md",
#  "movedTo":"data/deleted/x-articles/some-doc.md",
#  "reindex":{"x-articles":"started"},
#  "pollUrls":{"x-articles":"/api/collections/x-articles/update-status"}}
```

- **It deletes the SOURCE, not the derived JSON.** The listed document under
  `data/collections/<name>/documents/` is regenerated from the source markdown under
  the manifest's `reader.basePath`; deleting it leaves the index entry behind with a
  dangling `documentPath`. Removing the source is what sticks — the update's orphan
  pruning (`__prune_orphaned_documents`) then drops the index entries *and* the
  derived JSON. Never hand-edit `index_document_mapping.json`.
- **Soft delete.** The source is MOVED to `data/deleted/<collection>/<doc_id>`
  (`HUGINN_DELETED_DIR` overrides the root; `data/` is gitignored, so this is the only
  undo there is). A name collision is disambiguated (`x.1.md`), never overwritten.
  Move it back and reindex to restore.
- **Outside `basePath`, deliberately** — not a `.excluded/` folder inside it. Exclusion
  is purely `excludePatterns`, and none of the summary collections declare one; with
  `includePatterns: [".*"]` a file moved inside the tree is simply re-indexed under a
  new id. Moving out needs no manifest edit. The endpoint 400s if the deleted-dir
  resolves inside `basePath`.
- **Only real documents of the collection.** The doc id must be present in the
  collection's `reverse_index_document_mapping.json`, else **404** — `basePath` is not
  the collection. Several wikis' `basePath` is a live git repo root whose reader excludes
  most of what lives there (`CLAUDE.md`, `index.md`, dot-dirs) and skips `.git/` outright;
  without this check the endpoint would move those out of a real repository.
- **All collections sharing the `basePath` are reindexed**, not just the named one —
  `wiki` + `wiki-life`, the `nav-wiki*` family, `jira-issues` + baseline.
  Otherwise the siblings keep serving the deleted doc from a dangling index entry. `reindex` is therefore a **map** `{collection: status}` (named
  one first), and `pollUrls` a map for the `started` ones.
- **Not synchronous.** The move is immediate; the doc leaves search and the document
  listing only after the background update finishes — poll the collection's entry in
  `pollUrls`. A collection whose update was already running reports
  `skipped_already_running` and gets **no** `pollUrl` (the in-flight run's status would
  read `succeeded` without ever having seen this delete): the move still happened, and
  the caller must POST `/api/collections/{name}/update` for it later. It is never queued
  behind the running update, which may already be past its own enumeration step.
- **Last document of a collection is a no-op in the index.** `__prune_orphaned_documents`
  deliberately refuses to prune when the reader enumerates zero documents (a transient FS
  error or a mistyped pattern would otherwise wipe the whole index). So deleting a
  collection's *final* document moves the source and returns 200, but the index entry
  survives. Recreate the collection instead.
- **400s** (before anything moves) for a non-`localFiles` reader — query-based readers
  cannot enumerate their ids, so pruning never fires for them — an unresolvable
  `basePath`, or an unusable doc id: escaping `basePath` (realpath containment guard, so
  `../` and out-pointing symlinks both fail), being *itself* a symlink even when its
  target is inside (realpath would move the target and orphan the link), or containing an
  embedded NUL (`%00`). **404** for an unknown collection, a missing source file, or a
  file that is not an indexed document of the collection. A trailing slash in the id is
  normalized away.
- A relative `basePath` (e.g. x-articles' `./data/sources/x-articles`) resolves against
  the **server's CWD**, same as `FilesDocumentReader` and the update factory read it.
- CORS is unchanged (`GET`/`POST` only) — the endpoint is intended to be called
  server-side; a muninn proxy for it lands separately, and it is not meant for the
  browser extension.

## Build-time people aliasing (privacy)

Some collections index documents full of real colleagues' names. Those
collections must be copyable to another machine, so people are **aliased at
index build time**, inside `FilesDocumentConverter.convert()` — the built
collection under `data/collections/` never contains a real name, and the reverse
map stays local.

- `main/privacy/alias_registry.py` compiles the map into one regex pass:
  mapped people → their alias (`dev-06`), unmapped people → `[~ukjent-person]`,
  role nouns / test users / countries → left alone, `[A-Za-z]\d{6}` idents (bare
  or wrapped as `[~Q000124]`) → `[~person]`. A second pass turns any leftover
  dotted handle into `@person`, and in a dotted email *local part* the local part
  becomes `person` (`ola.nordmann@nav.no` → `person@nav.no`). `id` and `url` are
  never rewritten — they are the join keys to the source file and the index
  mapping. A **third** pass rewrites `/Users/<name>/` to `/Users/<user>/`: the
  account name in a macOS home path is a person, arriving through pasted stack
  traces and shell transcripts, and it is the one person the map is least likely
  to list because they are the operator rather than a document author. Every
  fingerprint check 10 found on the live collections was this shape. It needs
  the trailing `/` (`/Users/` at end of line has nothing to bound the segment
  against) and refuses a preceding word character, so `s3://bucket/Users/x` is
  untouched — check 10 keeps failing on the forms this deliberately leaves.
- **What that second pass leaves alone**, because the corpora are code-heavy:
  `@v1.2.3` versions; package paths under a known root (`@org.`, `@jakarta.`,
  `@kotlin.`, `@android.`, `@net.`, …); **code annotations**, i.e. a lowercase
  root plus a CamelCase segment (`@lombok.Setter`, `@dagger.Provides` — the
  annotation vocabulary is open-ended, so a root list alone cannot keep up); and
  hosts, decided by an **explicit** last-label list (TLDs plus `internal`,
  `local`, `test`, `dev`, `localhost`). The email rule additionally requires a
  real domain after the `@`, so Python's matmul (`np.eye@vec`, `A.T@B`) is not
  read as an address. There is deliberately no "short last segment looks like a
  TLD" heuristic: Norwegian surnames are short, and `@ola.berg` / `@kari.moe` /
  `@per.aas` survived into the built collection under it. The cost of the
  explicit list is that a Capitalized.Capitalized handle with no known root is a
  person, so `@Abac.Attr`-style shorthand redacts to `@person`.
- Multi-token variants match across a whitespace run **spanning at most one
  newline** (a blank line is a paragraph break, not a wrapped name) or `%20`, and
  their boundaries also accept `%2C`/`%20`/`%40`, so a name percent-encoded in a
  URL query string still matches. Bare single-token variants additionally block a
  preceding `.`, so a mononym cannot be welded onto a surviving dotted path
  segment — but not a preceding `-`, which is punctuation far more often than it
  is a path.
- **The map is validated at compile time** and a bad one refuses to build
  (`PrivacyMapInvalid`, a `PrivacyMapMissing`): blank literals, zero entries, an
  entry variant that is a bare given name (a single all-alphabetic token —
  hyphenated compounds like `Nord-Hansen` are fine), two distinct literals
  sharing a casefold key (`Weiss`/`Weiß`, where one would silently take the
  other's replacement), schema drift. Two discovered maps are `ambiguous` rather
  than first-wins.
- **Scope** is `main/privacy/scope.json` (public collection names + public source
  dirs) plus an optional private `huginn-*/privacy/scope.json` for the paths that
  are not public — the same glob convention as `graph_routing.json`. Matching is
  by collection name **or** reader `basePath`, so sibling/backup collections
  sharing a `basePath` arm too; `scoped_base_paths()` additionally reads the
  reader `basePath` out of every in-scope collection's built `manifest.json`,
  because scope is declared in two places and they drift. Everything else gets
  `alias_registry=None` and behaves exactly as before.
- **Discovery resolves against the repo root, never the CWD.** Every
  `huginn-*/privacy/*` glob and every relative `basePath` in a scope file goes
  through `alias_registry.REPO_ROOT` (derived from `__file__`;
  `HUGINN_PRIVACY_ROOT` relocates it for subprocess tests, and it fails closed).
  They used to resolve against the process CWD, so from anywhere but the repo
  root the whole scope evaporated — and `tag_documents.py`'s external-backend
  guard, whose entire job is to refuse, silently permitted the hosted backend.
- **Fail closed.** An in-scope collection whose map (`huginn-*/privacy/aliases.json`)
  is missing raises `PrivacyMapMissing` *before* the create path removes the
  collection folder. A clone with no private sub-repos simply builds nothing
  in scope.
- The manifest gets a `privacy: {policy_version, map_version, aliasedAt}` stamp on
  the **create branch only** — it asserts the whole index was built aliased, which
  a windowed nightly update (which re-converts only recent documents) cannot
  promise. The update branch preserves whatever stamp is already there and never
  adds one; that stamp is also what re-arms aliasing even if the scope files drift.
- **Rebuilding:** `.venv/bin/python scripts/audit/rebuild_aliased.py --collection
  <name>` builds into `<name>-aliased` (the create path deletes the target folder
  first, so never build in place), reusing the existing manifest's reader, indexers
  and contextual-prefix model, and passing the create adapter a `--contextual-cache`
  pointing at the REAL collection's cache. The build refuses to finish unless the
  new manifest's privacy stamp matches the current map/policy version and its
  reader block reproduces the source's. `--swap` re-checks the stamp, parks the
  live one under `data/prealias/<name>-<date>` (**outside** `data/collections/`, so
  no server glob serves it) and reloads the server — a failed reload exits non-zero.
  - **The swap rewrites the mapping's `documentPath` prefixes**
    (`<name>-aliased/documents/…` → `<name>/documents/…`, temp file +
    `os.replace`) — *this* is the canonical write-up of the incident behind it.
    The swap renames the *directory* and nothing else repointed those paths, so
    `_get_chunk_texts` read nothing, the cross-encoder scored empty strings as
    uniform noise and the confidence filter dropped every result: perfect
    retrieval, empty answers, no error anywhere. All three live collections
    shipped that way for weeks, past a clean gate at every scan. (Everywhere
    else — `rebuild_aliased.py`, check 12, `query_privacy.py` — points here.)
  - **Order: rename → rewrite → verify → reload.** The rewrite happens at SWAP
    time, after the rename, never at build time: until the directory is `<name>`,
    `<name>-aliased/documents/…` is the correct answer, and a temp collection
    served or scanned in place has to read its own aliased documents. The build
    therefore leaves a collection that names *itself* in both the manifest and
    every `documentPath` — so a refused build cannot leave a directory whose
    manifest claims to be another collection, which is exactly what check 12
    now fails on. `--swap` refuses **before** renaming if there is no
    `indexes/index_document_mapping.json` to repoint (after the rename there is
    no live collection to fall back to), and after the rewrite re-derives the
    stale set rather than trusting it: **any** `indexes/` JSON still spelling
    the temp name is a non-zero exit naming the parked copy. The mapping is the
    only built artifact holding the collection name — the reverse mapping is
    keyed by document id, `index_info.json` is a counter, and the manifest's
    `collectionName` is rewritten in the same step.
- **The distribution gate** is `main/privacy/index_scan.py`, driven by
  `.venv/bin/python scripts/audit/scan_index.py --collection <name>` (add
  `--compare <name>` for the pre-alias twin invariants, `--collections-dir` for a
  staged copy or an untarred package, `--map` to pin the map). *This script was
  called `verify_aliased_collection.py` until the checks were promoted into
  `main/privacy/` so the packager could call them as a library; the flags are
  unchanged.* It decodes JSON rather than grepping bytes, checks the BM25 token
  list separately — a byte-grep of the pickle finds nothing while both name
  tokens sit adjacent in `corpus_tokens` — and scans **every** file under the
  collection dir, failing on a `.bak` or an unrecognised binary rather than
  skipping it. With `--compare` it also asserts `numberOfDocuments` /
  `numberOfChunks` equal the twin's (`--allow-count-drift` downgrades that to a
  warning; a count the twin's manifest does not record at all is skipped).
  Everything printed and everything in `--json-report` is a shape or a count;
  `--candidates-out` is the one output with real text in it. Passed bare it goes
  to `data/privacy/scan_candidates_<collection>.json` (`data/` is gitignored
  wholesale); an explicit path is refused unless `git check-ignore` already
  covers it.
  - A file the scan cannot read — a truncated `documents/*.json`, an
    undecodable binary, an unreadable manifest — is a check-8 failure naming the
    **directory** and the reason, never a traceback and never the filename: a
    file under `documents/` is named after its document id, ids are never
    aliased (that is what check 7 is for), and this detail goes into
    `--json-report`. The scan can only certify what it has read. So is
    **any symlink** anywhere under the collection dir, file or directory, and
    the walk never follows one.
  - **Check 9, capitalised-bigram candidates, is the one that finds people the
    map never knew** — every other check works *from* the map and is blind to
    them. A `Capitalised Capitalised(+)` run is **retained** as a candidate when
    its first token is a plausible given name (public `main/privacy/given_names.txt`
    ∪ the private map's given names and `bare_given_name_residual`, unioned at
    runtime), then `non_person_labels` and a reviewed allow-list
    (`huginn-*/privacy/non_person_bigrams.json`, gitignored) are subtracted.
    **Retention, not subtraction** — a filter that drops "things that look like
    names" removes exactly the target category. EVERY adjacent pair of a
    capitalised run is evaluated, not just the head pair (a name behind an
    acronym, a heading word or another name was invisible: recall against the
    map's own names went 0.53 → 0.76 on the public gazetteer alone); the initial
    capital is `str.isupper()` rather than `[A-ZÆØÅ]`; a hyphenated given name is
    probed part by part; a dot may precede the pair and one newline may sit
    inside it. The union of gazetteer + map given names has its own floor
    (`MIN_GAZETTEER_ENTRIES`), because a truncated gazetteer retains nothing and
    "no candidates" is what a clean collection looks like.
    The public gazetteer must never grow corpus-specific names. Four structural
    tests hold it: one alphabetic token per line, sorted and duplicate-free, no
    two lines concatenating into a mapped full name **whose second token is a
    surname in the map** (a two-token literal is not automatically `First Last`
    — the wider rule deleted ten ordinary given names for Norwegian double given
    names, and only three of the ten were unioned back in from the map, so seven
    `<given> <surname>` pairs went undetectable), and a carve-out narrowed to
    the map's GIVEN-name set plus a capped, runtime-derived set of
    given-name/surname overlaps — the old carve-out exempted every mononym and
    bare surname the map knew (see the comment there for why a list of common
    names with holes in it is a *reverse* fingerprint).
  - **Checks 10 and 11** are new categories rather than map lookups: distributor
    fingerprints (an absolute `/Users/` path or a `.bak` reference anywhere in
    the unit — this found another person's macOS username in a pasted stack
    trace) and `main/privacy/sensitivity_scanner.py`. The sensitivity categories
    that BLOCK are the ones measured precise enough to block on AND about
    something that must not leave the machine: fødselsnummer, bankkonto,
    credential, ident, dotted handle. Email, plaintext-password patterns, phone
    numbers **and organisasjonsnummer** are reported and do not fail the gate —
    an organisation number identifies a company in a public register, is not
    personal data, is legitimate content in an issue about an employer, and is
    the category that fires most. Anchoring is what makes that affordable: 23
    findings across the three collections, against 334 for the same detectors
    without the keyword anchors and leading-digit rules
    (`benchmarks/pii/RESULTS.md`, whose §2 is now a reproducible bench). The new
    categories are **never** wired into `PiiSanitizer.sanitize` — that is the
    live Jira ingest write path, where a false positive mangles a stored
    document irreversibly.
  - **Check 12** is the only one that walks the *join* rather than the content:
    every `documentPath` in `index_document_mapping.json` must start with
    `<directory name>/documents/` and resolve to a file that exists. Every other
    check reads `documents/*.json` directly and is structurally blind to an index
    that no longer points at them — which is why the swap bug above passed a
    clean gate at every scan for weeks. Its detail carries the offending path
    *prefix* (a collection directory name) and counts, never a path: a document
    filename is a document id, and ids are never aliased; a path with no `/` in
    it is reported as `(unrecognised)` rather than by its filename. On PASS it
    says how many chunks resolved and under which prefix.
    - **It keys off the DIRECTORY name, and requires `manifest.collectionName`
      to equal it.** The searcher resolves `documentPath` against the
      collections dir and `KnowledgeStore` loads a collection by its directory
      name, so keying off `collectionName` made the check self-consistent
      instead of true: a directory renamed or copied under another name (a
      backup, a scratch copy) agrees with its own manifest and serves nothing.
      Four backup directories on this machine fail on that mismatch, correctly.
      The untar path is unaffected — a package unpacks to
      `data/collections/<name>/`, which is the name its mapping already spells.
    - **An empty mapping FAILS.** Every other check reads the documents
      directly, so an index that maps nothing passes all of them, and a check-12
      that only iterated the mapping passed vacuously too. A `documentPath` of
      `null` is a failure, not a traceback.
  - Bare given names standing alone stay out of scope (the map's
    `bare_given_name_residual`, a documented campaign decision).
- **Packaging is the only hand-off path.** `.venv/bin/python
  scripts/audit/package_collection.py --collection <name> [--out dir]
  [--compare <twin>] [--allow-count-drift] [--force]` runs the scan and writes
  `<name>-<date>-map<map_version>-policy<policy_version>.tar.gz` **only** when
  every check passes; a failure prints the report, writes nothing and exits
  non-zero. The date alone made two tarballs built the same day from different
  maps indistinguishable, and an existing file of the same name is a refusal
  unless `--force`. It takes the same flags as the CLI so it certifies the same
  check set, and refuses outright for a collection out of privacy scope or
  without a manifest `privacy` stamp. It also consults the **local sensitivity
  sweep** (subsection below): a report with unknown persons, or one older than
  the collection's last rebuild, refuses; no report at all only warns. Copying
  the collection directory by hand is not a supported hand-off. Four guarantees,
  each of which was missing:
  - **atomic** — built under a temp name in the destination directory and
    `os.replace`d into place. A half-written `.tar.gz` has the name of a
    certified package and the contents of an interrupted one;
  - **exactly what was scanned** — the member list comes from the report
    (`ScanReport.scanned_members`), not a second walk, and every member's size
    and mtime is re-checked immediately before the tar. A file that appeared or
    changed in that window is a `REFUSED:` line, because it would otherwise be
    tarred without having been read;
  - **no builder identity** — `uid`/`gid`/`uname`/`gname` are zeroed on every
    member, and the **gzip** member is opened by hand with `filename=""`,
    `mtime=0` rather than through `tarfile.open(path, "w:gz")`, which writes the
    path it is currently writing (`.<name>.tar.gz.tmp-<pid>`, i.e. the builder's
    process id) into the FNAME header. `tar -tvf` does not show it; `gzip -l`
    does. Same class of fingerprint as the absolute `/Users/` path check 10
    looks for, just in the header instead of the content — and zeroing MTIME
    also makes two packages of the same input compare byte for byte;
  - **a legible stamp** — `PACKAGE-STAMP.json` carries collection, scan date,
    policy/map version, document and file counts, `allowlistSha256` and
    `gazetteerSha256` (a tarball certified against a longer allow-list was
    certified against a weaker gate), and `scanChecks` as
    `{check: {passed, count, ran}}` for **every known check**. `ran: false` is
    what a run without `--compare` reports for checks 3b and 5; a stamp that
    simply omitted them read exactly like one where they had passed.
- **`scripts/tagging/tag_documents.py` refuses an external backend on a scoped
  tree.** It reads RAW source markdown and defaults to `--backend claude-cli`,
  which ships an excerpt of every file off the machine. When `--source` resolves
  inside an in-scope basePath (`alias_registry.path_in_scope`, realpath
  containment, repo-root-relative, including the basePaths read out of in-scope
  collections' manifests) and the backend is not `ollama`, it aborts before
  opening a single file. Only this script: the two graph extractors read the same
  trees but are deterministic and local.
- **Derived caches carry pre-alias text.**
  `.venv/bin/python scripts/audit/purge_prealias_caches.py` retires them (LLM graph
  caches deleted, dormant contextual caches renamed to `.pre-alias.bak`). The
  extractor's own policy check only discards a stale cache **in memory** — the file
  keeps the pre-alias extractions until this script removes it. The contextual cache
  of an actively-prefixed collection is kept on purpose: the pipeline invalidates
  exactly the documents aliasing changed (`ContextualCache.invalidate_doc`), instead
  of re-prefixing every chunk.
- **Query seam** (`main/privacy/query_privacy.py`). An aliased index cannot be
  searched by name, and a typed name must not come back out, so the query is
  rewritten with the **same** substituter — `dealias_query(q, registry)`, which is
  `registry.apply` and therefore idempotent. `KnowledgeStore` resolves one
  registry per served collection at load and at `POST
  /api/collections/{name}/reload` (`store.get_searchers_and_registries` hands
  the searchers and the registries back from **one** lock hold, so a reload
  cannot pair a rebuilt searcher with the previous registry); a missing map here
  **logs one error and serves the collection unaliased for querying** — the index
  on disk is already clean, and only *indexing* may fail closed. A collection
  whose manifest stamp names a different `map_version` than the map on disk logs
  one WARNING: queries de-alias with today's map, the index holds yesterday's.
  - Rewritten **per collection**, inside `search_pipeline.search_and_shape`, so a
    mixed request searches an aliased collection in alias space and an
    out-of-scope one in name space (the common case: `/api/search` with no
    `collection` hits all 30+).
  - **`title_boost_query` is name space for every collection.** Document
    filenames are never aliased (an id is a join key), so an alias token cannot
    legitimately match a title — but `apply_title_boost` tokenises with `\w+`,
    so `dev-01` contributes the token `01` and boosts every `…-01-…` filename in
    the collection.
  - If **any served** collection is armed — not just a targeted one — the
    request's *shared* text (trace `query.raw`, the query log, `augment_query`,
    `retryHints`, `corrective.queriesTried`) is the de-aliased one. Scoping a
    request to an out-of-scope collection does not make the typed name safe to
    log. The same holds for the MCP seam (`mcp_search_tool.build_search_tool_fn`
    takes both this collection's registry and the `shared_registry` across all
    served ones); both transports share one preamble,
    `query_privacy.prepare_aliased_request`.
  - **The twins.** Out-of-scope collections get the raw query plus the
    expansion terms *before* de-aliasing (`augment_query` returns both lists) —
    not the alias-space suffix, and not conditional on the query itself having
    changed, because the graph alone can introduce an alias token into a clean
    query. The corrective rescue query has no typed twin (it is derived from the
    hints), so it is mapped back through the map's own names
    (`AliasRegistry.to_names`). An out-of-scope index never receives an alias
    token on either leg.
  - **The knowledge graph is a pre-alias artifact** — its `*_llm_graph.json`
    files were extracted before the corpus was aliased, so node labels, ids
    (`entity:<name>`), expansion terms and entity contexts still spell people by
    name, and they reach `retryHints.relatedTerms`, `graph_answer`,
    `graph_context` and the trace's expansion. Measured on the 10-probe sweep:
    19 graph-contributed strings still spelled a mapped person before this seam
    and 0 after (per-probe table in PR #117). So `GraphSearchAugmenter`
    takes the same registry (`alias_registry=`) and every string it contributes
    is de-aliased — the trace's entity ids included, while the lookup keeps
    using the raw ones. A request that touches no aliased collection passes
    `None` and is byte-identical to before.
  - **Two term classes are dropped, not de-aliased**: the redaction tokens
    (`[~ukjent-person]`, `[~person]`, `@person`) and any term equal to a bare
    given name of a mapped entry (`AliasRegistry.given_names`). Bare given names
    are deliberately never substituted, so one surfacing in `relatedTerms`
    beside its own alias re-pairs the two.
  - **Other callers of the seam**: `collection_search_cmd_adapter.py` (CLI) and
    `routes/ingest.py::_find_similar_documents` — the ingest similarity query is
    document text, which carries names as readily as a typed query does.
  - **An alias query needs nothing else.** `fag-01` reranks and scores normally
    (measured live on the repaired indexes: 5 results, best 0.958). The "the
    cross-encoder scores every candidate as uniform noise" reading was a
    *symptom* of the swap bug above — `_get_chunk_texts` was reading nothing, so
    the CE was scoring empty strings. The hot path is unchanged by this seam,
    with one honest caveat: the reranker-skip decision
    (`_should_skip_reranker`) reads the word count and langdetect's verdict on
    the text that collection is actually searched with, so it can differ between
    collections in one mixed request.

### Local sensitivity sweep

`scripts/audit/sensitivity_sweep.py` is the **second opinion** on a built
collection: the deterministic gate proves no *listed* name survived, and this
asks a local model who is named that the map never heard of. Two questions
`index_scan.py` structurally cannot answer — a person in any form other than
the capitalised bigram check 9 retains (a mononym, a bare surname, an
initials-plus-surname byline, a role phrase naming one individual), and whether
a bare given name standing in prose is a person reference at all (the residual
is five figures of occurrences; a regex cannot read them).

```sh
.venv/bin/python scripts/audit/sensitivity_sweep.py --collection <name> --baseline
.venv/bin/python scripts/audit/sensitivity_sweep.py --collection <name>            # incremental
```

- **Local by requirement, not by preference.** The transport is
  `main/utils/ollama_cli.py` and there is no `--backend` flag. A sensitivity
  check that ships document text to a hosted model *is* the leak it is looking
  for. Same rule as `tag_documents.py`'s external-backend refusal, one step
  further: here there is no external backend to refuse.
- **It reads the DERIVED documents** (`data/collections/<name>/documents/*.json`
  — already aliased), never `data/sources/`. The point is what a recipient of
  the tarball would see.
- **Two modes.** `--baseline` asks about every document (`--limit N` bounds a
  priced sample to the first N). The default asks only about documents whose
  text hash changed since the cache — that is the mode the nightly job runs.
- **Model policy.** `--model` defaults to `main/utils/ollama_cli.DEFAULT_MODEL`
  (`qwen3.8:27b-mlx`), the general-purpose local model the tagging scripts also
  inherit. The knowledge-graph extractor and the contextual-prefix backend keep
  their own pins: their caches are keyed by document id, not by model, so a swap
  silently produces a hybrid artifact. The A/B behind that split (5 documents of
  `melosys-confluence-v3`, coding pin vs. this default) is in the PR that
  introduced the constant; it did not decide, so the extractor did not move.
- **Deterministic pre/post-processing does the judging, not the model.** A
  reference is dropped when it is not a verbatim substring of the document (an
  invented name is the finding a human spends the whole triage budget on), and
  when it is an exempt `non_person_labels` entry or a reviewed
  `non_person_bigrams.json` pair. What is left is bucketed `alias` (an alias or
  a redaction token — the substituter working), `mapped_residual` (a *bare*
  given name the map knows, via `AliasRegistry.given_names` plus the map's
  `bare_given_name_residual` keys — a deliberate non-substitution, counted not
  alarmed on) or `unknown_person`, which fails the sweep.
- **Cache.** `data/state/sensitivity/<collection>.json`, an envelope
  `{policy_version, model, entries: {doc_id: {hash, findings_count,
  unknown_count}}}` where `hash` is `sha256(text + POLICY_VERSION + model)`. All
  three change the answer, so all three are in the key; the envelope repeats the
  policy and model so a changed one is reported once rather than per entry (and
  an envelope is used rather than `_`-prefixed sibling keys because a doc id may
  itself start with `_`). A cached document's `unknown_count` still lands in the
  run total — an incremental that skipped the one dirty document and reported
  zero would flip the packaging gate from refuse to pass with nothing fixed.
- **An unread run is not a clean run.** The report counts the model answers this
  script could not parse, and a run past `MAX_PARSE_FAILURE_RATIO` (20% of
  calls) is `degraded` in the ledger, prints `RESULT: INCONCLUSIVE`, and is
  **warned on rather than passed** by the packaging gate. `unknownCount: 0` out
  of a run where the model answered nothing readable is indistinguishable from a
  genuinely clean collection — the same vacuous pass that `index_scan`'s
  map-entry and gazetteer floors exist to prevent. Measured rate on the real
  corpus: 5 of 117 calls (4.3%) on the nav-wiki `--limit 50` baseline. Reports
  written before this check carry neither field and stay readable.
- **Outputs.** A gitignored JSON report at
  `<private-sub-repo>/privacy/sweep_<collection>_<date>.json` (real strings, for
  triage; falls back to `data/privacy/`, and an explicit `--report-out` is
  refused unless `git check-ignore` already covers it). Everything on stdout is
  a count or a shape. Exit `0` clean, `2` on an `unknown_person`, `1` when
  Ollama is unreachable.
- **Ledger.** Every run is recorded under the collection key
  `sensitivity-audit` with a single `sweep` phase, `fatal: true` — `succeeded`
  clean, `degraded` on unknown persons, `failed` when the model was unreachable
  (a sweep that judged nothing must not read like one that found nothing). Not
  the swept collection's own key: a sweep is not an indexing run of it, and
  folding the two would put a `degraded` sweep on the row a reader uses to
  answer "is the index fresh". `/api/indexing/jobs` shows it with
  `loaded: false` by design. It POSTs to `/api/indexing/runs` and falls back to
  an in-process `IndexingRunLedger().append` — the same writer, same flock, that
  `scripts/lib/indexing_run.sh` shells out to.
- **Packaging interaction.** `package_collection.py` consults the latest report
  for the collection: an `unknownCount > 0` **refuses**, and so does a report
  older than the collection's `updatedTime` (a clean verdict about text that has
  since been rebuilt certifies nothing). **No report at all only WARNs** — the
  sweep is a second opinion, and making a hand-off depend on a local GPU being
  up is how a gate gets routed around. The verdict is stamped into
  `PACKAGE-STAMP.json` as `sensitivitySweep`.
- **Scheduling.** Wired into the helper-instrumented private daily scripts as a
  non-fatal `sensitivity` phase after the reindex. The Confluence and Jira daily
  scripts are **not** wired: they are not helper-instrumented and fire their
  reindex without polling it, so a sweep there would read a half-rebuilt index.

## LLM entity extraction (knowledge graph)

Extract entities and relationships from a collection using a local Ollama model. Outputs a `*_llm_graph.json` used for query expansion and graph context enrichment at search time.

```sh
.venv/bin/python scripts/knowledge_graph/extract_entities_llm.py --collection <collection-name>
.venv/bin/python scripts/knowledge_graph/extract_entities_llm.py --collection <collection-name> --limit 20  # test run
```

- Requires Ollama running locally with `qwen3.6:35b-a3b-coding-nvfp4` (or pass `--model`)
- Incremental: uses a `.cache.json` file, safe to stop and resume. It is written as
  `{"policy_version": N, "entries": {...}}` — an envelope, not a sentinel key beside
  the doc ids, because a doc id may itself start with `_`. For a collection the
  privacy registry arms, a cache under a different (or absent) policy version is
  discarded **in memory** rather than replaying pre-alias extractions; deleting the
  file is `purge_prealias_caches.py`'s job. Out-of-scope collections load their
  legacy flat cache unchanged.
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
  file ⇒ `schedule: null`, the designed degradation. The same file's optional
  `labelCollections` key (marker label → collections) is read by
  `scripts/backfill_indexing_runs.py` for the same reason: labels whose
  collection this repo names publicly are compiled in, the rest route
  privately. An empty or blank entry there is ignored rather than allowed
  to clobber a public default. A plist whose
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
  `(cd "$PROJECT_DIR" && ./.venv/bin/python -m main.runtime.indexing_run_ledger
  append --file -)` when the API is down — `uv run python -m ...` where the
  checkout has no `.venv`. The fallback MUST run from `PROJECT_DIR` (the `-m`
  import resolves `main` from cwd, and launchd runs jobs from `/`); if
  `PROJECT_DIR` is unset it logs one line to **stderr** and writes nothing rather
  than losing the record silently. Never `>>` the JSONL from shell: macOS has no
  `flock(1)`, so a redirect cannot take the `LOCK_EX` every other writer holds.
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
