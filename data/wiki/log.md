# Activity Log

## [2026-04-09] update | Moved wiki into Huginn

- Moved wiki from standalone vault (`/Users/rune/private/youtube-transcripts/`) into Huginn at `data/wiki/`
- Raw sources no longer duplicated — wiki reads from `data/sources/youtube-articles/` and `data/sources/x-feed/`
- Created `CLAUDE.md` schema file adapted for Huginn directory structure
- Added `!data/wiki/` exception to Huginn's `.gitignore` for version control
- Set up Obsidian vault at `huginn/data/` covering both sources and wiki
- Updated integration analysis with three-vault architecture (personal, work, Notion)

## [2026-04-09] create | Huginn-Muninn-Wiki Integration analysis

- Created `wiki/analyses/Huginn-Muninn-Wiki Integration.md` — architecture and implementation plan
- Describes 4 integration points: Huginn wiki collection, auto-ingest pipeline, Muninn bot query mode, cross-conversation memory
- Includes 6-phase implementation roadmap with priorities
- First analysis page in the wiki

## [2026-04-07] lint | Filename fix

- Renamed all 37 wiki files from kebab-case to title-case to match wikilink targets
- Fixed Obsidian phantom note issue where `[[Boris Cherny]]` created empty notes instead of resolving to `wiki/entities/boris-cherny.md`
- Updated CLAUDE.md filename convention to use page titles as filenames

## [2026-04-06] lint | Connectivity fix

- Found 8 orphan source pages with zero inbound links
- Found 2 missing entity pages (Gary Tan, Ray Kurzweil) — created both
- Added cross-links between source catalogs (health↔career↔parenting↔entertainment↔coding)
- Added source catalog links from concept/entity pages (Anthropic, OpenClaw, Future of SWE, LLM Fundamentals, Context Management, AI Industry, AGI Predictions)
- All pages now have at least 1 inbound link from another wiki page

## [2026-04-06] ingest | Remaining 9 categories (133 files)

- Bulk-ingested: ai/claude (28), ai/openclaw (17), ai/rag (7), tech (35), health (18), career (13), parenting (7), entertainment (5), coding (3)
- Created 8 source catalog pages: Claude Models, OpenClaw, RAG, Tech, Health, Career, Parenting, Entertainment, Coding
- Updated `wiki/index.md` — wiki now has 11 source catalogs covering all 417 raw sources
- All categories fully ingested

## [2026-04-06] ingest | AI General sources (106 files)

- Bulk-ingested 106 YouTube video summaries from `raw/ai/general/`
- Created 5 new concept pages: AGI and Singularity Predictions, Future of Software Engineering, AI Agent Design Patterns, AI Industry Landscape, LLM Fundamentals
- Created 4 new entity pages: Andrej Karpathy, Dario Amodei, OpenAI, NotebookLM
- Created 1 source catalog page organizing all 106 sources by theme (12 categories)
- Updated `wiki/index.md` — wiki now has 15 concepts, 9 entities, 2 source catalogs

## [2026-04-06] ingest | Claude Code sources (178 files)

- Bulk-ingested 178 YouTube video summaries from `raw/ai/claude-code/`
- Created 10 concept pages: Context Management, Skills System, Subagents, CLAUDE.md Configuration, Hooks and Automation, Model Context Protocol, Prompting for Claude, Verification-Driven Development, Agent Loops, AI Coding Workflows
- Created 5 entity pages: Claude Code, Anthropic, Boris Cherny, OpenClaw, GStack
- Created 1 source catalog page organizing all 178 sources by theme (15 categories)
- Updated `wiki/index.md` with all new pages

## [2026-04-06] create | Wiki initialized

- Created directory structure: `raw/`, `wiki/`, and subdirectories
- Created `CLAUDE.md` schema with ingest, query, and lint workflows
- Created `wiki/index.md` (empty catalog) and `wiki/log.md` (this file)
- Wiki is ready for first source ingest
