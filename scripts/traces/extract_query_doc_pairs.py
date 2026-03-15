#!/usr/bin/env python3
"""
Extract query-document pairs from Claude Code session transcripts.

Scans session JSONL files for MCP knowledge search and document fetch
tool calls, then pairs searches with subsequent document fetches to
create trace data for the replay benchmark.

A "session" is one JSONL file (one Claude Code conversation). Within
a session, each search query is paired with documents fetched after
that query (until the next search or end of session).

Usage:
    uv run scripts/traces/extract_query_doc_pairs.py
    uv run scripts/traces/extract_query_doc_pairs.py --projects melosys-api-claude melosys-eessi
    uv run scripts/traces/extract_query_doc_pairs.py --output ./huginn-nav/scripts/benchmarks/query-doc-pairs.jsonl
    uv run scripts/traces/extract_query_doc_pairs.py --since 2026-03-01
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# MCP tool name patterns
SEARCH_TOOLS = {"mcp__knowledge__search_knowledge", "mcp__melosys-confluence__search_melosys-confluence"}
FETCH_TOOLS = {"mcp__knowledge__get_document"}


def extract_tool_calls(session_path: Path) -> list[dict]:
    """Extract search and fetch tool calls from a session JSONL file.

    Returns list of dicts with keys: type (search|fetch), timestamp, and tool-specific fields.
    """
    calls = []
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            msg = entry.get("message", {})
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            timestamp = entry.get("timestamp", "")

            for c in content:
                if c.get("type") != "tool_use":
                    continue
                name = c.get("name", "")
                inp = c.get("input", {})

                if name in SEARCH_TOOLS:
                    query = inp.get("query", inp.get("q", ""))
                    if not query:
                        continue
                    collection = inp.get("collection", "")
                    tags = inp.get("tags", None)
                    calls.append({
                        "type": "search",
                        "timestamp": timestamp,
                        "query": query,
                        "collection": collection,
                        "tags": tags,
                    })

                elif name in FETCH_TOOLS:
                    collection = inp.get("collection", "")
                    doc_id = inp.get("doc_id", "")
                    if not doc_id:
                        continue
                    calls.append({
                        "type": "fetch",
                        "timestamp": timestamp,
                        "collection": collection,
                        "doc_id": doc_id,
                    })

    return calls


def pair_searches_with_fetches(calls: list[dict], session_id: str) -> list[dict]:
    """Pair each search with documents fetched after it (until the next search).

    Returns list of query-doc-pair entries ready for JSONL output.
    """
    pairs = []
    current_search = None
    fetched_docs = []

    def flush():
        nonlocal current_search, fetched_docs
        if current_search:
            collection = current_search["collection"]
            # If search had no collection, infer from fetched docs
            if not collection and fetched_docs:
                # Use the most common collection from fetched docs
                collections = [d["collection"] for d in fetched_docs if d.get("collection")]
                if collections:
                    collection = max(set(collections), key=collections.count)

            # Group fetched docs by collection
            docs_by_collection: dict[str, list[str]] = defaultdict(list)
            for d in fetched_docs:
                docs_by_collection[d["collection"]].append(d["doc_id"])

            if collection:
                # Emit one pair for the search's target collection
                pairs.append({
                    "trace_id": session_id,
                    "collection": collection,
                    "query": current_search["query"],
                    "tags": current_search["tags"],
                    "fetched_docs": docs_by_collection.get(collection, []),
                })
                # Also emit pairs for any other collections that had docs fetched
                for coll, docs in docs_by_collection.items():
                    if coll != collection:
                        pairs.append({
                            "trace_id": session_id,
                            "collection": coll,
                            "query": current_search["query"],
                            "tags": current_search["tags"],
                            "fetched_docs": docs,
                        })
            else:
                # No collection info at all — emit per-collection pairs from fetched docs
                for coll, docs in docs_by_collection.items():
                    pairs.append({
                        "trace_id": session_id,
                        "collection": coll,
                        "query": current_search["query"],
                        "tags": current_search["tags"],
                        "fetched_docs": docs,
                    })

                # If no docs were fetched at all, still emit the search with empty docs
                if not docs_by_collection:
                    pairs.append({
                        "trace_id": session_id,
                        "collection": "",
                        "query": current_search["query"],
                        "tags": current_search["tags"],
                        "fetched_docs": [],
                    })

        current_search = None
        fetched_docs = []

    for call in calls:
        if call["type"] == "search":
            flush()
            current_search = call
            fetched_docs = []
        elif call["type"] == "fetch":
            fetched_docs.append(call)

    flush()
    return pairs


def find_sessions(projects: list[str] | None, since: str | None) -> list[tuple[str, Path]]:
    """Find session files to process.

    Args:
        projects: List of project name substrings to filter by. None = all projects.
        since: ISO date string. Only include sessions modified after this date.

    Returns list of (project_name, session_path) tuples.
    """
    sessions = []
    if not CLAUDE_PROJECTS_DIR.exists():
        print(f"Warning: Claude projects directory not found: {CLAUDE_PROJECTS_DIR}")
        return sessions

    since_dt = None
    if since:
        since_dt = datetime.fromisoformat(since)
        # Ensure timezone-aware comparison
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name

        if projects:
            if not any(p in project_name for p in projects):
                continue

        for session_file in project_dir.glob("*.jsonl"):
            if since_dt:
                mtime = datetime.fromtimestamp(session_file.stat().st_mtime, tz=timezone.utc)
                if mtime < since_dt:
                    continue
            sessions.append((project_name, session_file))

    return sessions


def main():
    parser = argparse.ArgumentParser(description="Extract query-doc pairs from Claude sessions")
    parser.add_argument("--projects", nargs="*",
                        help="Project name substrings to include (default: all)")
    parser.add_argument("--since",
                        help="Only include sessions modified after this date (ISO format)")
    parser.add_argument("--output", default="./huginn-nav/scripts/benchmarks/query-doc-pairs.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file instead of overwriting")
    parser.add_argument("--min-fetched", type=int, default=0,
                        help="Only include pairs with at least this many fetched docs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without writing output")
    args = parser.parse_args()

    sessions = find_sessions(args.projects, args.since)
    print(f"Found {len(sessions)} session files to scan")

    all_pairs = []
    sessions_with_searches = 0
    existing_trace_ids = set()

    # Load existing trace IDs if appending
    output_path = Path(args.output)
    if args.append and output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                existing_trace_ids.add(entry.get("trace_id", ""))
        print(f"  {len(existing_trace_ids)} existing trace IDs (will skip)")

    for project_name, session_path in sessions:
        session_id = session_path.stem

        # Skip if already in the existing data
        if session_id in existing_trace_ids:
            continue

        calls = extract_tool_calls(session_path)
        if not calls:
            continue

        search_count = sum(1 for c in calls if c["type"] == "search")
        if search_count == 0:
            continue

        sessions_with_searches += 1
        pairs = pair_searches_with_fetches(calls, session_id)

        if args.min_fetched > 0:
            pairs = [p for p in pairs if len(p.get("fetched_docs", [])) >= args.min_fetched]

        all_pairs.extend(pairs)

    # Deduplicate: same (trace_id, collection, query) -> keep last
    seen = {}
    for p in all_pairs:
        key = (p["trace_id"], p["collection"], p["query"])
        seen[key] = p
    deduped_pairs = list(seen.values())

    # Stats
    pairs_with_docs = sum(1 for p in deduped_pairs if p.get("fetched_docs"))
    pairs_without_docs = sum(1 for p in deduped_pairs if not p.get("fetched_docs"))
    unique_traces = len(set(p["trace_id"] for p in deduped_pairs))
    collections = set(p["collection"] for p in deduped_pairs if p["collection"])

    print(f"\nResults:")
    print(f"  Sessions with searches: {sessions_with_searches}")
    print(f"  Total pairs: {len(deduped_pairs)} ({pairs_with_docs} with docs, {pairs_without_docs} empty)")
    print(f"  Unique sessions (trace_ids): {unique_traces}")
    print(f"  Collections: {collections}")

    if args.dry_run:
        print("\n(dry run — not writing output)")
        # Show some sample pairs
        for p in deduped_pairs[:5]:
            print(f"  [{p['collection']}] {p['query'][:60]} → {len(p.get('fetched_docs', []))} docs")
        return

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for p in deduped_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(output_path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"\nWritten to {output_path} ({total_lines} total pairs)")


if __name__ == "__main__":
    main()
