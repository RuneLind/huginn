#!/usr/bin/env python3
"""
Extract a knowledge graph from Jira issue markdown files.

Parses frontmatter and body text to build Epic and Issue nodes with
tilhører_epic and refererer_til edges.

Usage:
    uv run scripts/knowledge_graph/extract_jira_graph.py
    uv run scripts/knowledge_graph/extract_jira_graph.py --source ./data/sources/jira-issues
    uv run scripts/knowledge_graph/extract_jira_graph.py --output ./huginn-nav/scripts/knowledge_graph/jira_graph.json

Output auto-routes to private sub-repos when no --output is given:
    huginn-nav/scripts/knowledge_graph/jira_graph.json (NAV-internal data)
    huginn-jarvis/scripts/knowledge_graph/jira_graph.json (otherwise)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as a script (uv run scripts/...) by putting the repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main.utils.frontmatter import read_frontmatter_from_path, strip_frontmatter

ISSUE_KEY_RE = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')


def parse_body(filepath: str) -> str:
    """Read body text after frontmatter."""
    try:
        return strip_frontmatter(Path(filepath).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return ""


def extract_cross_references(body: str, self_key: str, epic_key: str) -> set[str]:
    """Find issue key references in body text, excluding self and epic."""
    refs = set()
    for m in ISSUE_KEY_RE.finditer(body):
        key = m.group(1)
        if key != self_key and key != epic_key:
            refs.add(key)
    return refs


def _default_output_path() -> Path:
    """Auto-route output to a private sub-repo so customer-confidential graphs
    don't land in the public working tree. Falls back to a NAV-only error if
    neither private repo is checked out."""
    nav_dir = Path("./huginn-nav/scripts/knowledge_graph")
    jarvis_dir = Path("./huginn-jarvis/scripts/knowledge_graph")
    if nav_dir.exists():
        return nav_dir / "jira_graph.json"
    if jarvis_dir.exists():
        return jarvis_dir / "jira_graph.json"
    return Path("")


def main():
    parser = argparse.ArgumentParser(description="Extract Jira knowledge graph from markdown files")
    parser.add_argument("--source", default="./data/sources/jira-issues",
                        help="Directory with Jira markdown files")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: auto-route to ./huginn-nav/ or ./huginn-jarvis/)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _default_output_path()
        if not output_path:
            print("Error: no --output given and no private sub-repo (huginn-nav/ or huginn-jarvis/) is checked out.")
            print("Pass --output explicitly. Do NOT write to ./scripts/knowledge_graph/ — that path leaks customer data into the public repo.")
            return

    print(f"Scanning {source_dir} for Jira markdown files...")

    # Phase 1: Parse all files, collect metadata
    issues = {}  # issue_key -> {metadata, body, cross_refs}
    epic_issues = defaultdict(list)  # epic_key -> [issue_key, ...]
    epic_summaries = {}  # epic_key -> summary text

    md_files = list(source_dir.rglob("*.md"))
    # Skip .excluded
    md_files = [f for f in md_files if ".excluded" not in f.parts]

    for filepath in md_files:
        meta = read_frontmatter_from_path(str(filepath))
        issue_key = meta.get("issue_key", "")
        if not issue_key:
            continue

        body = parse_body(str(filepath))
        epic_key = meta.get("epic_link", "")
        cross_refs = extract_cross_references(body, issue_key, epic_key)

        issues[issue_key] = {
            "summary": meta.get("title", meta.get("summary", "")),
            "status": meta.get("status", ""),
            "issue_type": meta.get("issue_type", ""),
            "epic_link": epic_key,
            "cross_refs": cross_refs,
        }

        if epic_key:
            epic_issues[epic_key].append(issue_key)
            epic_summary = meta.get("epic_summary", "")
            if epic_summary and epic_key not in epic_summaries:
                epic_summaries[epic_key] = epic_summary

    print(f"Parsed {len(issues)} issues, {len(epic_issues)} epics with children")

    # Phase 2: Build nodes
    nodes = []
    issue_keys_set = set(issues.keys())

    # Epic nodes
    for epic_key, child_issues in epic_issues.items():
        summary = epic_summaries.get(epic_key, "")
        nodes.append({
            "id": f"epic:{epic_key}",
            "type": "Epic",
            "label": f"{epic_key}: {summary}" if summary else epic_key,
            "properties": {
                "issue_count": len(child_issues),
                "summary": summary,
            }
        })

    # Issue nodes
    for issue_key, data in issues.items():
        nodes.append({
            "id": f"issue:{issue_key}",
            "type": "Issue",
            "label": f"{issue_key}: {data['summary']}" if data['summary'] else issue_key,
            "properties": {
                "status": data["status"],
                "issue_type": data["issue_type"],
            }
        })

    # Phase 3: Build edges
    edges = []
    seen_edges = set()

    def add_edge(source, target, edge_type):
        key = (source, target, edge_type)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"source": source, "target": target, "type": edge_type, "properties": {}})

    # Epic membership
    epic_node_ids = {n["id"] for n in nodes if n["type"] == "Epic"}
    for issue_key, data in issues.items():
        epic_key = data["epic_link"]
        if epic_key and f"epic:{epic_key}" in epic_node_ids:
            add_edge(f"issue:{issue_key}", f"epic:{epic_key}", "tilhører_epic")

    # Cross-references (only to issues that exist in the collection)
    cross_ref_count = 0
    for issue_key, data in issues.items():
        for ref_key in data["cross_refs"]:
            if ref_key in issue_keys_set:
                add_edge(f"issue:{issue_key}", f"issue:{ref_key}", "refererer_til")
                cross_ref_count += 1

    # Phase 4: Stats
    node_types = defaultdict(int)
    for n in nodes:
        node_types[n["type"]] += 1
    edge_types = defaultdict(int)
    for e in edges:
        edge_types[e["type"]] += 1

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "by_type": dict(node_types),
        "edge_types": dict(edge_types),
    }

    graph = {"nodes": nodes, "edges": edges, "stats": stats}

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGraph written to {output_path}")
    print(f"  Nodes: {stats['total_nodes']} ({', '.join(f'{t}: {c}' for t, c in node_types.items())})")
    print(f"  Edges: {stats['total_edges']} ({', '.join(f'{t}: {c}' for t, c in edge_types.items())})")
    print(f"  Cross-references: {cross_ref_count}")


if __name__ == "__main__":
    main()
