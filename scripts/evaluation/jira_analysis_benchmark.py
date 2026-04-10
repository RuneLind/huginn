#!/usr/bin/env python3
"""
Jira Analysis Quality Benchmark — autoresearch-style evaluation.

Measures how well different knowledge base configurations support Jira issue analysis.
Inspired by Karpathy's autoresearch: fixed input (Jira issues), variable treatment
(collection configs), measurable output (LLM-judged quality scores).

The loop: configure → search → score → compare → keep/discard.

Usage:
    .venv/bin/python scripts/evaluation/jira_analysis_benchmark.py
    .venv/bin/python scripts/evaluation/jira_analysis_benchmark.py --config scripts/evaluation/benchmark_config.json
    .venv/bin/python scripts/evaluation/jira_analysis_benchmark.py --config scripts/evaluation/benchmark_config.json --issues MELOSYS-7219 MELOSYS-6432
    .venv/bin/python scripts/evaluation/jira_analysis_benchmark.py --judge-model claude-sonnet-4-20250514

Requires:
    - Knowledge API Server running (default: http://localhost:8321)
    - Claude CLI available (for LLM judge scoring)
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

HUGINN_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    collection: str
    query: str
    results: list[dict]
    score: float | None = None  # best result relevance score


@dataclass
class IssueEvaluation:
    issue_key: str
    config_name: str
    search_results: list[SearchResult]
    graph_results: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    judge_reasoning: str = ""


# ── Knowledge API client ─────────────────────────────────────────────────────

class KnowledgeAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30)

    def search(self, collection: str, query: str, limit: int = 5) -> list[dict]:
        try:
            resp = self.client.get(
                f"{self.base_url}/api/search",
                params={"collection": collection, "q": query, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            log.warning(f"Search failed for {collection}/{query}: {e}")
            return []

    def get_graph_node(self, node_id: str) -> dict | None:
        try:
            resp = self.client.get(f"{self.base_url}/api/graph/{node_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def health(self) -> bool:
        try:
            resp = self.client.get(f"{self.base_url}/api/collections")
            return resp.status_code == 200
        except Exception:
            return False


# ── Query extraction ─────────────────────────────────────────────────────────

def extract_search_queries(issue_content: str, num_queries: int = 3) -> list[str]:
    """Extract meaningful search queries from Jira issue content.

    Builds queries from frontmatter fields and issue body to simulate
    what an AI agent would search for during analysis.
    """
    queries = []

    # Extract frontmatter fields
    fm_match = re.match(r"^---\s*\n(.*?)\n---", issue_content, re.DOTALL)
    frontmatter = {}
    if fm_match:
        for line in fm_match.group(1).split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                frontmatter[key.strip()] = val.strip().strip('"').strip("'")

    title = frontmatter.get("title", "")
    epic_link = frontmatter.get("epic_link", "")

    # Query 1: The issue title/summary — most direct search
    if title:
        queries.append(title)

    # Query 2: Key domain terms from the body (skip frontmatter)
    body = issue_content
    if fm_match:
        body = issue_content[fm_match.end():]

    # Extract domain-specific terms
    domain_terms = set()
    domain_patterns = [
        r"(?:BUC|SED)\s*[A-Z]\d{3}",           # BUC/SED codes
        r"(?:artikkel|art\.?)\s*\d+",             # Article references
        r"forordning\s*\d+/\d+",                  # Regulation references
        r"folketrygdloven\s*§\s*\d+",             # Norwegian law refs
        r"(?:lovvalg|trygdeavgift|medlemskap|vedtak|årsavregning)",  # Domain concepts
        r"MELOSYS-\d+",                            # Issue cross-refs
    ]
    for pattern in domain_patterns:
        for m in re.finditer(pattern, body, re.IGNORECASE):
            domain_terms.add(m.group(0))

    if domain_terms:
        # Combine a few domain terms into a query
        terms_list = sorted(domain_terms)[:4]
        queries.append(" ".join(terms_list))

    # Query 3: Epic context if available
    if epic_link:
        queries.append(f"epic {epic_link} {title[:50] if title else ''}")

    # Pad with title variants if we don't have enough queries
    while len(queries) < num_queries and title:
        # Try shorter/broader version of title
        words = title.split()
        if len(words) > 3:
            queries.append(" ".join(words[:len(words) // 2]))
            break
        else:
            break

    return queries[:num_queries]


# ── LLM Judge ────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of knowledge base search results for analyzing a Jira issue.

## Jira Issue
{issue_content}

## Search Results Retrieved
{search_results_text}

## Graph Context Retrieved
{graph_context}

## Scoring Rubric
Score each dimension 1-5 (1=poor, 3=adequate, 5=excellent):

{rubric_text}

## Instructions
- Score based on what an AI agent would need to produce a thorough analysis and work plan
- Consider both what IS found and what is MISSING
- Be specific about gaps in your reasoning

Respond in this exact JSON format (no markdown, no code fences):
{{"domain_understanding": <score>, "technical_context": <score>, "related_work": <score>, "actionability": <score>, "noise_ratio": <score>, "reasoning": "<2-3 sentences explaining the scores>"}}"""


def score_with_llm_judge(
    issue_content: str,
    search_results: list[SearchResult],
    graph_results: list[dict],
    rubric: dict,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[dict[str, float], str]:
    """Use Claude as an LLM judge to score search result quality."""

    # Format search results
    sr_parts = []
    for sr in search_results:
        sr_parts.append(f"\n### Collection: {sr.collection} | Query: \"{sr.query}\"")
        if not sr.results:
            sr_parts.append("  (no results)")
        for i, r in enumerate(sr.results[:5]):
            title = r.get("title", r.get("documentId", "untitled"))
            snippet = r.get("matchedContent", r.get("text", ""))[:200]
            score = r.get("score", "n/a")
            sr_parts.append(f"  {i + 1}. [{score}] {title}\n     {snippet}")
    search_results_text = "\n".join(sr_parts) if sr_parts else "(no search results)"

    # Format graph context
    graph_parts = []
    for g in graph_results:
        if g:
            graph_parts.append(json.dumps(g, indent=2, ensure_ascii=False)[:500])
    graph_context = "\n".join(graph_parts) if graph_parts else "(no graph context)"

    # Format rubric
    rubric_lines = []
    for dim in rubric["dimensions"]:
        rubric_lines.append(f"- **{dim['name']}** (weight {dim['weight']}): {dim['description']}")
    rubric_text = "\n".join(rubric_lines)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        issue_content=issue_content[:2000],  # truncate very long issues
        search_results_text=search_results_text,
        graph_context=graph_context,
        rubric_text=rubric_text,
    )

    try:
        result = subprocess.run(
            ["claude", "--model", model, "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        response = result.stdout.strip()

        # Parse JSON from response (handle potential markdown wrapping)
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            scores = {}
            for dim in rubric["dimensions"]:
                name = dim["name"]
                scores[name] = float(data.get(name, 3))
            reasoning = data.get("reasoning", "")
            return scores, reasoning

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        log.warning(f"LLM judge failed: {e}")

    # Fallback: neutral scores
    return {dim["name"]: 3.0 for dim in rubric["dimensions"]}, "Judge failed — neutral scores"


# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate_issue(
    api: KnowledgeAPIClient,
    issue_key: str,
    issue_content: str,
    config: dict,
    rubric: dict,
    search_limit: int = 5,
    num_queries: int = 3,
    judge_model: str = "claude-sonnet-4-20250514",
) -> IssueEvaluation:
    """Run one evaluation: search collections for an issue, then score results."""

    queries = extract_search_queries(issue_content, num_queries)
    log.info(f"  Queries for {issue_key}: {queries}")

    # Search each collection with each query
    search_results = []
    for collection in config["collections"]:
        for query in queries:
            results = api.search(collection, query, limit=search_limit)
            best_score = min((r.get("score", 0) for r in results), default=None) if results else None
            search_results.append(SearchResult(
                collection=collection,
                query=query,
                results=results,
                score=best_score,
            ))

    # Graph lookup if enabled
    graph_results = []
    if config.get("use_graph"):
        # Look up the issue itself and its epic
        for node_type in ["issue", "epic"]:
            node = api.get_graph_node(f"{node_type}:{issue_key}")
            if node:
                graph_results.append(node)

    # Score with LLM judge
    scores, reasoning = score_with_llm_judge(
        issue_content, search_results, graph_results, rubric, model=judge_model,
    )

    # Compute weighted score
    total_weight = sum(d["weight"] for d in rubric["dimensions"])
    weighted = sum(
        scores.get(d["name"], 3) * d["weight"] for d in rubric["dimensions"]
    ) / total_weight

    return IssueEvaluation(
        issue_key=issue_key,
        config_name=config["name"],
        search_results=search_results,
        graph_results=graph_results,
        scores=scores,
        weighted_score=weighted,
        judge_reasoning=reasoning,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(evaluations: list[IssueEvaluation], rubric: dict):
    """Print comparison table of evaluation results."""

    # Group by config
    by_config: dict[str, list[IssueEvaluation]] = {}
    for ev in evaluations:
        by_config.setdefault(ev.config_name, []).append(ev)

    dimensions = [d["name"] for d in rubric["dimensions"]]

    print("\n" + "=" * 80)
    print("JIRA ANALYSIS BENCHMARK RESULTS")
    print("=" * 80)

    # Per-config summary
    for config_name, evals in by_config.items():
        print(f"\n--- Configuration: {config_name} ---")
        print(f"{'Issue':<18} {'Weighted':>8}  " + "  ".join(f"{d[:12]:>12}" for d in dimensions))
        print("-" * (28 + 14 * len(dimensions)))

        for ev in sorted(evals, key=lambda e: e.issue_key):
            dim_scores = "  ".join(f"{ev.scores.get(d, 0):>12.1f}" for d in dimensions)
            print(f"{ev.issue_key:<18} {ev.weighted_score:>8.2f}  {dim_scores}")

        avg_weighted = sum(e.weighted_score for e in evals) / len(evals)
        avg_dims = "  ".join(
            f"{sum(e.scores.get(d, 0) for e in evals) / len(evals):>12.1f}"
            for d in dimensions
        )
        print("-" * (28 + 14 * len(dimensions)))
        print(f"{'AVERAGE':<18} {avg_weighted:>8.2f}  {avg_dims}")

    # Cross-config comparison
    if len(by_config) > 1:
        print(f"\n{'=' * 60}")
        print("CONFIGURATION COMPARISON")
        print(f"{'=' * 60}")
        print(f"{'Config':<20} {'Avg Score':>10}  {'vs Baseline':>12}")
        print("-" * 44)

        baseline_avg = None
        for config_name, evals in by_config.items():
            avg = sum(e.weighted_score for e in evals) / len(evals)
            if baseline_avg is None:
                baseline_avg = avg
                delta = "  (baseline)"
            else:
                diff = avg - baseline_avg
                delta = f"  {diff:+.2f} ({diff / baseline_avg * 100:+.1f}%)"
            print(f"{config_name:<20} {avg:>10.2f}{delta}")

    # Per-issue reasoning (abbreviated)
    print(f"\n{'=' * 60}")
    print("JUDGE REASONING (per issue, last config)")
    print(f"{'=' * 60}")
    for ev in evaluations:
        if ev.config_name == list(by_config.keys())[-1]:
            print(f"\n{ev.issue_key} [{ev.weighted_score:.2f}]: {ev.judge_reasoning}")


def save_results(evaluations: list[IssueEvaluation], output_path: Path):
    """Save detailed results to JSON for further analysis."""
    data = []
    for ev in evaluations:
        data.append({
            "issue_key": ev.issue_key,
            "config_name": ev.config_name,
            "scores": ev.scores,
            "weighted_score": ev.weighted_score,
            "judge_reasoning": ev.judge_reasoning,
            "num_search_results": sum(len(sr.results) for sr in ev.search_results),
            "graph_results_found": len(ev.graph_results),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluations": data,
        }, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Jira Analysis Quality Benchmark")
    parser.add_argument(
        "--config",
        default="scripts/evaluation/benchmark_config.json",
        help="Path to benchmark config JSON",
    )
    parser.add_argument(
        "--issues",
        nargs="*",
        help="Run only these issue keys (default: all)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        help="Run only these configuration names (default: all)",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-20250514",
        help="Model for LLM judge scoring",
    )
    parser.add_argument(
        "--output",
        default="scripts/evaluation/results/latest.json",
        help="Path for JSON results output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run searches but skip LLM judge scoring",
    )
    args = parser.parse_args()

    # Load config
    config_path = HUGINN_ROOT / args.config
    with open(config_path) as f:
        bench_config = json.load(f)

    api_url = bench_config["knowledge_api_url"]
    search_limit = bench_config.get("search_limit", 5)
    num_queries = bench_config.get("searches_per_issue", 3)
    rubric = bench_config["scoring_rubric"]

    # Filter issues/configs if requested
    issues = bench_config["benchmark_issues"]
    if args.issues:
        issues = [i for i in issues if i["key"] in args.issues]

    configs = bench_config["configurations"]
    if args.configs:
        configs = [c for c in configs if c["name"] in args.configs]

    # Check API
    api = KnowledgeAPIClient(api_url)
    if not api.health():
        log.error(f"Knowledge API not reachable at {api_url}")
        sys.exit(1)
    log.info(f"Knowledge API OK at {api_url}")

    # Run evaluations
    evaluations: list[IssueEvaluation] = []
    total = len(issues) * len(configs)
    done = 0

    for config in configs:
        log.info(f"\n=== Configuration: {config['name']} ({config['description']}) ===")

        for issue_info in issues:
            done += 1
            issue_key = issue_info["key"]
            issue_path = HUGINN_ROOT / issue_info["file"]

            if not issue_path.exists():
                log.warning(f"[{done}/{total}] Issue file not found: {issue_path}")
                continue

            log.info(f"[{done}/{total}] Evaluating {issue_key} with {config['name']}")
            issue_content = issue_path.read_text()

            ev = evaluate_issue(
                api=api,
                issue_key=issue_key,
                issue_content=issue_content,
                config=config,
                rubric=rubric,
                search_limit=search_limit,
                num_queries=num_queries,
                judge_model=args.judge_model if not args.dry_run else "skip",
            )

            if args.dry_run:
                # Just count search results, skip scoring
                total_results = sum(len(sr.results) for sr in ev.search_results)
                ev.scores = {d["name"]: 0 for d in rubric["dimensions"]}
                ev.judge_reasoning = f"dry-run: {total_results} search results, {len(ev.graph_results)} graph nodes"
                ev.weighted_score = 0

            evaluations.append(ev)

    # Report
    print_results(evaluations, rubric)
    save_results(evaluations, HUGINN_ROOT / args.output)


if __name__ == "__main__":
    main()
