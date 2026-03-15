#!/usr/bin/env python3
"""
Huginn Benchmark Runner

Usage:
    uv run benchmarks/runner.py                           # run all benchmarks
    uv run benchmarks/runner.py --suite speed              # speed benchmarks only
    uv run benchmarks/runner.py --suite quality            # quality only
    uv run benchmarks/runner.py --suite graph              # knowledge graph only
    uv run benchmarks/runner.py --suite pii                # PII only
    uv run benchmarks/runner.py --suite tagging            # tagging coverage only
    uv run benchmarks/runner.py --collection jira-issues   # specific collection
    uv run benchmarks/runner.py --compare                  # compare with last baseline
    uv run benchmarks/runner.py --skip-reranker            # skip loading reranker (faster)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.context import load_context
from benchmarks.results import (
    BenchmarkResult,
    create_run,
    load_latest,
    compare_runs,
    format_summary,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Default graph paths to try
DEFAULT_GRAPH_PATHS = [
    "./scripts/knowledge_graph/eessi_graph.json",
    "./scripts/knowledge_graph/jira_graph.json",
    "./data/eessi_graph.json",
    "./data/jira_graph.json",
]


def run_speed_benchmarks(ctx, collections: list[str]) -> list[BenchmarkResult]:
    """Run all speed benchmarks."""
    from benchmarks.speed.bench_model_loading import bench_collection_load
    from benchmarks.speed.bench_search import bench_search_latency, bench_search_scaling, bench_retriever_breakdown
    from benchmarks.speed.bench_indexing import bench_embedding_throughput, bench_indexing_speed

    results = []

    # Model loading benchmarks are expensive (reload models), so we skip them
    # and just measure collection loading and search latency

    for name in collections:
        print(f"  Speed: collection load ({name})")
        results.append(bench_collection_load(ctx.persister, name))

        print(f"  Speed: search latency ({name})")
        results.append(bench_search_latency(ctx, name))

        print(f"  Speed: search scaling ({name})")
        results.append(bench_search_scaling(ctx, name))

        print(f"  Speed: retriever breakdown ({name})")
        results.append(bench_retriever_breakdown(ctx, name))

    print("  Speed: embedding throughput")
    results.append(bench_embedding_throughput(ctx))

    for name in collections[:1]:  # Only first collection to save time
        print(f"  Speed: indexing speed ({name})")
        results.append(bench_indexing_speed(ctx, name))

    return results


def run_quality_benchmarks(ctx, collections: list[str]) -> list[BenchmarkResult]:
    """Run all quality benchmarks."""
    from benchmarks.quality.bench_self_retrieval import bench_self_retrieval
    from benchmarks.quality.bench_known_queries import bench_known_queries
    from benchmarks.quality.bench_search_components import bench_component_comparison
    from benchmarks.quality.bench_realistic_queries import bench_realistic_queries
    from benchmarks.quality.bench_trace_replay import bench_trace_replay, bench_session_replay

    results = []

    for name in collections:
        print(f"  Quality: self-retrieval ({name})")
        results.append(bench_self_retrieval(ctx, name))

        print(f"  Quality: known queries ({name})")
        results.append(bench_known_queries(ctx, name))

        print(f"  Quality: realistic queries ({name})")
        results.append(bench_realistic_queries(ctx, name))

        print(f"  Quality: trace replay ({name})")
        results.append(bench_trace_replay(ctx, name))

        print(f"  Quality: session replay ({name})")
        results.append(bench_session_replay(ctx, name))

        print(f"  Quality: component comparison ({name})")
        results.append(bench_component_comparison(ctx, name))

    return results


def run_graph_benchmarks(ctx) -> list[BenchmarkResult]:
    """Run all knowledge graph benchmarks."""
    from benchmarks.graph.bench_entity_detection import bench_entity_detection
    from benchmarks.graph.bench_query_expansion import bench_query_expansion
    from benchmarks.graph.bench_graph_answers import bench_graph_qa

    results = []

    print("  Graph: entity detection")
    results.append(bench_entity_detection(ctx))

    print("  Graph: graph Q&A")
    results.append(bench_graph_qa(ctx))

    # Query expansion needs a collection
    for name in ctx.collection_names:
        print(f"  Graph: query expansion ({name})")
        results.append(bench_query_expansion(ctx, name))

    return results


def run_pii_benchmarks(ctx, collections: list[str]) -> list[BenchmarkResult]:
    """Run all PII benchmarks."""
    from benchmarks.pii.bench_pii_detection import bench_pii_detection, bench_pii_collection_scan

    results = []

    print("  PII: detection accuracy")
    results.append(bench_pii_detection(ctx))

    # Scan Jira collections for PII leaks
    jira_collections = [c for c in collections if "jira" in c.lower()]
    for name in jira_collections:
        print(f"  PII: collection scan ({name})")
        results.append(bench_pii_collection_scan(ctx, name))

    return results


def run_tagging_benchmarks(ctx, collections: list[str]) -> list[BenchmarkResult]:
    """Run all tagging benchmarks."""
    from benchmarks.tagging.bench_tagging_coverage import bench_tagging_coverage

    results = []

    for name in collections:
        print(f"  Tagging: coverage ({name})")
        results.append(bench_tagging_coverage(ctx, name))

    return results


def main():
    parser = argparse.ArgumentParser(description="Huginn Benchmark Runner")
    parser.add_argument("--suite", action="append", dest="suites",
                        choices=["speed", "quality", "graph", "pii", "tagging"],
                        help="Run specific benchmark suite(s). Can be repeated.")
    parser.add_argument("--collection", action="append", dest="collections",
                        help="Only benchmark specific collection(s)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with latest baseline after running")
    parser.add_argument("--data-path", default="./data/collections",
                        help="Path to collections directory")
    parser.add_argument("--graph-path", action="append", dest="graph_paths",
                        help="Path to knowledge graph JSON file(s)")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="Skip loading reranker for faster startup")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    # Resolve graph paths
    graph_paths = args.graph_paths or [p for p in DEFAULT_GRAPH_PATHS if Path(p).exists()]

    # Load context
    print("Loading models and collections...")
    t0 = time.monotonic()
    ctx = load_context(
        data_path=args.data_path,
        collection_filter=args.collections,
        graph_paths=graph_paths,
        skip_reranker=args.skip_reranker,
    )
    load_time = (time.monotonic() - t0) * 1000
    print(f"Loaded {len(ctx.collection_names)} collections in {load_time:.0f}ms: {', '.join(ctx.collection_names)}")
    if ctx.graph:
        print(f"Knowledge graph: {ctx.graph.node_count()} nodes, {ctx.graph.edge_count()} edges")
    print()

    collections = ctx.collection_names
    all_results: list[BenchmarkResult] = []

    # Run selected suites
    suites = args.suites if args.suites else ["speed", "quality", "graph", "pii", "tagging"]

    for suite in suites:
        print(f"Running {suite} benchmarks...")
        t0 = time.monotonic()

        if suite == "speed":
            all_results.extend(run_speed_benchmarks(ctx, collections))
        elif suite == "quality":
            all_results.extend(run_quality_benchmarks(ctx, collections))
        elif suite == "graph":
            all_results.extend(run_graph_benchmarks(ctx))
        elif suite == "pii":
            all_results.extend(run_pii_benchmarks(ctx, collections))
        elif suite == "tagging":
            all_results.extend(run_tagging_benchmarks(ctx, collections))

        elapsed = (time.monotonic() - t0) * 1000
        print(f"  Done in {elapsed:.0f}ms\n")

    # Create run and save
    run = create_run(all_results)
    filepath = run.save(RESULTS_DIR)
    print(f"Results saved to: {filepath}")

    # Compare with baseline
    comparison = None
    if args.compare:
        # Find the second-most-recent result file to use as baseline
        result_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.name)
        result_files = [f for f in result_files if f.name != "latest.json"]
        if len(result_files) >= 2:
            baseline_file = result_files[-2]
            from benchmarks.results import BenchmarkRun
            baseline = BenchmarkRun.from_json(baseline_file.read_text())
            comparison = compare_runs(run, baseline)
            print(f"Comparing with baseline: {baseline_file.name}")
        else:
            print("No previous baseline found for comparison.")

    # Print summary
    print()
    print(format_summary(run, comparison))


if __name__ == "__main__":
    main()
