"""Benchmark: graph query answering correctness."""

import time

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult

# Test cases for graph Q&A
GRAPH_QA_CASES = [
    # Relational queries — should produce answers
    {
        "query": "Hvilke SEDer inneholder LA_BUC_01?",
        "entities": ["buc:LA_BUC_01"],
        "should_answer": True,
        "expected_in_answer": ["A001"],
    },
    {
        "query": "Hvilke SEDer inneholder LA_BUC_02?",
        "entities": ["buc:LA_BUC_02"],
        "should_answer": True,
        "expected_in_answer": ["A003"],
    },
    {
        "query": "Which BUCs contain A003?",
        "entities": ["sed:A003"],
        "should_answer": True,
        "expected_in_answer": ["LA_BUC_02"],
    },
    {
        "query": "Hva er hjemmelen for LA_BUC_01?",
        "entities": ["buc:LA_BUC_01"],
        "should_answer": True,
        "expected_in_answer": ["Artikkel"],
    },
    # Non-relational queries — should return None
    {
        "query": "Forklar hva LA_BUC_01 betyr",
        "entities": ["buc:LA_BUC_01"],
        "should_answer": False,
    },
    {
        "query": "Hva er status på dette?",
        "entities": [],
        "should_answer": False,
    },
]


def bench_graph_qa(ctx: BenchmarkContext) -> BenchmarkResult:
    """Test graph Q&A accuracy: relational queries should get answers, others shouldn't."""
    graph = ctx.graph
    if not graph:
        return BenchmarkResult(
            name="graph_qa",
            category="graph",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "No knowledge graph loaded"},
        )

    t_start = time.monotonic()

    correct_answers = 0
    correct_non_answers = 0
    wrong_answers = 0
    missed_answers = 0
    content_checks_passed = 0
    content_checks_total = 0
    details = []

    # Combine static and dynamic cases
    all_cases = list(GRAPH_QA_CASES)
    all_cases.extend(_dynamic_qa_cases(graph))

    for case in all_cases:
        # Only use entities that exist in the graph
        entities = [e for e in case["entities"] if e in graph.nodes]

        answer = graph.answer_graph_query(entities, case["query"])

        if case["should_answer"]:
            if answer is not None:
                correct_answers += 1
                # Check expected content
                expected = case.get("expected_in_answer", [])
                for exp in expected:
                    content_checks_total += 1
                    if exp in answer:
                        content_checks_passed += 1
                    else:
                        details.append({
                            "query": case["query"],
                            "issue": f"Expected '{exp}' not found in answer",
                            "answer_snippet": answer[:200],
                        })
            else:
                missed_answers += 1
                details.append({"query": case["query"], "issue": "Expected answer but got None"})
        else:
            if answer is None:
                correct_non_answers += 1
            else:
                wrong_answers += 1
                details.append({
                    "query": case["query"],
                    "issue": "Expected None but got answer",
                    "answer_snippet": answer[:200],
                })

    total_duration = (time.monotonic() - t_start) * 1000
    n = len(all_cases)

    metrics = {
        "accuracy": (correct_answers + correct_non_answers) / n if n else 0,
        "answer_rate": correct_answers / sum(1 for c in all_cases if c["should_answer"]) if any(c["should_answer"] for c in all_cases) else 0,
        "non_answer_rate": correct_non_answers / sum(1 for c in all_cases if not c["should_answer"]) if any(not c["should_answer"] for c in all_cases) else 0,
        "content_accuracy": content_checks_passed / content_checks_total if content_checks_total else 1.0,
        "test_cases": n,
    }

    return BenchmarkResult(
        name="graph_qa",
        category="graph",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"details": details},
    )


def _dynamic_qa_cases(graph) -> list[dict]:
    """Generate Q&A test cases from actual graph nodes."""
    import random
    rng = random.Random(42)
    cases = []

    # Find epics with child issues (relational queries that should work)
    epics_with_issues = []
    for node_id, node in graph.nodes.items():
        if node["type"] == "Epic":
            children = [
                e for e in graph.incoming.get(node_id, [])
                if e["type"] == "tilhører_epic"
            ]
            if children:
                epics_with_issues.append((node_id, node, children))

    for epic_id, epic, children in rng.sample(epics_with_issues, min(3, len(epics_with_issues))):
        key = epic_id.split(":", 1)[1]
        child_key = children[0]["source"].split(":", 1)[1]
        cases.append({
            "query": f"Hvilke issues tilhører {key}?",
            "entities": [epic_id],
            "should_answer": True,
            "expected_in_answer": [child_key],
        })

    # Non-relational queries about real entities (should return None)
    issues = [nid for nid in graph.nodes if graph.nodes[nid]["type"] == "Issue"]
    for issue_id in rng.sample(issues, min(2, len(issues))):
        key = issue_id.split(":", 1)[1]
        cases.append({
            "query": f"Forklar hva {key} handler om",
            "entities": [issue_id],
            "should_answer": False,
        })

    return cases
