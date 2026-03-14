"""Benchmark: knowledge graph entity detection accuracy."""

import json
import time

from benchmarks.context import BenchmarkContext
from benchmarks.results import BenchmarkResult

# Inline test cases — also loadable from JSON
DEFAULT_TEST_CASES = [
    # BUC patterns
    {"text": "Hva er LA_BUC_01?", "expected": ["buc:LA_BUC_01"]},
    {"text": "LA BUC 02 prosessen", "expected": ["buc:LA_BUC_02"]},
    {"text": "LABUC01 detaljer", "expected": ["buc:LA_BUC_01"]},
    # SED patterns
    {"text": "Hva inneholder A003?", "expected": ["sed:A003"]},
    {"text": "X001 brukes til forespørsler", "expected": ["sed:X001"]},
    {"text": "hva er a003?", "expected": ["sed:A003"]},
    {"text": "x001 info", "expected": ["sed:X001"]},
    # Artikkel patterns
    {"text": "artikkel 13 gjelder arbeid i flere land", "expected": ["artikkel:13"]},
    {"text": "art. 12 utsendte arbeidstakere", "expected": ["artikkel:12"]},
    {"text": "art 13 nr. 1 regler", "expected": ["artikkel:13.1", "artikkel:13"]},
    # Forordning
    {"text": "forordning 883/2004 artikkel 12", "expected": ["forordning:883/2004", "artikkel:12"]},
    # Multiple entities
    {"text": "LA_BUC_01 inneholder A001 og A002", "expected": ["buc:LA_BUC_01", "sed:A001", "sed:A002"]},
    # Jira keys
    {"text": "Se MELOSYS-5203 for detaljer", "expected_pattern": "issue:MELOSYS-5203|epic:MELOSYS-5203"},
    # Negatives (should detect nothing)
    {"text": "A999 finnes ikke", "expected": []},
    {"text": "ingen entiteter her", "expected": []},
    {"text": "dette er en vanlig setning", "expected": []},
]


def bench_entity_detection(ctx: BenchmarkContext) -> BenchmarkResult:
    """Measure entity detection precision and recall."""
    graph = ctx.graph
    if not graph:
        return BenchmarkResult(
            name="entity_detection",
            category="graph",
            metrics={"skipped": 1},
            duration_ms=0,
            metadata={"reason": "No knowledge graph loaded"},
        )

    # Load test cases from JSON if available, otherwise use defaults
    cases_file = ctx.find_data_file("entity_test_cases.json")
    if cases_file:
        cases = json.loads(cases_file.read_text())["cases"]
    else:
        cases = list(DEFAULT_TEST_CASES)

    # Add dynamic test cases from the actual graph
    cases.extend(_dynamic_cases_from_graph(graph))

    t_start = time.monotonic()

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_expected = 0
    total_detected = 0
    detection_times_us = []
    fp_details = []
    fn_details = []

    for case in cases:
        text = case["text"]

        t0 = time.monotonic()
        detected = graph.detect_entities(text)
        detection_times_us.append((time.monotonic() - t0) * 1_000_000)

        expected = set(case.get("expected", []))
        expected_pattern = case.get("expected_pattern")

        if expected_pattern:
            # Pattern-based matching (for Jira keys that could be issue or epic)
            import re
            pattern = re.compile(expected_pattern)
            matched = [d for d in detected if pattern.match(d)]
            if matched:
                true_positives += 1
                total_expected += 1
                total_detected += len(detected)
            elif not detected:
                false_negatives += 1
                total_expected += 1
                fn_details.append({"text": text, "expected_pattern": expected_pattern})
            continue

        detected_set = set(detected)
        total_expected += len(expected)
        total_detected += len(detected_set)

        tp = len(expected & detected_set)
        fp = len(detected_set - expected)
        fn = len(expected - detected_set)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

        if fp > 0:
            fp_details.append({"text": text, "unexpected": list(detected_set - expected)})
        if fn > 0:
            fn_details.append({"text": text, "missed": list(expected - detected_set)})

    total_duration = (time.monotonic() - t_start) * 1000

    precision = true_positives / total_detected if total_detected else 1.0
    recall = true_positives / total_expected if total_expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "test_cases": len(cases),
        "detection_speed_us_mean": sum(detection_times_us) / len(detection_times_us) if detection_times_us else 0,
    }

    return BenchmarkResult(
        name="entity_detection",
        category="graph",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"fp_details": fp_details[:5], "fn_details": fn_details[:5]},
    )


def _dynamic_cases_from_graph(graph) -> list[dict]:
    """Generate test cases from actual graph nodes to ensure coverage."""
    import random
    rng = random.Random(42)
    cases = []

    # Sample some Jira issue/epic keys from the real graph
    epics = [nid for nid in graph.nodes if nid.startswith("epic:")]
    issues = [nid for nid in graph.nodes if nid.startswith("issue:")]

    for epic_id in rng.sample(epics, min(3, len(epics))):
        key = epic_id.split(":", 1)[1]
        cases.append({
            "text": f"Se {key} for detaljer om epic",
            "expected": [epic_id],
        })

    for issue_id in rng.sample(issues, min(3, len(issues))):
        key = issue_id.split(":", 1)[1]
        cases.append({
            "text": f"{key} er en viktig oppgave",
            "expected": [issue_id],
        })

    # Sample BUC/SED if they exist
    bucs = [nid for nid in graph.nodes if nid.startswith("buc:")]
    seds = [nid for nid in graph.nodes if nid.startswith("sed:")]

    for buc_id in rng.sample(bucs, min(2, len(bucs))):
        label = graph.nodes[buc_id].get("label", buc_id)
        # Extract the BUC code from label (e.g., "LA_BUC_01")
        import re
        m = re.search(r'LA[_ ]?BUC[_ ]?\d{1,2}', label)
        if m:
            cases.append({"text": f"Hva inneholder {m.group(0)}?", "expected": [buc_id]})

    for sed_id in rng.sample(seds, min(2, len(seds))):
        label = graph.nodes[sed_id].get("label", sed_id)
        import re
        m = re.search(r'[AX]\d{3}', label)
        if m:
            cases.append({"text": f"Detaljer om {m.group(0)}", "expected": [sed_id]})

    return cases
