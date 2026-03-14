"""Benchmark: PII detection accuracy and speed."""

import json
import time
from pathlib import Path

from benchmarks.context import BenchmarkContext, load_documents_for_collection
from benchmarks.results import BenchmarkResult
from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer

DATA_DIR = Path(__file__).parent.parent / "data"

# Test vectors for PII detection — true positives and true negatives
# Valid Norwegian fødselsnummer must pass mod-11 checksum.
# These are synthetic test numbers that pass the checksum validation.
DEFAULT_PII_CASES = {
    "true_positives": [
        {"text": "Fødselsnummer: 01010100050", "category": "personnummer",
         "description": "Standard fnr format (valid mod-11 checksum)"},
        {"text": "kontakt ola.nordmann@firma.no for info", "category": "email",
         "description": "Norwegian email"},
        {"text": "passord er hemmelig123", "category": "password",
         "description": "Norwegian password pattern"},
        {"text": "password is secret456", "category": "password",
         "description": "English password pattern"},
        {"text": "password: MyS3cret!", "category": "password",
         "description": "Password with colon"},
        {"text": "bruker@navikt.no sendte melding", "category": "email",
         "description": "Work email"},
    ],
    "true_negatives": [
        {"text": "GitHub run 12345678901", "description": "11-digit GitHub run ID"},
        {"text": "order ID: 98765432101", "description": "11 digits, invalid checksum"},
        {"text": "referanse 00000000000", "description": "All zeros"},
        {"text": "foo@example.com is test", "description": "Allowlisted domain"},
        {"text": "Foo@file.xsl reference", "description": "File-like reference"},
        {"text": "bar@test.com placeholder", "description": "Allowlisted test domain"},
        {"text": "Template@config.json", "description": "Config file reference"},
        {"text": "dette er en vanlig setning uten PII", "description": "No PII at all"},
        {"text": "telefon: 22334455", "description": "8-digit phone number, not fnr"},
    ],
}


def bench_pii_detection(ctx: BenchmarkContext) -> BenchmarkResult:
    """Measure PII detection accuracy: TP, FP, TN, FN rates."""
    sanitizer = PiiSanitizer()

    # Load test cases from JSON if available
    cases_file = DATA_DIR / "pii_test_cases.json"
    if cases_file.exists():
        cases = json.loads(cases_file.read_text())
    else:
        cases = DEFAULT_PII_CASES

    t_start = time.monotonic()

    tp = 0  # True positives: PII correctly detected
    fn = 0  # False negatives: PII missed
    tn = 0  # True negatives: non-PII correctly ignored
    fp = 0  # False positives: non-PII incorrectly flagged
    detection_times_us = []
    errors = []

    # True positives
    for case in cases["true_positives"]:
        t0 = time.monotonic()
        findings = sanitizer.detect(case["text"])
        detection_times_us.append((time.monotonic() - t0) * 1_000_000)

        if findings:
            # Check category matches
            found_categories = {f.category for f in findings}
            if case["category"] in found_categories:
                tp += 1
            else:
                fn += 1
                errors.append({
                    "type": "wrong_category",
                    "text": case["text"],
                    "expected": case["category"],
                    "found": list(found_categories),
                })
        else:
            fn += 1
            errors.append({
                "type": "false_negative",
                "text": case["text"],
                "expected": case["category"],
                "description": case.get("description", ""),
            })

    # True negatives
    for case in cases["true_negatives"]:
        t0 = time.monotonic()
        findings = sanitizer.detect(case["text"])
        detection_times_us.append((time.monotonic() - t0) * 1_000_000)

        if not findings:
            tn += 1
        else:
            fp += 1
            errors.append({
                "type": "false_positive",
                "text": case["text"],
                "found_categories": [f.category for f in findings],
                "description": case.get("description", ""),
            })

    total_duration = (time.monotonic() - t_start) * 1000

    total_positive = tp + fn
    total_negative = tn + fp

    metrics = {
        "true_positive_rate": tp / total_positive if total_positive else 1.0,
        "false_negative_rate": fn / total_positive if total_positive else 0.0,
        "true_negative_rate": tn / total_negative if total_negative else 1.0,
        "false_positive_rate": fp / total_negative if total_negative else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "total_cases": len(cases["true_positives"]) + len(cases["true_negatives"]),
        "detection_speed_us_mean": sum(detection_times_us) / len(detection_times_us) if detection_times_us else 0,
    }

    return BenchmarkResult(
        name="pii_detection",
        category="pii",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"errors": errors},
    )


def bench_pii_collection_scan(ctx: BenchmarkContext, collection_name: str) -> BenchmarkResult:
    """Scan a Jira collection to verify no PII leaked through sanitization.

    After sanitization during ingestion, documents should contain no detectable PII.
    """
    sanitizer = PiiSanitizer()
    documents = load_documents_for_collection(ctx.persister, collection_name)

    t_start = time.monotonic()

    files_scanned = 0
    files_with_pii = 0
    findings_by_category: dict[str, int] = {}
    sample_findings = []

    for doc in documents:
        text = doc.get("text", "")
        if not text:
            continue

        files_scanned += 1
        findings = sanitizer.detect(text)

        if findings:
            files_with_pii += 1
            for f in findings:
                findings_by_category[f.category] = findings_by_category.get(f.category, 0) + 1
            if len(sample_findings) < 5:
                sample_findings.append({
                    "doc_id": doc.get("id", "unknown"),
                    "categories": [f.category for f in findings],
                    "count": len(findings),
                })

    total_duration = (time.monotonic() - t_start) * 1000

    metrics = {
        "files_scanned": files_scanned,
        "files_with_pii": files_with_pii,
        "pii_free_rate": (files_scanned - files_with_pii) / files_scanned if files_scanned else 1.0,
        "total_findings": sum(findings_by_category.values()),
        "scan_speed_docs_per_second": files_scanned / (total_duration / 1000) if total_duration > 0 else 0,
    }

    # Add per-category counts
    for cat, count in findings_by_category.items():
        metrics[f"findings_{cat}"] = count

    return BenchmarkResult(
        name=f"pii_collection_scan_{collection_name}",
        category="pii",
        metrics=metrics,
        duration_ms=total_duration,
        metadata={"collection": collection_name, "sample_findings": sample_findings},
    )
