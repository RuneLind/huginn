#!/usr/bin/env python3
"""
Scan and sanitize PII from Jira issue markdown files.

Modes:
  --dryRun   Report findings without modifying files (default)
  --apply    Redact PII in-place

Scans both main directory and .excluded/ subdirectory.

Usage:
    # Dry run — report only
    uv run scripts/jira/sanitizers/sanitize_jira_files.py \
        --path ./data/sources/jira-issues

    # Apply redactions in-place
    uv run scripts/jira/sanitizers/sanitize_jira_files.py \
        --path ./data/sources/jira-issues --apply

    # Write findings report to file
    uv run scripts/jira/sanitizers/sanitize_jira_files.py \
        --path ./data/sources/jira-issues --report ./logs/pii_report.log
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.jira.sanitizers.pii_sanitizer import PiiSanitizer, PiiFinding

logger = logging.getLogger(__name__)


def scan_directory(
    base_path: Path,
    sanitizer: PiiSanitizer,
    apply: bool = False,
) -> dict:
    """Scan all .md files under base_path (including .excluded/).

    Returns summary dict with stats and per-file findings.
    """
    stats = {
        "files_scanned": 0,
        "files_with_pii": 0,
        "files_modified": 0,
        "total_findings": 0,
        "findings_by_category": {},
        "affected_files": [],
    }

    md_files = sorted(base_path.rglob("*.md"))
    total = len(md_files)

    for i, md_file in enumerate(md_files, 1):
        try:
            original = md_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read {md_file}: {e}")
            continue

        stats["files_scanned"] += 1
        result = sanitizer.sanitize(original)

        if not result.has_pii:
            continue

        stats["files_with_pii"] += 1
        stats["total_findings"] += len(result.findings)

        rel_path = str(md_file.relative_to(base_path))
        file_entry = {
            "file": rel_path,
            "findings": [],
        }

        for f in result.findings:
            cat = f.category
            stats["findings_by_category"][cat] = stats["findings_by_category"].get(cat, 0) + 1
            file_entry["findings"].append({
                "category": f.category,
                "matched": f.matched_text,
                "redacted": f.redacted_text,
                "line": f.line_number,
            })

        stats["affected_files"].append(file_entry)

        if apply and result.changed:
            # Preserve file mtime (critical for incremental update cutoff)
            stat = md_file.stat()
            md_file.write_text(result.sanitized_text, encoding="utf-8")
            os.utime(md_file, (stat.st_atime, stat.st_mtime))
            stats["files_modified"] += 1

        if i % 500 == 0:
            logger.info(f"  Progress: {i}/{total} files scanned...")

    return stats


def print_report(stats: dict, apply: bool):
    """Print human-readable report to stdout."""
    mode = "APPLY (in-place redaction)" if apply else "DRY RUN (report only)"
    print(f"\n{'=' * 70}")
    print(f"PII SANITIZATION REPORT — {mode}")
    print(f"{'=' * 70}")
    print(f"  Files scanned:    {stats['files_scanned']}")
    print(f"  Files with PII:   {stats['files_with_pii']}")
    if apply:
        print(f"  Files modified:   {stats['files_modified']}")
    print(f"  Total findings:   {stats['total_findings']}")
    print()

    if stats["findings_by_category"]:
        print("  Findings by category:")
        for cat, count in sorted(stats["findings_by_category"].items()):
            print(f"    {cat:20s} {count:5d}")
        print()

    if stats["affected_files"]:
        print(f"  Affected files ({len(stats['affected_files'])}):")
        for entry in stats["affected_files"]:
            findings = entry["findings"]
            cats = {}
            for f in findings:
                cats[f["category"]] = cats.get(f["category"], 0) + 1
            cat_summary = ", ".join(f"{c}:{n}" for c, n in sorted(cats.items()))
            print(f"    {entry['file']}")
            print(f"      [{cat_summary}]")
            for f in findings:
                print(f"      L{f['line']:4d} {f['category']:15s} {f['matched'][:30]:30s} → {f['redacted']}")
        print()

    print(f"{'=' * 70}")
    if not apply and stats["total_findings"] > 0:
        print("  Run with --apply to redact PII in-place.")
    print()


def write_findings_log(stats: dict, log_path: Path, apply: bool):
    """Write structured findings log (for daily update logging)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": "apply" if apply else "dry_run",
        "files_scanned": stats["files_scanned"],
        "files_with_pii": stats["files_with_pii"],
        "files_modified": stats.get("files_modified", 0),
        "total_findings": stats["total_findings"],
        "findings_by_category": stats["findings_by_category"],
        "affected_files": [
            {
                "file": e["file"],
                "findings": [
                    {"category": f["category"], "line": f["line"], "matched": f["matched"]}
                    for f in e["findings"]
                ],
            }
            for e in stats["affected_files"]
        ],
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Findings log appended to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan and sanitize PII from Jira issue markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path", required=True,
        help="Base path to Jira markdown files (e.g. ./data/sources/jira-issues)",
    )
    parser.add_argument(
        "--apply", action="store_true", default=False,
        help="Apply redactions in-place (default: dry run)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to write structured findings log (JSON lines, appended)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_path = Path(args.path)
    if not base_path.is_dir():
        logger.error(f"Directory not found: {base_path}")
        sys.exit(1)

    sanitizer = PiiSanitizer()

    logger.info(f"Scanning {base_path} ({'APPLY' if args.apply else 'DRY RUN'})...")
    stats = scan_directory(base_path, sanitizer, apply=args.apply)

    print_report(stats, apply=args.apply)

    if args.report:
        write_findings_log(stats, Path(args.report), apply=args.apply)


if __name__ == "__main__":
    main()
