"""
Move empty/stub Notion markdown files to .excluded/ subfolder.

Scans .md files for YAML frontmatter, classifies body content, and moves
files that have no meaningful content. Writes an excluded_manifest.json
for use with --excludeManifest during re-fetch.

Usage:
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion --dryRun          # preview
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion                    # move files
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion --minContentLength 100  # stricter
"""

import os
import re
import argparse
import logging

from main.sources.cleanup import md_cleanup
from main.utils.frontmatter import read_frontmatter_and_body
from main.utils.logger import setup_root_logger

setup_root_logger()

EXCLUDED_DIR = ".excluded"

# Patterns that don't count as real content
REFERENCE_PATTERNS = [
    re.compile(r"^\[Child database:.*\]$"),
    re.compile(r"^\[Child page:.*\]$"),
    re.compile(r"^\[Unsupported:.*\]$"),
]


def classify_body(body_text, min_content_length):
    """Classify body content. Returns reason string or None if content is fine."""
    stripped = body_text.strip()

    if not stripped:
        return "empty_stub"

    # Check if body is only reference markers
    non_reference_lines = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(p.match(line) for p in REFERENCE_PATTERNS):
            continue
        non_reference_lines.append(line)

    if not non_reference_lines:
        return "reference_only"

    actual_text = " ".join(non_reference_lines)
    if len(actual_text) < min_content_length:
        return "minimal_content"

    return None


def main():
    ap = argparse.ArgumentParser(description="Move empty/stub Notion .md files to .excluded/")
    ap.add_argument("--saveMd", required=True, help="Directory with .md files")
    ap.add_argument("--dryRun", action="store_true", default=False, help="Preview without moving")
    ap.add_argument("--minContentLength", type=int, default=50,
                    help="Minimum chars of actual text to keep a file (default: 50)")
    args = ap.parse_args()

    save_md_path = args.saveMd
    excluded_path = os.path.join(save_md_path, EXCLUDED_DIR)
    min_content_length = args.minContentLength

    logging.info(f"Scanning {save_md_path} for .md files (minContentLength={min_content_length})...")

    manifest_entries = []
    category_counts = {"empty_stub": 0, "reference_only": 0, "minimal_content": 0}
    total_scanned = 0

    for filepath, rel_path in md_cleanup.iter_markdown_files(save_md_path):
        total_scanned += 1

        try:
            metadata, body = read_frontmatter_and_body(filepath)
        except Exception as e:
            logging.warning(f"Could not parse {filepath}: {e}")
            continue

        reason = classify_body(body, min_content_length)
        if reason is None:
            continue

        category_counts[reason] += 1

        manifest_entries.append({
            "notion_id": metadata.get("notion_id", ""),
            "last_edited_time": metadata.get("last_edited_time", ""),
            "reason": reason,
            "original_path": rel_path,
            "title": metadata.get("title", ""),
        })

        if args.dryRun:
            logging.info(f"[DRY RUN] [{reason}] {rel_path}")
        else:
            md_cleanup.move_to_excluded(filepath, rel_path, excluded_path)

    # Write manifest
    if not args.dryRun:
        md_cleanup.write_excluded_manifest(excluded_path, manifest_entries, "notion_id")

    # Clean up empty directories
    if not args.dryRun:
        md_cleanup.remove_empty_dirs(save_md_path)

    action = "Would move" if args.dryRun else "Moved"
    total_excluded = sum(category_counts.values())
    logging.info(
        f"Done. Scanned {total_scanned} files. "
        f"{action} {total_excluded}: "
        f"empty_stub={category_counts['empty_stub']}, "
        f"reference_only={category_counts['reference_only']}, "
        f"minimal_content={category_counts['minimal_content']}"
    )


if __name__ == "__main__":
    main()
