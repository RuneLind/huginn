"""
Move empty/stub Notion markdown files to .excluded/ subfolder.

Scans .md files for YAML frontmatter, classifies body content, and moves
files that have no meaningful content. Writes an excluded_manifest.json
for use with --excludeManifest during re-fetch.

Usage:
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion --dryRun          # preview
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion                    # move files
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion --minContentLength 100  # stricter
    uv run notion_cleanup_md.py --saveMd ./data/sources/my-notion --minWordCount 30       # exclude thin pages
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


# Per-source line policy: drop reference-marker lines (child page/database,
# unsupported blocks).
NOTION_LINE_FILTERS = [
    lambda line: any(p.match(line) for p in REFERENCE_PATTERNS),
]


def classify_body(body_text, min_content_length, min_word_count=0):
    """Classify body content. Returns reason string or None if content is fine.

    Notion policy: strip reference-marker lines; a page with nothing but those
    is ``reference_only``.

    ``min_word_count`` was silently absent from notion's copy of this function
    (an accidental drift from its confluence/jira siblings); PR F restores it
    via the shared ``md_cleanup.classify_body`` skeleton. Default 0 keeps the
    word-count gate disabled, so cleanup at the previous defaults is byte-stable;
    the new ``--minWordCount`` flag activates it (now yields ``low_word_count``).
    """
    return md_cleanup.classify_body(
        body_text,
        min_content_length,
        NOTION_LINE_FILTERS,
        min_word_count=min_word_count,
        filtered_empty_reason="reference_only",
    )


def main():
    ap = argparse.ArgumentParser(description="Move empty/stub Notion .md files to .excluded/")
    ap.add_argument("--saveMd", required=True, help="Directory with .md files")
    ap.add_argument("--dryRun", action="store_true", default=False, help="Preview without moving")
    ap.add_argument("--minContentLength", type=int, default=50,
                    help="Minimum chars of actual text to keep a file (default: 50)")
    ap.add_argument("--minWordCount", type=int, default=0,
                    help="Minimum word count to keep a file (0 = disabled)")
    args = ap.parse_args()

    save_md_path = args.saveMd
    excluded_path = os.path.join(save_md_path, EXCLUDED_DIR)
    min_content_length = args.minContentLength
    min_word_count = args.minWordCount

    logging.info(f"Scanning {save_md_path} for .md files (minContentLength={min_content_length}, minWordCount={min_word_count})...")

    manifest_entries = []
    category_counts = {"empty_stub": 0, "reference_only": 0, "minimal_content": 0, "low_word_count": 0}
    total_scanned = 0

    for filepath, rel_path in md_cleanup.iter_markdown_files(save_md_path):
        total_scanned += 1

        try:
            metadata, body = read_frontmatter_and_body(filepath)
        except Exception as e:
            logging.warning(f"Could not parse {filepath}: {e}")
            continue

        reason = classify_body(body, min_content_length, min_word_count)
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
        f"minimal_content={category_counts['minimal_content']}, "
        f"low_word_count={category_counts['low_word_count']}"
    )


if __name__ == "__main__":
    main()
