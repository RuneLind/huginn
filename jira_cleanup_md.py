"""
Move empty/stub and noisy Jira markdown files to .excluded/ subfolder.

Scans .md files for YAML frontmatter, classifies body content, and moves
files that have no meaningful content. Supports status-based, title-based,
and label-based noise filtering via configurable JSON patterns.

Writes an excluded_manifest.json for use with --excludeManifest during re-fetch.

Usage:
    uv run jira_cleanup_md.py --saveMd ./data/sources/jira-issues --dryRun              # preview
    uv run jira_cleanup_md.py --saveMd ./data/sources/jira-issues                        # move files
    uv run jira_cleanup_md.py --saveMd ./data/sources/jira-issues --minWordCount 30      # stricter
    uv run jira_cleanup_md.py --saveMd ./data/sources/jira-issues --noiseConfig ./jira_noise_patterns.json
"""

import os
import re
import json
import shutil
import argparse
import logging
from datetime import datetime, timedelta

from main.sources.cleanup import md_cleanup
from main.utils.frontmatter import read_frontmatter_and_body
from main.utils.logger import setup_root_logger
from main.utils.manifest import merge_manifest_entries

setup_root_logger()

EXCLUDED_DIR = ".excluded"


def load_noise_patterns(config_path):
    """Load noise patterns from a JSON config file.

    Config format: list of entries, each with a "reason" and one of:
    - {"status_pattern": "<substring>", "reason": "..."} — status-based (case-insensitive)
    - {"title_pattern": "<substring>", "reason": "..."} — title-based (case-insensitive)
    - {"label_pattern": "<substring>", "reason": "..."} — label-based (case-insensitive)
    - {"type_pattern": "<substring>", "reason": "..."} — issue type-based (case-insensitive)
    """
    if not config_path or not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_status_pattern(status, noise_patterns):
    """Check if a status matches any status-based noise pattern. Returns reason or None."""
    if not status:
        return None
    status_lower = status.lower()
    for entry in noise_patterns:
        if "status_pattern" in entry and entry["status_pattern"].lower() in status_lower:
            return entry["reason"]
    return None


def match_title_pattern(title, noise_patterns):
    """Check if a title matches any title-based noise pattern. Returns reason or None."""
    if not title:
        return None
    title_lower = title.lower()
    for entry in noise_patterns:
        if "title_pattern" in entry and entry["title_pattern"].lower() in title_lower:
            return entry["reason"]
    return None


def match_label_pattern(labels_str, noise_patterns):
    """Check if any label matches a label-based noise pattern. Returns reason or None."""
    if not labels_str:
        return None
    labels_lower = labels_str.lower()
    for entry in noise_patterns:
        if "label_pattern" in entry and entry["label_pattern"].lower() in labels_lower:
            return entry["reason"]
    return None


def match_type_pattern(issue_type, noise_patterns):
    """Check if issue type matches a type-based noise pattern. Returns reason or None."""
    if not issue_type:
        return None
    type_lower = issue_type.lower()
    for entry in noise_patterns:
        if "type_pattern" in entry and entry["type_pattern"].lower() in type_lower:
            return entry["reason"]
    return None


def compute_exclusion(metadata, body, age_cutoff, noise_patterns,
                      min_content_length, min_word_count):
    """Classify a file against all exclusion rules.

    Returns (category, reason) where reason is None if the file should be kept.
    Pulled out of the main walk so we can run it in two passes — once to
    determine tentative exclusions, then again to preserve parents-of-kept
    subtasks before actually moving files.
    """
    if age_cutoff:
        updated_str = metadata.get("modifiedTime", metadata.get("updated", ""))
        if updated_str:
            try:
                updated_dt = datetime.fromisoformat(updated_str).replace(tzinfo=None)
                if updated_dt < age_cutoff:
                    return ("too_old", f"last updated {updated_str[:10]}")
            except (ValueError, TypeError):
                pass

    reason = match_status_pattern(metadata.get("status", ""), noise_patterns)
    if reason:
        return ("noise_status", reason)
    reason = match_type_pattern(metadata.get("issue_type", ""), noise_patterns)
    if reason:
        return ("noise_type", reason)
    reason = match_title_pattern(metadata.get("title", ""), noise_patterns)
    if reason:
        return ("noise_title", reason)
    reason = match_label_pattern(metadata.get("labels", ""), noise_patterns)
    if reason:
        return ("noise_label", reason)

    body_reason = classify_body(body, min_content_length, min_word_count)
    if body_reason:
        return (body_reason, body_reason)

    return (None, None)


def classify_body(body_text, min_content_length, min_word_count=0):
    """Classify body content. Returns reason string or None if content is fine."""
    stripped = body_text.strip()

    if not stripped:
        return "empty_stub"

    # Strip markdown headings and formatting for content analysis
    meaningful_lines = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip the issue title heading (# KEY: title)
        if re.match(r'^#\s+\S+-\d+:', line):
            continue
        # Skip Epic reference line
        if line.startswith("**Epic:**"):
            continue
        # Skip section headings
        if re.match(r'^#{1,6}\s+', line):
            continue
        meaningful_lines.append(line)

    if not meaningful_lines:
        return "empty_stub"

    actual_text = " ".join(meaningful_lines)
    if len(actual_text) < min_content_length:
        return "minimal_content"

    if min_word_count > 0:
        word_count = len(actual_text.split())
        if word_count < min_word_count:
            return "low_word_count"

    return None


def main():
    ap = argparse.ArgumentParser(description="Move empty/stub and noisy Jira .md files to .excluded/")
    ap.add_argument("--saveMd", required=True, help="Directory with .md files")
    ap.add_argument("--dryRun", action="store_true", default=False, help="Preview without moving")
    ap.add_argument("--minContentLength", type=int, default=50,
                    help="Minimum chars of actual text to keep a file (default: 50)")
    ap.add_argument("--minWordCount", type=int, default=0,
                    help="Minimum word count to keep a file (0 = disabled)")
    ap.add_argument("--maxAgeYears", type=float, default=0,
                    help="Exclude issues not updated in this many years (0 = disabled, e.g. 2)")
    ap.add_argument("--noiseConfig", default=None,
                    help="JSON file with noise patterns")
    args = ap.parse_args()

    save_md_path = args.saveMd
    excluded_path = os.path.join(save_md_path, EXCLUDED_DIR)
    min_content_length = args.minContentLength
    min_word_count = args.minWordCount
    max_age_years = args.maxAgeYears
    age_cutoff = None
    if max_age_years > 0:
        age_cutoff = datetime.now() - timedelta(days=max_age_years * 365.25)
        logging.info(f"Age cutoff: excluding issues not updated since {age_cutoff.isoformat()[:10]}")

    # Auto-detect noise config
    noise_config_path = args.noiseConfig
    if noise_config_path is None:
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jira_noise_patterns.json")
        if os.path.exists(default_config):
            noise_config_path = default_config

    noise_patterns = load_noise_patterns(noise_config_path)
    if noise_patterns:
        logging.info(f"Loaded {len(noise_patterns)} noise patterns from {noise_config_path}")

    logging.info(f"Scanning {save_md_path} for .md files (minContentLength={min_content_length}, minWordCount={min_word_count})...")

    # Load the exclude manifest once. Pass 2b removes resurrected entries
    # and Pass 3 appends new ones; we write a single time at the end.
    manifest_path = os.path.join(excluded_path, "excluded_manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest_changed = False

    new_manifest_entries = []
    category_counts = {
        "empty_stub": 0, "minimal_content": 0, "low_word_count": 0,
        "too_old": 0,
        "noise_status": 0, "noise_title": 0, "noise_label": 0, "noise_type": 0,
        "preserved_as_parent": 0,
    }
    total_scanned = 0

    # Pass 1 — classify every file and remember its tentative verdict.
    # We need a second pass to preserve issues whose key is referenced as
    # `parent` by another kept file (subtask → parent), so the knowledge
    # graph keeps its er_subtask_av edges intact.
    classifications = []  # list of (filepath, rel_path, metadata, category, reason)

    for filepath, rel_path in md_cleanup.iter_markdown_files(save_md_path):
        total_scanned += 1

        try:
            metadata, body = read_frontmatter_and_body(filepath)
        except Exception as e:
            logging.warning(f"Could not parse {filepath}: {e}")
            continue

        category, reason = compute_exclusion(
            metadata, body, age_cutoff, noise_patterns,
            min_content_length, min_word_count,
        )
        classifications.append((filepath, rel_path, metadata, category, reason))

    # Pass 2 — collect parent keys referenced by files that survive Pass 1.
    preserved_parents = set()
    for _, _, metadata, _, reason in classifications:
        if reason is None:
            parent_key = (metadata.get("parent", "") or "").strip()
            if parent_key:
                preserved_parents.add(parent_key)

    # Pass 2b — resurrect previously-excluded parents that are referenced by
    # kept subtasks. Without this, parents that failed an earlier word-count
    # check stay in .excluded/ forever, leaving subtask edges dangling.
    resurrected = 0
    if preserved_parents and manifest:
        kept_in_manifest = []
        for entry in manifest:
            key = entry.get("issue_key", "")
            original_path = entry.get("original_path", "")
            if key and original_path and key in preserved_parents:
                src_path = os.path.join(excluded_path, original_path)
                dest_path = os.path.join(save_md_path, original_path)
                if os.path.exists(src_path) and not os.path.exists(dest_path):
                    if args.dryRun:
                        logging.info(
                            f"[DRY RUN] [RESURRECT as parent-of-kept-subtask] {original_path}"
                        )
                    else:
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.move(src_path, dest_path)
                    resurrected += 1
                    continue
            kept_in_manifest.append(entry)
        if resurrected > 0:
            manifest = kept_in_manifest
            manifest_changed = True

    # Pass 3 — act on exclusions, but override for parents of kept subtasks.
    for filepath, rel_path, metadata, category, reason in classifications:
        if reason is None:
            continue

        issue_key = (metadata.get("issue_key", "") or "").strip()
        if issue_key and issue_key in preserved_parents:
            category_counts["preserved_as_parent"] += 1
            if args.dryRun:
                logging.info(
                    f"[DRY RUN] [PRESERVED as parent (would have been {category}: {reason})] {rel_path}"
                )
            continue

        category_counts[category] = category_counts.get(category, 0) + 1
        tag = f"{category}: {reason}" if category.startswith("noise_") else reason
        new_manifest_entries.append({
            "issue_key": metadata.get("issue_key", ""),
            "modifiedTime": metadata.get("modifiedTime", metadata.get("updated", "")),
            "reason": tag,
            "original_path": rel_path,
            "title": metadata.get("title", metadata.get("summary", "")),
            "status": metadata.get("status", ""),
            "project": metadata.get("project", ""),
        })

        if args.dryRun:
            logging.info(f"[DRY RUN] [{tag}] {rel_path}")
        else:
            md_cleanup.move_to_excluded(filepath, rel_path, excluded_path)

    if new_manifest_entries:
        merged = merge_manifest_entries(manifest, new_manifest_entries, "issue_key")
        if len(merged) > len(manifest):
            manifest = merged
            manifest_changed = True

    if not args.dryRun and manifest_changed:
        os.makedirs(excluded_path, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logging.info(f"Wrote manifest with {len(manifest)} entries to {manifest_path}")

    # Clean up empty directories
    if not args.dryRun:
        md_cleanup.remove_empty_dirs(save_md_path)

    action = "Would move" if args.dryRun else "Moved"
    preserved = category_counts.get("preserved_as_parent", 0)
    total_excluded = sum(v for k, v in category_counts.items() if k != "preserved_as_parent")
    parts = [f"{k}={v}" for k, v in category_counts.items() if v > 0]
    suffix_bits = []
    if preserved:
        suffix_bits.append(f"preserved {preserved} parents-of-kept-subtasks")
    if resurrected:
        verb = "would resurrect" if args.dryRun else "resurrected"
        suffix_bits.append(f"{verb} {resurrected} from .excluded/")
    suffix = f" ({'; '.join(suffix_bits)})" if suffix_bits else ""
    logging.info(
        f"Done. Scanned {total_scanned} files. "
        f"{action} {total_excluded}: {', '.join(parts) if parts else 'none'}{suffix}"
    )


if __name__ == "__main__":
    main()
