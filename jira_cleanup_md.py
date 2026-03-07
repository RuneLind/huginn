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

from main.utils.logger import setup_root_logger

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


def parse_frontmatter_and_body(filepath):
    """Parse YAML frontmatter and return (metadata_dict, body_text)."""
    metadata = {}
    body_lines = []
    in_frontmatter = False
    frontmatter_ended = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not in_frontmatter and not frontmatter_ended and line.strip() == "---":
                in_frontmatter = True
                continue
            if in_frontmatter:
                if line.strip() == "---":
                    in_frontmatter = False
                    frontmatter_ended = True
                    continue
                stripped = line.strip()
                # List item (e.g. "  - frontend") — no colon required
                if stripped.startswith("- "):
                    item = stripped[2:].strip()
                    if item:
                        existing = metadata.get("labels", "")
                        metadata["labels"] = (existing + "," + item) if existing else item
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"')
                    # Key with no value — likely start of a YAML list (e.g. "labels:")
                    if key and not value:
                        continue
                    metadata[key] = value
            else:
                body_lines.append(line)

    return metadata, "".join(body_lines)


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

    manifest_entries = []
    category_counts = {
        "empty_stub": 0, "minimal_content": 0, "low_word_count": 0,
        "too_old": 0,
        "noise_status": 0, "noise_title": 0, "noise_label": 0, "noise_type": 0,
    }
    total_scanned = 0

    for root, dirs, files in os.walk(save_md_path):
        dirs[:] = [d for d in dirs if d != EXCLUDED_DIR]

        for filename in files:
            if not filename.endswith(".md"):
                continue

            total_scanned += 1
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, save_md_path)

            try:
                metadata, body = parse_frontmatter_and_body(filepath)
            except Exception as e:
                logging.warning(f"Could not parse {filepath}: {e}")
                continue

            # Check age cutoff (updated/modifiedTime too old)
            if age_cutoff:
                updated_str = metadata.get("modifiedTime", metadata.get("updated", ""))
                if updated_str:
                    try:
                        updated_dt = datetime.fromisoformat(updated_str)
                        # Strip timezone for comparison
                        updated_dt = updated_dt.replace(tzinfo=None)
                        if updated_dt < age_cutoff:
                            reason = f"last updated {updated_str[:10]}"
                            category = "too_old"
                            category_counts[category] += 1
                            manifest_entries.append({
                                "issue_key": metadata.get("issue_key", ""),
                                "modifiedTime": metadata.get("modifiedTime", metadata.get("updated", "")),
                                "reason": f"too_old: {reason}",
                                "original_path": rel_path,
                                "title": metadata.get("title", metadata.get("summary", "")),
                                "status": metadata.get("status", ""),
                                "project": metadata.get("project", ""),
                            })
                            if args.dryRun:
                                logging.info(f"[DRY RUN] [too_old: {reason}] {rel_path}")
                            else:
                                dest = os.path.join(excluded_path, rel_path)
                                os.makedirs(os.path.dirname(dest), exist_ok=True)
                                shutil.move(filepath, dest)
                            continue
                    except (ValueError, TypeError):
                        pass

            # Check status-based noise
            reason = match_status_pattern(metadata.get("status", ""), noise_patterns)
            if reason:
                category = "noise_status"
            else:
                # Check type-based noise
                reason = match_type_pattern(metadata.get("issue_type", ""), noise_patterns)
                if reason:
                    category = "noise_type"
                else:
                    # Check title-based noise
                    reason = match_title_pattern(metadata.get("title", ""), noise_patterns)
                    if reason:
                        category = "noise_title"
                    else:
                        # Check label-based noise
                        reason = match_label_pattern(metadata.get("labels", ""), noise_patterns)
                        if reason:
                            category = "noise_label"
                        else:
                            # Content-based classification
                            reason = classify_body(body, min_content_length, min_word_count)
                            category = reason

            if reason is None:
                continue

            category_counts[category] = category_counts.get(category, 0) + 1

            manifest_entries.append({
                "issue_key": metadata.get("issue_key", ""),
                "modifiedTime": metadata.get("modifiedTime", metadata.get("updated", "")),
                "reason": f"{category}: {reason}" if category.startswith("noise_") else reason,
                "original_path": rel_path,
                "title": metadata.get("title", metadata.get("summary", "")),
                "status": metadata.get("status", ""),
                "project": metadata.get("project", ""),
            })

            if args.dryRun:
                tag = f"{category}: {reason}" if category.startswith("noise_") else reason
                logging.info(f"[DRY RUN] [{tag}] {rel_path}")
            else:
                dest = os.path.join(excluded_path, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(filepath, dest)

    # Write manifest
    if not args.dryRun and manifest_entries:
        os.makedirs(excluded_path, exist_ok=True)
        manifest_path = os.path.join(excluded_path, "excluded_manifest.json")

        # Merge with existing manifest
        existing = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_keys = {e["issue_key"] for e in existing}
            manifest_entries = existing + [e for e in manifest_entries if e["issue_key"] not in existing_keys]

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2, ensure_ascii=False)
        logging.info(f"Wrote manifest with {len(manifest_entries)} entries to {manifest_path}")

    # Clean up empty directories
    if not args.dryRun:
        for root, dirs, files in os.walk(save_md_path, topdown=False):
            if EXCLUDED_DIR in root.split(os.sep):
                continue
            for d in dirs:
                if d == EXCLUDED_DIR:
                    continue
                dir_path = os.path.join(root, d)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except Exception:
                    pass

    action = "Would move" if args.dryRun else "Moved"
    total_excluded = sum(category_counts.values())
    parts = [f"{k}={v}" for k, v in category_counts.items() if v > 0]
    logging.info(
        f"Done. Scanned {total_scanned} files. "
        f"{action} {total_excluded}: {', '.join(parts) if parts else 'none'}"
    )


if __name__ == "__main__":
    main()
