"""
Move empty/stub and noisy Confluence markdown files to .excluded/ subfolder.

Scans .md files for YAML frontmatter, classifies body content, and moves
files that have no meaningful content. Also supports path-based noise filtering
via glob patterns to exclude entire directory subtrees (meeting notes, archives,
status reports, etc.) that add noise to vector search indexes.

Writes an excluded_manifest.json for use with --excludeManifest during re-fetch.

Usage:
    uv run confluence_cleanup_md.py --saveMd ./data/sources/my-confluence --dryRun          # preview
    uv run confluence_cleanup_md.py --saveMd ./data/sources/my-confluence                    # move files
    uv run confluence_cleanup_md.py --saveMd ./data/sources/my-confluence --minContentLength 100  # stricter
    uv run confluence_cleanup_md.py --saveMd ./data/sources/my-confluence --noiseConfig ./confluence_noise_patterns.json
    uv run confluence_cleanup_md.py --saveMd ./data/sources/my-confluence --minWordCount 30  # exclude diagram-only pages
    uv run confluence_cleanup_md.py --saveMd ./data/sources/my-confluence --sanitize         # strip WIP markers
"""

import os
import re
import json
import shutil
import argparse
import logging
from fnmatch import fnmatch

from main.utils.logger import setup_root_logger

setup_root_logger()

EXCLUDED_DIR = ".excluded"

# Patterns that don't count as real content in Confluence pages
REFERENCE_PATTERNS = [
    re.compile(r"^https?://\S+$"),                      # bare URL on its own line
    re.compile(r"^\[.+\]\(https?://\S+\)$"),            # single markdown link
    re.compile(r"^Sjekk .*:.*https?://"),                # "Sjekk Teamkatalogen: <url>"
    re.compile(r"^\*\*Spaceeier:\*\*\s*$"),              # empty "Spaceeier:" label
]

# Headings that don't count as real content (common Confluence boilerplate)
BOILERPLATE_HEADINGS = re.compile(r"^#{1,6}\s+(Aktivitet|Bidragsytere|Spaceeier)\s*$")


def load_noise_patterns(config_path):
    """Load noise patterns from a JSON config file.

    Config format: list of entries, each with a "reason" and one of:
    - {"pattern": "<glob>", "reason": "..."} — path-based (fnmatch)
    - {"title_pattern": "<substring>", "reason": "..."} — title-based (case-insensitive)
    """
    if not config_path or not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_noise_pattern(rel_path, noise_patterns):
    """Check if a relative path matches any path-based noise pattern. Returns reason or None."""
    rel_lower = rel_path.lower()
    for entry in noise_patterns:
        if "pattern" in entry and fnmatch(rel_lower, entry["pattern"].lower()):
            return entry["reason"]
    return None


def match_title_noise_pattern(title, noise_patterns):
    """Check if a title matches any title-based noise pattern. Returns reason or None."""
    if not title:
        return None
    title_lower = title.lower()
    for entry in noise_patterns:
        if "title_pattern" in entry and entry["title_pattern"].lower() in title_lower:
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
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip().strip('"')
            else:
                body_lines.append(line)

    return metadata, "".join(body_lines)


def classify_body(body_text, min_content_length, min_word_count=0):
    """Classify body content. Returns reason string or None if content is fine."""
    stripped = body_text.strip()

    if not stripped:
        return "empty_stub"

    # Filter out boilerplate headings and reference-only lines
    meaningful_lines = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if BOILERPLATE_HEADINGS.match(line):
            continue
        if any(p.match(line) for p in REFERENCE_PATTERNS):
            continue
        meaningful_lines.append(line)

    if not meaningful_lines:
        return "reference_only"

    actual_text = " ".join(meaningful_lines)
    if len(actual_text) < min_content_length:
        return "minimal_content"

    if min_word_count > 0:
        word_count = len(actual_text.split())
        if word_count < min_word_count:
            return "low_word_count"

    return None


SANITIZE_LINE_PATTERNS = [
    re.compile(r"@@"),                                          # inline TODO/mention markers
    re.compile(r"^\s*\S+\s+(følger opp|tar det videre|sjekker|undersøker)\b", re.IGNORECASE),  # personal action items
]

# Patterns that indicate a document is work-in-progress
WIP_TITLE_PATTERN = re.compile(r"(- )?under arbeid|utkast|\bWIP\b", re.IGNORECASE)
WIP_BODY_PATTERNS = [
    re.compile(r"^#{1,6}\s+(UNDER ARBEID|WIP)\b", re.IGNORECASE),  # WIP headings
    re.compile(r"^(Under arbeid|WIP)\s*$", re.IGNORECASE),          # standalone WIP line
]


def sanitize_content(body_text):
    """Strip personal action items and TODO markers from body text.

    Returns (cleaned_text, removed_count).
    """
    cleaned_lines = []
    removed = 0
    for line in body_text.splitlines(keepends=True):
        if any(p.search(line) for p in SANITIZE_LINE_PATTERNS):
            removed += 1
        else:
            cleaned_lines.append(line)
    return "".join(cleaned_lines), removed


def detect_wip(title, body_text):
    """Detect if a document is work-in-progress based on title or body markers."""
    if WIP_TITLE_PATTERN.search(title):
        return True
    for line in body_text.splitlines():
        if any(p.search(line) for p in WIP_BODY_PATTERNS):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Move empty/stub and noisy Confluence .md files to .excluded/")
    ap.add_argument("--saveMd", required=True, help="Directory with .md files")
    ap.add_argument("--dryRun", action="store_true", default=False, help="Preview without moving")
    ap.add_argument("--minContentLength", type=int, default=50,
                    help="Minimum chars of actual text to keep a file (default: 50)")
    ap.add_argument("--minWordCount", type=int, default=0,
                    help="Minimum word count to keep a file (0 = disabled)")
    ap.add_argument("--noiseConfig", default=None,
                    help="JSON file with noise patterns (list of {pattern|title_pattern, reason})")
    ap.add_argument("--sanitize", action="store_true", default=False,
                    help="Strip WIP markers (@@) and personal action items from remaining files")
    args = ap.parse_args()

    save_md_path = args.saveMd
    excluded_path = os.path.join(save_md_path, EXCLUDED_DIR)
    min_content_length = args.minContentLength
    min_word_count = args.minWordCount

    # Auto-detect noise config: look next to this script if not specified
    noise_config_path = args.noiseConfig
    if noise_config_path is None:
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confluence_noise_patterns.json")
        if os.path.exists(default_config):
            noise_config_path = default_config

    noise_patterns = load_noise_patterns(noise_config_path)
    if noise_patterns:
        logging.info(f"Loaded {len(noise_patterns)} noise patterns from {noise_config_path}")

    logging.info(f"Scanning {save_md_path} for .md files (minContentLength={min_content_length}, minWordCount={min_word_count})...")

    manifest_entries = []
    category_counts = {"empty_stub": 0, "reference_only": 0, "minimal_content": 0, "low_word_count": 0, "noise_path": 0, "noise_title": 0}
    total_scanned = 0

    for root, dirs, files in os.walk(save_md_path):
        # Skip the .excluded directory
        dirs[:] = [d for d in dirs if d != EXCLUDED_DIR]

        for filename in files:
            if not filename.endswith(".md"):
                continue

            total_scanned += 1
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, save_md_path)

            # Check path-based noise patterns first (cheap, no file read needed)
            noise_reason = match_noise_pattern(rel_path, noise_patterns)
            if noise_reason:
                category_counts["noise_path"] += 1
                try:
                    metadata, _ = parse_frontmatter_and_body(filepath)
                except Exception:
                    metadata = {}

                manifest_entries.append({
                    "page_id": metadata.get("page_id", ""),
                    "modifiedTime": metadata.get("modifiedTime", ""),
                    "reason": f"noise_path: {noise_reason}",
                    "original_path": rel_path,
                    "title": metadata.get("title", ""),
                    "space": metadata.get("space", ""),
                })

                if args.dryRun:
                    logging.info(f"[DRY RUN] [noise_path: {noise_reason}] {rel_path}")
                else:
                    dest = os.path.join(excluded_path, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.move(filepath, dest)
                continue

            # Content-based classification (also used for title noise check)
            try:
                metadata, body = parse_frontmatter_and_body(filepath)
            except Exception as e:
                logging.warning(f"Could not parse {filepath}: {e}")
                continue

            # Check title-based noise patterns
            title_noise_reason = match_title_noise_pattern(metadata.get("title", ""), noise_patterns)
            if title_noise_reason:
                category_counts["noise_title"] += 1
                manifest_entries.append({
                    "page_id": metadata.get("page_id", ""),
                    "modifiedTime": metadata.get("modifiedTime", ""),
                    "reason": f"noise_title: {title_noise_reason}",
                    "original_path": rel_path,
                    "title": metadata.get("title", ""),
                    "space": metadata.get("space", ""),
                })
                if args.dryRun:
                    logging.info(f"[DRY RUN] [noise_title: {title_noise_reason}] {rel_path}")
                else:
                    dest = os.path.join(excluded_path, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.move(filepath, dest)
                continue

            reason = classify_body(body, min_content_length, min_word_count)
            if reason is None:
                continue

            category_counts[reason] += 1

            manifest_entries.append({
                "page_id": metadata.get("page_id", ""),
                "modifiedTime": metadata.get("modifiedTime", ""),
                "reason": reason,
                "original_path": rel_path,
                "title": metadata.get("title", ""),
                "space": metadata.get("space", ""),
            })

            if args.dryRun:
                logging.info(f"[DRY RUN] [{reason}] {rel_path}")
            else:
                dest = os.path.join(excluded_path, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(filepath, dest)

    # Write manifest
    if not args.dryRun and manifest_entries:
        os.makedirs(excluded_path, exist_ok=True)
        manifest_path = os.path.join(excluded_path, "excluded_manifest.json")

        # Merge with existing manifest if present
        existing = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_ids = {e["page_id"] for e in existing}
            manifest_entries = existing + [e for e in manifest_entries if e["page_id"] not in existing_ids]

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
    logging.info(
        f"Done. Scanned {total_scanned} files. "
        f"{action} {total_excluded}: "
        f"noise_path={category_counts['noise_path']}, "
        f"noise_title={category_counts['noise_title']}, "
        f"empty_stub={category_counts['empty_stub']}, "
        f"reference_only={category_counts['reference_only']}, "
        f"minimal_content={category_counts['minimal_content']}, "
        f"low_word_count={category_counts['low_word_count']}"
    )

    # Sanitization + WIP detection pass
    if args.sanitize:
        sanitized_count = 0
        total_lines_removed = 0
        wip_flagged = 0
        for root, dirs, files in os.walk(save_md_path):
            dirs[:] = [d for d in dirs if d != EXCLUDED_DIR]
            for filename in files:
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    metadata, body = parse_frontmatter_and_body(filepath)
                except Exception:
                    continue
                cleaned_body, removed = sanitize_content(body)
                is_wip = detect_wip(metadata.get("title", ""), body)
                needs_wip_flag = is_wip and metadata.get("wip") != "true"
                if removed > 0 or needs_wip_flag:
                    if args.dryRun:
                        rel_path = os.path.relpath(filepath, save_md_path)
                        parts = []
                        if removed > 0:
                            parts.append(f"remove {removed} lines")
                        if needs_wip_flag:
                            parts.append("flag wip: true")
                        logging.info(f"[DRY RUN] [sanitize] {rel_path}: would {', '.join(parts)}")
                    else:
                        with open(filepath, "r", encoding="utf-8") as f:
                            raw = f.read()
                        # Locate frontmatter boundaries once
                        fm_end = 0
                        if raw.startswith("---"):
                            # Find closing --- on its own line (not inside values)
                            close_idx = raw.find("\n---\n", 3)
                            if close_idx == -1:
                                close_idx = raw.find("\n---", 3)  # handle EOF without trailing newline
                            if close_idx < 0:
                                # Malformed frontmatter — skip this file to avoid data loss
                                continue
                            second_sep = close_idx + 1  # skip the \n, point at ---
                            # Inject wip: true before the closing --- if needed
                            if needs_wip_flag:
                                raw = raw[:second_sep] + "wip: true\n" + raw[second_sep:]
                                second_sep += len("wip: true\n")
                            fm_end = second_sep + 3
                            if fm_end < len(raw) and raw[fm_end] == "\n":
                                fm_end += 1
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(raw[:fm_end] + cleaned_body)
                    sanitized_count += 1
                    total_lines_removed += removed
                    if needs_wip_flag:
                        wip_flagged += 1

        action_s = "Would sanitize" if args.dryRun else "Sanitized"
        logging.info(f"{action_s} {sanitized_count} files ({total_lines_removed} lines removed, {wip_flagged} flagged as WIP)")


if __name__ == "__main__":
    main()
