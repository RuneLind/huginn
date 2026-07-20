"""Shared filesystem mechanics for the confluence/jira/notion cleanup adapters.

Each cleanup script walks a tree of ``.md`` files, moves the ones its domain
classifier rejects into a sibling ``.excluded/`` mirror, writes an
``excluded_manifest.json``, and prunes the directories left empty. The walk, the
move, the manifest write, and the empty-dir prune are identical across all three
scripts — only the per-source classification differs. This module owns the
identical parts so a change to the move/manifest mechanics lands once.
"""
import json
import logging
import os
import shutil

from main.utils.manifest import merge_manifest_entries

EXCLUDED_DIR = ".excluded"
MANIFEST_NAME = "excluded_manifest.json"

logger = logging.getLogger(__name__)


def iter_markdown_files(save_md_path):
    """Yield ``(filepath, rel_path)`` for every ``.md`` file under save_md_path,
    skipping the ``.excluded/`` mirror."""
    for root, dirs, files in os.walk(save_md_path):
        dirs[:] = [d for d in dirs if d != EXCLUDED_DIR]
        for filename in files:
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            yield filepath, os.path.relpath(filepath, save_md_path)


def move_to_excluded(filepath, rel_path, excluded_path):
    """Move filepath into excluded_path, preserving its relative location."""
    dest = os.path.join(excluded_path, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(filepath, dest)
    return dest


def write_excluded_manifest(excluded_path, entries, id_field):
    """Merge entries into excluded_path's manifest (deduped by id_field) and write.

    Returns the manifest path written, or None when there are no entries.
    """
    if not entries:
        return None
    os.makedirs(excluded_path, exist_ok=True)
    manifest_path = os.path.join(excluded_path, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        entries = merge_manifest_entries(existing, entries, id_field)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote manifest with {len(entries)} entries to {manifest_path}")
    return manifest_path


def classify_body(
    body_text,
    min_content_length,
    line_filters=(),
    *,
    min_word_count=0,
    empty_reason="empty_stub",
    filtered_empty_reason="reference_only",
):
    """Shared body-content classifier skeleton for the cleanup adapters.

    The *machinery* — strip, drop blank/boilerplate lines, then apply the
    content thresholds — is identical across confluence/jira/notion. Only the
    per-source *policy* differs, and that policy is injected here rather than
    forked into three near-copies:

    - ``line_filters`` — an iterable of predicates ``(line) -> truthy``. Each
      surviving (non-blank, pre-stripped) line is dropped if *any* predicate
      matches. This is where a source declares what it considers noise:
      reference-only links, boilerplate headings, an issue-title heading, an
      epic pointer, child-page markers, etc.
    - ``min_content_length`` / ``min_word_count`` — the two thresholds, applied
      to the space-joined surviving lines. ``min_word_count <= 0`` disables the
      word-count gate (the confluence/jira default; now also available to
      notion).
    - ``empty_reason`` — returned when the body is blank after ``strip()``
      (all three sources use ``"empty_stub"``).
    - ``filtered_empty_reason`` — returned when every line was filtered out.
      Jira deliberately reports ``"empty_stub"`` here (an issue with nothing but
      its title/epic heading is a stub, not a reference), while confluence and
      notion report ``"reference_only"``. This divergence is preserved, not
      unified.

    Returns the reason string, or ``None`` when the body should be kept.
    """
    stripped = body_text.strip()
    if not stripped:
        return empty_reason

    meaningful_lines = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(predicate(line) for predicate in line_filters):
            continue
        meaningful_lines.append(line)

    if not meaningful_lines:
        return filtered_empty_reason

    actual_text = " ".join(meaningful_lines)
    if len(actual_text) < min_content_length:
        return "minimal_content"

    if min_word_count > 0 and len(actual_text.split()) < min_word_count:
        return "low_word_count"

    return None


def remove_empty_dirs(save_md_path):
    """Remove directories left empty after moves (skips the ``.excluded/`` mirror)."""
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
