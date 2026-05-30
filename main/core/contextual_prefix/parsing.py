"""Shared JSON-array parsing for contextual-prefix backends.

All three real backends (Ollama, Claude Code CLI, Anthropic SDK) ask the model to
return a JSON array of one prefix string per chunk. Models occasionally wrap the
array in markdown fences, trail a comma, or return an object with the array under
a key. `parse_prefix_array` tolerates those variants and falls back to logging +
empty-list rather than raising, so a single doc's parse failure can't take down
the whole indexing run.
"""

import json
import logging
import os
import re
import time


logger = logging.getLogger(__name__)


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])\s*\Z")

# Dumping raw model output to disk writes confidential document content (the
# contextual-prefix prompt embeds the chunk text), so it is OFF by default and
# must be opted into via CONTEXTUAL_PREFIX_DEBUG_DUMP. When enabled, the dump
# directory is capped to the most-recent N files so it can't grow without bound.
DEFAULT_DUMP_DIR = "./data/contextual_caches/parse_failures"
DEFAULT_DUMP_MAX_FILES = 20
DUMP_FILE_PREFIX = "parse-fail-"


def parse_prefix_array(raw: str, expected_count: int) -> list[str]:
    if not raw:
        return []

    cleaned = JSON_FENCE_RE.sub("", raw).strip()
    cleaned = TRAILING_COMMA_RE.sub(r"\1", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        dump_path = _dump_parse_failure(cleaned, exc)
        logger.warning(
            "Could not parse JSON from model output (len=%d, expected_count=%d, err=%s).\n"
            "  first 400: %s\n"
            "  last  400: %s\n"
            "  raw saved to: %s",
            len(cleaned), expected_count, exc,
            cleaned[:400].replace("\n", "\\n"),
            cleaned[-400:].replace("\n", "\\n"),
            dump_path,
        )
        return []

    if isinstance(parsed, dict):
        for key in ("prefixes", "results", "chunks"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        logger.warning("Expected JSON array of prefixes, got %s", type(parsed).__name__)
        return []

    prefixes = [str(p).strip() for p in parsed]

    if len(prefixes) != expected_count:
        logger.warning("Got %d prefixes, expected %d", len(prefixes), expected_count)

    return prefixes


def _dump_enabled() -> bool:
    return os.environ.get("CONTEXTUAL_PREFIX_DEBUG_DUMP", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _dump_max_files() -> int:
    raw = os.environ.get("CONTEXTUAL_PREFIX_DEBUG_MAX_FILES")
    if raw is None:
        return DEFAULT_DUMP_MAX_FILES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DUMP_MAX_FILES


def _prune_old_dumps(dump_dir: str, keep: int) -> None:
    """Remove oldest dumps so that writing one more leaves the dir at `keep` files.

    File names embed a millisecond timestamp, so a lexical sort is chronological.
    """
    try:
        existing = sorted(
            f for f in os.listdir(dump_dir)
            if f.startswith(DUMP_FILE_PREFIX) and f.endswith(".txt")
        )
    except OSError:
        return
    excess = len(existing) - (keep - 1)
    for name in existing[:max(0, excess)]:
        try:
            os.remove(os.path.join(dump_dir, name))
        except OSError:
            pass


def _dump_parse_failure(raw: str, exc: Exception) -> str:
    if not _dump_enabled():
        return "(disabled; set CONTEXTUAL_PREFIX_DEBUG_DUMP=1 to capture raw output)"

    keep = _dump_max_files()
    if keep <= 0:
        return "(disabled; CONTEXTUAL_PREFIX_DEBUG_MAX_FILES<=0)"

    dump_dir = os.environ.get("CONTEXTUAL_PREFIX_DEBUG_DIR", DEFAULT_DUMP_DIR)
    try:
        os.makedirs(dump_dir, exist_ok=True)
        _prune_old_dumps(dump_dir, keep)
        path = os.path.join(dump_dir, f"{DUMP_FILE_PREFIX}{int(time.time() * 1000)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# JSONDecodeError: {exc}\n")
            f.write(f"# raw output length: {len(raw)}\n")
            f.write("# -----\n")
            f.write(raw)
        return path
    except OSError as e:
        logger.warning("Failed to dump parse failure: %s", e)
        return "(dump failed)"
