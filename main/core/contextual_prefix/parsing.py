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
PARSE_FAILURE_DUMP_DIR = os.environ.get("CONTEXTUAL_PREFIX_DEBUG_DIR", "./data/contextual_caches/parse_failures")


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


def _dump_parse_failure(raw: str, exc: Exception) -> str:
    try:
        os.makedirs(PARSE_FAILURE_DUMP_DIR, exist_ok=True)
        path = os.path.join(PARSE_FAILURE_DUMP_DIR, f"parse-fail-{int(time.time() * 1000)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# JSONDecodeError: {exc}\n")
            f.write(f"# raw output length: {len(raw)}\n")
            f.write("# -----\n")
            f.write(raw)
        return path
    except OSError as e:
        logger.warning("Failed to dump parse failure: %s", e)
        return "(dump failed)"
