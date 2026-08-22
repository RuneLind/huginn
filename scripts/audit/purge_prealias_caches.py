#!/usr/bin/env python3
"""Retire derived caches built from PRE-alias document text.

Two caches outlive a rebuild and would otherwise replay real names back into
freshly aliased artifacts:

* the LLM knowledge-graph extraction cache (`*_llm_graph.cache.json`), keyed by
  doc_id alone. The extractor now mixes the privacy policy version into the file
  so a stale cache misses on its own, but the file still *contains* extracted
  real names, so it is deleted rather than left lying around.
* dormant contextual-prefix caches (`data/contextual_caches/<collection>.json`)
  for in-scope collections whose manifest has no `contextualPrefix` block. They
  are not consulted today, but re-enabling prefixing later would replay
  pre-alias prefixes. They are renamed to `.pre-alias.bak`, not deleted: the
  entries are expensive to regenerate and nothing reads a `.bak`.

A cache whose collection IS actively prefixed (melosys-confluence-v3) is left
alone on purpose. The rebuild invalidates exactly the documents the aliasing
changed, per document — wiping the file would re-prefix every chunk through the
LLM instead of the handful that actually moved.

    scripts/audit/purge_prealias_caches.py --dry-run
    scripts/audit/purge_prealias_caches.py
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import load_scope  # noqa: E402

CONTEXTUAL_CACHES = REPO_ROOT / "data" / "contextual_caches"
GRAPH_CACHE_GLOBS = ["huginn-*/scripts/knowledge_graph/*_llm_graph.cache.json",
                     "scripts/knowledge_graph/*_llm_graph.cache.json"]


def in_scope_collections() -> set:
    """Collections in privacy scope: named in a scope file, or sharing a scoped basePath."""
    named, base_paths = load_scope()
    collections = set(named)
    for manifest_path in (REPO_ROOT / "data" / "collections").glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base_path = manifest.get("reader", {}).get("basePath")
        if not base_path:
            continue
        resolved = Path(base_path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / base_path
        if str(resolved.resolve()) in base_paths:
            collections.add(manifest_path.parent.name)
    return collections


def actively_prefixed(collection: str) -> bool:
    """True only for a collection that is BOTH prefixed and still rebuilt.

    A `supersededBy` collection is never reindexed again (nav-wiki-v2 says so in
    its own manifest note), so its prefix cache can never be invalidated by a
    rebuild — it is exactly as dormant as a collection with no prefix block.
    """
    manifest_path = REPO_ROOT / "data" / "collections" / collection / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("supersededBy"):
        return False
    return bool(manifest.get("contextualPrefix"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    collections = in_scope_collections()
    print(f"in-scope collections ({len(collections)}): {', '.join(sorted(collections))}\n")

    for pattern in GRAPH_CACHE_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            collection = path.name[: -len("_llm_graph.cache.json")]
            if collection not in collections:
                continue
            size = path.stat().st_size
            print(f"delete graph cache {path.relative_to(REPO_ROOT)} ({size} bytes)")
            if not args.dry_run:
                path.unlink()

    for path in sorted(CONTEXTUAL_CACHES.glob("*.json")):
        collection = path.stem
        if collection not in collections:
            continue
        if actively_prefixed(collection):
            print(f"keep contextual cache {path.relative_to(REPO_ROOT)} "
                  f"({collection} is actively prefixed; the rebuild invalidates per document)")
            continue
        target = path.with_suffix(".json.pre-alias.bak")
        print(f"rename contextual cache {path.relative_to(REPO_ROOT)} -> {target.name}")
        if not args.dry_run:
            path.rename(target)


if __name__ == "__main__":
    main()
