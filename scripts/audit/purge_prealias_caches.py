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

    .venv/bin/python scripts/audit/purge_prealias_caches.py --dry-run
    .venv/bin/python scripts/audit/purge_prealias_caches.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import PrivacyMapMissing, load_scope, resolve_registry  # noqa: E402

COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"
CONTEXTUAL_CACHES = REPO_ROOT / "data" / "contextual_caches"
GRAPH_CACHE_GLOBS = ["huginn-*/scripts/knowledge_graph/*_llm_graph.cache.json",
                     "scripts/knowledge_graph/*_llm_graph.cache.json"]


def in_scope_collections() -> set:
    """Collections the BUILD would alias — asked of resolve_registry, not reimplemented.

    A second copy of the scope rules is a second thing to keep in sync, and the
    consequence of drift here is a pre-alias cache left in place for a
    collection that is in fact aliased.
    """
    named, _ = load_scope()
    candidates = named | {p.parent.name for p in COLLECTIONS_DIR.glob("*/manifest.json")}
    collections = set()
    for name in sorted(candidates):
        manifest_path = COLLECTIONS_DIR / name / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        try:
            armed = resolve_registry(name, (manifest.get("reader") or {}).get("basePath"),
                                     armed_by_manifest=bool(manifest.get("privacy"))) is not None
        except PrivacyMapMissing:
            armed = True          # in scope; the map being unloadable does not unscope it
        if armed:
            collections.add(name)
    return collections


def actively_prefixed(collection: str) -> bool:
    """True only for a collection that is BOTH prefixed and still rebuilt.

    A `supersededBy` collection is never reindexed again (nav-wiki-v2 says so in
    its own manifest note), so its prefix cache can never be invalidated by a
    rebuild — it is exactly as dormant as a collection with no prefix block.
    """
    manifest_path = COLLECTIONS_DIR / collection / "manifest.json"
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

    # Relative basePaths in a manifest resolve against the process CWD, the same
    # way FilesDocumentReader and resolve_registry read them; launchd and a
    # caller in another directory would otherwise get a different scope answer.
    os.chdir(REPO_ROOT)

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
