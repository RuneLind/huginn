#!/usr/bin/env python3
"""CLI over the LOCAL sensitivity sweep (``main/privacy/sensitivity_sweep.py``).

    .venv/bin/python scripts/audit/sensitivity_sweep.py --collection nav-wiki
    .venv/bin/python scripts/audit/sensitivity_sweep.py --collection nav-wiki --baseline
    .venv/bin/python scripts/audit/sensitivity_sweep.py --collection nav-wiki --baseline --limit 50

The library module carries the reasoning: what the sweep is for, why the
transport is local-only, how a reference is classified, and what the packaging
gate does with the report. This file is argparse and ``main()``.

Outputs:

* a **gitignored** JSON report (real strings, for a human to triage) at
  ``<private-sub-repo>/privacy/sweep_<collection>_<date>_<mode>[-limitN].json``,
  falling back to ``data/privacy/``;
* a stdout summary of counts and SHAPES only, safe to paste;
* a run record under the ledger collection key ``sensitivity-audit``.

Exit codes: ``0`` clean, ``2`` when an ``unknown_person`` survived the filters,
``1`` when the model could not be reached or the collection / map is missing.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import (  # noqa: E402
    POLICY_VERSION, AliasRegistry, PrivacyMapMissing, discover_map_path,
)
from main.privacy.index_scan import load_allowed_bigrams  # noqa: E402
from main.privacy.sensitivity_sweep import (  # noqa: E402
    LEDGER_COLLECTION, MAPPED_RESIDUAL, ROLE, UNKNOWN_PERSON,
    ReferenceClassifier, allowlist_sha256, answers_are_readable, cache_path_for,
    ledger_record, load_cache, ollama_reachable, report_path, report_run, run_sweep,
    summary_lines, write_cache,
)
from main.runtime.indexing_run_ledger import now_iso  # noqa: E402
from main.utils.ollama_cli import DEFAULT_MODEL, call_ollama  # noqa: E402
from scripts.audit.scan_index import (  # noqa: E402
    DEFAULT_COLLECTIONS_DIR, refuse_if_tracked, resolve_allowlist,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--collection", required=True)
    ap.add_argument("--collections-dir", default=str(DEFAULT_COLLECTIONS_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Local Ollama model (default: {DEFAULT_MODEL}). There is no hosted "
                         f"backend here and there must not be one.")
    ap.add_argument("--baseline", action="store_true",
                    help="Ask about every document, ignoring the cache. The default mode asks "
                         "only about documents whose text hash changed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Bound the run to the first N documents by id (0 = all). Allowed in "
                         "either mode; the report is then marked limited, and a limited report "
                         "never certifies a hand-off.")
    ap.add_argument("--map", default=None, help="Alias map path (default: the discovered one)")
    ap.add_argument("--allowed-bigrams", default=None,
                    help="Reviewed non-person bigram allow-list (default: the discovered one)")
    ap.add_argument("--report-out", default=None,
                    help="Where the report goes. THIS FILE CONTAINS REAL TEXT; any path is "
                         "refused unless git already ignores it.")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-cache", action="store_true", help="Neither read nor write the cache")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8321"))
    ap.add_argument("--job", default=None, help="launchd job label, for the run record")
    ap.add_argument("--trigger", default="cli", help="scheduled/manual/cli")
    ap.add_argument("--no-ledger", action="store_true", help="Do not record a run")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    collection_dir = Path(args.collections_dir).resolve() / args.collection
    if not (collection_dir / "manifest.json").exists():
        print(f"REFUSED: no such collection: {collection_dir}", file=sys.stderr)
        return 1

    manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        map_path = discover_map_path(args.map)
        registry = AliasRegistry.load(map_path)
    except (PrivacyMapMissing, OSError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    alias_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
    allowlist_path = resolve_allowlist(args.allowed_bigrams)
    classifier = ReferenceClassifier(registry, alias_map,
                                     load_allowed_bigrams(allowlist_path))
    allowlist_sha = allowlist_sha256(allowlist_path)

    mode = "baseline" if args.baseline else "incremental"
    # BOTH paths through the guard, not just the explicit one. The default lands
    # in a private sub-repo, and whether that sub-repo ignores its own
    # `privacy/` is a fact about that repo — not something this script may
    # assume on the strength of the directory's name.
    destination = refuse_if_tracked(
        args.report_out or report_path(args.collection, mode=mode, limit=args.limit))
    destination.parent.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path_for(args.collection, args.cache_dir)
    cache = ({} if args.no_cache
             else load_cache(cache_file, args.model, registry.map_version, allowlist_sha))

    started_at = now_iso()
    started = time.time()

    def record(status, detail, error=None):
        if args.no_ledger:
            return None
        return report_run(
            ledger_record(status, started_at=started_at, finished_at=now_iso(),
                          duration=int(time.time() - started), detail=detail,
                          collection=args.collection, baseline=args.baseline,
                          job=args.job, trigger=args.trigger, error=error),
            args.api_url)

    if not ollama_reachable():
        message = "Ollama is not reachable at localhost:11434 — nothing was judged"
        print(f"FAILED: {message}", file=sys.stderr)
        record("failed", {"model": args.model, "mode": mode}, error=message)
        return 1

    def progress(index, total, doc_id, findings):
        unknown = sum(1 for f in findings if f["classification"] == UNKNOWN_PERSON)
        print(f"  [{index}/{total}] {len(findings)} refs, {unknown} unknown", flush=True)

    # `call=call_ollama` explicitly rather than relying on the default argument:
    # a default is bound at def time, so a test monkeypatching the module symbol
    # would drive the real transport. Passing it here resolves the global on
    # every call, which is what makes the CLI path testable at all.
    try:
        result = run_sweep(collection_dir, classifier, model=args.model,
                           baseline=args.baseline, limit=args.limit, cache=cache,
                           timeout=args.timeout, call=call_ollama, progress=progress,
                           map_version=registry.map_version, allowlist_sha=allowlist_sha)
    except Exception as e:                                          # noqa: BLE001
        # A crash mid-run used to leave NO ledger record at all, which reads on
        # the dashboard exactly like a night the sweep was never scheduled. The
        # record is the only durable trace an unattended run leaves.
        message = f"{type(e).__name__}: {e}"
        print(f"FAILED: the sweep crashed — {message}", file=sys.stderr)
        record("failed", {"model": args.model, "mode": mode}, error=message)
        raise

    if not args.no_cache:
        write_cache(cache_file, args.model, result.pop("entries"),
                    registry.map_version, allowlist_sha)
    else:
        result.pop("entries", None)

    generated_at = now_iso()
    report = {
        "collection": args.collection,
        "generatedAt": generated_at,
        "model": args.model,
        "policyVersion": POLICY_VERSION,
        "mapVersion": registry.map_version,
        "mode": mode,
        "limit": args.limit or None,
        "documentsExpected": manifest.get("numberOfDocuments"),
        "collectionUpdatedTime": manifest.get("updatedTime"),
        "collectionLastModifiedDocumentTime": manifest.get("lastModifiedDocumentTime"),
        **{k: v for k, v in result.items() if k != "findings"},
        "findings": result["findings"],
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ncollection: {args.collection}  model: {args.model}  "
          f"policy v{POLICY_VERSION}  map v{registry.map_version}\n")
    for line in summary_lines(result):
        print(line)
    print(f"\nreport: {destination}")

    unknown = result["unknownCount"]
    readable = answers_are_readable(result["windows"], result["parseFailures"])
    status = "degraded" if (unknown or not readable) else "succeeded"
    where = record(status, {"model": args.model, "mode": mode,
                            "documents": result["documents"],
                            "documentsExpected": manifest.get("numberOfDocuments"),
                            "documentsAsked": result["documentsAsked"],
                            "limit": args.limit or None,
                            "unknown": unknown,
                            "mappedResidual": result["counts"][MAPPED_RESIDUAL],
                            "role": result["counts"][ROLE],
                            "parseFailures": result["parseFailures"]})
    if where:
        print(f"run recorded under {LEDGER_COLLECTION} via {where}")

    if unknown:
        print(f"\nRESULT: {unknown} unknown person reference(s) — triage the report before "
              f"packaging this collection.")
        return 2
    if not readable:
        # Exit 0 still: the documented codes are 0/2/1 and the nightly phase is
        # non-fatal. The degradation travels where it is acted on — the ledger
        # row and the packaging gate, which both decline to call this clean.
        print(f"\nRESULT: INCONCLUSIVE — {result['parseFailures']} of {result['windows']} "
              f"model answers were unreadable, so 'no unknown persons' is not evidence. "
              f"Re-run before relying on this report.")
        return 0
    print("\nRESULT: clean — the local model found no person reference the map does not "
          "already account for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
