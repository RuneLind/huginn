#!/usr/bin/env python3
"""CLI over the distribution gate (``main/privacy/index_scan.py``).

    .venv/bin/python scripts/audit/scan_index.py --collection nav-wiki
    .venv/bin/python scripts/audit/scan_index.py --collection jira-issues-aliased \
        --compare jira-issues            # adds the exempt-label / ident-exception invariant
    .venv/bin/python scripts/audit/scan_index.py --collection nav-wiki \
        --collections-dir /tmp/unpacked  # scan an untarred package
    .venv/bin/python scripts/audit/scan_index.py --collection nav-wiki \
        --json-report report.json --candidates-out <private-sub-repo>/privacy/candidates.json

This file used to be ``verify_aliased_collection.py``; it was renamed when the
checks were promoted into ``main/privacy/index_scan.py`` so
``scripts/audit/package_collection.py`` could call them as a library. The flags
are unchanged and ``--compare`` still means the same thing.

``--candidates-out`` is the ONE output that contains real text: the capitalised
pairs check 9 could not clear, for a human to triage. Write it inside a
gitignored private sub-repo. Everything printed to stdout, and everything in
``--json-report``, is a shape or a count.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.index_scan import scan_collection  # noqa: E402

DEFAULT_COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"
MAP_GLOB = "huginn-*/privacy/aliases.json"
EXCEPTIONS_GLOB = "huginn-*/privacy/ident_exceptions.json"
BIGRAM_ALLOWLIST_GLOB = "huginn-*/privacy/non_person_bigrams.json"


# Each of these reads REPO_ROOT at CALL time rather than binding it as a default,
# so a test can point the whole discovery layer at a tmp dir with one monkeypatch.

def resolve_map(explicit: str | None, repo_root: Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    map_paths = sorted((repo_root or REPO_ROOT).glob(MAP_GLOB))
    if len(map_paths) != 1:
        sys.exit(f"Expected exactly one alias map ({MAP_GLOB}), found {len(map_paths)}")
    return map_paths[0]


def load_exceptions(repo_root: Path | None = None) -> set:
    exceptions = set()
    for path in sorted((repo_root or REPO_ROOT).glob(EXCEPTIONS_GLOB)):
        exceptions |= {t.lower() for t in json.loads(path.read_text(encoding="utf-8"))["tokens"]}
    return exceptions


def resolve_allowlist(explicit: str | None, repo_root: Path | None = None):
    if explicit:
        return Path(explicit)
    found = sorted((repo_root or REPO_ROOT).glob(BIGRAM_ALLOWLIST_GLOB))
    return found[0] if len(found) == 1 else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--compare", default=None,
                    help="Pre-alias collection, for the exemption invariant and the BM25 control")
    ap.add_argument("--collections-dir", default=str(DEFAULT_COLLECTIONS_DIR),
                    help="Scan a staged copy or an untarred package instead of data/collections")
    ap.add_argument("--map", default=None, help="Alias map path (default: the discovered one)")
    ap.add_argument("--allow-count-drift", action="store_true",
                    help="Downgrade a numberOfDocuments/numberOfChunks difference against "
                         "--compare to a warning. For a source tree that has grown since the "
                         "live collection was last built.")
    ap.add_argument("--allowed-bigrams", default=None,
                    help=f"Reviewed non-person bigram allow-list (default: the discovered "
                         f"{BIGRAM_ALLOWLIST_GLOB})")
    ap.add_argument("--json-report", default=None,
                    help="Write the machine-readable report (shapes and counts only) here")
    ap.add_argument("--candidates-out", default=None,
                    help="Write the unreviewed capitalised pairs here for triage. THIS FILE "
                         "CONTAINS REAL TEXT — put it in a gitignored private sub-repo.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    collections_dir = Path(args.collections_dir).resolve()
    map_path = resolve_map(args.map)

    report = scan_collection(
        collections_dir / args.collection,
        map_path,
        compare_dir=(collections_dir / args.compare) if args.compare else None,
        exceptions=load_exceptions(),
        allow_count_drift=args.allow_count_drift,
        allowed_bigrams_path=resolve_allowlist(args.allowed_bigrams),
    )

    print(f"collection: {report.collection}  map v{report.map_version}  "
          f"policy v{report.policy_version}  entries: {report.map_entries}\n")
    for line in report.format_lines():
        print(line)

    if args.json_report:
        Path(args.json_report).write_text(
            json.dumps(report.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    if args.candidates_out:
        Path(args.candidates_out).write_text(
            json.dumps({"collection": report.collection,
                        "candidates": report.candidates}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\ncandidates written to {args.candidates_out} ({len(report.candidates)})")

    # The PASS line states what it does and does NOT certify: an unqualified
    # "PASS" would read as "no real name survives", which is not what was checked.
    print("\nRESULT: PASS — no listed variant / ident / handle, no unreviewed "
          "capitalised pair, no fingerprint; bare given names standing alone are out "
          "of scope (map `bare_given_name_residual`)" if report.passed else "\nRESULT: FAIL")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
