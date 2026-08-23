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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy import DEFAULT_PRIVACY_DIR  # noqa: E402
from main.privacy.alias_registry import (  # noqa: E402
    PrivacyMapMissing, _load_ident_exceptions, discover_map_path,
)
from main.privacy.index_scan import scan_collection  # noqa: E402

DEFAULT_COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"
BIGRAM_ALLOWLIST_GLOB = "huginn-*/privacy/non_person_bigrams.json"

# The map and the ident exceptions are discovered by ``alias_registry`` itself,
# not by a second copy of the globs here. The gate has to verify with the map the
# BUILD substituted with; two independent resolvers is how it ends up verifying
# with a different one. `alias_registry.REPO_ROOT` is what a test repoints.
DEFAULT_CANDIDATES_DIR = DEFAULT_PRIVACY_DIR


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
    ap.add_argument("--candidates-out", nargs="?", const="", default=None,
                    help="Write the unreviewed capitalised pairs for triage. THIS FILE "
                         "CONTAINS REAL TEXT. With no value it goes to "
                         "data/privacy/scan_candidates_<collection>.json; an explicit path "
                         "is refused unless git already ignores it.")
    return ap


def enclosing_repo(path: Path):
    """The nearest ancestor that is a git checkout, or None.

    ``.git`` as a FILE counts — that is a worktree or a submodule, and both are
    repositories with their own ignore rules.
    """
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def refuse_if_tracked(explicit) -> Path:
    """Resolve an output path, or exit if git would track it.

    Shared with ``scripts/audit/sensitivity_sweep.py``, which writes the other
    output that contains real names. One copy, because a second implementation of
    a guard like this drifts in exactly the direction that costs nothing to write
    and everything to discover.

    Absolute FIRST. `check-ignore` used to run with `cwd=REPO_ROOT` while the
    write resolves against the process CWD, so a relative path had the two steps
    answering about two different files: `--candidates-out out.json` from inside
    `data/` was refused because `<repo>/out.json` is tracked territory, and the
    mirror image — ignored at the root, tracked where the caller stands — would
    have been allowed and then written into a tracked directory.

    And the question is asked INSIDE the repository that owns the path. The
    default report and candidate directories live in private `huginn-*/`
    sub-repos, which are separate checkouts: run from this repo, `check-ignore`
    consults THIS repo's rules (`.git/info/exclude`, `core.excludesFile`, the
    root `.gitignore`) about a file another repository will actually commit or
    not. A sub-repo that ignores its own `privacy/` via `.git/info/exclude` was
    reported as tracked and the write refused; the mirror image — a path this
    repo ignores because `huginn-*/` is ignored here, while the sub-repo tracks
    it — would have been allowed, and that one ships real names.

    A path with no enclosing checkout is allowed: nothing can track it.
    """
    path = Path(explicit).expanduser().resolve()
    repo = enclosing_repo(path)
    if repo is None:
        return path
    ignored = subprocess.run(["git", "check-ignore", "-q", str(path)],
                             cwd=repo, capture_output=True)
    # 0 = ignored, 1 = not ignored, 128 = not a git checkout / outside the repo.
    if ignored.returncode == 1:
        sys.exit(f"REFUSED: {path} is not gitignored in {repo}, and it is an output that "
                 f"contains real names. Write it under data/ or inside a private "
                 f"huginn-*/ sub-repo that ignores it.")
    return path


def candidates_destination(explicit: str | None, collection: str) -> Path:
    """Where the capitalised-pair candidate list may be written.

    Default: ``data/privacy/`` — `data/` is gitignored wholesale. EVERY path goes
    through `refuse_if_tracked`, the default included: "gitignored wholesale" is
    a property of a `.gitignore` this script does not own, and checking only the
    explicit path means the one route nobody thinks about is the unchecked one.
    The campaign has already had to rewrite history once over a real name in this
    repo; a `--candidates-out report.json` typo is the same mistake with a
    shorter fuse.
    """
    if not explicit:
        DEFAULT_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        return refuse_if_tracked(
            DEFAULT_CANDIDATES_DIR / f"scan_candidates_{collection}.json")
    return refuse_if_tracked(explicit)


def main() -> None:
    args = build_parser().parse_args()
    collections_dir = Path(args.collections_dir).resolve()
    candidates_out = (candidates_destination(args.candidates_out, args.collection)
                      if args.candidates_out is not None else None)
    try:
        map_path = discover_map_path(args.map)
    except PrivacyMapMissing as e:
        sys.exit(f"REFUSED: {e}")

    report = scan_collection(
        collections_dir / args.collection,
        map_path,
        compare_dir=(collections_dir / args.compare) if args.compare else None,
        exceptions=_load_ident_exceptions(),
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
    if candidates_out is not None:
        candidates_out.write_text(
            json.dumps({"collection": report.collection,
                        "candidates": report.candidates}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\ncandidates written to {candidates_out} ({len(report.candidates)})")

    # The PASS line states what it does and does NOT certify: an unqualified
    # "PASS" would read as "no real name survives", which is not what was checked.
    print("\nRESULT: PASS — no listed variant / ident / handle, no unreviewed "
          "capitalised pair, no fingerprint; bare given names standing alone are out "
          "of scope (map `bare_given_name_residual`)" if report.passed else "\nRESULT: FAIL")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
