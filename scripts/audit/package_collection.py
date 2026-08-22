#!/usr/bin/env python3
"""Produce the hand-off tarball for an aliased collection. THE ONLY WAY TO SHIP.

    .venv/bin/python scripts/audit/package_collection.py --collection nav-wiki
    .venv/bin/python scripts/audit/package_collection.py --collection nav-wiki --out /tmp/pkg

Copying `data/collections/<name>/` by hand — scp, rsync, a zip, AirDrop — is not
a supported hand-off and never was. The gate has to *produce* the artifact,
because a scanner someone must remember to run before copying is advisory, and
the one time it is skipped is the time it mattered. This script runs
``main.privacy.index_scan.scan_collection`` and writes the tarball **only** when
every check passes; on a failure it prints the report and writes nothing at all.

What the tarball contains, and nothing else:

    PACKAGE-STAMP.json                 <- what was scanned, and with which map
    data/collections/<name>/…          <- the collection, at the path it unpacks to

Not the alias map (the reverse map never leaves this machine), not
``data/prealias/`` (that is the collection WITH the real names in it), not the
contextual or graph caches (they carry pre-alias text until
``purge_prealias_caches.py`` retires them). The tarball is built from the one
collection directory, so none of those can be swept in by accident.

Two refusals before the scan even runs — both about packaging the wrong thing
rather than about what is inside it:

* the collection must be in privacy scope (``resolve_registry`` returns a
  registry for it). An out-of-scope collection was never aliased at build time,
  so a clean scan against a map that does not apply to it would certify nothing;
* its manifest must carry the ``privacy`` stamp, which the create branch writes
  and a windowed update cannot promise. No stamp means "some of this index was
  built before aliasing", and no scan can tell which part.
"""
import argparse
import json
import sys
import tarfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import PrivacyMapMissing, resolve_registry  # noqa: E402
from main.privacy.index_scan import scan_collection  # noqa: E402
from scripts.audit.scan_index import (  # noqa: E402
    DEFAULT_COLLECTIONS_DIR, load_exceptions, resolve_allowlist, resolve_map,
)

STAMP_NAME = "PACKAGE-STAMP.json"


def refusal(collection_dir: Path, collection: str) -> str | None:
    """Why this collection must not be packaged at all, or None."""
    manifest_path = collection_dir / "manifest.json"
    if not manifest_path.exists():
        return f"no such collection: {collection_dir}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("privacy"):
        return (f"{collection}: no privacy stamp in the manifest — this index was not built "
                f"by the aliasing create path, so a clean scan would certify nothing")
    try:
        registry = resolve_registry(collection, (manifest.get("reader") or {}).get("basePath"))
    except PrivacyMapMissing as e:
        return f"{collection}: {e}"
    if registry is None:
        return (f"{collection}: not in privacy scope (main/privacy/scope.json plus any private "
                f"extension) — nothing aliased it, so there is nothing to certify")
    return None


def archive_members(collection_dir: Path, collection: str):
    """(path on disk, path inside the tarball) for every regular file, sorted.

    Regular files only. A symlink inside the collection would be archived as a
    link that resolves against the *recipient's* filesystem, and a device node or
    fifo has no business in an index; either one means the directory is not what
    this script thinks it is, so it refuses rather than shipping it.
    """
    prefix = Path("data") / "collections" / collection
    for path in sorted(collection_dir.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"refusing to package a non-regular file: "
                             f"{path.relative_to(collection_dir)}")
        yield path, (prefix / path.relative_to(collection_dir)).as_posix()


def package_stamp(report, manifest: dict) -> dict:
    return {
        "collection": report.collection,
        "scanDate": date.today().isoformat(),
        "policy_version": report.policy_version,
        "map_version": report.map_version,
        "numberOfDocuments": manifest.get("numberOfDocuments"),
        "scanChecks": report.counts(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "packages"),
                    help="Directory the tarball is written to (created if missing)")
    ap.add_argument("--collections-dir", default=str(DEFAULT_COLLECTIONS_DIR),
                    help="Package a staged copy instead of data/collections")
    ap.add_argument("--map", default=None, help="Alias map path (default: the discovered one)")
    ap.add_argument("--allowed-bigrams", default=None,
                    help="Reviewed non-person bigram allow-list (default: the discovered one)")
    args = ap.parse_args()

    collections_dir = Path(args.collections_dir).resolve()
    collection_dir = collections_dir / args.collection

    problem = refusal(collection_dir, args.collection)
    if problem:
        sys.exit(f"REFUSED: {problem}")

    report = scan_collection(
        collection_dir,
        resolve_map(args.map),
        exceptions=load_exceptions(),
        allowed_bigrams_path=resolve_allowlist(args.allowed_bigrams),
    )
    print(f"collection: {report.collection}  map v{report.map_version}  "
          f"policy v{report.policy_version}  entries: {report.map_entries}\n")
    for line in report.format_lines():
        print(line)

    if not report.passed:
        failed = [c.name for c in report.checks if not c.passed]
        sys.exit(f"\nREFUSED: the scan failed ({', '.join(failed)}). Nothing was written. "
                 f"Run scripts/audit/scan_index.py --candidates-out … to triage.")

    manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / f"{args.collection}-{date.today().isoformat()}.tar.gz"

    members = list(archive_members(collection_dir, args.collection))
    stamp = json.dumps(package_stamp(report, manifest), indent=2, ensure_ascii=False)
    stamp_path = out_dir / f".{args.collection}-{STAMP_NAME}"
    stamp_path.write_text(stamp + "\n", encoding="utf-8")
    try:
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(stamp_path, arcname=STAMP_NAME)
            for path, arcname in members:
                tar.add(path, arcname=arcname)
    finally:
        stamp_path.unlink(missing_ok=True)

    print(f"\nPASS — wrote {tarball} ({tarball.stat().st_size} bytes, "
          f"{len(members)} files + {STAMP_NAME})")


if __name__ == "__main__":
    main()
