#!/usr/bin/env python3
"""Produce the hand-off tarball for an aliased collection. THE ONLY WAY TO SHIP.

    .venv/bin/python scripts/audit/package_collection.py --collection nav-wiki
    .venv/bin/python scripts/audit/package_collection.py --collection nav-wiki --out /tmp/pkg
    .venv/bin/python scripts/audit/package_collection.py --collection nav-wiki-aliased \\
        --compare nav-wiki --allow-count-drift

Copying `data/collections/<name>/` by hand — scp, rsync, a zip, AirDrop — is not
a supported hand-off and never was. The gate has to *produce* the artifact,
because a scanner someone must remember to run before copying is advisory, and
the one time it is skipped is the time it mattered. This script runs
``main.privacy.index_scan.scan_collection`` and writes the tarball **only** when
every check passes; on a failure it prints the report and writes nothing at all.

    <name>-<date>-map<map_version>-policy<policy_version>.tar.gz
      PACKAGE-STAMP.json                 <- what was scanned, and with which map
      data/collections/<name>/…          <- the collection, at the path it unpacks to

Not the alias map (the reverse map never leaves this machine), not
``data/prealias/`` (that is the collection WITH the real names in it), not the
contextual or graph caches (they carry pre-alias text until
``purge_prealias_caches.py`` retires them).

Four properties the tarball has to have, each of which was missing:

* **the same check set as the CLI.** ``--compare`` and ``--allow-count-drift``
  are accepted and passed through, and the stamp records a row for EVERY known
  check — `ran: false` for the ones a run without a twin never performed, which
  a stamp that simply omitted them could not distinguish from `passed`;
* **exactly what was scanned.** The member list comes from the report, not from
  a second walk, and every member's size and mtime is re-checked immediately
  before the tar. Between the scan and the tar the collection is unlocked, and a
  nightly reindex finishing in that window would put an unscanned document
  inside a tarball stamped as scanned;
* **atomic.** Built under a temp name in the destination directory and
  ``os.replace``d into place. A half-written `.tar.gz` has the name of a
  certified package and the contents of an interrupted one;
* **no builder identity.** ``uid``/``gid``/``uname``/``gname`` are zeroed on
  every member. That is the same class of fingerprint as the absolute `/Users/`
  path check 10 looks for, just in the header instead of the content.

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
import hashlib
import io
import json
import os
import sys
import tarfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import (  # noqa: E402
    PrivacyMapMissing, _load_ident_exceptions, discover_map_path, resolve_registry,
)
from main.privacy.index_scan import (  # noqa: E402
    CHECK_NAMES, GIVEN_NAMES_FILE, scan_collection,
)
from scripts.audit.scan_index import DEFAULT_COLLECTIONS_DIR, resolve_allowlist  # noqa: E402

STAMP_NAME = "PACKAGE-STAMP.json"


class Refused(Exception):
    """A reason not to ship, printed as one `REFUSED:` line rather than raised."""


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


def verify_members(collection_dir: Path, members: list) -> list:
    """Everything about the directory that no longer matches what was scanned.

    Size and mtime rather than a re-hash: the window is seconds wide and the
    question is "did anything move", not "is this byte-identical to a baseline".
    A file that appeared is as disqualifying as one that changed — it would be
    tarred without ever having been read.
    """
    problems = []
    expected = {}
    for member in members:
        expected[member["path"]] = (member["size"], member["mtime"])

    on_disk = set()
    for dirpath, dirnames, filenames in os.walk(collection_dir, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (here / name).is_symlink()]
        for name in filenames:
            on_disk.add((here / name).relative_to(collection_dir).as_posix())

    for relative in sorted(on_disk - set(expected)):
        problems.append(f"{relative} appeared after the scan")
    for relative in sorted(set(expected) - on_disk):
        problems.append(f"{relative} disappeared after the scan")
    for relative in sorted(on_disk & set(expected)):
        try:
            stat = os.lstat(collection_dir / relative)
        except OSError as e:
            problems.append(f"{relative} is no longer readable ({type(e).__name__})")
            continue
        if (stat.st_size, stat.st_mtime_ns) != expected[relative]:
            problems.append(f"{relative} changed after the scan")
    return problems


def archive_members(collection_dir: Path, collection: str, members: list):
    """(path on disk, path inside the tarball) for every scanned member, sorted.

    Regular files only. A symlink inside the collection would be archived as a
    link that resolves against the *recipient's* filesystem, and a device node or
    fifo has no business in an index; either one means the directory is not what
    this script thinks it is, so it refuses rather than shipping it. (The scan's
    check 8 already fails on both — this is the belt to its braces, and the one
    that runs against the list actually being tarred.)
    """
    prefix = Path("data") / "collections" / collection
    for member in members:
        path = collection_dir / member["path"]
        if path.is_symlink() or not path.is_file():
            raise Refused(f"refusing to package a non-regular file: {member['path']}")
        yield path, (prefix / member["path"]).as_posix()


def _reset_identity(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Zero the builder's uid, gid and login name on a member."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _sha256(path) -> str | None:
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tarball_name(collection: str, report, today: str) -> str:
    """`<name>-<date>-map<map>-policy<policy>.tar.gz`.

    The date alone made two tarballs built the same day from different maps
    indistinguishable, which is exactly the pair a recipient needs to tell apart.
    """
    return (f"{collection}-{today}-map{report.map_version}"
            f"-policy{report.policy_version}.tar.gz")


def package_stamp(report, manifest: dict, allowlist_path, gazetteer_path) -> dict:
    return {
        "collection": report.collection,
        "scanDate": date.today().isoformat(),
        "policy_version": report.policy_version,
        "map_version": report.map_version,
        "numberOfDocuments": manifest.get("numberOfDocuments"),
        "numberOfFiles": len(report.scanned_members),
        # A row per KNOWN check, so "did not run" is legible. The inputs that
        # decide what check 9 subtracts are hashed rather than named: a tarball
        # certified against a longer allow-list was certified against a weaker
        # gate, and the sha is the only way a recipient can tell.
        "scanChecks": report.check_summary(),
        "allowlistSha256": _sha256(allowlist_path),
        "gazetteerSha256": _sha256(gazetteer_path),
    }


def _write_tarball(tarball: Path, stamp: str, members) -> int:
    """Build under a temp name in the same directory, then `os.replace`."""
    temp = tarball.with_name(f".{tarball.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with tarfile.open(temp, "w:gz") as tar:
            payload = stamp.encode("utf-8")
            info = _reset_identity(tarfile.TarInfo(STAMP_NAME))
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            for path, arcname in members:
                tar.add(path, arcname=arcname, filter=_reset_identity)
                count += 1
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    os.replace(temp, tarball)
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "packages"),
                    help="Directory the tarball is written to (created if missing)")
    ap.add_argument("--collections-dir", default=str(DEFAULT_COLLECTIONS_DIR),
                    help="Package a staged copy instead of data/collections")
    ap.add_argument("--map", default=None, help="Alias map path (default: the discovered one)")
    ap.add_argument("--compare", default=None,
                    help="Pre-alias collection, for the exemption invariant and the BM25 "
                         "control. Without it those checks are stamped as not run.")
    ap.add_argument("--allow-count-drift", action="store_true",
                    help="Downgrade a numberOfDocuments/numberOfChunks difference against "
                         "--compare to a warning.")
    ap.add_argument("--allowed-bigrams", default=None,
                    help="Reviewed non-person bigram allow-list (default: the discovered one)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing tarball of the same name.")
    args = ap.parse_args()

    collections_dir = Path(args.collections_dir).resolve()
    collection_dir = collections_dir / args.collection

    problem = refusal(collection_dir, args.collection)
    if problem:
        sys.exit(f"REFUSED: {problem}")

    try:
        map_path = discover_map_path(args.map)
    except PrivacyMapMissing as e:
        sys.exit(f"REFUSED: {e}")
    allowlist_path = resolve_allowlist(args.allowed_bigrams)

    try:
        report = scan_collection(
            collection_dir,
            map_path,
            compare_dir=(collections_dir / args.compare) if args.compare else None,
            exceptions=_load_ident_exceptions(),
            allow_count_drift=args.allow_count_drift,
            allowed_bigrams_path=allowlist_path,
        )
    except Refused as e:
        sys.exit(f"REFUSED: {e}")
    print(f"collection: {report.collection}  map v{report.map_version}  "
          f"policy v{report.policy_version}  entries: {report.map_entries}\n")
    for line in report.format_lines():
        print(line)

    if not report.passed:
        failed = [c.name for c in report.checks if not c.passed]
        sys.exit(f"\nREFUSED: the scan failed ({', '.join(failed)}). Nothing was written. "
                 f"Run scripts/audit/scan_index.py --candidates-out … to triage.")

    drifted = verify_members(collection_dir, report.scanned_members)
    if drifted:
        sys.exit(f"\nREFUSED: the collection changed between the scan and the tar, so the "
                 f"tarball would carry files nothing certified: {'; '.join(drifted[:5])}")

    manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / tarball_name(args.collection, report, date.today().isoformat())
    if tarball.exists() and not args.force:
        sys.exit(f"\nREFUSED: {tarball} already exists. A tarball someone has already copied "
                 f"out must keep matching the one on disk — pass --force to overwrite.")

    stamp = json.dumps(package_stamp(report, manifest, allowlist_path, GIVEN_NAMES_FILE),
                       indent=2, ensure_ascii=False) + "\n"
    try:
        members = list(archive_members(collection_dir, args.collection,
                                       report.scanned_members))
        written = _write_tarball(tarball, stamp, members)
    except Refused as e:
        sys.exit(f"\nREFUSED: {e}")

    print(f"\nPASS — wrote {tarball} ({tarball.stat().st_size} bytes, "
          f"{written} files + {STAMP_NAME})")


if __name__ == "__main__":
    main()
