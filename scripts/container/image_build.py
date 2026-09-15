#!/usr/bin/env python3
"""Build the serve-only huginn image from a staging folder, then check it.

    .venv/bin/python scripts/container/image_build.py --commit <sha> \\
        --package data/packages/<a>.tar.gz [--package ...] \\
        --platform linux/arm64 --tag huginn-serve:<tag>

1. Refuses a package whose stamp does not pass (``sensitivitySweep.status`` is
   ``pass`` and every check that ran passed), and prints each package's
   ``lastModifiedDocumentTime`` beside the live collection's.
2. Stages ``$TMPDIR/huginn-image-<sha>-*/``: ``src/`` is ``git archive`` of the
   commit, ``collections/`` the packages unpacked one at a time, each stamp at
   ``data/collections/<name>/PACKAGE-STAMP.json``.
3. Refuses before building if any staged file is not a git blob or a package
   member with the same hash. The Dockerfile copies only from this folder.
4. ``docker buildx build --load`` from the staging folder, which is deleted
   afterwards whatever happens.
5. Confirms from the build log that only ``torch`` came from
   ``download.pytorch.org``, then runs ``image_check.py``.

Prints each tarball's sha256 as a record for the push; nothing stored at
packaging time exists to compare it against. Exit 0 built and passed, 1 the
image check refused, 2 refused before building, 3 the build failed.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container import image_check  # noqa: E402
from scripts.container.provenance import (  # noqa: E402
    Refused,
    extract_package,
    git_tree,
    read_package,
    verify_staging,
)

PYTORCH_OWN = {"torch"}
# uv resolves from download.pytorch.org and fetches wheels from download-r2.pytorch.org.
_PYTORCH_WHEEL = re.compile(r"https://(?:[\w-]+\.)*pytorch\.org/\S*?/([A-Za-z0-9_.]+?)-\d[^/\s]*?\.whl")
_INSTALL_STEP = re.compile(r"^#(\d+) \[[^\]]*\] RUN --mount=from=ghcr\.io/astral-sh/uv")


def pytorch_index_packages(log: str) -> tuple[set[str] | None, str]:
    """Package names downloaded from download.pytorch.org, or None when the
    install step was cached and the log cannot answer."""
    step = next((m.group(1) for line in log.splitlines() if (m := _INSTALL_STEP.match(line))), None)
    if step is None:
        return None, "install step not found in the build log"
    if re.search(rf"^#{step} CACHED$", log, re.M):
        return None, "install step was cached; rebuild with --no-cache to read the index"
    names = {m.group(1).lower().replace("_", "-") for m in _PYTORCH_WHEEL.finditer(log)}
    return names, f"{len(names)} package(s) from download.pytorch.org"


def pytorch_refusal(names: set[str] | None) -> str | None:
    """None when the log cannot answer (cached step) or shows exactly PyTorch's own."""
    if names is None or (names and names <= PYTORCH_OWN):
        return None
    return f"expected only {sorted(PYTORCH_OWN)} from *.pytorch.org, the log shows {sorted(names)}"


def input_refusals(packages) -> list[str]:
    if not packages:
        return ["no package given"]
    names = [p.collection for p in packages]
    return [f"two packages name {n}" for n in sorted({n for n in names if names.count(n) > 1})]


def _live_last_modified(collection: str) -> str:
    try:
        manifest = json.loads((REPO_ROOT / "data/collections" / collection / "manifest.json").read_text())
        return str(manifest.get("lastModifiedDocumentTime"))
    except (OSError, ValueError):
        return "(no live collection)"


def stage(commit: str, packages, staging: Path) -> None:
    archive = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", commit],
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(staging / "src", filter="data")
    for package in packages:
        extract_package(package, staging / "collections")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--package", action="append", required=True, dest="packages")
    ap.add_argument("--platform", required=True, choices=["linux/arm64", "linux/amd64"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--builder", default="multiplatform")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--log", help="Build log path (default: $TMPDIR/huginn-image-<sha>.build.log)")
    args = ap.parse_args(argv)

    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", f"{args.commit}^{{commit}}"],
                            capture_output=True, check=True, text=True).stdout.strip()
    try:
        packages = [read_package(p) for p in args.packages]
    except Refused as exc:
        print(f"REFUSED before building: {exc}")
        return 2
    refusals = input_refusals(packages)
    if refusals:
        print("REFUSED before building: " + "; ".join(refusals))
        return 2
    for p in packages:
        print(f"package {p.collection}: documents {p.manifest.get('numberOfDocuments')}, "
              f"chunks {p.manifest.get('numberOfChunks')}, "
              f"lastModifiedDocumentTime {p.manifest.get('lastModifiedDocumentTime')} "
              f"(live {_live_last_modified(p.collection)}), sha256 {p.sha256}")

    tree = git_tree(commit, str(REPO_ROOT))
    if "Dockerfile" not in tree:
        print(f"REFUSED before building: {commit[:12]} has no Dockerfile")
        return 2
    log_path = Path(args.log or Path(tempfile.gettempdir()) / f"huginn-image-{commit[:12]}.build.log")
    staging = Path(tempfile.mkdtemp(prefix=f"huginn-image-{commit[:12]}-"))
    try:
        try:
            stage(commit, packages, staging)
            refusals = verify_staging(staging, tree, packages)
        except (OSError, tarfile.TarError) as exc:
            # The type and errno only: the message can name a document's path.
            print(f"REFUSED [staging] {type(exc).__name__} errno {getattr(exc, 'errno', None)}")
            return 2
        if refusals:
            for refusal in refusals:
                print(f"REFUSED [staging] {refusal}")
            return 2
        print(f"staging verified: {sum(1 for f in staging.rglob('*') if not f.is_dir())} files")
        cmd = ["docker", "buildx", "build", "--builder", args.builder, "--platform", args.platform,
               "--provenance=false", "--sbom=false", "--load", "--progress=plain",
               "-f", str(staging / "src" / "Dockerfile"), "-t", args.tag, str(staging)]
        if args.no_cache:
            cmd.insert(3, "--no-cache")
        started = time.monotonic()
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"build {'succeeded' if result.returncode == 0 else 'FAILED'} in "
              f"{time.monotonic() - started:.0f} s; log {log_path}")
        if result.returncode != 0:
            return 3
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    names, note = pytorch_index_packages(log_path.read_text(errors="replace"))
    print(f"pytorch index: {note}{'' if names is None else ' ' + str(sorted(names))}")
    status = image_check.run(args.tag, commit, args.packages)
    refusal = pytorch_refusal(names)
    if refusal:
        print(f"REFUSED [pytorch-index] {refusal}")
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
