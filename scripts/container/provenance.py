"""Where every file in the serve image is allowed to come from.

Two sources exist: the git tree of the pinned commit, and the members of the
``package_collection.py`` tarballs. ``image_build.py`` verifies the staging
folder against them before ``docker build``; ``image_check.py`` verifies the
built image's layers against the same answers. Stdlib only.

Nothing here prints a path under a collection's ``documents/``: a document
file is named after its document id, and ids are never aliased.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

STAMP = "PACKAGE-STAMP.json"
COLLECTIONS_PREFIX = "data/collections/"
_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHUNK = 1 << 20


class Refused(Exception):
    """A package, staging folder or image that must not be built or shipped."""


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hash_stream(stream, size: int) -> tuple[str, str]:
    """(git blob id, sha256) of ``size`` bytes read from ``stream``."""
    sha1 = hashlib.sha1(b"blob %d\0" % size)
    sha256 = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(_CHUNK, remaining))
        if not chunk:
            raise Refused(f"stream ended {remaining} bytes early")
        sha1.update(chunk)
        sha256.update(chunk)
        remaining -= len(chunk)
    return sha1.hexdigest(), sha256.hexdigest()


def git_tree(commit: str, repo: str = ".") -> dict[str, tuple[str, str]]:
    """``{path: (mode, blob id)}`` for every blob in ``commit``."""
    out = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "-z", "--full-tree", commit],
        capture_output=True, check=True,
    ).stdout
    tree = {}
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        if kind == "blob":
            tree[path.decode()] = (mode, oid)
    return tree


def safe_display(key: str) -> str:
    """``key`` with anything under a collection's ``documents/`` withheld."""
    parts = key.split("/")
    for i in range(len(parts) - 1):
        if parts[i] == "documents" and "collections" in parts[:i]:
            return "/".join(parts[: i + 1]) + "/<document>"
    return key


def stamp_refusals(stamp) -> list[str]:
    """Why a ``PACKAGE-STAMP.json`` fails; empty when it passes.

    Passes when ``sensitivitySweep.status == "pass"``, at least one
    ``scanChecks`` entry ran, and every entry that ran has passed. Checks 3b
    and 5 report ``ran: false`` when a package was made without ``--compare``,
    so requiring every check to have run would refuse every real stamp. An
    entry without ``ran`` counts as run. Anything malformed is a refusal.
    """
    if not isinstance(stamp, dict):
        return ["stamp is not a JSON object"]
    reasons = []
    sweep = stamp.get("sensitivitySweep")
    status = sweep.get("status") if isinstance(sweep, dict) else None
    if status != "pass":
        reasons.append(f"sensitivitySweep.status is {status!r}, not 'pass'")
    checks = stamp.get("scanChecks")
    if not isinstance(checks, dict) or not checks:
        return reasons + ["scanChecks is missing or empty"]
    ran = 0
    for name, check in sorted(checks.items()):
        if not isinstance(check, dict):
            reasons.append(f"check {name} is not an object")
            continue
        if check.get("ran", True) is False:
            continue
        ran += 1
        if check.get("passed") is not True:
            reasons.append(f"check {name} ran and did not pass")
    if ran == 0:
        reasons.append("no check ran")
    return reasons


@dataclass
class Package:
    path: str
    collection: str
    sha256: str
    stamp: dict
    manifest: dict
    # "data/collections/<name>/…" -> sha256; the stamp at its relocated path
    members: dict[str, str] = field(default_factory=dict)

    @property
    def stamp_key(self) -> str:
        return f"{COLLECTIONS_PREFIX}{self.collection}/{STAMP}"


def _member_key(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise Refused(f"member escapes the package root: {safe_display(name)}")
    return "/".join(p for p in path.parts if p != ".")


def read_package(path) -> Package:
    """Read and validate one tarball, hashing every member. Raises ``Refused``."""
    path = str(path)
    with open(path, "rb") as fh:
        tar_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    files: dict[str, str] = {}
    small: dict[str, bytes] = {}
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            key = _member_key(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise Refused(f"{Path(path).name}: member is not a regular file: {safe_display(key)}")
            if key in files:
                raise Refused(f"{Path(path).name}: duplicate member {safe_display(key)}")
            stream = tar.extractfile(member)
            if key == STAMP or key.endswith("/manifest.json"):
                data = stream.read()
                small[key] = data
                files[key] = hashlib.sha256(data).hexdigest()
            else:
                files[key] = hash_stream(stream, member.size)[1]
    name = Path(path).name
    if STAMP not in small:
        raise Refused(f"{name}: no {STAMP} at the package root")
    stamp = json.loads(small[STAMP])
    reasons = stamp_refusals(stamp)
    if reasons:
        raise Refused(f"{name}: stamp refused: " + "; ".join(reasons))
    collection = stamp.get("collection")
    if not isinstance(collection, str) or not _COLLECTION_NAME.match(collection):
        raise Refused(f"{name}: stamp names no usable collection")
    prefix = f"{COLLECTIONS_PREFIX}{collection}/"
    package = Package(path=path, collection=collection, sha256=tar_sha256, stamp=stamp, manifest={})
    for key, digest in files.items():
        if key == STAMP:
            continue
        if not key.startswith(prefix):
            raise Refused(f"{name}: member outside {prefix}: {safe_display(key)}")
        package.members[key] = digest
    if package.stamp_key in package.members:
        raise Refused(f"{name}: a member already sits at the stamp's relocated path")
    package.members[package.stamp_key] = files[STAMP]
    manifest_key = f"{prefix}manifest.json"
    if manifest_key not in small:
        raise Refused(f"{name}: no {manifest_key}")
    package.manifest = json.loads(small[manifest_key])
    count = stamp.get("numberOfDocuments")
    if not isinstance(count, int) or package.manifest.get("numberOfDocuments") != count:
        raise Refused(f"{name}: stamp numberOfDocuments differs from the manifest's")
    return package


def extract_package(package: Package, root: Path) -> None:
    """Write the package's members under ``root``, the stamp at its relocated path."""
    with tarfile.open(package.path, "r:*") as tar:
        for member in tar:
            if member.isdir():
                continue
            key = _member_key(member.name)
            target = root / (package.stamp_key if key == STAMP else key)
            if target.exists() or target.is_symlink():
                raise Refused(f"{package.collection}: staging already holds {safe_display(key)}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, "xb") as dst:
                while chunk := src.read(_CHUNK):
                    dst.write(chunk)
            # Fixed modes, not the umask's: the image check refuses writable bits.
            os.chmod(target, 0o644)
            for parent in target.relative_to(root).parents:
                os.chmod(root / parent, 0o755)


def verify_staging(staging: Path, tree: dict, packages: list[Package]) -> list[str]:
    """Every file under ``staging`` must be a git blob (``src/``) or a package
    member (``collections/``) with the same hash. Returns refusals."""
    expected = {}
    for package in packages:
        expected.update(package.members)
    refusals = []
    for dirpath, dirnames, filenames in os.walk(staging):
        base = Path(dirpath)
        for entry in [*dirnames, *filenames]:
            full = base / entry
            rel = full.relative_to(staging).as_posix()
            if full.is_dir() and not full.is_symlink():
                continue
            top, _, key = rel.partition("/")
            if top == "src" and key:
                refusal = _verify_src_entry(full, key, tree)
            elif top == "collections" and key:
                refusal = _verify_collection_entry(full, key, expected)
            else:
                refusal = "neither src/ nor collections/"
            if refusal:
                refusals.append(f"{safe_display(rel)}: {refusal}")
    return refusals


def _verify_src_entry(full: Path, key: str, tree: dict) -> str | None:
    if key not in tree:
        return "not in the pinned commit"
    mode, oid = tree[key]
    if mode == "120000":
        if not full.is_symlink():
            return "the commit has a symlink here"
        return None if git_blob_id(os.readlink(full).encode()) == oid else "symlink target differs from the commit"
    if full.is_symlink() or not full.is_file():
        return "not a regular file"
    return None if git_blob_id(full.read_bytes()) == oid else "content differs from the commit"


def _verify_collection_entry(full: Path, key: str, expected: dict) -> str | None:
    if key not in expected:
        return "not a member of any package"
    if full.is_symlink() or not full.is_file():
        return "not a regular file"
    with open(full, "rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    return None if digest == expected[key] else "content differs from the package member"
