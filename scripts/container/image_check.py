#!/usr/bin/env python3
"""Check a built serve image before it runs or is pushed.

    .venv/bin/python scripts/container/image_check.py --image huginn-serve:<tag> \\
        --commit <sha> --package data/packages/<a>.tar.gz [--package ...]

Reads every layer of ``docker save`` and refuses on any of:

- **name**: an ``aliases.json``, a ``huginn-*`` folder, ``data/sources/`` or
  ``data/prealias/``, a ``*_graph.json``, an archive (``.zip``, ``.tar`` and
  compressed tars), an ``nvidia_*`` dist-info; not exactly one stamp per
  collection, or a stamp whose ``numberOfDocuments`` differs from the manifest's.
- **whiteout**: any ``.wh.*`` entry. A file deleted in a later layer is gone
  from the running filesystem but still in the layer the registry serves.
- **provenance**, for ``/app``: every file outside ``/app/hf-cache/`` matches
  the pinned commit's git tree or a package member by hash; every file under
  it belongs to a pinned model revision in ``models.lock.json``, except one
  ``refs/main`` per model holding that revision. Checked in every layer, not
  only the final filesystem, for the reason whiteouts are refused.
- **base**: the config's layer IDs must start with the pinned base image's
  (``base.lock.json``, per architecture) and match the save's layer count.
- **links**: a symlink or hardlink is allowed only in a base layer, or as a
  symlink under ``/app/hf-cache`` (where provenance then accepts model snapshot
  links only). An entry in a later layer whose parent resolves through a base
  link elsewhere is refused. Links under ``/app/hf-cache`` are not added to the
  link map, so the map cannot change after the base; entries below them are
  left to provenance.
- **scan_index**: ``scripts/audit/scan_index.py`` over the collections read
  out of the image.

Then a non-blocking report: files with a mapped-person needle hit, counted per
area. Model tokenizers and ``main/privacy/given_names.txt`` are known hits.

Declared limit: ``site-packages`` and the OS layer get the name and whiteout
checks only. They come from hash-pinned wheels and a digest-pinned base image,
and the other way in is a Dockerfile step, which code review covers.

Output is paths, counts and check names; a path under a collection's
``documents/`` is withheld. Exit 0 pass, 1 refused.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container.provenance import (  # noqa: E402
    COLLECTIONS_PREFIX,
    STAMP,
    Package,
    Refused,
    git_tree,
    hash_stream,
    read_package,
    safe_display,
)

# A tripwire for archive NAMES, not a content check: a gzip of a tar named .gz,
# a .whl or a .jar passes it. A clean image holds 129 bare .gz/.bz2/.xz files
# (man pages, apt logs, site-packages test data), so single-file compression is
# not refused. Under /app provenance refuses any file; elsewhere the Dockerfile
# is the boundary (declared limit).
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tbz", ".tbz2", ".txz", ".tzst", ".7z", ".rar")
_COMPRESSED_TAR = re.compile(r"\.tar\.([a-z0-9.]+)$")
# `.tar.<...>` is a compressed, split or renamed tar unless its final extension
# is a source or text file (`codec.tar.py`).
_NOT_ARCHIVE_AFTER_TAR = {"py", "pyi", "txt", "md", "rst", "json", "html", "h", "c"}
_NVIDIA_DIST_INFO = re.compile(r"^nvidia_.*\.dist-info$", re.I)
_TEXT_REPORT_MAX_BYTES = 20 * 1024 * 1024
APP = "app/"
HF_CACHE = "app/hf-cache/"


@dataclass
class Report:
    refusals: list[tuple[str, str]] = field(default_factory=list)
    verified: Counter = field(default_factory=Counter)
    layers: int = 0
    # path -> ("sym", target as written) or ("hard", target path), recorded from
    # the base layers only; any non-link entry at the same path removes it.
    links: dict[str, tuple[str, str]] = field(default_factory=dict)
    base_layers: int = 0

    def refuse(self, check: str, detail: str) -> None:
        self.refusals.append((check, detail))


def normalize(name: str) -> str:
    return "/".join(p for p in name.split("/") if p not in ("", "."))


_MAX_LINK_HOPS = 40  # Linux's MAXSYMLINKS


def resolve(path: str, links: dict) -> str | None:
    """The path the kernel reaches, walked component by component from the root.

    A symlink component is replaced by its target (absolute from the root,
    relative from the link's folder); ``..`` stops at the root; a hardlink to a
    symlink is that symlink. Returns None after too many hops (a loop).
    """
    pending = [p for p in path.split("/") if p]
    resolved: list[str] = []
    hops = 0
    while pending:
        part = pending.pop(0)
        if part == ".":
            continue
        if part == "..":
            if resolved:
                resolved.pop()
            continue
        link = links.get("/".join([*resolved, part]))
        while link and link[0] == "hard":
            hops += 1
            if hops > _MAX_LINK_HOPS:
                return None
            link = links.get(link[1])
        if link is None:
            resolved.append(part)
            continue
        hops += 1
        if hops > _MAX_LINK_HOPS:
            return None
        if link[1].startswith("/"):
            resolved = []
        pending = [p for p in link[1].split("/") if p] + pending
    return "/".join(resolved)


def name_refusal(key: str, is_dir: bool) -> str | None:
    """Case-insensitive: a case-insensitive filesystem or a copy by hand does not
    keep the spelling the rule was written for."""
    key = key.lower()
    parts = key.split("/")
    base = parts[-1]
    folders = parts if is_dir else parts[:-1]
    if base == "aliases.json":
        return "an alias map file name"
    if any(p.startswith("huginn-") for p in folders):
        return "a huginn-* folder"
    padded = f"/{key}/"
    if "/data/sources/" in padded or "/data/prealias/" in padded:
        return "a raw-source or pre-alias path"
    if not is_dir and base.endswith("_graph.json"):
        return "a knowledge graph file name"
    tar = _COMPRESSED_TAR.search(base)
    if not is_dir and (base.endswith(ARCHIVE_SUFFIXES) or (tar and tar.group(1).rsplit(".", 1)[-1] not in _NOT_ARCHIVE_AFTER_TAR)):
        return "an archive"
    if any(_NVIDIA_DIST_INFO.match(p) for p in parts):
        return "an nvidia_* dist-info"
    return None


def _layer_paths(outer: tarfile.TarFile) -> tuple[list[str], str]:
    """The layer blob paths, in order, and the config blob path."""
    names = set(outer.getnames())
    if "manifest.json" in names:
        manifest = json.load(outer.extractfile("manifest.json"))
        if len(manifest) != 1:
            raise Refused(f"docker save holds {len(manifest)} images, expected 1")
        return list(manifest[0]["Layers"]), manifest[0]["Config"]

    def blob(digest):
        algo, value = digest.split(":", 1)
        return f"blobs/{algo}/{value}"

    index = json.load(outer.extractfile("index.json"))
    node = index["manifests"]
    while True:
        if len(node) != 1:
            raise Refused(f"image index lists {len(node)} manifests, expected 1")
        doc = json.load(outer.extractfile(blob(node[0]["digest"])))
        if "manifests" in doc:
            node = doc["manifests"]
            continue
        return [blob(layer["digest"]) for layer in doc["layers"]], blob(doc["config"]["digest"])


def _base_layer_count(config: dict, layer_count: int, base: dict | None, report: Report) -> int:
    """How many leading layers are the pinned base image; 0 refuses every link."""
    ids = config["rootfs"]["diff_ids"]
    if len(ids) != layer_count:
        report.refuse("base", f"the config lists {len(ids)} layers, the save holds {layer_count}")
        return 0
    if base is None:
        return 0
    architecture = config.get("architecture")
    pinned = base["diffIds"].get(architecture)
    if not pinned:
        report.refuse("base", f"no pinned base layers for architecture {architecture!r}")
        return 0
    if ids[:len(pinned)] != pinned:
        report.refuse("base", f"the first {len(pinned)} layer(s) are not the pinned base image")
        return 0
    return len(pinned)


class _Models:
    def __init__(self, lock: dict):
        self.by_folder = {}
        for model in lock["models"]:
            files = {f["path"]: (f.get("lfsSha256") or f["blobId"]) for f in model["files"]}
            self.by_folder["models--" + model["repo"].replace("/", "--")] = (model, files)

    def dir_allowed(self, sub: list[str], model: dict, files: dict) -> bool:
        if sub in ([], ["blobs"], ["refs"], ["snapshots"], ["snapshots", model["revision"]]):
            return True
        if len(sub) > 2 and sub[:2] == ["snapshots", model["revision"]]:
            prefix = "/".join(sub[2:]) + "/"
            return any(path.startswith(prefix) for path in files)
        return False


def _hf_refusal(key: str, member: tarfile.TarInfo, read, models: _Models) -> str | None:
    rest = key[len(HF_CACHE):].split("/") if key != HF_CACHE.rstrip("/") else []
    if not rest:
        return None
    if rest[0] != "hub":
        return "outside hf-cache/hub"
    if len(rest) == 1:
        return None if member.isdir() else "a file where hub/ belongs"
    entry = models.by_folder.get(rest[1])
    if entry is None:
        return "not a pinned model folder"
    model, files = entry
    sub = rest[2:]
    if member.isdir():
        return None if models.dir_allowed(sub, model, files) else "a folder no pinned revision has"
    if sub == ["refs", "main"]:
        if not member.isfile():
            return "refs/main is not a regular file"
        return None if read() == model["revision"].encode() else "refs/main names another revision"
    if len(sub) == 2 and sub[0] == "blobs":
        if not member.isfile():
            return "a blob that is not a regular file"
        etags = {etag: path for path, etag in files.items()}
        if sub[1] not in etags:
            return "a blob no pinned file has"
        blob_id, sha256 = read(hashes=True)
        lfs = len(sub[1]) == 64
        return None if (sha256 if lfs else blob_id) == sub[1] else "blob content differs from its name"
    if len(sub) > 2 and sub[:2] == ["snapshots", model["revision"]]:
        path = "/".join(sub[2:])
        if path not in files:
            return "a snapshot file the pinned revision does not list"
        if not member.issym():
            return "a snapshot entry that is not a symlink"
        target = posixpath.normpath(posixpath.join(posixpath.dirname("/".join(sub)), member.linkname))
        return None if target == f"blobs/{files[path]}" else "a snapshot symlink to the wrong blob"
    return "not a pinned model file"


def check_layers(image_tar, tree: dict, packages: list[Package], lock: dict,
                 extract_to: Path | None = None, needle_scanner=None,
                 base: dict | None = None) -> tuple[Report, Counter]:
    """The name, whiteout, base, link and provenance checks over every layer.

    ``base`` is ``base.lock.json``. Without it no layer counts as base, so any
    link outside the model snapshots is refused (fails closed).
    """
    report = Report()
    needle_files: Counter = Counter()
    if not packages:
        report.refuse("name", "no packages to check the image against")
    names = [p.collection for p in packages]
    for name in sorted({n for n in names if names.count(n) > 1}):
        # Members are keyed by path, so the last package would silently win.
        report.refuse("name", f"two packages name {name}")
    expected = {}
    for package in packages:
        expected.update(package.members)
    models = _Models(lock)
    stamps: Counter = Counter()
    small: dict[str, bytes] = {}
    seen_collections: set[str] = set()
    seen_members: set[str] = set()
    allowed_dirs = _implied_dirs([*tree, *expected])

    index = 0
    try:
        with tarfile.open(image_tar) as outer:
            layer_paths, config_path = _layer_paths(outer)
            report.base_layers = _base_layer_count(json.load(outer.extractfile(config_path)),
                                                   len(layer_paths), base, report)
            for index, layer_path in enumerate(layer_paths):
                report.layers += 1
                with tarfile.open(fileobj=outer.extractfile(layer_path), mode="r|*") as layer:
                    for member in layer:
                        _check_member(index, member, layer, report, tree, expected, models, stamps,
                                      small, seen_collections, seen_members, allowed_dirs,
                                      extract_to, needle_scanner, needle_files)
    except (Refused, tarfile.TarError, OSError, EOFError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # Added to the refusals found so far rather than raised, so those stay
        # visible. Refused messages carry counts only; for anything else the
        # type only, since an OSError's message can carry a path.
        reason = str(exc) if isinstance(exc, Refused) else type(exc).__name__
        report.refuse("unreadable", f"unreadable image at layer {index}: {reason}")
        return report, needle_files

    for package in packages:
        missing = len(set(package.members) - seen_members)
        if missing:
            report.refuse("provenance", f"{package.collection}: {missing} package member(s) "
                                        f"missing from the image")
    wanted = {p.collection for p in packages}
    if seen_collections != wanted:
        report.refuse("name", f"collections in the image {sorted(seen_collections)} "
                              f"differ from the packages {sorted(wanted)}")
    for name in sorted(seen_collections):
        prefix = f"{APP}{COLLECTIONS_PREFIX}{name}/"
        if stamps[name] != 1:
            report.refuse("name", f"{name}: {stamps[name]} stamps, expected exactly 1")
            continue
        try:
            stamp = json.loads(small[prefix + STAMP])
            manifest = json.loads(small[prefix + "manifest.json"])
        except (KeyError, ValueError):
            report.refuse("name", f"{name}: stamp or manifest missing or unreadable")
            continue
        if stamp.get("numberOfDocuments") != manifest.get("numberOfDocuments"):
            report.refuse("name", f"{name}: stamp numberOfDocuments differs from the manifest's")
    return report, needle_files


def _implied_dirs(paths) -> set[str]:
    """Every folder under ``app/`` that a git path or package member lives in."""
    dirs = {"app"}
    for path in paths:
        parts = path.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add(APP + "/".join(parts[:i]))
    return dirs


def _check_member(index, member, layer, report, tree, expected, models, stamps, small,
                  seen_collections, seen_members, allowed_dirs, extract_to, needle_scanner,
                  needle_files):
    if ".." in member.name.split("/"):
        # Layer apply cleans the path, so `usr/../app/x` lands in /app while
        # skipping every app/ rule, and an extract would write outside its folder.
        report.refuse("name", f"layer {index}: an entry with a '..' path segment")
        return
    key = normalize(member.name)
    if not key:
        return
    shown = f"layer {index}: {safe_display(key)}"
    if key.split("/")[-1].startswith(".wh."):
        report.refuse("whiteout", shown)
        return
    refusal = name_refusal(key, member.isdir())
    if refusal:
        report.refuse("name", f"{shown} ({refusal})")
    # Links are recorded only from the base layers. The layers the build adds
    # may hold no link except a symlink under app/hf-cache/, which this rule
    # exempts and _hf_refusal then limits to model snapshot links. So the link
    # map is fixed once the base is read: no link added later can redirect a
    # path.
    is_link = member.issym() or member.islnk()
    if index < report.base_layers:
        if is_link:
            report.links[key] = ("sym", member.linkname) if member.issym() else ("hard", normalize(member.linkname))
        else:
            report.links.pop(key, None)
    else:
        # The rules read paths as written; an entry whose parent resolves
        # through a base link would land where those rules never looked.
        parent = posixpath.dirname(key)
        if parent and resolve(parent, report.links) != parent:
            report.refuse("name", f"{shown} (an entry below a symlink)")
        if is_link and not (member.issym() and key.startswith(HF_CACHE)):
            report.refuse("name", f"{shown} (a link in a layer the image build adds)")
        elif not is_link:
            report.links.pop(key, None)

    cache = {}
    collection_prefix = APP + COLLECTIONS_PREFIX
    # A layer is read as a stream, so a member can be read once only: a large
    # collection file is written out while it is hashed, never re-read.
    spool = (extract_to / key[len(APP):]
             if extract_to is not None and member.isfile() and key.startswith(collection_prefix)
             else None)

    def read(hashes=False):
        if "data" not in cache and "hashes" not in cache:
            stream = layer.extractfile(member)
            if member.size <= _TEXT_REPORT_MAX_BYTES:
                cache["data"] = stream.read()
            elif spool is not None:
                spool.parent.mkdir(parents=True, exist_ok=True)
                with open(spool, "wb") as out:
                    cache["hashes"] = hash_stream(_Tee(stream, out), member.size)
                cache["spooled"] = True
            else:
                cache["hashes"] = hash_stream(stream, member.size)
        if hashes:
            if "hashes" not in cache:
                import hashlib
                data = cache["data"]
                cache["hashes"] = (hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
                                   hashlib.sha256(data).hexdigest())
            return cache["hashes"]
        return cache.get("data")

    if key.startswith(APP) or key == APP.rstrip("/"):
        refusal = _app_refusal(key, member, read, tree, expected, models, allowed_dirs)
        if refusal:
            report.refuse("provenance", f"{shown} ({refusal})")
        elif member.isfile():
            area = "hf-cache" if key.startswith(HF_CACHE) else (
                "collections" if key.startswith(APP + COLLECTIONS_PREFIX) else "app")
            report.verified[area] += 1

    if key.startswith(collection_prefix) and member.isfile():
        name, _, inner = key[len(collection_prefix):].partition("/")
        seen_collections.add(name)
        seen_members.add(key[len(APP):])
        if inner == STAMP:
            stamps[name] += 1
        if inner in (STAMP, "manifest.json"):
            data = read()
            small[key] = spool.read_bytes() if data is None and cache.get("spooled") else data
        if spool is not None:
            data = read()
            if data is not None:
                spool.parent.mkdir(parents=True, exist_ok=True)
                spool.write_bytes(data)

    if needle_scanner is not None and member.isfile() and 0 < member.size <= _TEXT_REPORT_MAX_BYTES:
        data = read()
        if data is not None and b"\0" not in data[:8192]:
            text = data.decode("utf-8", errors="ignore")
            if needle_scanner.search(text):
                needle_files["/".join(key.split("/")[:2])] += 1


class _Tee:
    """A read-only stream that copies what is read into ``out``."""

    def __init__(self, stream, out):
        self._stream, self._out = stream, out

    def read(self, size):
        chunk = self._stream.read(size)
        self._out.write(chunk)
        return chunk


def _app_refusal(key, member, read, tree, expected, models, allowed_dirs) -> str | None:
    in_hf_cache = key.startswith(HF_CACHE) or key == HF_CACHE.rstrip("/")
    if member.isdir():
        if member.mode & 0o022:
            return "a group- or world-writable folder"
        if in_hf_cache:
            return _hf_refusal(key, member, read, models)
        return None if key in allowed_dirs else "a folder no commit path or package member implies"
    if member.islnk() or not (member.isfile() or member.issym()):
        return "not a regular file or symlink"
    if member.isfile() and member.mode & 0o022:
        return "a group- or world-writable file"
    if in_hf_cache:
        if member.isfile() and member.mode & 0o111:
            return "an executable model file"
        return _hf_refusal(key, member, read, models)
    rel = key[len(APP):]
    if rel.startswith(COLLECTIONS_PREFIX):
        if rel not in expected:
            return "not a member of any package"
        if not member.isfile():
            return "not a regular file"
        if member.mode & 0o111:
            return "an executable collection file"
        return None if read(hashes=True)[1] == expected[rel] else "differs from the package member"
    if rel not in tree:
        return "not in the pinned commit"
    mode, oid = tree[rel]
    if mode == "120000":
        if not member.issym():
            return "the commit has a symlink here"
        return None if _blob_id(member.linkname.encode()) == oid else "symlink target differs"
    if not member.isfile():
        return "not a regular file"
    if bool(member.mode & 0o111) != (mode == "100755"):
        return "executable bit differs from the commit"
    return None if read(hashes=True)[0] == oid else "content differs from the commit"


def _blob_id(data: bytes) -> str:
    from scripts.container.provenance import git_blob_id
    return git_blob_id(data)


def _needle_scanner(map_path):
    """The mapped-person needles, or None when no map is on this machine."""
    try:
        from main.privacy.alias_registry import discover_map_path
        from main.privacy.index_scan import NeedleScanner, build_needles

        path = discover_map_path(map_path)
        return NeedleScanner(build_needles(json.loads(Path(path).read_text(encoding="utf-8")))), path
    except Exception as exc:  # noqa: BLE001 - a report, never a gate
        return None, f"no map ({type(exc).__name__})"


def _scan_index(collections_dir: Path, names, map_path) -> list[tuple[str, str]]:
    refusals = []
    for name in sorted(names):
        cmd = [sys.executable, str(REPO_ROOT / "scripts/audit/scan_index.py"),
               "--collection", name, "--collections-dir", str(collections_dir)]
        if map_path:
            cmd += ["--map", map_path]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        verdict = lines[-1] if lines else "(no output)"
        print(f"  scan_index {name}: exit {result.returncode}: {verdict}")
        if result.returncode != 0:
            refusals.append(("scan_index", f"{name}: exit {result.returncode}"))
    return refusals


def _at_commit(commit, path) -> dict | None:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"],
                            capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else None


def run(image, commit, package_paths, lock=None, map_path=None, save=None, base_lock=None) -> int:
    tree = git_tree(commit, str(REPO_ROOT))
    if lock is None:
        lock = _at_commit(commit, "scripts/container/models.lock.json")
    base = json.loads(Path(base_lock).read_text()) if base_lock else _at_commit(
        commit, "scripts/container/base.lock.json")
    if lock is None or base is None:
        print(f"REFUSED (lock): {commit[:12]} has no models.lock.json or base.lock.json; pass --base-lock")
        return 1
    try:
        packages = [read_package(p) for p in package_paths]
    except Refused as exc:
        print(f"REFUSED (package): {exc}")
        return 1
    scanner, map_note = _needle_scanner(map_path)
    with tempfile.TemporaryDirectory(prefix="huginn-image-check-") as tmp:
        tmp = Path(tmp)
        image_tar = save
        if image_tar is None:
            image_tar = tmp / "image.tar"
            subprocess.run(["docker", "save", "-o", str(image_tar), image], check=True)
        report, needle_files = check_layers(image_tar, tree, packages, lock, extract_to=tmp / "extracted",
                                            needle_scanner=scanner, base=base)
        print(f"image {image}: {report.layers} layers; verified files {dict(report.verified)}")
        if not report.refusals:
            report.refusals += _scan_index(tmp / "extracted" / COLLECTIONS_PREFIX.rstrip("/"),
                                           {p.collection for p in packages}, map_path)
    print(f"needle report (not blocking, {'map loaded' if scanner else map_note}): "
          f"files with a hit per area {dict(sorted(needle_files.items()))}")
    for check, detail in report.refusals:
        print(f"REFUSED [{check}] {detail}")
    by_check = Counter(check for check, _ in report.refusals)
    print(("PASS" if not report.refusals else f"REFUSED {dict(by_check)}") + f" — {image}")
    return 0 if not report.refusals else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--commit", required=True, help="The commit the image was staged from")
    ap.add_argument("--package", action="append", required=True, dest="packages")
    ap.add_argument("--map", help="Alias map for scan_index and the needle report (default: discovered)")
    ap.add_argument("--save", help="Read an existing `docker save` tar instead of saving --image")
    ap.add_argument("--base-lock", help="base.lock.json path (default: the one in --commit)")
    args = ap.parse_args(argv)
    return run(args.image, args.commit, args.packages, map_path=args.map, save=args.save,
               base_lock=args.base_lock)


if __name__ == "__main__":
    sys.exit(main())
