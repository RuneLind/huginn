#!/usr/bin/env python3
"""Rebuild an in-scope collection with build-time aliasing, into a temp name.

Why a temp name: DocumentCollectionCreator.__create_collection() starts with
`persister.remove_folder(name)`, so building in place would destroy the live
collection before the first embedding is computed. We build into
`<name>-aliased`, verify it, and only then swap.

    .venv/bin/python scripts/audit/rebuild_aliased.py --collection jira-issues
    .venv/bin/python scripts/audit/rebuild_aliased.py --collection jira-issues --swap

The build reuses the existing manifest verbatim — reader block, indexers and
contextual-prefix model — with two deliberate edits:

* the create adapter's `--contextual-cache` is pointed at the REAL collection's
  cache file (`data/contextual_caches/<name>.json`), because the temp collection
  name would otherwise start from an empty cache and re-prefix every chunk
  through the LLM;
* an absolute `basePath` is rewritten relative to the repo root. An absolute path
  is a distributor fingerprint (it carries the builder's home directory) and this
  whole campaign is about making the built collection copyable.

Both build and swap refuse to proceed unless the rebuilt manifest carries a
privacy stamp matching the CURRENT map and policy version, and unless its reader
block reproduces the source's. Those two checks are the difference between "the
swap moved an aliased index into place" and "the swap moved whatever happened to
be there" — a scope file edited between build and swap, or an `includePatterns`
list quietly dropped, produce a collection that looks fine on disk.

`--swap` is a separate, explicit step: two renames plus a reload of the running
server, with the pre-alias collection parked under `data/prealias/<name>-<date>`
— outside `data/collections/`, so a server started with a glob does not serve it
and the audit sweep does not scan it as live.

The temp name is not in main/privacy/scope.json's collection list, and does not
need to be: the scope also matches on the reader's basePath, which the temp build
shares with the real collection. That is what arms aliasing for `<name>-aliased`.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main.privacy.alias_registry import POLICY_VERSION, AliasRegistry  # noqa: E402

COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"
PREALIAS_DIR = REPO_ROOT / "data" / "prealias"
MAP_GLOB = "huginn-*/privacy/aliases.json"


def load_manifest(name: str) -> dict:
    path = COLLECTIONS_DIR / name / "manifest.json"
    if not path.exists():
        sys.exit(f"No such collection: {name} ({path} missing)")
    return json.loads(path.read_text(encoding="utf-8"))


def current_map_version() -> int:
    maps = sorted(glob.glob(str(REPO_ROOT / MAP_GLOB)))
    if len(maps) != 1:
        sys.exit(f"Expected exactly one alias map ({MAP_GLOB}), found {len(maps)}")
    return AliasRegistry.load(maps[0]).map_version


def relative_base_path(base_path: str) -> str:
    """Absolute paths inside the repo become './…'; anything else is left alone."""
    path = Path(base_path)
    if not path.is_absolute():
        return base_path
    try:
        return "./" + str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return base_path


def parking_path(name: str) -> Path:
    return PREALIAS_DIR / f"{name}-{date.today().isoformat()}"


INDEX_MAPPING = "indexes/index_document_mapping.json"


def rewrite_document_paths(collection_dir: Path, temp_name: str, name: str) -> int:
    """Repoint every ``documentPath`` from the temp build name to the real one.

    The build runs under `<name>-aliased`, so the mapping it writes records
    `<name>-aliased/documents/<id>.json`. `swap()` renames the *directory*, and
    nothing used to rewrite those paths — all three live collections then served
    empty chunk texts, silently, at every reranked query (the incident is
    written up in CLAUDE.md, "Build-time people aliasing").

    Called by `swap()` AFTER the rename, never by the build: until the directory
    is actually `<name>`, `<name>-aliased/documents/…` is the *correct* answer,
    and a temp collection that is served or scanned in place has to read its own
    aliased documents.

    Written to a sibling temp file and `os.replace`d, so an interrupted rewrite
    cannot leave a mapping that is half one prefix and half the other.

    Returns the number of entries changed (0 when already rewritten).
    """
    path = collection_dir / INDEX_MAPPING
    mapping = json.loads(path.read_text(encoding="utf-8"))
    stale, real = f"{temp_name}/", f"{name}/"
    changed = 0
    for entry in mapping.values():
        document_path = entry.get("documentPath", "")
        if document_path.startswith(stale):
            entry["documentPath"] = real + document_path[len(stale):]
            changed += 1
    if changed:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    return changed


def stale_temp_paths(collection_dir: Path, temp_name: str) -> list:
    """Relative paths of `indexes/` JSON still spelling the temp collection name.

    `rewrite_document_paths` only touches the mapping, because that is the one
    built artifact stamped with the collection name (the reverse mapping is
    keyed by document id, `index_info.json` holds a counter, and the manifest's
    `collectionName` is rewritten separately). Rather than trust that list to
    stay true, the swap re-derives it: any `indexes/` JSON still carrying
    `<temp_name>/` is a refusal. Only JSON — the indexer blobs beside it are
    pickles, and searching them for a name would be a decode error, not a check.
    """
    needle = f"{temp_name}/".encode()
    return sorted(path.relative_to(collection_dir).as_posix()
                  for path in (collection_dir / "indexes").rglob("*.json")
                  if needle in path.read_bytes())


def stamp_mismatch(manifest: dict, policy_version: int, map_version) -> str | None:
    """Why this manifest is not a build of the current map, or None."""
    stamp = manifest.get("privacy")
    if not stamp:
        return "no privacy stamp — this build did not alias anything"
    if stamp.get("policy_version") != policy_version:
        return f"policy_version {stamp.get('policy_version')} != {policy_version}"
    if stamp.get("map_version") != map_version:
        return f"map_version {stamp.get('map_version')} != {map_version}"
    return None


def reader_mismatch(source_reader: dict, built_reader: dict) -> str | None:
    """Why the rebuilt reader block differs from the source's, or None.

    basePath is compared after normalization (the rebuild deliberately rewrites
    an absolute path); everything else must be reproduced exactly, because a
    dropped excludePattern indexes documents the live collection never had.
    """
    def normalized(reader):
        out = dict(reader)
        if out.get("basePath"):
            out["basePath"] = str((REPO_ROOT / out["basePath"]).resolve())
        return out

    expected, actual = normalized(source_reader), normalized(built_reader)
    differing = sorted(k for k in set(expected) | set(actual)
                       if expected.get(k) != actual.get(k))
    if differing:
        return f"reader block differs on {differing}"
    return None


def build_command(manifest: dict, name: str, temp_name: str, workers: int) -> list:
    """The create-adapter argv for this rebuild.

    Empty list values are omitted rather than passed: `-indexers` with no values
    is an argparse error, and `-includePatterns` with none would read nothing.
    """
    reader = manifest["reader"]
    contextual_model = (manifest.get("contextualPrefix") or {}).get("model", "none")
    cmd = [
        sys.executable, "files_collection_create_cmd_adapter.py",
        "-collection", temp_name,
        "-basePath", relative_base_path(reader["basePath"]),
        "-includePatterns", *(reader.get("includePatterns") or [".*"]),
        "--contextual-model", contextual_model,
        # The real collection's cache, not the temp name's: unchanged chunks must
        # keep hitting, or a rebuild re-generates thousands of prefixes.
        "--contextual-cache", f"./data/contextual_caches/{name}.json",
        "--contextual-workers", str(workers),
    ]
    indexers = [i["name"] for i in (manifest.get("indexers") or [])]
    if indexers:
        cmd += ["-indexers", *indexers]
    exclude_patterns = reader.get("excludePatterns") or []
    if exclude_patterns:
        cmd += ["-excludePatterns", *exclude_patterns]
    if reader.get("failFast"):
        cmd.append("-failFast")
    return cmd


def build(name: str, temp_name: str, workers: int) -> None:
    manifest = load_manifest(name)
    reader = manifest["reader"]
    if reader.get("type") != "localFiles":
        sys.exit(f"{name}: only localFiles collections can be rebuilt this way (got {reader.get('type')})")

    cmd = build_command(manifest, name, temp_name, workers)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    temp_manifest_path = COLLECTIONS_DIR / temp_name / "manifest.json"
    temp_manifest = json.loads(temp_manifest_path.read_text(encoding="utf-8"))
    for problem in (stamp_mismatch(temp_manifest, POLICY_VERSION, current_map_version()),
                    reader_mismatch(reader, temp_manifest.get("reader", {}))):
        if problem:
            sys.exit(f"{temp_name}: {problem}. Refusing to leave a build that cannot be swapped.")

    # Nothing is renamed here. The build leaves a collection that names ITSELF —
    # manifest `collectionName` and every `documentPath` say `<name>-aliased` —
    # so it is internally consistent as it sits: servable, and scannable (the
    # gate's check 12 keys off the directory name and requires the manifest to
    # agree with it). Rewriting either one before the directory moves would
    # leave, on any later refusal, a directory whose own manifest claims to be
    # another collection. `swap()` renames first and repoints after.
    print(f"\n{temp_name}: {temp_manifest['numberOfDocuments']} docs, "
          f"{temp_manifest['numberOfChunks']} chunks, privacy={temp_manifest.get('privacy')}")


def swap(name: str, temp_name: str, api_base: str) -> None:
    live = COLLECTIONS_DIR / name
    aliased = COLLECTIONS_DIR / temp_name
    if not aliased.exists():
        sys.exit(f"Nothing to swap: {aliased} does not exist. Build first.")

    # Re-checked here, not just after the build: the map or the scope files may
    # have moved in between, and this is the last point before the live
    # collection is replaced.
    problem = stamp_mismatch(json.loads((aliased / "manifest.json").read_text(encoding="utf-8")),
                             POLICY_VERSION, current_map_version())
    if problem:
        sys.exit(f"{temp_name}: {problem}. Refusing to swap it over {name}.")

    # Everything the post-rename rewrite needs, checked BEFORE anything moves.
    # The rewrite cannot run earlier (until the directory is renamed the temp
    # prefixes are correct), so a build that never produced an index must be
    # refused here — after the rename there is no live collection to fall back
    # to, and a mapping still pointing into `<temp_name>/documents/` resolves to
    # nothing: empty chunk texts, no error anywhere.
    if not (aliased / "indexes").is_dir() or not (aliased / INDEX_MAPPING).is_file():
        sys.exit(f"{temp_name}: no {INDEX_MAPPING} — this is not a finished build, "
                 f"and the swap could not repoint it. Refusing to touch {name}.")

    parked = parking_path(name)
    if parked.exists():
        sys.exit(f"Refusing to overwrite an existing parked collection: {parked}")
    parked.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(live), str(parked))
    shutil.move(str(aliased), str(live))
    print(f"swapped: {name} -> {parked}, {temp_name} -> {name}")

    # Rename, then repoint, then verify. The collection now IS `<name>`, so this
    # is the first moment at which `<name>/documents/…` is the right answer.
    manifest_path = live / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collectionName"] = name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"repointed {rewrite_document_paths(live, temp_name, name)} documentPath prefixes")

    stale = stale_temp_paths(live, temp_name)
    if stale:
        # Non-zero, loudly: the directory is in place but unreadable, and the
        # parked copy is the way back.
        sys.exit(f"{name}: {stale} still spell {temp_name} after the rewrite — the "
                 f"collection would read no chunk texts. The pre-alias copy is at "
                 f"{parked}; move it back and investigate.")

    request = urllib.request.Request(f"{api_base}/api/collections/{name}/reload", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print("reload:", response.read().decode())
    except (urllib.error.URLError, OSError) as e:
        # Non-zero: the swap succeeded but the running server is still serving
        # the pre-alias index from memory, which is exactly the state the
        # campaign must not silently end in.
        sys.exit(f"reload failed ({e}); the swap is done — restart the API server to pick up {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="Contextual-prefix workers (only matters when the manifest has a model)")
    ap.add_argument("--swap", action="store_true",
                    help="Skip building; park the live collection and move <name>-aliased into place")
    ap.add_argument("--api-base", default=os.environ.get("HUGINN_API_BASE", "http://127.0.0.1:8321"))
    args = ap.parse_args()

    temp_name = f"{args.collection}-aliased"
    if args.swap:
        swap(args.collection, temp_name, args.api_base)
    else:
        build(args.collection, temp_name, args.workers)


if __name__ == "__main__":
    main()
