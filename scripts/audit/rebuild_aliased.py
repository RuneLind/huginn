#!/usr/bin/env python3
"""Rebuild an in-scope collection with build-time aliasing, into a temp name.

Why a temp name: DocumentCollectionCreator.__create_collection() starts with
`persister.remove_folder(name)`, so building in place would destroy the live
collection before the first embedding is computed. We build into
`<name>-aliased`, verify it, and only then swap.

    scripts/audit/rebuild_aliased.py --collection jira-issues
    scripts/audit/rebuild_aliased.py --collection jira-issues --swap

The build reuses the existing manifest verbatim — reader block, indexers and
contextual-prefix model — with two deliberate edits:

* the contextual-prefix cache is pointed at the REAL collection's cache file
  (`data/contextual_caches/<name>.json`), because the temp collection name would
  otherwise start from an empty cache and re-prefix every chunk through the LLM;
* an absolute `basePath` is rewritten relative to the repo root. An absolute path
  is a distributor fingerprint (it carries the builder's home directory) and this
  whole campaign is about making the built collection copyable.

`--swap` is a separate, explicit step: two renames plus a reload of the running
server, with the pre-alias collection kept as `<name>-prealias-<date>`.

The temp name is not in main/privacy/scope.json's collection list, and does not
need to be: the scope also matches on the reader's basePath, which the temp build
shares with the real collection. That is what arms aliasing for `<name>-aliased`.
"""
import argparse
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
COLLECTIONS_DIR = REPO_ROOT / "data" / "collections"


def load_manifest(name: str) -> dict:
    path = COLLECTIONS_DIR / name / "manifest.json"
    if not path.exists():
        sys.exit(f"No such collection: {name} ({path} missing)")
    return json.loads(path.read_text(encoding="utf-8"))


def relative_base_path(base_path: str) -> str:
    """Absolute paths inside the repo become './…'; anything else is left alone."""
    path = Path(base_path)
    if not path.is_absolute():
        return base_path
    try:
        return "./" + str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return base_path


def build(name: str, temp_name: str, workers: int) -> None:
    manifest = load_manifest(name)
    reader = manifest["reader"]
    if reader.get("type") != "localFiles":
        sys.exit(f"{name}: only localFiles collections can be rebuilt this way (got {reader.get('type')})")

    contextual_model = (manifest.get("contextualPrefix") or {}).get("model", "none")
    cmd = [
        sys.executable, "files_collection_create_cmd_adapter.py",
        "-collection", temp_name,
        "-basePath", relative_base_path(reader["basePath"]),
        "-includePatterns", *reader.get("includePatterns", [".*"]),
        "-indexers", *[i["name"] for i in manifest["indexers"]],
        "--contextual-model", contextual_model,
        # The real collection's cache, not the temp name's: unchanged chunks must
        # keep hitting, or a rebuild re-generates thousands of prefixes.
        "--contextual-cache", f"./data/contextual_caches/{name}.json",
        "--contextual-workers", str(workers),
    ]
    exclude_patterns = reader.get("excludePatterns", [])
    if exclude_patterns:
        cmd += ["-excludePatterns", *exclude_patterns]
    if reader.get("failFast"):
        cmd.append("-failFast")

    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    # The temp collection records its own name; the swap makes it the real one.
    temp_manifest_path = COLLECTIONS_DIR / temp_name / "manifest.json"
    temp_manifest = json.loads(temp_manifest_path.read_text(encoding="utf-8"))
    temp_manifest["collectionName"] = name
    temp_manifest_path.write_text(json.dumps(temp_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{temp_name}: {temp_manifest['numberOfDocuments']} docs, "
          f"{temp_manifest['numberOfChunks']} chunks, privacy={temp_manifest.get('privacy')}")


def swap(name: str, temp_name: str, api_base: str) -> None:
    live = COLLECTIONS_DIR / name
    aliased = COLLECTIONS_DIR / temp_name
    if not aliased.exists():
        sys.exit(f"Nothing to swap: {aliased} does not exist. Build first.")
    parked = COLLECTIONS_DIR / f"{name}-prealias-{date.today().isoformat()}"
    if parked.exists():
        sys.exit(f"Refusing to overwrite an existing parked collection: {parked}")

    shutil.move(str(live), str(parked))
    shutil.move(str(aliased), str(live))
    print(f"swapped: {name} -> {parked.name}, {temp_name} -> {name}")

    request = urllib.request.Request(f"{api_base}/api/collections/{name}/reload", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print("reload:", response.read().decode())
    except (urllib.error.URLError, OSError) as e:
        print(f"reload failed ({e}); restart the API server to pick up {name}")


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
