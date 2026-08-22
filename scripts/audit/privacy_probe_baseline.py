#!/usr/bin/env python3
"""Capture the PR-acceptance baseline for build-time aliasing.

Picks a probe set of mapped people from the private alias map and records, per
probe, the top-N document ids the LIVE (pre-alias) index returns for their real
name. After the swap the same probe run with the *alias* must return the same
documents — that is the campaign's named acceptance test, and it cannot be
checked without a before-picture.

Real names are read from the gitignored map at runtime and written to the
gitignored output file; nothing about a real person is ever printed to stdout or
stored in this repo.

    .venv/bin/python scripts/audit/privacy_probe_baseline.py     # writes beside the discovered map
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_GLOB = "huginn-*/privacy/aliases.json"
CORPUS_CACHE = "huginn-*/privacy/.sources_cache.txt"


def word_count(text: str, needle: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text, re.I))


def pick_probes(map_data: dict, corpus: str, size: int) -> list[dict]:
    """Deterministic probe set: corpus-attested entries spread over the roles.

    Guarantees at least one `require_full_name` entry and one whose corpus
    attestation is the deactivated-account `Name [X]` form, because those are the
    two shapes most likely to be missed by a naive substituter.
    """
    scored = []
    for entry in map_data["entries"]:
        hits = word_count(corpus, entry["name"])
        if not hits:
            continue
        bracket = word_count(corpus, f"{entry['name']} [X]")
        scored.append({"alias": entry["alias"], "name": entry["name"], "role": entry["role"],
                       "hits": hits, "bracketHits": bracket,
                       "requireFullName": entry.get("require_full_name", False)})
    scored.sort(key=lambda p: (-p["hits"], p["alias"]))

    picked, by_role, chosen = [], {}, set()

    def take(probe):
        if probe["alias"] not in chosen:
            chosen.add(probe["alias"])
            picked.append(probe)

    for probe in scored:                       # round 1: spread over roles
        if len(by_role.get(probe["role"], [])) < max(1, size // 3):
            by_role.setdefault(probe["role"], []).append(probe)
            take(probe)
    for probe in scored:                       # round 2: fill up
        if len(picked) >= size:
            break
        take(probe)

    # Both required shapes are collected BEFORE truncating, and the truncation
    # happens once: appending them one at a time cut the first one back off,
    # which silently produced a baseline with no require_full_name probe at all.
    required = []
    for requirement in (lambda p: p["requireFullName"], lambda p: p["bracketHits"] > 0):
        if any(requirement(p) for p in picked) or any(requirement(p) for p in required):
            continue
        extra = next((p for p in scored if requirement(p) and p["alias"] not in chosen), None)
        if extra:
            chosen.add(extra["alias"])
            required.append(extra)
    if required:
        picked = picked[:max(0, size - len(required))] + required
    return picked[:size]


def search(api_base: str, collection: str, query: str, limit: int) -> list[str]:
    url = (f"{api_base}/api/search?" + urllib.parse.urlencode(
        {"q": query, "collection": collection, "limit": limit, "brief": "true"}))
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode())
    return [r.get("id") or r.get("documentId") or r.get("url") for r in payload.get("results", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Output path (default: probe_baseline.json beside the discovered alias map)")
    ap.add_argument("--collections", nargs="+",
                    default=["melosys-confluence-v3", "jira-issues", "nav-wiki"])
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--api-base", default="http://127.0.0.1:8321")
    args = ap.parse_args()

    map_paths = sorted(REPO_ROOT.glob(MAP_GLOB))
    if not map_paths:
        sys.exit(f"No alias map found ({MAP_GLOB})")
    map_data = json.loads(map_paths[0].read_text(encoding="utf-8"))
    out_path = Path(args.out) if args.out else map_paths[0].with_name("probe_baseline.json")

    corpus_paths = sorted(REPO_ROOT.glob(CORPUS_CACHE))
    if not corpus_paths:
        sys.exit(f"No corpus cache ({CORPUS_CACHE}); run the private map lint first to populate it")
    corpus = corpus_paths[0].read_text(encoding="utf-8", errors="ignore")

    probes = pick_probes(map_data, corpus, args.probes)
    print(f"probe set ({len(probes)}): " + ", ".join(
        f"{p['alias']}/{p['role']}{'*' if p['requireFullName'] else ''}"
        f"{'[X]' if p['bracketHits'] else ''}" for p in probes))

    out = {"apiBase": args.api_base, "limit": args.limit, "probes": []}
    for probe in probes:
        record = {k: probe[k] for k in ("alias", "name", "role", "hits", "bracketHits", "requireFullName")}
        record["baseline"] = {}
        for collection in args.collections:
            ids = search(args.api_base, collection, probe["name"], args.limit)
            record["baseline"][collection] = ids
        out["probes"].append(record)
        print(f"  {probe['alias']}: " + " ".join(
            f"{c}={len(record['baseline'][c])}" for c in args.collections))

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
