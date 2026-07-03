#!/usr/bin/env python3
"""Report which previously-excluded ("trapped") Jira issues were recovered by a
wide re-fetch, using a snapshot of the exclude manifest taken BEFORE recovery.

Context: content-based (low_word_count/empty_stub/minimal_content) and age-based
(too_old) exclusions are dynamic — a stub can grow into a real issue and an old
issue can be revived. Before the fetcher's exclude-manifest handling was fixed
to treat only stable noise_* reasons as permanent skips, such issues were
trapped in .excluded/ forever. This script quantifies the recovery.

An issue is "recovered" if it was in the pre-recovery exclude manifest but now
lives as a real .md file in the saveMd root (cleanup did NOT re-exclude it this
run — it grew into a substantial issue, or an old issue was revived).

Typical use (after a wide re-fetch + cleanup + reindex):
    python scripts/jira/report_recovered_exclusions.py \
        --snapshot logs/exclude_manifest_snapshot_pre_recovery.json \
        --saveMd data/sources/jira-issues \
        --csv logs/recovered_exclusions.csv

Take the snapshot BEFORE recovery:
    cp data/sources/jira-issues/.excluded/excluded_manifest.json \
       logs/exclude_manifest_snapshot_pre_recovery.json
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Put the repo root on sys.path so `main.*` imports resolve when run from anywhere.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from main.utils.frontmatter import read_frontmatter_from_path as _read_fm
except Exception:  # pragma: no cover - standalone fallback
    def _read_fm(path):
        meta = {}
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return meta
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            return meta
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip().strip('"')
        return meta


def _bucket(reason: str) -> str:
    r = (reason or "").lower()
    if any(k in r for k in ("low_word_count", "empty_stub", "minimal_content")):
        return "content"
    if "too_old" in r or "last updated" in r:
        return "age"
    if r.startswith("noise_"):
        return "noise"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True,
                    help="Pre-recovery exclude_manifest.json snapshot")
    ap.add_argument("--saveMd", required=True, help="Jira sources directory")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    with open(args.snapshot, encoding="utf-8") as f:
        baseline = json.load(f)

    save_md = Path(args.saveMd)
    excluded_dir = save_md / ".excluded"

    # Map issue_key -> current .md path in the saveMd ROOT (kept, indexed files).
    kept_by_key = {}
    for md in save_md.glob("*.md"):
        meta = _read_fm(md)
        key = meta.get("issue_key")
        if key:
            kept_by_key[key] = (md, meta)

    recovered, still_excluded = [], []
    bucket_recovered = {"content": 0, "age": 0, "noise": 0, "other": 0}
    bucket_total = {"content": 0, "age": 0, "noise": 0, "other": 0}

    for entry in baseline:
        key = entry.get("issue_key")
        if not key:
            continue
        b = _bucket(entry.get("reason", ""))
        bucket_total[b] += 1
        if key in kept_by_key:
            _, meta = kept_by_key[key]
            recovered.append({
                "issue_key": key,
                "bucket": b,
                "old_reason": entry.get("reason", ""),
                "old_status": entry.get("status", ""),
                "new_status": meta.get("status", ""),
                "new_epic": meta.get("epic_link", ""),
                "new_summary": meta.get("summary", "") or meta.get("title", ""),
            })
            bucket_recovered[b] += 1
        else:
            still_excluded.append(key)

    recovered.sort(key=lambda r: (r["bucket"], r["issue_key"]))

    print("=" * 78)
    print(f"RECOVERY REPORT  (baseline excluded: {len(baseline)})")
    print("=" * 78)
    print(f"Recovered (now kept & indexed): {len(recovered)}")
    for b in ("content", "age", "noise", "other"):
        if bucket_total[b]:
            print(f"    {b:8s}: {bucket_recovered[b]:5d} recovered / {bucket_total[b]:5d} excluded")
    print(f"Still excluded / not re-fetched: {len(still_excluded)}")
    print("-" * 78)
    print(f"{'ISSUE':14s} {'BUCKET':8s} {'OLD STATUS':22s} {'NEW STATUS':22s} EPIC")
    print("-" * 78)
    for r in recovered:
        print(f"{r['issue_key']:14s} {r['bucket']:8s} "
              f"{(r['old_status'] or '-')[:21]:22s} "
              f"{(r['new_status'] or '-')[:21]:22s} {r['new_epic']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "issue_key", "bucket", "old_reason", "old_status",
                "new_status", "new_epic", "new_summary"])
            w.writeheader()
            w.writerows(recovered)
        print("-" * 78)
        print(f"CSV written: {args.csv}  ({len(recovered)} rows)")


if __name__ == "__main__":
    main()
