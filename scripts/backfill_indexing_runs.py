#!/usr/bin/env python3
"""One-off backfill of the indexing run ledger from the daily job logs.

The launchd jobs write per-run text logs to ``logs/daily_*.log``. Those logs are
pruned at 30 days, so this recovers roughly three weeks of history the ledger
would otherwise start without. Records are stamped ``backfilled: true`` and carry
whole-job durations only — the per-phase split (tagging vs reindex) is not
recoverable from these logs, which is precisely what PR 2 adds going forward.

Design notes worth keeping:

* Markers are NOT uniform — ten distinct start-marker forms exist across the
  families, so this keys on MARKER TEXT rather than on the log filename family.
  That also handles the three ``daily_update_2026-05-12_*.log`` files that
  survived from two months before the retention window: every prune pattern is
  family-scoped (``-name 'daily_wiki_*.log'`` and friends) and none of them
  matches that filename shape. Keying on marker text picks them up as the Notion
  runs they are, with no filename special-case.
* Three families (Jira, Confluence, Notion) carry no collection name in the
  marker and have no installed plist, so they need the explicit map below.
* The wiki marker covers two collections and is split into two records.
* Unterminated runs are real (one family has 22 starts vs 21 finishes). They are
  recorded with status "unknown", no finishedAt and no duration, rather than
  guessed at or dropped.
* trigger is "unknown", never "scheduled": the logs contain obvious manual runs
  (one collection has 18:12, 21:38 and 22:20 starts alongside the scheduled
  09:15), and labelling those as scheduled would corrupt the schedule-drift
  signal the dashboard exists to show.
* x-feed is excluded: it appends to one 322MB ``logs/x_feed_update.log`` with no
  per-run files and no start/finish markers, so its runs are not delimited.
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.runtime.indexing_run_ledger import (  # noqa: E402
    IndexingRunLedger,
    duration_seconds,
    to_iso_z,
)

TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

START_RE = re.compile(r"===\s*(?P<label>.+?)\s+update started(?:\s*\((?P<args>[^)]*)\))?\s*===")
FINISH_RE = re.compile(r"===\s*(?P<label>.+?)\s+update finished")
FAILED_RE = re.compile(r"===\s*FAILED:")

# Marker labels that name no collection and have no installed plist to consult.
# These three collection names are deliberately spelled out: all three appear in
# this repo's public CLAUDE.md "Common collections" table, so they carry no
# private information (unlike the scheduled names the schedule module keeps out
# of the public repo).
LABEL_COLLECTIONS = {
    "Daily Jira": ["jira-issues"],
    "Daily Confluence": ["melosys-confluence-v3"],
    # start.sh currently serves "capra-notion", but the fetch script writes
    # "capra-notion-v9" — a backfill record describes what the script did.
    "Daily Notion": ["capra-notion-v9"],
}


def _collections_for(label, args):
    """Collections a start marker refers to.

    ``args`` is the parenthesised part: ``collection=wiki`` or
    ``collections=wiki wiki-life``. The plural form is split into one record per
    collection.
    """
    if args:
        for key in ("collections=", "collection="):
            if key in args:
                value = args.split(key, 1)[1].strip()
                return [c for c in value.split() if c]
    return LABEL_COLLECTIONS.get(label.strip(), [])


def _parse_time(line):
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    naive = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return naive.replace(tzinfo=timezone.utc)


def parse_log(path):
    """Runs found in one log file. A file may contain several."""
    runs = []
    open_runs = []  # (collections, started_at)
    failed = False
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return runs

    for line in lines:
        moment = _parse_time(line)
        if FAILED_RE.search(line):
            failed = True
        start = START_RE.search(line)
        if start and moment:
            collections = _collections_for(start.group("label"), start.group("args"))
            if collections:
                open_runs.append((collections, moment))
            continue
        finish = FINISH_RE.search(line)
        if finish and moment and open_runs:
            collections, started_at = open_runs.pop(0)
            for collection in collections:
                runs.append(_record(collection, started_at, moment,
                                    "failed" if failed else "succeeded", path))
            failed = False

    for collections, started_at in open_runs:
        # Started, never finished — killed, machine slept, or (the single real
        # case in the current logs) the script bailed out on a critical error and
        # printed "=== FAILED:" instead of a finish marker. A seen FAILED marker
        # is positive evidence, so those are recorded as failed rather than
        # unknown; finishedAt still stays null because the job printed no end
        # timestamp to trust.
        status = "failed" if failed else "unknown"
        for collection in collections:
            runs.append(_record(collection, started_at, None, status, path))
    return runs


def _record(collection, started_at, finished_at, status, source_path):
    started_iso = to_iso_z(started_at)
    finished_iso = to_iso_z(finished_at)
    return {
        "runId": f"backfill-{collection}-{started_iso}",
        "collection": collection,
        "job": None,
        "trigger": "unknown",
        "variant": "incremental",
        "startedAt": started_iso,
        "finishedAt": finished_iso,
        "durationSeconds": duration_seconds(started_iso, finished_iso),
        "status": status,
        "phases": [],
        "documentCount": None,
        "chunkCount": None,
        "error": None,
        "backfilled": True,
        "source": "backfill",
        "sourceLog": os.path.basename(source_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="./logs")
    parser.add_argument("--runs-dir", default="./data/state/runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print a summary without writing to the ledger")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.logs_dir, "daily_*.log")))
    runs = []
    for path in paths:
        runs.extend(parse_log(path))
    runs.sort(key=lambda r: r["startedAt"])

    summary = {}
    for run in runs:
        bucket = summary.setdefault(run["collection"], {"runs": 0, "unknown": 0, "failed": 0})
        bucket["runs"] += 1
        if run["status"] == "unknown":
            bucket["unknown"] += 1
        elif run["status"] == "failed":
            bucket["failed"] += 1

    if not args.dry_run:
        ledger = IndexingRunLedger(runs_dir=args.runs_dir)
        existing = {}
        for collection in set(r["collection"] for r in runs):
            existing[collection] = {
                r.get("runId") for r in ledger.recent(collection, limit=None)
            }
        written = 0
        for run in runs:
            # Re-running the backfill must not duplicate history.
            if run["runId"] in existing.get(run["collection"], set()):
                continue
            ledger.append(run)
            written += 1
        print(f"Wrote {written} new records ({len(runs)} parsed) to {args.runs_dir}")
    else:
        print(f"Parsed {len(runs)} runs from {len(paths)} log files (dry run)")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
