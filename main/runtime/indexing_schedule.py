"""launchd schedule discovery for the indexing jobs.

Reads installed ``~/Library/LaunchAgents/com.huginn.*.plist`` files and maps each
job to the collections it indexes. Every failure mode here degrades to ``None``;
the jobs endpoint must never fail because a plist is missing or malformed.

The job → collection mapping is an explicit table keyed on the script BASENAME,
not derived from the plist. Installed plists carry only ``/bin/bash <script>`` —
no collection appears anywhere in them — and the scripts themselves declare their
collections in three different shell syntaxes (``COLLECTION="mimir"``,
``COLLECTIONS=("wiki" "wiki-life")``, ``COLLECTIONS="${COLLECTIONS:-jira-issues}"``).
Parsing that is more fragile than a ten-line table. Basename-keyed because the
script directories are private sub-repos and gitignored.

The table itself lives in those private sub-repos, not here, mirroring the
``graph_routing.json`` precedent: each carries a ``scripts/schedule_routing.json``
mapping script basename → collections. Five of the seven scheduled collection
names were never public, and one of them is customer-adjacent, which this repo's
own conventions ban outright. Absent routing files ⇒ ``schedule: null``, which is
the designed degradation.
"""
import glob
import json
import logging
import os
import plistlib

logger = logging.getLogger(__name__)

LAUNCH_AGENTS_GLOB = os.path.expanduser("~/Library/LaunchAgents/com.huginn.*.plist")

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUTING_GLOBS = (
    os.path.join(_HERE, "huginn-*", "scripts", "schedule_routing.json"),
    os.path.join(_HERE, "scripts", "schedule_routing.json"),
)

# Intentionally empty in the public repo — see the module docstring. A private
# deployment supplies its own names through schedule_routing.json.
SCRIPT_COLLECTIONS = {}


def load_script_collections(globs=ROUTING_GLOBS):
    """script basename -> collections, merged across the private routing files.

    Best effort in the same sense as everything else here: an unreadable or
    malformed routing file costs that file's schedules, never the endpoint.
    """
    mapping = dict(SCRIPT_COLLECTIONS)
    for pattern in globs:
        try:
            paths = sorted(glob.glob(pattern))
        except OSError:
            continue
        for path in paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                logger.debug("Could not read schedule routing %s", path, exc_info=True)
                continue
            entries = data.get("scriptCollections")
            if not isinstance(entries, dict):
                continue
            for script, collections in entries.items():
                if isinstance(collections, list) and collections:
                    mapping[script] = collections
    return mapping


def _schedule_from_plist(data):
    calendar = data.get("StartCalendarInterval")
    if isinstance(calendar, dict):
        return {
            "kind": "calendar",
            "hour": calendar.get("Hour"),
            "minute": calendar.get("Minute"),
            "weekday": calendar.get("Weekday"),
        }
    if isinstance(calendar, list) and calendar:
        first = calendar[0]
        hours = {e.get("Hour") for e in calendar if isinstance(e, dict)}
        minutes = {e.get("Minute") for e in calendar if isinstance(e, dict)}
        # x-feed is 24 entries at the same minute — hourly, expressed as a
        # calendar array because StartInterval is load-time-relative and cannot
        # be pinned to an offset. Reporting only calendar[0] would render that
        # as "daily at 00:35", understating the cadence 24-fold and making its
        # "next run" wrong for 23 hours out of every 24.
        if len(minutes) == 1 and hours == set(range(24)):
            return {"kind": "hourly", "minute": minutes.pop()}
        return {
            "kind": "calendar",
            "hour": first.get("Hour"),
            "minute": first.get("Minute"),
            "weekday": first.get("Weekday"),
            "entries": len(calendar),
        }
    interval = data.get("StartInterval")
    if isinstance(interval, int):
        # No installed job uses this yet; supported so an interval-scheduled job
        # gets a schedule instead of a null. "Next run" for these derives from
        # lastRun.finishedAt + seconds rather than a wall-clock time.
        return {"kind": "interval", "seconds": interval}
    return None


# Cache keyed on an MTIME SIGNATURE, not a TTL. The GET /api/indexing/jobs
# endpoint calls this per request and a dashboard polls it every 15s, sharing a
# process with search; each uncached call globs two routing patterns and JSON-
# parses every match on top of globbing LaunchAgents and plistlib-loading every
# plist. A stat-per-file is far cheaper than open-and-parse, and unlike a TTL it
# invalidates the instant a routing file or plist actually changes — so a freshly
# installed plist shows up at once, with no staleness window to explain away.
_SCHEDULE_CACHE = {}


def _mtime_signature(patterns):
    """Sorted ``(path, mtime_ns)`` over every file matching any pattern.

    Changes when any matched file is touched, and — because the matched SET is
    part of the tuple — when a file is added or removed. A file that vanishes
    between glob and stat simply drops out, same as it would from the parse.
    """
    signature = []
    for pattern in patterns:
        try:
            paths = glob.glob(pattern)
        except OSError:
            continue
        for path in paths:
            try:
                signature.append((path, os.stat(path).st_mtime_ns))
            except OSError:
                continue
    return tuple(sorted(signature))


def load_schedules(pattern=LAUNCH_AGENTS_GLOB, routing_globs=ROUTING_GLOBS):
    """collection name -> {job, schedule}, best effort, cached on an mtime signature.

    Returns an empty mapping rather than raising if the LaunchAgents directory is
    unreadable or every plist is malformed. The result is cached and only re-read
    when a plist or routing file changes (see ``_mtime_signature``).
    """
    cache_key = (pattern, tuple(routing_globs))
    signature = _mtime_signature((pattern, *routing_globs))
    cached = _SCHEDULE_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    result = _load_schedules_uncached(pattern, routing_globs)
    _SCHEDULE_CACHE[cache_key] = (signature, result)
    return result


def _load_schedules_uncached(pattern, routing_globs):
    schedules = {}
    script_collections = load_script_collections(globs=routing_globs)
    try:
        paths = sorted(glob.glob(pattern))
    except OSError:
        return schedules

    for path in paths:
        try:
            with open(path, "rb") as handle:
                data = plistlib.load(handle)
        except Exception:
            logger.debug("Could not read launchd plist %s", path, exc_info=True)
            continue

        arguments = data.get("ProgramArguments") or []
        script = next(
            (os.path.basename(a) for a in arguments
             if isinstance(a, str) and a.endswith(".sh")),
            None,
        )
        collections = script_collections.get(script)
        if not collections:
            continue

        entry = {
            "job": data.get("Label") or os.path.basename(path),
            "schedule": _schedule_from_plist(data),
        }
        for collection in collections:
            schedules[collection] = entry
    return schedules
